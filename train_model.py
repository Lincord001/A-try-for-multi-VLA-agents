#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import shutil
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def _build_save_step_set(cfg: TrainPipelineConfig) -> set[int]:
    if not getattr(cfg, "save_by_steps", True):
        return set()
    return set(getattr(cfg, "save_steps", []))


def _get_best_loss_range(cfg: TrainPipelineConfig) -> tuple[int, int] | None:
    if not getattr(cfg, "save_best_in_range", True):
        return None
    loss_range = getattr(cfg, "best_loss_checkpoint_range", [])
    if len(loss_range) != 2:
        return None
    return int(loss_range[0]), int(loss_range[1])


def _format_loss_for_checkpoint_name(loss_value: float) -> str:
    return f"{loss_value:.6f}"


def _get_amp_dtype(device: torch.device) -> torch.dtype:
    # Prefer bf16 on modern NVIDIA GPUs (e.g. A800) for wider numeric range.
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        try:
            wandb_logger = WandBLogger(cfg)
        except Exception as e:
            if "API key" in str(e) or "UsageError" in str(type(e).__name__) or "wandb.errors.errors.UsageError" in str(type(e)):
                logging.warning(
                    colored(
                        f"WandB authentication failed: {e}. Continuing without WandB logging.",
                        "yellow",
                        attrs=["bold"],
                    )
                )
                wandb_logger = None
            else:
                raise
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    amp_dtype = _get_amp_dtype(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    # 只有在 yaml 中没有指定 pretrained_path 时才使用默认值
    # 这样 yaml 中的配置可以覆盖默认值
    pretrained_path_set = getattr(cfg.policy, 'pretrained_path', None)
    if pretrained_path_set is None or pretrained_path_set == '':
        if cfg.policy.type == "pi0":
            cfg.policy.pretrained_path = os.environ.get(
                "PALIGEMMA_PRETRAINED_PATH",
                "google/paligemma-3b-pt-224",
            )
            logging.info(f"Using default pretrained_path for pi0: {cfg.policy.pretrained_path}")
        elif cfg.policy.type == 'smolvla':
            cfg.policy.pretrained_path = 'lerobot/smolvla_base'
            logging.info(f"Using default pretrained_path for smolvla: {cfg.policy.pretrained_path}")
    else:
        logging.info(f"Using pretrained_path from config: {cfg.policy.pretrained_path}")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp and amp_dtype == torch.float16)
    if cfg.policy.use_amp:
        logging.info(f"Using AMP dtype: {amp_dtype}")

    step = 0  # number of policy updates (forward + backward + optim)
    save_by_freq = getattr(cfg, "save_by_freq", True)
    save_steps = _build_save_step_set(cfg)
    best_loss_range = _get_best_loss_range(cfg)
    best_loss_ckpt_dir = None
    best_loss_value = float("inf")
    best_loss_step = None

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    logging.info(
        "Checkpoint logic switches: "
        f"save_by_freq={save_by_freq}, "
        f"save_by_steps={getattr(cfg, 'save_by_steps', True)}, "
        f"save_best_in_range={getattr(cfg, 'save_best_in_range', True)}"
    )
    if save_steps:
        logging.info(f"Extra checkpoint steps from config: {sorted(save_steps)}")
    if best_loss_range is not None:
        logging.info(
            "Best-loss checkpointing enabled for range "
            f"[{best_loss_range[0]}, {best_loss_range[1]}] (keep only one best checkpoint)."
        )

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            amp_dtype=amp_dtype,
        )
        # Capture current-step loss before any metric reset for logging.
        step_loss_value = float(train_tracker.loss.val)

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = (
            (save_by_freq and (step % cfg.save_freq == 0 or step == cfg.steps))
            or step in save_steps
        )
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if cfg.save_checkpoint and best_loss_range is not None and best_loss_range[0] <= step <= best_loss_range[1]:
            current_loss = step_loss_value
            if current_loss == current_loss and current_loss < best_loss_value:
                step_id = get_step_identifier(step, cfg.steps)
                range_label = f"{best_loss_range[0]}_{best_loss_range[1]}"
                loss_label = _format_loss_for_checkpoint_name(current_loss)
                new_best_ckpt_dir = (
                    cfg.output_dir
                    / "checkpoints"
                    / f"best_loss_range_{range_label}_step_{step_id}_loss_{loss_label}"
                )
                save_checkpoint(new_best_ckpt_dir, step, cfg, policy, optimizer, lr_scheduler)

                if best_loss_ckpt_dir is not None and best_loss_ckpt_dir != new_best_ckpt_dir:
                    shutil.rmtree(best_loss_ckpt_dir, ignore_errors=True)

                best_loss_value = current_loss
                best_loss_step = step
                best_loss_ckpt_dir = new_best_ckpt_dir
                logging.info(
                    f"Updated best-loss checkpoint in range at step {step}, loss={current_loss:.6f}, "
                    f"path={new_best_ckpt_dir}"
                )

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}, loss={step_loss_value:.6f}")
            step_id = get_step_identifier(step, cfg.steps)
            loss_label = _format_loss_for_checkpoint_name(step_loss_value)
            checkpoint_dir = cfg.output_dir / "checkpoints" / f"{step_id}_loss_{loss_label}"
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type, dtype=amp_dtype) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    if best_loss_ckpt_dir is not None:
        logging.info(
            f"Final best-loss checkpoint in range: step={best_loss_step}, "
            f"loss={best_loss_value:.6f}, path={best_loss_ckpt_dir}"
        )
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
