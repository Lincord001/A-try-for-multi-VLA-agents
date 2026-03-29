import threading
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class PerformanceMonitor:
    """性能监控工具，记录各步骤的执行时间。"""

    def __init__(self, window_size=50):
        self.window_size = window_size
        self.main_thread_times = {
            "grab_image": [],
            "data_copy": [],
            "get_action": [],
            "step_env": [],
            "render": [],
            "total_loop": [],
        }
        self.inference_thread_times = {
            "lock_read": [],
            "img_resize": [],
            "img_totensor": [],
            "img_todevice": [],
            "normalize_inputs": [],
            "prepare_images": [],
            "prepare_state": [],
            "prepare_language": [],
            "sample_actions": [],
            "postprocess": [],
            "total_inference": [],
        }
        self.lock = threading.Lock()

    def record_main(self, stage, duration):
        """记录主线程各阶段耗时。"""
        with self.lock:
            if stage in self.main_thread_times:
                self.main_thread_times[stage].append(duration)
                if len(self.main_thread_times[stage]) > self.window_size:
                    self.main_thread_times[stage].pop(0)

    def record_inference(self, stage, duration):
        """记录推理线程各阶段耗时。"""
        with self.lock:
            if stage in self.inference_thread_times:
                self.inference_thread_times[stage].append(duration)
                if len(self.inference_thread_times[stage]) > self.window_size:
                    self.inference_thread_times[stage].pop(0)

    def get_stats(self, stage, times_list):
        """获取统计信息。"""
        if len(times_list) == 0:
            return 0.0, 0.0, 0.0
        arr = np.array(times_list)
        return np.mean(arr), np.min(arr), np.max(arr)

    def print_stats(self, step, mode="arm"):
        """打印统计信息。"""
        with self.lock:
            print(f"\n{'=' * 80}")
            print(f"📊 Performance Stats (Step {step}, Mode: {mode.upper()})")
            print(f"{'=' * 80}")

            print("\n[主线程] 时间统计 (ms):")
            for stage, times in self.main_thread_times.items():
                mean, min_val, max_val = self.get_stats(stage, times)
                count = len(times)
                if count > 0:
                    print(
                        f"  {stage:15s}: 平均={mean * 1000:6.2f} "
                        f"最小={min_val * 1000:6.2f} 最大={max_val * 1000:6.2f} (样本={count})"
                    )

            print("\n[推理线程] 时间统计 (ms):")
            for stage, times in self.inference_thread_times.items():
                mean, min_val, max_val = self.get_stats(stage, times)
                count = len(times)
                if count > 0:
                    print(
                        f"  {stage:15s}: 平均={mean * 1000:6.2f} "
                        f"最小={min_val * 1000:6.2f} 最大={max_val * 1000:6.2f} (样本={count})"
                    )

            total_inference = self.inference_thread_times.get("total_inference", [])
            if len(total_inference) > 0:
                mean_total = np.mean(total_inference)
                print(f"\n🎯 推理线程总耗时: {mean_total * 1000:.2f}ms (目标: <250ms)")
            print(f"{'=' * 80}\n")

    def reset(self):
        """重置统计信息。"""
        with self.lock:
            for key in self.main_thread_times:
                self.main_thread_times[key] = []
            for key in self.inference_thread_times:
                self.inference_thread_times[key] = []


class ActionSmoother:
    """
    动作平滑器 - 使用指数移动平均(EMA)减少颤抖。

    功能：
    1. 关节角度平滑：减少模型输出的高频噪声
    2. 夹爪迟滞控制：防止夹爪在阈值附近频繁切换
    """

    def __init__(
        self,
        joint_dim=6,
        alpha_joints=0.4,
        alpha_gripper=0.3,
        smoothing_enabled=True,
        gripper_hysteresis_enabled=True,
        gripper_open_thresh=0.7,
        gripper_close_thresh=0.25,
    ):
        self.joint_dim = joint_dim
        self.alpha_joints = alpha_joints
        self.alpha_gripper = alpha_gripper
        self.smoothing_enabled = smoothing_enabled
        self.gripper_hysteresis_enabled = gripper_hysteresis_enabled
        self.gripper_open_thresh = gripper_open_thresh
        self.gripper_close_thresh = gripper_close_thresh

        # 历史状态
        self.last_joint_angles = None
        self.last_gripper_cmd = None
        self.gripper_state = False  # True=打开, False=关闭

    def reset(self):
        """重置平滑器状态（环境重置时调用）。"""
        self.last_joint_angles = None
        self.last_gripper_cmd = None
        self.gripper_state = False

    def prime_from_state(self, current_state):
        """用当前真实机器人状态对齐平滑器，避免恢复接管时第一帧跳变。"""
        if current_state is None:
            self.reset()
            return
        state = np.asarray(current_state, dtype=np.float64).reshape(-1)
        if state.shape[0] < self.joint_dim:
            self.reset()
            return
        self.last_joint_angles = state[: self.joint_dim].copy()
        if state.shape[0] > self.joint_dim:
            gripper_cmd = float(state[self.joint_dim])
            self.last_gripper_cmd = gripper_cmd
            if self.gripper_hysteresis_enabled:
                if gripper_cmd > self.gripper_open_thresh:
                    self.gripper_state = True
                elif gripper_cmd < self.gripper_close_thresh:
                    self.gripper_state = False
            else:
                self.gripper_state = gripper_cmd > 0.5

    def smooth_action(self, raw_action):
        """
        平滑处理动作。

        Args:
            raw_action: 原始动作 (7,) - [6关节角度, 1夹爪]

        Returns:
            smoothed_action: 平滑后的动作 (7,)
            gripper_state: 夹爪状态 (bool)
        """
        if raw_action is None:
            return None, self.gripper_state

        joint_angles = raw_action[: self.joint_dim].copy()
        gripper_cmd = raw_action[self.joint_dim] if len(raw_action) > self.joint_dim else 0.0

        # ========== EMA 平滑关节角度 ==========
        if self.smoothing_enabled:
            if self.last_joint_angles is not None:
                joint_angles = (
                    self.alpha_joints * joint_angles
                    + (1 - self.alpha_joints) * self.last_joint_angles
                )
            self.last_joint_angles = joint_angles.copy()
        else:
            self.last_joint_angles = joint_angles.copy()

        # ========== EMA 平滑夹爪 + 迟滞控制 ==========
        if self.smoothing_enabled:
            if self.last_gripper_cmd is not None:
                gripper_cmd = (
                    self.alpha_gripper * gripper_cmd
                    + (1 - self.alpha_gripper) * self.last_gripper_cmd
                )
            self.last_gripper_cmd = gripper_cmd
        else:
            self.last_gripper_cmd = gripper_cmd

        if self.gripper_hysteresis_enabled:
            if gripper_cmd > self.gripper_open_thresh:
                self.gripper_state = True
            elif gripper_cmd < self.gripper_close_thresh:
                self.gripper_state = False
        else:
            self.gripper_state = gripper_cmd > 0.5

        smoothed_action = np.concatenate([joint_angles, [gripper_cmd]])
        return smoothed_action, self.gripper_state


class AsyncInferenceRunner:
    """通用异步推理运行器（ARM/BASE 参数化）。"""

    def __init__(
        self,
        policy,
        device,
        img_transform,
        control_dt,
        camera_keys,
        mode_label,
        mode_emoji,
        config,
        action_horizon,
        chunk_threshold,
        perf_monitor=None,
    ):
        self.policy = policy
        self.device = device
        self.img_transform = img_transform
        self.control_dt = control_dt
        self.camera_keys = tuple(camera_keys)
        self.mode_label = mode_label
        self.mode_emoji = mode_emoji
        self.action_horizon = action_horizon
        self.chunk_threshold = chunk_threshold
        self.perf_monitor = perf_monitor
        self.image_size = config["image_size"]

        self.lock = threading.Lock()

        self.latest_raw_images = None
        self.latest_state = None
        self.latest_task = None
        self.latest_obs_timestamp = 0

        self.latest_action_chunk = None
        self.chunk_start_timestamp = 0

        self.current_step_index = 0
        self.chunk_id = 0
        self.last_processed_timestamp = 0

        self.running = False
        self.thread = None

    def start(self):
        if self.running and self.thread and self.thread.is_alive():
            print(f"⚠️ [{self.mode_label} AsyncRunner] 发现旧线程仍在运行，先停止它")
            self.stop()

        self.reset_state()

        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print(f"{self.mode_emoji} [{self.mode_label} AsyncRunner] 推理线程已启动 (状态已重置)")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        print(f"🛑 [{self.mode_label} AsyncRunner] 推理线程已停止")

    def reset_state(self):
        """重置所有状态（不停止线程）- 可在环境重置时调用。"""
        with self.lock:
            self.latest_raw_images = None
            self.latest_state = None
            self.latest_task = None
            self.latest_obs_timestamp = 0
            self.latest_action_chunk = None
            self.chunk_start_timestamp = 0
            self.current_step_index = 0
            self.chunk_id = 0
            self.last_processed_timestamp = 0

    def update_observation(self, images_dict, state, task, timestamp):
        """主线程调用：更新最新的观测数据。"""
        t0 = time.perf_counter()

        images_copy = {}
        for key, value in images_dict.items():
            if isinstance(value, np.ndarray):
                images_copy[key] = value.copy()

        state_copy = np.array(state, copy=True) if state is not None else None
        task_copy = task.copy() if isinstance(task, list) else task

        t1 = time.perf_counter()
        if self.perf_monitor:
            self.perf_monitor.record_main("data_copy", t1 - t0)

        with self.lock:
            self.latest_raw_images = images_copy
            self.latest_state = state_copy
            self.latest_task = task_copy
            self.latest_obs_timestamp = timestamp

    def get_action_at_time(self, current_time):
        """
        主线程调用：顺序获取下一个动作（不使用延迟补偿）。

        返回: (action, status_msg)
        """
        with self.lock:
            if self.latest_action_chunk is None:
                return None, "Wait for init"

            chunk = self.latest_action_chunk
            chunk_len = chunk.shape[0]
            step_index = self.current_step_index
            effective_horizon = min(self.action_horizon, chunk_len)

            if step_index < effective_horizon:
                action = chunk[step_index]
                self.current_step_index += 1
                return action, f"OK (Step {step_index}/{effective_horizon})"

            last_idx = effective_horizon - 1 if effective_horizon > 0 else 0
            return chunk[last_idx], f"Hold (Horizon {effective_horizon} reached)"

    def _inference_loop(self):
        """后台线程：执行推理。"""
        while self.running:
            t_lock_start = time.perf_counter()
            raw_images = None
            state = None
            task = None
            obs_timestamp = 0

            with self.lock:
                t_lock_read = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference("lock_read", t_lock_read - t_lock_start)

                if self.latest_raw_images is not None:
                    need_inference = False
                    if self.latest_action_chunk is None:
                        need_inference = True
                    else:
                        chunk_len = self.latest_action_chunk.shape[0]
                        effective_horizon = min(self.action_horizon, chunk_len)
                        remaining_steps = effective_horizon - self.current_step_index
                        if remaining_steps <= self.chunk_threshold:
                            need_inference = True

                    if need_inference and self.latest_obs_timestamp > self.last_processed_timestamp:
                        raw_images = self.latest_raw_images
                        state = self.latest_state
                        task = self.latest_task
                        obs_timestamp = self.latest_obs_timestamp
                        self.last_processed_timestamp = obs_timestamp
                    else:
                        raw_images = None

            if raw_images is None:
                time.sleep(0.001)
                continue

            t_inference_start = time.perf_counter()
            try:
                t0 = time.perf_counter()
                resized_images = {
                    key: Image.fromarray(raw_images[key]).resize(
                        (self.image_size, self.image_size),
                        resample=Image.BILINEAR,
                    )
                    for key in self.camera_keys
                }
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference("img_resize", t1 - t0)

                t0 = time.perf_counter()
                image_tensors = {
                    key: self.img_transform(img).unsqueeze(0) for key, img in resized_images.items()
                }
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference("img_totensor", t1 - t0)

                t0 = time.perf_counter()
                image_tensors = {key: tensor.to(self.device) for key, tensor in image_tensors.items()}
                state_tensor = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference("img_todevice", t1 - t0)

                batch = {
                    "observation.state": state_tensor,
                    "task": task,
                }
                for key, tensor in image_tensors.items():
                    batch[f"observation.images.{key}"] = tensor

                with torch.no_grad():
                    t0 = time.perf_counter()
                    batch = self.policy.normalize_inputs(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference("normalize_inputs", t1 - t0)

                    t0 = time.perf_counter()
                    images, img_masks = self.policy.prepare_images(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference("prepare_images", t1 - t0)

                    t0 = time.perf_counter()
                    state_processed = self.policy.prepare_state(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference("prepare_state", t1 - t0)

                    t0 = time.perf_counter()
                    lang_tokens, lang_masks = self.policy.prepare_language(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference("prepare_language", t1 - t0)

                    t0 = time.perf_counter()
                    actions = self.policy.model.sample_actions(
                        images, img_masks, lang_tokens, lang_masks, state_processed
                    )
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference("sample_actions", t1 - t0)

                    t0 = time.perf_counter()
                    original_action_dim = self.policy.config.action_feature.shape[0]
                    actions = actions[:, :, :original_action_dim]
                    actions = self.policy.unnormalize_outputs({"action": actions})["action"]

                    if self.policy.config.adapt_to_pi_aloha:
                        actions = self.policy._pi_aloha_encode_actions(actions)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference("postprocess", t1 - t0)

                chunk_np = actions[0].cpu().numpy()
                t_inference_end = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference(
                        "total_inference", t_inference_end - t_inference_start
                    )

                with self.lock:
                    self.latest_action_chunk = chunk_np
                    self.chunk_start_timestamp = obs_timestamp
                    self.current_step_index = 0
                    self.chunk_id += 1

            except Exception as e:
                print(f"[{self.mode_label}] Inference Error: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(0.1)


def get_default_transform():
    """返回标准图像变换。"""
    return transforms.Compose([transforms.ToTensor()])
