"""
control_loop.py
---------------
每帧控制步进函数和自动结果检测函数。

所有函数接受 DeployState 和相关依赖，直接修改 state，不使用 nonlocal。

特殊返回值：
  check_auto_result() 返回 True 表示已达到 TASK_LOOP_COUNT，调用方应 break。
"""

import csv
import time

import numpy as np
import torch
from PIL import Image

from mujoco_env.instruction_utils import (
    apply_instruction_from_group as _apply_instruction_from_group,
)

from .deploy_state import DeployState
from .task_runtime import get_target_cup_init, perform_auto_reset
from .config import (
    TASK_TIMEOUT_SEC,
    TASK_LOOP_COUNT,
    STEP_LOG_PATH,
    ARM_CONFIG,
    ARM_SYNC_INFERENCE,
    ARM_EXEC_HORIZON,
)
from mujoco_env.visualization import (
    append_step_log,
    ensure_step_log_header,
)
from mujoco_env.y_env7 import EXPERT_Y_GRASP_OFFSET


# =============================================================================
# 自动检测成功 + 超时判定
# =============================================================================

def check_auto_result(
    state: DeployState,
    env,
    arm_policy,
    arm_runner,
    arm_smoother,
) -> bool:
    """
    检测当前任务是否成功或超时，处理重置并更新统计数据。
    仅在 state.auto_check_enabled 为 True 时执行。

    Returns:
        True  → 已达到 TASK_LOOP_COUNT，调用方应 break 退出主循环。
        False → 继续执行。
    """
    if not state.auto_check_enabled:
        return False

    elapsed = time.time() - state.task_start_time

    # ---- 自动检测成功 ----
    if env.check_success():
        target_color, cup_init = get_target_cup_init(env)
        append_step_log(
            STEP_LOG_PATH, state.control_mode, 'success',
            state.step - state.task_start_step, target_color, cup_init,
        )
        state.task_stats[target_color]['x'].append(float(cup_init[0]))
        state.task_stats[target_color]['y'].append(float(cup_init[1]))
        state.task_stats[target_color]['z'].append(float(cup_init[2]))
        state.task_stats['grasp_center_y'].append(
            float(cup_init[1] + EXPERT_Y_GRASP_OFFSET)
        )
        if state.task_tb3_init is not None and np.all(np.isfinite(state.task_tb3_init)):
            state.tb3_init_stats['x'].append(float(state.task_tb3_init[0]))
            state.tb3_init_stats['y'].append(float(state.task_tb3_init[1]))
            state.tb3_init_stats['z'].append(float(state.task_tb3_init[2]))

        state.task_completed_count += 1
        state.task_success_count += 1
        success_rate = (
            state.task_success_count / state.task_completed_count * 100
        ) if state.task_completed_count > 0 else 0.0
        loop_str = TASK_LOOP_COUNT if TASK_LOOP_COUNT > 0 else '∞'
        print(
            f"\n✅ Task SUCCESS (Auto-detected). "
            f"Task {state.task_completed_count}/{loop_str} | "
            f"Success: {state.task_success_count}/{state.task_completed_count} "
            f"({success_rate:.1f}%). Resetting for next task..."
        )

        perform_auto_reset(
            env=env,
            control_mode=state.control_mode,
            instruction_group_indices=state.instruction_group_indices,
            last_instruction_by_mode=state.last_instruction_by_mode,
            arm_runner=arm_runner,
            arm_policy=arm_policy,
            arm_smoother=arm_smoother,
            auto_mode_arm=state.auto_mode_arm,
            arm_sync_inference=ARM_SYNC_INFERENCE,
            apply_instruction_from_group=_apply_instruction_from_group,
            reset_options=state.auto_reset_options,
        )
        state.arm_action_chunk = None
        state.arm_chunk_step_index = 0
        state.step = 0
        state.reset_task_timer(env)

        if TASK_LOOP_COUNT > 0 and state.task_completed_count >= TASK_LOOP_COUNT:
            print(f"\n🎯 Reached target task count ({TASK_LOOP_COUNT}). Exiting...")
            return True

    # ---- 超时判定 ----
    elif elapsed >= TASK_TIMEOUT_SEC:
        ensure_step_log_header(STEP_LOG_PATH)
        with open(STEP_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                state.control_mode, 'fail',
                state.step - state.task_start_step,
                'unknown', 'nan', 'nan', 'nan',
            ])

        state.task_completed_count += 1
        state.task_fail_count += 1
        success_rate = (
            state.task_success_count / state.task_completed_count * 100
        ) if state.task_completed_count > 0 else 0.0
        loop_str = TASK_LOOP_COUNT if TASK_LOOP_COUNT > 0 else '∞'
        print(
            f"\n⏱️ Task TIMEOUT ({TASK_TIMEOUT_SEC}s). "
            f"Task {state.task_completed_count}/{loop_str} | "
            f"Success: {state.task_success_count}/{state.task_completed_count} "
            f"({success_rate:.1f}%). Resetting for next task..."
        )

        perform_auto_reset(
            env=env,
            control_mode=state.control_mode,
            instruction_group_indices=state.instruction_group_indices,
            last_instruction_by_mode=state.last_instruction_by_mode,
            arm_runner=arm_runner,
            arm_policy=arm_policy,
            arm_smoother=arm_smoother,
            auto_mode_arm=state.auto_mode_arm,
            arm_sync_inference=ARM_SYNC_INFERENCE,
            apply_instruction_from_group=_apply_instruction_from_group,
            reset_options=state.auto_reset_options,
        )
        state.arm_action_chunk = None
        state.arm_chunk_step_index = 0
        state.step = 0
        state.reset_task_timer(env)

        if TASK_LOOP_COUNT > 0 and state.task_completed_count >= TASK_LOOP_COUNT:
            print(f"\n🎯 Reached target task count ({TASK_LOOP_COUNT}). Exiting...")
            return True

    return False


# =============================================================================
# ARM 自动控制步进（同步 + 异步）
# =============================================================================

def step_arm_auto(
    state: DeployState,
    env,
    arm_policy,
    arm_runner,
    arm_smoother,
    img_transform,
    device,
):
    """ARM 自动控制模式：推理一步并执行动作。"""
    # 1. 收集观测数据
    robot_state = env.get_joint_state()   # (7,) 包含夹爪状态
    images_dict = env.grab_image()        # {'agent', 'wrist'}

    action_step = None

    if ARM_SYNC_INFERENCE:
        # ========== 同步推理（按 chunk 执行） ==========
        agent_img = Image.fromarray(images_dict['agent']).resize(
            (ARM_CONFIG['image_size'], ARM_CONFIG['image_size']),
            resample=Image.BILINEAR,
        )
        wrist_img = Image.fromarray(images_dict['wrist']).resize(
            (ARM_CONFIG['image_size'], ARM_CONFIG['image_size']),
            resample=Image.BILINEAR,
        )
        agent_tensor = img_transform(agent_img).unsqueeze(0).to(device)
        wrist_tensor = img_transform(wrist_img).unsqueeze(0).to(device)

        data = {
            'observation.state': torch.tensor(
                [robot_state], dtype=torch.float32
            ).to(device),
            'observation.images.agent': agent_tensor,
            'observation.images.wrist': wrist_tensor,
            'task': [env.instruction],
        }

        need_new_chunk = state.arm_action_chunk is None
        if not need_new_chunk:
            current_horizon = min(ARM_EXEC_HORIZON, state.arm_action_chunk.shape[0])
            need_new_chunk = state.arm_chunk_step_index >= current_horizon

        if need_new_chunk:
            # 不用 select_action()，避免其内部 action queue 导致"64步全消耗后才重推理"
            with torch.no_grad():
                batch = arm_policy.normalize_inputs(data)
                images, img_masks = arm_policy.prepare_images(batch)
                state_processed = arm_policy.prepare_state(batch)
                lang_tokens, lang_masks = arm_policy.prepare_language(batch)

                actions = arm_policy.model.sample_actions(
                    images, img_masks, lang_tokens, lang_masks, state_processed
                )

                original_action_dim = arm_policy.config.action_feature.shape[0]
                actions = actions[:, :, :original_action_dim]
                actions = arm_policy.unnormalize_outputs({"action": actions})["action"]

                if arm_policy.config.adapt_to_pi_aloha:
                    actions = arm_policy._pi_aloha_encode_actions(actions)

            action_np = actions.detach().cpu().numpy()
            if action_np.ndim == 3:
                chunk_np = action_np[0]
            elif action_np.ndim == 2:
                chunk_np = action_np
            elif action_np.ndim == 1:
                chunk_np = action_np[None, :]
            else:
                raise RuntimeError(
                    f"Unexpected action tensor shape: {action_np.shape}"
                )

            if chunk_np.shape[0] > 0:
                state.arm_action_chunk = chunk_np[:, :7]
                state.arm_chunk_step_index = 0

        if (
            state.arm_action_chunk is not None
            and state.arm_action_chunk.shape[0] > 0
        ):
            current_horizon = min(ARM_EXEC_HORIZON, state.arm_action_chunk.shape[0])
            if state.arm_chunk_step_index < current_horizon:
                action_step = state.arm_action_chunk[state.arm_chunk_step_index]
                state.arm_chunk_step_index += 1

    else:
        # ========== 异步推理 ==========
        obs_capture_time = time.time()
        arm_runner.update_observation(
            images_dict, robot_state, [env.instruction], obs_capture_time
        )
        action_step, _status_msg = arm_runner.get_action_at_time(time.time())
        if action_step is None:
            # 没有新动作时保持当前位置，避免突然跳变
            action_step = robot_state.copy()

    # 2. 执行动作（使用平滑器）
    if action_step is not None:
        smoothed_action, gripper_state = arm_smoother.smooth_action(action_step)
        env.step(smoothed_action, mode='arm')
        env.gripper_state = gripper_state

    # 3. 更新 p0 和 R0（用于保持 eef_pose 状态同步）
    env.p0, env.R0 = env.env.get_pR_body(body_name='tcp_link')

    # 4. 渲染 & 步数
    env.render(teleop=False, idx=state.step)
    state.step += 1

    if state.step % 50 == 0:
        mode_str = "SYNC" if ARM_SYNC_INFERENCE else "ASYNC"
        print(f"[ARM-{mode_str}] Step {state.step} | Task: {env.instruction}")


# =============================================================================
# ARM 手动控制步进
# =============================================================================

def step_arm_manual(state: DeployState, env, teleop):
    """ARM 手动（遥控）控制模式：读取遥控动作并执行。"""
    action, reset = teleop.get_action(mode='arm')
    if reset:
        _apply_instruction_from_group(
            env,
            'arm',
            state.instruction_group_indices,
            state.last_instruction_by_mode,
            log_prefix="🔄 [ARM RESET]",
            reinitialize_arm=True,
        )
        state.step = 0
        state.reset_task_timer(env)
    else:
        env.step(action, mode='arm', action_type='eef_pose')

    env.render(teleop=True, idx=state.step)
    state.step += 1


# =============================================================================
# BASE 自动控制步进
# =============================================================================

def step_base_auto(state: DeployState, env, base_runner, base_postproc):
    """BASE 自动控制模式：异步推理一步并执行动作。"""
    # 1. 收集观测数据
    robot_state = env.get_base_state()   # (4,) [轮速 + 朝向sin/cos]
    images_dict = env.grab_image()        # {'front', 'left', 'right'}

    obs_capture_time = time.time()

    # 2. 更新观测到推理线程
    base_runner.update_observation(
        images_dict, robot_state, [env.instruction], obs_capture_time
    )

    # 3. 获取动作
    action_step, _status_msg = base_runner.get_action_at_time(time.time())

    # 4. 执行动作
    if action_step is not None:
        yaw = float(np.arctan2(float(robot_state[2]), float(robot_state[3])))
        action_step = base_postproc.process(action_step, yaw)
        env.step(action_step, mode='base')
    else:
        base_postproc.reset()
        env.step(np.array([0.0, 0.0]), mode='base')

    # 5. 渲染 & 步数
    env.render(teleop=False, idx=state.step)
    state.step += 1

    if state.step % 50 == 0:
        print(f"[BASE] Step {state.step} | Task: {env.instruction}")


# =============================================================================
# BASE 手动控制步进
# =============================================================================

def step_base_manual(state: DeployState, env, teleop, base_runner, base_postproc):
    """BASE 手动（遥控）控制模式：读取遥控动作并执行。"""
    action, reset = teleop.get_action(mode='base')
    if reset:
        _apply_instruction_from_group(
            env,
            'base',
            state.instruction_group_indices,
            state.last_instruction_by_mode,
            log_prefix="🔄 [BASE RESET]",
        )
        env.reset(mode='base')
        teleop.reset()
        base_postproc.reset()
        if base_runner:
            base_runner.reset_state()
        state.step = 0
        state.reset_task_timer(env)
    else:
        env.step(action, mode='base')

    env.render(teleop=True, idx=state.step)
    state.step += 1
