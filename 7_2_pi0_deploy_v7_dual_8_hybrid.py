#!/usr/bin/env python3
"""
V6 环境部署脚本 - 支持 Arm 和 Base 双模式异步推理

功能说明：
- 按 C: 切换 arm/base 控制模式（不重置环境）
- 按 N: 启动当前模式的 pi0 控制（异步推理）
- 按 M: 恢复人类遥控模式
- 按 Z: 重置环境
- 按 Q: 退出

Arm 模式：
- 数据集: demo_data_arm_v4
- 权重: ckpt/pi0_arm/pretrained_model_arm_v4
- 输入: 2个相机(agent/wrist, 224x224) + 6维关节角度
- 输出: 7维 (6关节角度 + 1夹爪状态) - 绝对量

Base 模式：
- 数据集: demo_data_base_ver_3
- 权重: ckpt/pi0_base/pretrained_model_ver_3/pretrained_model
- 输入: 3个相机(front/left/right, 256x256) + 2维轮速度
- 输出: 2维轮速度指令 - 绝对量
"""

import os
import sys

print("Setting up environment variables for Hugging Face...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from deploy_v7_dual.config import (
    TASK_TIMEOUT_SEC,
    TASK_LOOP_COUNT,
    TASK_STATS_OUTPUT_DIR,
    ARM_CONFIG,
    BASE_CONFIG,
    CONTROL_FREQUENCY,
    CONTROL_DT,
    ARM_INFERENCE_MODE,
    ARM_SYNC_INFERENCE,
    ACTION_HORIZON,
    CHUNK_THRESHOLD,
    BASE_ACTION_HORIZON,
    BASE_CHUNK_THRESHOLD,
    BASE_POSTPROC_ENABLED,
    BASE_POSTPROC_HEADING_HOLD_ENABLED,
    BASE_POSTPROC_KP_YAW,
    BASE_POSTPROC_MAX_TURN_V,
    BASE_POSTPROC_YAW_DEADBAND,
    BASE_POSTPROC_STRAIGHT_DELTA_TH,
    BASE_POSTPROC_MIN_ABS_SPEED,
    BASE_POSTPROC_MAX_WHEEL_ABS,
    BASE_FORWARD_SPEED_SCALE_ENABLED,
    BASE_FORWARD_SPEED_SCALE,
    ARM_EXEC_HORIZON,
    SMOOTHING_ENABLED,
    SMOOTHING_ALPHA_JOINTS,
    SMOOTHING_ALPHA_GRIPPER,
    GRIPPER_HYSTERESIS_ENABLED,
    GRIPPER_OPEN_THRESH,
    GRIPPER_CLOSE_THRESH,
    LOAD_ARM_MODEL,
    LOAD_BASE_MODEL,
    ARM_PILOT_RUN_MODE,
    RANDOM_INIT_ENABLED,
    RANDOM_INIT_GRIPPER_OPEN,
    TB3_X_GAUSSIAN_ENABLED,
    TB3_X_CENTER,
    TB3_X_OFFSET_STD,
    TB3_X_OFFSET_MIN,
    TB3_X_OFFSET_MAX,
)
from deploy_v7_dual.runtime_components import (
    ActionSmoother,
    AsyncInferenceRunner,
    get_default_transform,
)
from deploy_v7_dual.deploy_state import DeployState
from deploy_v7_dual.key_handlers import (
    handle_key_c,
    handle_arrow_keys,
    handle_key_n,
    handle_key_m,
    handle_key_z,
    handle_key_k,
    handle_key_l,
    handle_key_g,
)
from deploy_v7_dual.control_loop import (
    check_auto_result,
    step_arm_auto,
    step_arm_manual,
    step_base_auto,
    step_base_manual,
)
from deploy_v7_dual.ui_prints import (
    print_startup_banner,
    print_model_loading_config,
    print_action_smoother_config,
    print_controls_guide,
)

from mujoco_env.instruction_utils import (
    validate_instruction_groups as _validate_instruction_groups,
    apply_instruction_from_group as _apply_instruction_from_group,
)

# 导入 LeRobot 和 MuJoCo 环境
try:
    from deploy_v7_dual.policy_loader import load_policy
    from mujoco_env.y_env7 import SimpleEnv7, EXPERT_Y_GRASP_OFFSET
    from mujoco_env.teleop import TeleopAgent
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

from mujoco_env.action_utils import BaseActionPostProcessor
from mujoco_env.visualization import (
    extract_model_version,
    plot_task_stats,
    plot_tb3_init_stats,
)


# ==========================================
# 主程序
# ==========================================
def main():
    _validate_instruction_groups()

    print_startup_banner()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print_model_loading_config(
        load_arm_model=LOAD_ARM_MODEL,
        load_base_model=LOAD_BASE_MODEL,
        arm_inference_mode=ARM_INFERENCE_MODE,
        arm_sync_inference=ARM_SYNC_INFERENCE,
        chunk_threshold=CHUNK_THRESHOLD,
        arm_exec_horizon=ARM_EXEC_HORIZON,
        action_horizon=ACTION_HORIZON,
        base_action_horizon=BASE_ACTION_HORIZON,
        base_chunk_threshold=BASE_CHUNK_THRESHOLD,
        base_postproc_enabled=BASE_POSTPROC_ENABLED,
        base_postproc_heading_hold_enabled=BASE_POSTPROC_HEADING_HOLD_ENABLED,
        base_postproc_kp_yaw=BASE_POSTPROC_KP_YAW,
        base_postproc_max_turn_v=BASE_POSTPROC_MAX_TURN_V,
        base_forward_speed_scale_enabled=BASE_FORWARD_SPEED_SCALE_ENABLED,
        base_forward_speed_scale=BASE_FORWARD_SPEED_SCALE,
        arm_pilot_run_mode=ARM_PILOT_RUN_MODE,
        random_init_enabled=RANDOM_INIT_ENABLED,
        random_init_gripper_open=RANDOM_INIT_GRIPPER_OPEN,
        tb3_x_gaussian_enabled=TB3_X_GAUSSIAN_ENABLED,
        tb3_x_center=TB3_X_CENTER,
        tb3_x_offset_std=TB3_X_OFFSET_STD,
        tb3_x_offset_min=TB3_X_OFFSET_MIN,
        tb3_x_offset_max=TB3_X_OFFSET_MAX,
        task_timeout_sec=TASK_TIMEOUT_SEC,
        task_loop_count=TASK_LOOP_COUNT,
        task_stats_output_dir=TASK_STATS_OUTPUT_DIR,
    )

    # 1. 根据配置加载模型
    arm_policy = None
    base_policy = None

    if LOAD_ARM_MODEL:
        arm_policy = load_policy(ARM_CONFIG, device, label='ARM', emoji='🤖')
    else:
        print("\n⏭️  Skipping ARM model loading (LOAD_ARM_MODEL=False)")

    if LOAD_BASE_MODEL:
        base_policy = load_policy(BASE_CONFIG, device, label='BASE', emoji='🚗')
    else:
        print("\n⏭️  Skipping BASE model loading (LOAD_BASE_MODEL=False)")

    if arm_policy is None and base_policy is None:
        print("❌ Both policies failed to load or were disabled. Exiting.")
        return

    # 2. 初始化环境 (使用 y7 场景)
    print("\n" + "="*60)
    print("🌍 Initializing MuJoCo Environment (V6)...")
    print("="*60)

    xml_path = './asset/example_scene_y7.xml'
    PnPEnv = SimpleEnv7(
        xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
        random_init_enabled=RANDOM_INIT_ENABLED,
        random_init_gripper_open=RANDOM_INIT_GRIPPER_OPEN,
        tb3_x_gaussian_enabled=TB3_X_GAUSSIAN_ENABLED,
        tb3_x_center=TB3_X_CENTER,
        tb3_x_offset_std=TB3_X_OFFSET_STD,
        tb3_x_offset_min=TB3_X_OFFSET_MIN,
        tb3_x_offset_max=TB3_X_OFFSET_MAX,
    )

    control_mode = 'arm'
    PnPEnv.reset(mode=control_mode)
    teleop = TeleopAgent(PnPEnv)

    # 3. 初始化共享状态
    state = DeployState(
        control_mode=control_mode,
        instruction_group_indices={'arm': 0, 'base': 0},
        last_instruction_by_mode={'arm': None, 'base': None},
    )
    _apply_instruction_from_group(
        PnPEnv,
        state.control_mode,
        state.instruction_group_indices,
        state.last_instruction_by_mode,
        log_prefix="[INIT]",
        reinitialize_arm=(state.control_mode == 'arm'),
    )
    state.reset_task_timer(PnPEnv)

    # 4. 初始化图像预处理
    IMG_TRANSFORM = get_default_transform()

    # 5. 初始化推理器
    arm_runner = None
    base_runner = None

    if arm_policy is not None:
        if ARM_SYNC_INFERENCE:
            print("🔄 [ARM] Using SYNC inference mode")
        else:
            arm_runner = AsyncInferenceRunner(
                arm_policy,
                device,
                IMG_TRANSFORM,
                control_dt=CONTROL_DT,
                camera_keys=['agent', 'wrist'],
                mode_label='ARM',
                mode_emoji='🤖',
                config=ARM_CONFIG,
                action_horizon=ACTION_HORIZON,
                chunk_threshold=CHUNK_THRESHOLD,
                perf_monitor=None,
            )
            print("🔄 [ARM] Using ASYNC inference mode")
            print(f"   - ACTION_HORIZON: {ACTION_HORIZON}")
            print(f"   - CHUNK_THRESHOLD: {CHUNK_THRESHOLD}")

    if base_policy is not None:
        base_runner = AsyncInferenceRunner(
            base_policy,
            device,
            IMG_TRANSFORM,
            control_dt=CONTROL_DT,
            camera_keys=['front', 'left', 'right'],
            mode_label='BASE',
            mode_emoji='🚗',
            config=BASE_CONFIG,
            action_horizon=BASE_ACTION_HORIZON,
            chunk_threshold=BASE_CHUNK_THRESHOLD,
            perf_monitor=None,
        )
        print("🔄 [BASE] Using ASYNC inference mode")
        print(f"   - BASE_ACTION_HORIZON: {BASE_ACTION_HORIZON}")
        print(f"   - BASE_CHUNK_THRESHOLD: {BASE_CHUNK_THRESHOLD}")

    # 6. 初始化动作平滑器（防颤抖）和 Base 后处理器
    arm_smoother = ActionSmoother(
        joint_dim=6,
        alpha_joints=SMOOTHING_ALPHA_JOINTS,
        alpha_gripper=SMOOTHING_ALPHA_GRIPPER,
        smoothing_enabled=SMOOTHING_ENABLED,
        gripper_hysteresis_enabled=GRIPPER_HYSTERESIS_ENABLED,
        gripper_open_thresh=GRIPPER_OPEN_THRESH,
        gripper_close_thresh=GRIPPER_CLOSE_THRESH,
    )
    base_postproc = BaseActionPostProcessor(
        postproc_enabled=BASE_POSTPROC_ENABLED,
        heading_hold_enabled=BASE_POSTPROC_HEADING_HOLD_ENABLED,
        kp_yaw=BASE_POSTPROC_KP_YAW,
        max_turn_v=BASE_POSTPROC_MAX_TURN_V,
        yaw_deadband=BASE_POSTPROC_YAW_DEADBAND,
        straight_delta_th=BASE_POSTPROC_STRAIGHT_DELTA_TH,
        min_abs_speed=BASE_POSTPROC_MIN_ABS_SPEED,
        max_wheel_abs=BASE_POSTPROC_MAX_WHEEL_ABS,
        speed_scale_enabled=BASE_FORWARD_SPEED_SCALE_ENABLED,
        speed_scale=BASE_FORWARD_SPEED_SCALE,
    )
    print_action_smoother_config(
        smoothing_enabled=SMOOTHING_ENABLED,
        smoothing_alpha_joints=SMOOTHING_ALPHA_JOINTS,
        smoothing_alpha_gripper=SMOOTHING_ALPHA_GRIPPER,
        gripper_hysteresis_enabled=GRIPPER_HYSTERESIS_ENABLED,
        gripper_open_thresh=GRIPPER_OPEN_THRESH,
        gripper_close_thresh=GRIPPER_CLOSE_THRESH,
    )

    print_controls_guide(state.control_mode)

    try:
        while PnPEnv.env.is_viewer_alive():
            # [A] 物理环境步进
            PnPEnv.step_env()

            # [B] 控制循环
            if PnPEnv.env.loop_every(HZ=CONTROL_FREQUENCY):

                # --- 键位处理 ---
                handle_key_c(state, PnPEnv, arm_policy, arm_runner, base_runner,
                             arm_smoother, base_postproc)
                handle_arrow_keys(state, PnPEnv)
                handle_key_n(state, PnPEnv, arm_policy, arm_runner,
                             arm_smoother, base_runner, base_postproc)
                handle_key_m(state, PnPEnv, arm_runner, arm_smoother,
                             base_runner, base_postproc)
                handle_key_z(state, PnPEnv, arm_policy, arm_runner,
                             base_runner, arm_smoother, base_postproc)
                handle_key_k(state, PnPEnv, base_postproc)
                handle_key_l(state, PnPEnv, arm_policy, arm_runner, arm_smoother)
                handle_key_g(state, PnPEnv)

                # --- 控制逻辑 ---
                if state.control_mode == 'arm':
                    # 自动检测成功 + 超时判定（仅在L键开启后生效）
                    if check_auto_result(state, PnPEnv, arm_policy,
                                         arm_runner, arm_smoother):
                        break

                    if state.auto_mode_arm and arm_policy is not None:
                        step_arm_auto(state, PnPEnv, arm_policy, arm_runner,
                                      arm_smoother, IMG_TRANSFORM, device)
                    else:
                        step_arm_manual(state, PnPEnv, teleop)

                else:  # base mode
                    if state.auto_mode_base and base_runner:
                        step_base_auto(state, PnPEnv, base_runner, base_postproc)
                    else:
                        step_base_manual(state, PnPEnv, teleop,
                                         base_runner, base_postproc)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        # 只有当有统计数据时才输出统计图（仅在arm模式的L键自动检测模式下才有数据）
        total_tasks = state.task_success_count + state.task_fail_count
        if total_tasks > 0:
            version_suffix = ''
            if LOAD_ARM_MODEL and ARM_CONFIG.get('model_path'):
                version_suffix = extract_model_version(ARM_CONFIG['model_path'])
            plot_task_stats(
                state.task_stats,
                TASK_STATS_OUTPUT_DIR,
                state.task_success_count,
                state.task_fail_count,
                version_suffix=version_suffix,
                grasp_y_offset=EXPERT_Y_GRASP_OFFSET,
            )
            plot_tb3_init_stats(
                state.tb3_init_stats,
                TASK_STATS_OUTPUT_DIR,
                state.task_success_count,
                state.task_fail_count,
                version_suffix=version_suffix,
            )
            print(f"\n📊 Final Statistics:")
            print(f"   Total Tasks: {total_tasks}")
            print(f"   Success: {state.task_success_count}")
            print(f"   Fail: {state.task_fail_count}")
            print(f"   Success Rate: {state.task_success_count/total_tasks*100:.1f}%")
        else:
            print("\n📊 No task statistics to output (L-key auto-detection mode was not used).")

        # 清理
        if arm_runner and arm_runner.running:
            arm_runner.stop()
        if base_runner and base_runner.running:
            base_runner.stop()
        if PnPEnv.env.viewer:
            PnPEnv.env.close_viewer()
        print("🛑 Environment closed.")


if __name__ == "__main__":
    main()
