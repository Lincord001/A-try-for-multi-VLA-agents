"""
key_handlers.py
---------------
所有 GLFW 按键处理函数。

每个函数接受 DeployState 和相关依赖，直接修改 state，不使用 nonlocal。
"""

import glfw

from mujoco_env.instruction_utils import (
    INSTRUCTION_GROUPS,
    get_group_info as _get_group_info,
    apply_instruction_from_group as _apply_instruction_from_group,
)

from .deploy_state import DeployState
from .config import ARM_SYNC_INFERENCE


# =============================================================================
# C 键：切换 arm/base 模式（不重置环境，支持串行接力）
# =============================================================================

def handle_key_c(state: DeployState, env, arm_policy, arm_runner, base_runner,
                 arm_smoother, base_postproc):
    if not env.env.is_key_pressed_once(key=glfw.KEY_C):
        return

    # 停止当前运行的推理
    if state.control_mode == 'arm':
        state.deactivate_arm_auto(arm_runner, arm_smoother,
                                  disable_auto_check=True, reset_runner_state=False)
    else:
        state.deactivate_base_auto(base_runner, base_postproc, reset_runner_state=False)

    # 重置 runner/policy 状态（清除旧数据）
    state.clear_runtime_state(arm_policy, arm_runner, base_runner,
                              arm_smoother, base_postproc)

    # 切换模式
    if state.control_mode == 'arm':
        state.control_mode = 'base'
        state.auto_check_enabled = False  # Base 模式不支持自动检测
    else:
        state.control_mode = 'arm'

    # 同步环境模式并刷新任务文本（不重置环境）
    env.control_mode = state.control_mode
    _apply_instruction_from_group(
        env,
        state.control_mode,
        state.instruction_group_indices,
        state.last_instruction_by_mode,
        log_prefix="🔄 [MODE SWITCH]",
    )

    print(f"\n🔄 Mode Switched to: {state.control_mode.upper()} (env preserved)")
    print(f"   Task: {env.instruction}")


# =============================================================================
# 左右键：切换当前模式的指令组
# =============================================================================

def handle_arrow_keys(state: DeployState, env):
    left_pressed = env.env.is_key_pressed_once(key=glfw.KEY_LEFT)
    right_pressed = env.env.is_key_pressed_once(key=glfw.KEY_RIGHT)
    if not (left_pressed or right_pressed):
        return

    groups = INSTRUCTION_GROUPS[state.control_mode]
    if len(groups) <= 1:
        idx, total, group_name, _ = _get_group_info(
            state.control_mode, state.instruction_group_indices
        )
        print(f"\nℹ️ [{state.control_mode.upper()}] Only one instruction group: "
              f"{group_name} ({idx + 1}/{total})")
    else:
        if right_pressed:
            state.instruction_group_indices[state.control_mode] = (
                (state.instruction_group_indices[state.control_mode] + 1) % len(groups)
            )
        else:
            state.instruction_group_indices[state.control_mode] = (
                (state.instruction_group_indices[state.control_mode] - 1) % len(groups)
            )
        _apply_instruction_from_group(
            env,
            state.control_mode,
            state.instruction_group_indices,
            state.last_instruction_by_mode,
            log_prefix="🔁 [GROUP SWITCH]",
        )


# =============================================================================
# N 键：启动自动控制
# =============================================================================

def handle_key_n(state: DeployState, env, arm_policy, arm_runner,
                 arm_smoother, base_runner, base_postproc):
    if not env.env.is_key_pressed_once(key=glfw.KEY_N):
        return

    if state.control_mode == 'arm':
        if arm_policy is not None and not state.auto_mode_arm:
            state.activate_arm_auto(arm_policy, arm_runner, arm_smoother,
                                    enable_auto_check=False, reset_timer=False)
            mode_str = "SYNC" if ARM_SYNC_INFERENCE else "ASYNC"
            print(f"\n🤖 [ARM] PI0 Auto Control Started! (Mode: {mode_str})")
        elif arm_policy is None:
            print("\n⚠️ ARM policy not loaded!")
    else:  # base mode
        if base_runner is not None and not state.auto_mode_base:
            state.auto_mode_base = True
            base_postproc.reset()
            base_runner.start()
            print("\n🚗 [BASE] PI0 Auto Control Started!")
        elif base_runner is None:
            print("\n⚠️ BASE policy not loaded!")


# =============================================================================
# M 键：恢复手动控制
# =============================================================================

def handle_key_m(state: DeployState, env, arm_runner, arm_smoother,
                 base_runner, base_postproc):
    if not env.env.is_key_pressed_once(key=glfw.KEY_M):
        return

    if state.control_mode == 'arm' and state.auto_mode_arm:
        state.deactivate_arm_auto(arm_runner, arm_smoother,
                                  disable_auto_check=True, reset_runner_state=True)
        print("\n👤 [ARM] Switched to Manual Control")
    elif state.control_mode == 'base' and state.auto_mode_base:
        state.deactivate_base_auto(base_runner, base_postproc, reset_runner_state=True)
        print("\n👤 [BASE] Switched to Manual Control")


# =============================================================================
# Z 键：重置环境
# =============================================================================

def handle_key_z(state: DeployState, env, arm_policy, arm_runner,
                 base_runner, arm_smoother, base_postproc):
    if not env.env.is_key_pressed_once(key=glfw.KEY_Z):
        return

    # 停止自动控制
    state.deactivate_arm_auto(arm_runner, arm_smoother,
                              disable_auto_check=False, reset_runner_state=False)
    state.deactivate_base_auto(base_runner, base_postproc, reset_runner_state=False)

    # 关闭自动检测（手动重置时）
    state.auto_check_enabled = False

    # 重置 runner/policy 状态（清除旧数据）
    state.clear_runtime_state(arm_policy, arm_runner, base_runner, arm_smoother, base_postproc)

    _apply_instruction_from_group(
        env,
        state.control_mode,
        state.instruction_group_indices,
        state.last_instruction_by_mode,
        log_prefix="🔄 [RESET]",
        reinitialize_arm=(state.control_mode == 'arm'),
    )
    if state.control_mode != 'arm':
        env.reset(mode=state.control_mode)

    state.step = 0
    state.reset_task_timer(env)

    print(f"\n🔄 Environment Reset. Mode: {state.control_mode.upper()}")
    print(f"   Task: {env.instruction}")


# =============================================================================
# K 键：循环 Base 轮速缩放档位（OFF -> 0.75x -> 0.50x）
# =============================================================================

def handle_key_k(state: DeployState, env, base_postproc):
    if not env.env.is_key_pressed_once(key=glfw.KEY_K):
        return

    if not base_postproc.speed_scale_enabled:
        base_postproc.speed_scale_enabled = True
        base_postproc.speed_scale = 0.75
    elif abs(base_postproc.speed_scale - 0.75) < 1e-6:
        base_postproc.speed_scale = 0.5
    else:
        base_postproc.speed_scale_enabled = False

    if base_postproc.speed_scale_enabled:
        print(f"\n⚙️ [BASE] Forward speed scaling: ON ({base_postproc.speed_scale:.2f}x)")
    else:
        print("\n⚙️ [BASE] Forward speed scaling: OFF")


# =============================================================================
# L 键：开启/关闭自动检测+自动控制功能（开关）
# =============================================================================

def handle_key_l(state: DeployState, env, arm_policy, arm_runner, arm_smoother):
    if not env.env.is_key_pressed_once(key=glfw.KEY_L):
        return

    if state.control_mode == 'arm':
        if not state.auto_check_enabled:
            # 开启自动检测 + 自动控制
            if arm_policy is None:
                print("\n⚠️ [L] ARM policy not loaded! Cannot start auto control.")
            else:
                state.activate_arm_auto(arm_policy, arm_runner, arm_smoother,
                                        enable_auto_check=True, reset_timer=True, env=env)
                print("\n✅ [L] Auto-control + Auto-detection ENABLED!")
                print("   → Model will auto-execute tasks, check success/fail, and auto-reset.")
        else:
            # 关闭自动检测 + 自动控制
            state.deactivate_arm_auto(arm_runner, arm_smoother,
                                      disable_auto_check=True, reset_runner_state=True)
            print("\n⏸️ [L] Auto-control + Auto-detection DISABLED.")
    else:
        print("\n⚠️ [L] Auto-control + Auto-detection only available in ARM mode.")


# =============================================================================
# G 键：传送小车和杯子
# =============================================================================

def handle_key_g(state: DeployState, env):
    if not env.env.is_key_pressed_once(key=glfw.KEY_G):
        return

    print("\n🚀 [G] Teleporting base and cup to (4.25, 3.5, 0) yaw=57.1")
    env.teleport_base_and_cups(4.25, 3.5, 0.0, -57.1)
