"""
key_handlers.py
---------------
所有 GLFW 按键处理函数。

每个函数接受 DeployState 和相关依赖，直接修改 state，不使用 nonlocal。
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

import glfw
import numpy as np
from PIL import Image

from mujoco_env.instruction_utils import (
    INSTRUCTION_GROUPS,
    get_group_info as _get_group_info,
    apply_instruction_from_group as _apply_instruction_from_group,
)
from orchestration.task_decomposer import build_navigation_scene_context

from .deploy_state import DeployState
from .config import ARM_SYNC_INFERENCE
from .task_sequence import start_task_sequence_from_file

_TASK_DECOMP_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task_decomp")
_PRESET_TASK_DECOMP_INSTRUCTION = (
    "First, have the mobile base transport the blue mug that is already loaded on its tray to the robotic arm workbench area. "
    "Then have the arm place the red mug from the workbench onto the mobile base tray, and have the mobile base transport that red mug to the bedroom."
)
_G_PRESET_BASE_X = 0.285
_G_PRESET_BASE_Y = 7.5
_G_PRESET_BASE_YAW_DEG = 90.0


def _submit_vla_retrieval(vla_instruction_rag, request_id, query, cluster_caption, target_caption, target_image_path):
    result = vla_instruction_rag.retrieve_instruction(
        query=query,
        cluster_caption=cluster_caption,
        target_caption=target_caption,
        target_image_path=target_image_path,
    )
    result["request_id"] = int(request_id)
    return result


def _prepare_blue_mug_tray_scene(env) -> None:
    env.pending_tray_init_color = "blue"
    env.reset(
        mode="arm",
        preserve_instruction=True,
        options={
            "random_init_enabled": 0,
            "tb3_x_random_enabled": False,
            "tb3_x_min": float(_G_PRESET_BASE_X),
            "tb3_x_max": float(_G_PRESET_BASE_X),
        },
    )

    p_tray_before, R_tray_before = env.env.get_pR_body("tb3_tray")
    p_mug_before, R_mug_before = env.env.get_pR_body("body_obj_mug_6")
    mug_rel_p = R_tray_before.T @ (p_mug_before - p_tray_before)
    mug_rel_R = R_tray_before.T @ R_mug_before

    env.set_base_pose(
        float(_G_PRESET_BASE_X),
        float(_G_PRESET_BASE_Y),
        np.deg2rad(float(_G_PRESET_BASE_YAW_DEG)),
        z=0.0,
    )

    p_tray_after, R_tray_after = env.env.get_pR_body("tb3_tray")
    p_mug_after = p_tray_after + R_tray_after @ mug_rel_p
    R_mug_after = R_tray_after @ mug_rel_R
    env.env.set_p_base_body("body_obj_mug_6", p_mug_after, forward=False)
    env.env.set_R_base_body("body_obj_mug_6", R_mug_after)

    support_body_name = env.tray_support_body_names.get("body_obj_mug_6")
    if support_body_name:
        env._set_tray_support_local_pos(
            support_body_name,
            np.array([mug_rel_p[0], mug_rel_p[1], 0.0], dtype=np.float64),
            forward=False,
        )
        env.tray_support_active["body_obj_mug_6"] = True
        env.env.forward(increase_tick=False)

    for _ in range(20):
        env.step_env()

    try:
        env.grab_image()
    except Exception:
        pass


def _prompt_user_task_input() -> str:
    print("\n📝 [Y] Enter your natural-language user task:")
    return input("   > ").strip()


def _submit_task_decomposition(task_decomposer, user_instruction: str, agent_image_path: str, navigation_context):
    return task_decomposer.decompose_user_task(
        user_instruction,
        agent_image_path=agent_image_path,
        navigation_context=navigation_context,
    )


def _reset_task_decomposition_session(
    state: DeployState,
    base_runner,
    base_postproc,
    arm_runner,
    arm_smoother,
    trace_manager=None,
    arm_orchestrator=None,
) -> bool:
    input_future = state.task_decomp_input_future
    if input_future is not None:
        if not input_future.done() and not input_future.cancel():
            print("\n⚠️ [Y] Cannot interrupt task input while the terminal input thread is still waiting.")
            return False
        state.task_decomp_input_future = None

    decomp_future = state.task_decomp_future
    if decomp_future is not None:
        if not decomp_future.done() and not decomp_future.cancel():
            print("\n⚠️ [Y] Cannot interrupt task decomposition while the background worker is still running.")
            return False
        state.task_decomp_future = None

    if state.task_sequence_active:
        state.stop_task_sequence("manual_key_y_restart")

    if trace_manager is not None:
        trace_manager.stop_arm_episode(reason="manual_key_y_restart")
        trace_manager.stop_base_episode(reason="manual_key_y_restart")
    if arm_orchestrator is not None:
        arm_orchestrator.on_auto_stop("manual_key_y_restart")

    state.deactivate_arm_auto(
        arm_runner,
        arm_smoother,
        disable_auto_check=True,
        reset_runner_state=True,
    )
    state.deactivate_base_auto(
        base_runner,
        base_postproc,
        reset_runner_state=True,
    )
    state.stop_navigation()
    state.rag_vla_handoff_waiting = False
    state.task_decomp_pending_instruction = None
    state.task_decomp_error = None
    return True


def process_pending_vla_retrieval(state: DeployState):
    future = state.rag_vla_pending_future
    if future is None or not state.rag_vla_retrieval_pending:
        return
    if not future.done():
        return

    request_id = int(state.rag_vla_pending_request_id)
    state.rag_vla_pending_future = None
    state.rag_vla_retrieval_pending = False

    try:
        vla_result = future.result()
    except Exception as exc:
        state.rag_vla_instruction = None
        state.rag_vla_instruction_meta = {}
        state.rag_vla_retrieval_error = str(exc)
        print(f"\n⚠️ [VLA-RAG] Async retrieval failed: {exc}")
        return

    if int(vla_result.get("request_id", -1)) != request_id:
        return

    state.rag_vla_instruction_meta = dict(vla_result)
    state.rag_vla_retrieval_error = None
    if not vla_result.get("matched", True) or not vla_result.get("instruction"):
        state.rag_vla_instruction = None
        print("\n⚠️ [VLA-RAG] Async retrieval finished: no matching VLA instruction.")
        if vla_result.get("llm_reason"):
            print(f"   LLM reason: {vla_result['llm_reason']}")
        return

    state.rag_vla_instruction = str(vla_result["instruction"])
    print(
        f"\n🧠 [VLA-RAG] Async retrieval ready: {state.rag_vla_instruction} "
        f"| group={vla_result['group_name']} | score={vla_result['score']:.6f}"
    )
    if vla_result.get("llm_reason"):
        print(f"   LLM reason: {vla_result['llm_reason']}")


def process_pending_task_decomposition(
    state: DeployState,
    env,
    task_decomposer,
    rag_navigator=None,
):
    input_future = state.task_decomp_input_future
    if input_future is not None and input_future.done():
        state.task_decomp_input_future = None
        try:
            user_instruction = str(input_future.result()).strip()
        except Exception as exc:
            state.task_decomp_error = str(exc)
            print(f"\n❌ [Y] Failed to read task input: {exc}")
            user_instruction = ""

        if not user_instruction:
            if state.task_decomp_error is None:
                print("⚠️ [Y] Empty instruction ignored.")
        else:
            state.task_decomp_pending_instruction = user_instruction
            state.task_decomp_error = None
            print(f"\n📝 [Y] Task input received: {user_instruction}")

    if state.task_decomp_future is None and state.task_decomp_pending_instruction:
        if task_decomposer is None:
            print("\n⚠️ [Y] Task decomposer unavailable.")
            state.task_decomp_pending_instruction = None
            return
        try:
            agent_image_path = _write_agent_view_snapshot(env, prefix="deploy_task_decomposer_agent")
        except Exception as exc:
            state.task_decomp_error = str(exc)
            state.task_decomp_pending_instruction = None
            print(f"\n❌ [Y] Failed to capture current agent view: {exc}")
            return

        navigation_context = None
        if rag_navigator is not None:
            try:
                current_pose = _get_current_base_pose(env)
                navigation_context = build_navigation_scene_context(
                    current_pose=current_pose,
                    rag_navigator=rag_navigator,
                ).to_dict()
            except Exception as exc:
                print(f"\n⚠️ [Y] Failed to build navigation context, continuing without it: {exc}")
                navigation_context = None

        user_instruction = str(state.task_decomp_pending_instruction)
        print("\n🧩 [Y] Task decomposition started in background.")
        state.task_decomp_future = _TASK_DECOMP_EXECUTOR.submit(
            _submit_task_decomposition,
            task_decomposer,
            user_instruction,
            agent_image_path,
            navigation_context,
        )
        state.task_decomp_pending_instruction = None

    future = state.task_decomp_future
    if future is None or not future.done():
        return

    state.task_decomp_future = None
    try:
        result = future.result()
    except Exception as exc:
        state.task_decomp_error = str(exc)
        print(f"\n❌ [Y] Task decomposition failed: {exc}")
        return

    print("\n================ Task Decomposition Report ================")
    print(f"【用户任务】{result.user_instruction}")
    print(f"【是否可执行】{result.feasible}")
    print(f"【总结】{result.summary_for_user}")
    print("【逐步诊断】")
    for item in result.diagnostics:
        print(
            f"  - step={int(item['step_index']) + 1} "
            f"executor={item.get('executor') or 'UNKNOWN'} "
            f"status={item.get('status')} "
            f"reason={item.get('reason_code')}: {item.get('reason_text')}"
        )
    if result.llm_debug_records:
        print("【模型原始输出】")
        for idx, record in enumerate(result.llm_debug_records, start=1):
            stage = str(record.get("stage", "")).strip() or "unknown_stage"
            model = str(record.get("model", "")).strip() or "unknown_model"
            mode = "multimodal" if record.get("multimodal") else "text"
            raw_text = str(record.get("raw_text", "")).strip()
            print(f"  - #{idx} stage={stage} model={model} mode={mode}")
            if raw_text:
                print("    raw_text:")
                print(raw_text)
            print("    parsed_payload:")
            print(json.dumps(record.get("parsed_payload", {}), ensure_ascii=False, indent=2))
    if result.feasible:
        print("【任务队列】")
        for idx, task in enumerate(result.normalized_tasks, start=1):
            print(f"  - #{idx}: {task}")
    print("==========================================================")

    if not result.feasible:
        print("   → Queue not started because the task is not fully executable.")
        return

    if state.task_sequence_active:
        state.stop_task_sequence("task_decomposer_replace_queue")

    state.start_task_sequence(tasks=result.normalized_tasks, source_path="[task_decomposer]")
    print(
        f"\n▶️ [TASK QUEUE] Loaded {len(result.normalized_tasks)} tasks from task decomposer."
    )
    print("   → The queue will begin automatically on the next control tick.")


def _write_agent_view_snapshot(env, *, prefix: str = "task_decomposer_agent") -> str:
    env.grab_image()
    rgb_agent = getattr(env, "rgb_agent", None)
    if rgb_agent is None:
        raise RuntimeError("当前环境中没有可用的 agent view 图像。")
    array = np.asarray(rgb_agent)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise RuntimeError(f"agent view 图像格式非法: shape={array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array[:, :, :3])
    output_dir = Path(tempfile.gettempdir()) / "task_decomposer_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}.png"
    image.save(output_path)
    return str(output_path.resolve())


def _get_current_base_pose(env) -> np.ndarray:
    p_tb3, R_tb3 = env.env.get_pR_body("tb3_base")
    yaw = float(np.arctan2(float(R_tb3[1, 0]), float(R_tb3[0, 0])))
    return np.array([float(p_tb3[0]), float(p_tb3[1]), yaw], dtype=np.float64)


# =============================================================================
# C 键：切换 arm/base 模式（不重置环境，支持串行接力）
# =============================================================================

def handle_key_c(state: DeployState, env, arm_policy, arm_runner, base_runner,
                 arm_smoother, base_postproc, trace_manager=None, arm_orchestrator=None):
    if not env.env.is_key_pressed_once(key=glfw.KEY_C):
        return

    state.stop_task_sequence("manual_key_c_mode_switch")

    # 停止当前运行的推理
    if state.control_mode == 'arm':
        if trace_manager is not None:
            trace_manager.stop_arm_episode(reason='mode_switch')
        if arm_orchestrator is not None:
            arm_orchestrator.on_auto_stop('mode_switch')
        state.deactivate_arm_auto(arm_runner, arm_smoother,
                                  disable_auto_check=True, reset_runner_state=False)
    else:
        if trace_manager is not None:
            trace_manager.stop_base_episode(reason='mode_switch')
        state.deactivate_base_auto(base_runner, base_postproc, reset_runner_state=False)

    # 重置 runner/policy 状态（清除旧数据）
    state.clear_runtime_state(arm_policy, arm_runner, base_runner,
                              arm_smoother, base_postproc)
    state.stop_navigation()
    state.rag_vla_handoff_waiting = False

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
                 arm_smoother, base_runner, base_postproc, trace_manager=None, arm_orchestrator=None):
    if not env.env.is_key_pressed_once(key=glfw.KEY_N):
        return

    if state.control_mode == 'arm':
        if arm_policy is not None and not state.auto_mode_arm:
            state.activate_arm_auto(arm_policy, arm_runner, arm_smoother,
                                    enable_auto_check=False, reset_timer=False)
            if trace_manager is not None:
                trace_manager.start_arm_episode(env, step=state.step, reason='manual_key_n_start')
            if arm_orchestrator is not None:
                arm_orchestrator.on_auto_start(env)
            mode_str = "SYNC" if ARM_SYNC_INFERENCE else "ASYNC"
            print(f"\n🤖 [ARM] PI0 Auto Control Started! (Mode: {mode_str})")
        elif arm_policy is None:
            print("\n⚠️ ARM policy not loaded!")
    else:  # base mode
        if base_runner is not None and not state.auto_mode_base:
            state.stop_navigation()
            state.auto_mode_base = True
            state.base_nudge_active = False
            state.base_nudge_start_time = 0.0
            state.base_nudge_start_xy = None
            state.base_nudge_heading_yaw = 0.0
            state.base_nudge_last_progress_time = 0.0
            state.base_nudge_last_progress_dist = 0.0
            if state.base_execution_tracker is not None:
                state.base_execution_tracker.reset(mode="base_vla", instruction=getattr(env, "instruction", None))
            base_postproc.reset()
            base_runner.start()
            if trace_manager is not None:
                trace_manager.start_base_episode(env, step=state.step, reason='manual_key_n_start')
            print("\n🚗 [BASE] PI0 Auto Control Started!")
        elif base_runner is None:
            print("\n⚠️ BASE policy not loaded!")


# =============================================================================
# M 键：恢复手动控制
# =============================================================================

def handle_key_m(state: DeployState, env, arm_runner, arm_smoother,
                 base_runner, base_postproc, trace_manager=None, arm_orchestrator=None):
    if not env.env.is_key_pressed_once(key=glfw.KEY_M):
        return

    state.stop_task_sequence("manual_key_m_stop")

    if state.control_mode == 'arm' and state.auto_mode_arm:
        if trace_manager is not None:
            trace_manager.stop_arm_episode(reason='manual_key_m_stop')
        if arm_orchestrator is not None:
            arm_orchestrator.on_auto_stop('manual_key_m_stop')
        state.deactivate_arm_auto(arm_runner, arm_smoother,
                                  disable_auto_check=True, reset_runner_state=True)
        print("\n👤 [ARM] Switched to Manual Control")
    elif state.control_mode == 'base' and state.auto_mode_base:
        if trace_manager is not None:
            trace_manager.stop_base_episode(reason='manual_key_m_stop')
        state.deactivate_base_auto(base_runner, base_postproc, reset_runner_state=True)
        print("\n👤 [BASE] Switched to Manual Control")
    elif state.control_mode == 'base' and state.nav_mode_active:
        state.stop_navigation()
        state.rag_vla_handoff_waiting = False
        print("\n👤 [BASE] Stopped RAG navigation, switched to Manual Control")
    elif state.control_mode == 'base' and state.rag_vla_handoff_waiting:
        state.rag_vla_handoff_waiting = False
        print("\n👤 [BASE] Cancelled RAG->VLA waiting state, switched to Manual Control")


# =============================================================================
# Z 键：重置环境
# =============================================================================

def handle_key_z(state: DeployState, env, arm_policy, arm_runner,
                 base_runner, arm_smoother, base_postproc, trace_manager=None, arm_orchestrator=None):
    if not env.env.is_key_pressed_once(key=glfw.KEY_Z):
        return

    state.stop_task_sequence("manual_key_z_reset")

    # 停止自动控制
    if trace_manager is not None:
        trace_manager.stop_arm_episode(reason='manual_key_z_reset')
        trace_manager.stop_base_episode(reason='manual_key_z_reset')
    if arm_orchestrator is not None:
        arm_orchestrator.on_auto_stop('manual_key_z_reset')
    state.deactivate_arm_auto(arm_runner, arm_smoother,
                              disable_auto_check=False, reset_runner_state=False)
    state.deactivate_base_auto(base_runner, base_postproc, reset_runner_state=False)
    state.stop_navigation()
    state.rag_vla_handoff_waiting = False

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

def handle_key_l(state: DeployState, env, arm_policy, arm_runner, arm_smoother, trace_manager=None, arm_orchestrator=None):
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
                if trace_manager is not None:
                    trace_manager.start_arm_episode(env, step=state.step, reason='auto_check_enabled')
                if arm_orchestrator is not None:
                    arm_orchestrator.on_auto_start(env)
                print("\n✅ [L] Auto-control + Auto-detection ENABLED!")
                print("   → Model will auto-execute tasks, check success/fail, and auto-reset.")
        else:
            # 关闭自动检测 + 自动控制
            if trace_manager is not None:
                trace_manager.stop_arm_episode(reason='auto_check_disabled')
            if arm_orchestrator is not None:
                arm_orchestrator.on_auto_stop('auto_check_disabled')
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

    state.stop_task_sequence("manual_key_g_scene_preset")
    _prepare_blue_mug_tray_scene(env)
    state.control_mode = "arm"
    env.control_mode = "arm"
    state.step = 0
    state.task_decomp_pending_instruction = None
    state.task_decomp_error = None
    print(
        "\n🚀 [G] Scene preset ready: blue mug loaded on tray, "
        f"base reset to x={_G_PRESET_BASE_X:.3f}, y={_G_PRESET_BASE_Y:.3f}, yaw={_G_PRESET_BASE_YAW_DEG:.1f}°."
    )


# =============================================================================
# T/U 键：输入自然语言检索
#   - BASE 模式: T 键 -> 导航 RAG
#   - ARM  模式: U 键 -> arm instruction RAG
# =============================================================================

def handle_key_t(
    state: DeployState,
    env,
    rag_navigator,
    vla_instruction_rag=None,
    vla_executor: ThreadPoolExecutor | None = None,
    arm_instruction_rag=None,
):
    if state.control_mode == 'arm':
        if not env.env.is_key_pressed_once(key=glfw.KEY_U):
            return
        if arm_instruction_rag is None:
            print("\n⚠️ [U] ARM instruction retriever unavailable.")
            return

        print("\n📝 [U] Enter your natural-language arm task query:")
        query_text = input("   > ").strip()
        if not query_text:
            print("⚠️ [U] Empty query ignored.")
            return

        try:
            result = arm_instruction_rag.retrieve_instruction(query_text)
            print("\n================ ARM Instruction RAG 报告 ================")
            print(f"【用户指令】{query_text}")
            if result.get("matched", True) and result.get("instruction"):
                instruction = str(result["instruction"])
                env.set_instruction(given=instruction, task_type='arm')
                state.last_instruction_by_mode['arm'] = instruction
                print(f"【标准化指令】{instruction}")
                print(f"【所属分组】{result.get('group_name', 'N/A')}")
                print(f"【检索分数】{float(result.get('score', 0.0)):.6f}")
            else:
                print("【标准化指令】NO_MATCH")
            if result.get("llm_reason"):
                print(f"【LLM理由】{result['llm_reason']}")
            print("=========================================================")
        except Exception as e:
            print(f"❌ [U] ARM retrieval failed: {e}")
        return

    if not env.env.is_key_pressed_once(key=glfw.KEY_T):
        return

    if rag_navigator is None:
        print("\n⚠️ [T] RAG navigator unavailable. Check DASHSCOPE_API_KEY / forest graph config.")
        return

    print("\n📝 [T] Enter your natural-language navigation query:")
    query_text = input("   > ").strip()
    if not query_text:
        print("⚠️ [T] Empty query ignored.")
        return

    try:
        result = rag_navigator.retrieve_top_leaf(query_text)
        state.update_retrieval_result(query_text, result)
        query = str(result.get("query", query_text))
        cluster_id = str(result.get("cluster_id", "N/A"))
        cluster_caption = str(result.get("cluster_caption", ""))
        target_node = str(result.get("target_node", "N/A"))
        target_caption = str(result.get("target_caption", ""))
        target_image_path = result.get("target_image_path")
        score = float(result.get("score", 0.0))
        target_xy = result.get("target_xy")

        print("\n================ Embodied-RAG 检索报告 ================")
        print(f"【用户指令】{query}")
        print(f"【宏观决策】{cluster_id} | 区域摘要: {cluster_caption}")
        print(f"【微观匹配】{target_node} | 描述: {target_caption} | 余弦相似度: {score:.6f}")
        if isinstance(target_xy, (list, tuple)) and len(target_xy) >= 2:
            print(
                f"【最终导航系坐标】(x, y) = "
                f"({float(target_xy[0]):.6f}, {float(target_xy[1]):.6f})"
            )
        print("======================================================")
        state.rag_vla_query_text = None
        state.rag_vla_instruction = None
        state.rag_vla_instruction_meta = {}
        state.rag_vla_retrieval_error = None
        state.rag_vla_handoff_waiting = False
        if vla_instruction_rag is not None:
            if vla_executor is None:
                print("【VLA微调指令】未提供后台检索执行器")
            else:
                state.rag_vla_request_seq += 1
                request_id = int(state.rag_vla_request_seq)
                state.rag_vla_pending_request_id = request_id
                state.rag_vla_retrieval_pending = True
                state.rag_vla_pending_future = vla_executor.submit(
                    _submit_vla_retrieval,
                    vla_instruction_rag,
                    request_id,
                    query,
                    cluster_caption,
                    target_caption,
                    target_image_path,
                )
                print(
                    "【VLA微调指令】后台检索已启动 "
                    f"| weights=query/cluster/target="
                    f"{vla_instruction_rag.args.query_weight:.2f}/"
                    f"{vla_instruction_rag.args.cluster_weight:.2f}/"
                    f"{vla_instruction_rag.args.target_weight:.2f}"
                )
        else:
            state.rag_vla_query_text = None
            state.rag_vla_instruction = None
            state.rag_vla_instruction_meta = {}
            print("【VLA微调指令】未启用 VLA Instruction RAG")
        print("   → Press [R] to start navigation to this retrieved target.")
    except Exception as e:
        print(f"❌ [T] Retrieval failed: {e}")


# =============================================================================
# P 键：加载并启动 JSON 串行任务队列
# =============================================================================

def handle_key_p(state: DeployState, env, task_sequence_json: str):
    if not env.env.is_key_pressed_once(key=glfw.KEY_P):
        return

    if state.task_sequence_active:
        state.stop_task_sequence("manual_key_p_toggle_stop")
        return

    try:
        start_task_sequence_from_file(state, task_sequence_json)
        print("   → The queue will begin automatically on the next control tick.")
    except Exception as exc:
        print(f"\n❌ [P] Failed to load task queue: {exc}")


def handle_key_y(
    state: DeployState,
    env,
    arm_runner,
    arm_smoother,
    base_runner,
    base_postproc,
    task_decomposer,
    rag_navigator=None,
    trace_manager=None,
    arm_orchestrator=None,
):
    if not env.env.is_key_pressed_once(key=glfw.KEY_Y):
        return

    if task_decomposer is None:
        print("\n⚠️ [Y] Task decomposer unavailable.")
        return

    has_active_task_session = (
        state.task_decomp_input_future is not None
        or state.task_decomp_future is not None
        or state.task_decomp_pending_instruction is not None
        or state.task_sequence_active
        or state.auto_mode_arm
        or state.auto_mode_base
        or state.nav_mode_active
    )

    if has_active_task_session:
        stopped = _reset_task_decomposition_session(
            state,
            base_runner=base_runner,
            base_postproc=base_postproc,
            arm_runner=arm_runner,
            arm_smoother=arm_smoother,
            trace_manager=trace_manager,
            arm_orchestrator=arm_orchestrator,
        )
        if not stopped:
            return
        print("\n🔄 [Y] Previous task session stopped. Starting a new preset task session.")

    state.task_decomp_error = None
    state.task_decomp_input_future = None
    state.task_decomp_pending_instruction = _PRESET_TASK_DECOMP_INSTRUCTION
    print("\n🧩 [Y] Preset user task queued for decomposition:")
    print(f"   {_PRESET_TASK_DECOMP_INSTRUCTION}")


# =============================================================================
# R 键：触发/停止 RAG 导航执行
# =============================================================================

def handle_key_r(state: DeployState, env, base_runner, base_postproc, rag_executor, rag_target_node, rag_output_json, trace_manager=None):
    if not env.env.is_key_pressed_once(key=glfw.KEY_R):
        return

    if state.control_mode != 'base':
        print("\n⚠️ [R] RAG navigation only available in BASE mode. Press [C] to switch.")
        return

    if state.nav_mode_active:
        state.stop_navigation()
        env.step(np.array([0.0, 0.0]), mode='base')
        print("\n⏸️ [R] RAG navigation stopped.")
        return

    if state.auto_mode_base:
        if trace_manager is not None:
            trace_manager.stop_base_episode(reason='rag_navigation_takeover')
        state.deactivate_base_auto(base_runner, base_postproc, reset_runner_state=False)
        print("\n🔁 [R] BASE PI0 auto control stopped before RAG navigation.")

    try:
        target_node = state.rag_retrieved_target_node or rag_target_node
        target_source = "retrieved" if state.rag_retrieved_target_node else "default"
        p_tb3, R_tb3 = env.env.get_pR_body('tb3_base')
        yaw = float(np.arctan2(float(R_tb3[1, 0]), float(R_tb3[0, 0])))
        start_pose = np.array([float(p_tb3[0]), float(p_tb3[1]), yaw], dtype=np.float64)

        result = rag_executor.plan_dense_waypoints(start_pose=start_pose, target_node=target_node)
        rag_executor.save_result(result, rag_output_json)
        state.start_navigation(result["dense_waypoints"], target_node)
        base_postproc.reset()
        print(
            f"\n🧭 [R] RAG navigation started: target={target_node} ({target_source}) "
            f"| path_nodes={len(result['path_nodes'])} "
            f"| dense_waypoints={result['dense_waypoints_count']}"
        )
    except Exception as e:
        state.stop_navigation()
        print(f"\n❌ [R] Failed to start RAG navigation: {e}")
