"""
key_handlers.py
---------------
所有 GLFW 按键处理函数。

每个函数接受 DeployState 和相关依赖，直接修改 state，不使用 nonlocal。
"""

from concurrent.futures import ThreadPoolExecutor

import glfw
import numpy as np

from mujoco_env.instruction_utils import (
    INSTRUCTION_GROUPS,
    get_group_info as _get_group_info,
    apply_instruction_from_group as _apply_instruction_from_group,
)

from .deploy_state import DeployState
from .config import ARM_SYNC_INFERENCE
from .task_sequence import start_task_sequence_from_file


def _submit_vla_retrieval(vla_instruction_rag, request_id, query, cluster_caption, target_caption, target_image_path):
    result = vla_instruction_rag.retrieve_instruction(
        query=query,
        cluster_caption=cluster_caption,
        target_caption=target_caption,
        target_image_path=target_image_path,
    )
    result["request_id"] = int(request_id)
    return result


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

    print("\n🚀 [G] Teleporting base and cup to (4.25, 3.5, 0) yaw=57.1")
    env.teleport_base_and_cups(4.25, 3.5, 0.0, -57.1)


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
