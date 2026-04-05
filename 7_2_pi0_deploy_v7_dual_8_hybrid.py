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

模块职责（deploy_v7_dual）：
  - deploy_v7_dual/config.py              (部署参数与开关配置)
  - deploy_v7_dual/policy_loader.py       (策略模型加载与初始化)
  - deploy_v7_dual/runtime_components.py  (异步推理器/动作平滑器/图像预处理)
  - deploy_v7_dual/deploy_state.py        (部署共享状态与任务计时统计)
  - deploy_v7_dual/key_handlers.py        (按键事件处理：模式切换/启停/重置等)
  - deploy_v7_dual/control_loop.py        (arm/base 自动与手动控制主逻辑)
  - deploy_v7_dual/ui_prints.py           (启动信息与控制说明输出)
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

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
    POLICY_SERVER_ENABLED,
    POLICY_SERVER_HOST,
    POLICY_SERVER_PORT,
    POLICY_SERVER_AUTHKEY,
    ARM_PILOT_RUN_MODE,
    RANDOM_INIT_ENABLED,
    RANDOM_INIT_GRIPPER_OPEN,
    TB3_X_GAUSSIAN_ENABLED,
    TB3_X_CENTER,
    TB3_X_OFFSET_STD,
    TB3_X_OFFSET_MIN,
    TB3_X_OFFSET_MAX,
    RAG_TOPOLOGY_JSON,
    RAG_TARGET_NODE,
    RAG_DENSE_OUTPUT_JSON,
    RAG_FOREST_JSON,
    RAG_MACRO_MODEL,
    RAG_EMBEDDING_MODEL,
    RAG_RETRIEVE_MAX_RETRY,
    RAG_RETRIEVE_RETRY_WAIT,
    VLA_RAG_CACHE_JSON,
    VLA_RAG_SELECTION_VLM_MODEL,
    VLA_RAG_TOP_K,
    VLA_RAG_QUERY_WEIGHT,
    VLA_RAG_CLUSTER_WEIGHT,
    VLA_RAG_TARGET_WEIGHT,
    ARM_RAG_CACHE_JSON,
    ARM_RAG_TOP_K,
    ARM_RAG_SELECTION_MODEL,
    EXECUTION_TRACE_ENABLED,
    EXECUTION_TRACE_OUTPUT_DIR,
    EXECUTION_TRACE_FLUSH_EVERY,
    EXECUTION_TRACE_EVALUATE_TRACKER,
    TASK_SEQUENCE_JSON,
    ARM_VLM_ORCHESTRATION_ENABLED,
    ARM_VLM_CHECK_OUTPUT_DIR,
    ARM_VLM_MODEL,
    TASK_DECOMPOSER_ENABLED,
    TASK_DECOMPOSER_TEXT_MODEL,
    TASK_DECOMPOSER_VISION_MODEL,
    TASK_DECOMPOSER_NAV_TOP_K,
    TASK_DECOMPOSER_EMBEDDING_MODEL,
    TASK_DECOMPOSER_STREAM_OUTPUT,
    TASK_REPLANNER_ENABLED,
    TASK_REPLANNER_AUTO_APPLY,
    TASK_REPLANNER_MODEL,
    TASK_REPLANNER_MAX_NEW_TASKS,
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
    handle_key_r,
    handle_key_t,
    handle_key_p,
    handle_key_y,
    process_pending_vla_retrieval,
    process_pending_task_decomposition,
)
from deploy_v7_dual.control_loop import (
    check_auto_result,
    handle_arm_orchestrator_event,
    step_arm_auto,
    step_arm_recovery,
    step_arm_manual,
    step_base_auto,
    step_base_manual,
    step_base_nav,
    step_base_wait_vla_handoff,
)
from deploy_v7_dual.ui_prints import (
    print_startup_banner,
    print_model_loading_config,
    print_action_smoother_config,
    print_controls_guide,
)
from deploy_v7_dual.execution_trace_manager import ExecutionTraceManager
from deploy_v7_dual.arm_vlm_orchestrator import ArmVLMOrchestrator
from deploy_v7_dual.task_sequence import advance_task_sequence
from orchestration.task_decomposer import TaskDecomposer, DashScopeTaskDecompositionBackend
from orchestration.task_replanner import TaskReplanner, DashScopeTaskReplanningBackend

from mujoco_env.instruction_utils import (
    INSTRUCTION_GROUPS,
    validate_instruction_groups as _validate_instruction_groups,
    apply_instruction_from_group as _apply_instruction_from_group,
)

# 导入 LeRobot 和 MuJoCo 环境
try:
    from deploy_v7_dual.policy_loader import load_policy
    from deploy_v7_dual.policy_backends import RemotePolicyClient
    from mujoco_env.y_env7 import SimpleEnv7, EXPERT_Y_GRASP_OFFSET
    from mujoco_env.teleop import TeleopAgent
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

from mujoco_env.action_utils import BaseActionPostProcessor
from rag_pipeline.rag_executor import TrajectoryExecutor
from rag_pipeline.rag_navigator import RAGNavigator, NavigatorArgs
from rag_pipeline.vla_instruction_rag import VLAInstructionRAG, VLAInstructionRAGArgs
from rag_pipeline.arm_instruction_rag import ArmInstructionRAG, ArmInstructionRAGArgs
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
        if POLICY_SERVER_ENABLED:
            arm_policy = RemotePolicyClient(
                mode="arm",
                host=POLICY_SERVER_HOST,
                port=POLICY_SERVER_PORT,
                authkey=POLICY_SERVER_AUTHKEY,
            )
            arm_policy.ping()
            print(
                "\n🧠 [ARM] Using remote policy server: "
                f"{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}"
            )
        else:
            arm_policy = load_policy(ARM_CONFIG, device, label='ARM', emoji='🤖')
    else:
        print("\n⏭️  Skipping ARM model loading (LOAD_ARM_MODEL=False)")

    if LOAD_BASE_MODEL:
        if POLICY_SERVER_ENABLED:
            base_policy = RemotePolicyClient(
                mode="base",
                host=POLICY_SERVER_HOST,
                port=POLICY_SERVER_PORT,
                authkey=POLICY_SERVER_AUTHKEY,
            )
            base_policy.ping()
            print(
                "\n🧠 [BASE] Using remote policy server: "
                f"{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}"
            )
        else:
            base_policy = load_policy(BASE_CONFIG, device, label='BASE', emoji='🚗')
    else:
        print("\n⏭️  Skipping BASE model loading (LOAD_BASE_MODEL=False)")

    if arm_policy is None and base_policy is None:
        print("\n⚠️ Both policies are not loaded (or disabled).")
        print("   Running in MANUAL-ONLY mode: environment will still open.")

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
    reset_report = PnPEnv.reset(mode=control_mode)
    if getattr(reset_report, 'warnings', None):
        print(f"⚠️ Reset warnings ({len(reset_report.warnings)}):")
        for msg in reset_report.warnings:
            print(f"   - {msg}")
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
    trace_manager = ExecutionTraceManager(
        enabled=EXECUTION_TRACE_ENABLED,
        output_dir=EXECUTION_TRACE_OUTPUT_DIR,
        flush_every=EXECUTION_TRACE_FLUSH_EVERY,
        evaluate_tracker=EXECUTION_TRACE_EVALUATE_TRACKER,
        metadata={
            "control_frequency": CONTROL_FREQUENCY,
            "arm_inference_mode": ARM_INFERENCE_MODE,
            "arm_model_path": ARM_CONFIG.get("model_path"),
            "base_model_path": BASE_CONFIG.get("model_path"),
        },
    )
    arm_orchestrator = ArmVLMOrchestrator(
        enabled=ARM_VLM_ORCHESTRATION_ENABLED,
        output_dir=ARM_VLM_CHECK_OUTPUT_DIR,
        model=ARM_VLM_MODEL,
    )

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
    rag_executor = TrajectoryExecutor(RAG_TOPOLOGY_JSON)
    rag_navigator = None
    vla_instruction_rag = None
    vla_instruction_executor = None
    arm_instruction_rag = None
    arm_instruction_executor = None
    try:
        rag_navigator = RAGNavigator(
            NavigatorArgs(
                query="runtime_query",
                input_json=RAG_FOREST_JSON,
                macro_model=RAG_MACRO_MODEL,
                embedding_model=RAG_EMBEDDING_MODEL,
                max_retry=RAG_RETRIEVE_MAX_RETRY,
                retry_wait=RAG_RETRIEVE_RETRY_WAIT,
                log_level="INFO",
            )
        )
        print(f"🧠 [RAG] Navigator loaded: {RAG_FOREST_JSON}")
    except Exception as e:
        print(f"⚠️ [RAG] Navigator init failed, T-key retrieval disabled: {e}")
    try:
        vla_instruction_rag = VLAInstructionRAG(
            VLAInstructionRAGArgs(
                instruction_groups=list(INSTRUCTION_GROUPS.get('base', [])),
                embedding_model=RAG_EMBEDDING_MODEL,
                selection_model=VLA_RAG_SELECTION_VLM_MODEL,
                cache_json=VLA_RAG_CACHE_JSON,
                top_k=VLA_RAG_TOP_K,
                query_weight=VLA_RAG_QUERY_WEIGHT,
                cluster_weight=VLA_RAG_CLUSTER_WEIGHT,
                target_weight=VLA_RAG_TARGET_WEIGHT,
                max_retry=RAG_RETRIEVE_MAX_RETRY,
                retry_wait=RAG_RETRIEVE_RETRY_WAIT,
            )
        )
        vla_instruction_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="base_vla_rag",
        )
        print(f"🧠 [VLA-RAG] Instruction retriever loaded: {VLA_RAG_CACHE_JSON}")
    except Exception as e:
        print(f"⚠️ [VLA-RAG] Instruction retriever init failed, auto handoff retrieval disabled: {e}")
    try:
        arm_instruction_rag = ArmInstructionRAG(
            ArmInstructionRAGArgs(
                instruction_groups=list(INSTRUCTION_GROUPS.get('arm', [])),
                embedding_model=RAG_EMBEDDING_MODEL,
                selection_model=ARM_RAG_SELECTION_MODEL,
                cache_json=ARM_RAG_CACHE_JSON,
                top_k=ARM_RAG_TOP_K,
                max_retry=RAG_RETRIEVE_MAX_RETRY,
                retry_wait=RAG_RETRIEVE_RETRY_WAIT,
            )
        )
        arm_instruction_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="arm_instruction_rag",
        )
        print(f"🧠 [ARM-RAG] Instruction retriever loaded: {ARM_RAG_CACHE_JSON}")
    except Exception as e:
        print(f"⚠️ [ARM-RAG] Instruction retriever init failed, arm query normalization disabled: {e}")
    task_decomposer = None
    task_replanner = None
    if TASK_DECOMPOSER_ENABLED:
        try:
            task_decomposer = TaskDecomposer(
                backend=DashScopeTaskDecompositionBackend(
                    text_model=TASK_DECOMPOSER_TEXT_MODEL,
                    vision_model=TASK_DECOMPOSER_VISION_MODEL,
                    stream_output=TASK_DECOMPOSER_STREAM_OUTPUT,
                ),
                nav_top_k=TASK_DECOMPOSER_NAV_TOP_K,
                embedding_model=TASK_DECOMPOSER_EMBEDDING_MODEL,
            )
            print("🧩 [TASK-DECOMP] Task decomposer loaded.")
        except Exception as e:
            print(f"⚠️ [TASK-DECOMP] Task decomposer init failed: {e}")
    if TASK_REPLANNER_ENABLED:
        try:
            task_replanner = TaskReplanner(
                backend=DashScopeTaskReplanningBackend(
                    model=TASK_REPLANNER_MODEL,
                ),
            )
            print("🔁 [TASK-REPLAN] Task replanner loaded.")
        except Exception as e:
            print(f"⚠️ [TASK-REPLAN] Task replanner init failed: {e}")
    state.task_replanner_enabled = bool(TASK_REPLANNER_ENABLED and task_replanner is not None)
    state.task_replanner_auto_apply = bool(TASK_REPLANNER_AUTO_APPLY)
    state.task_replanner = task_replanner
    state.task_replanner_max_new_tasks = int(TASK_REPLANNER_MAX_NEW_TASKS)

    try:
        while PnPEnv.env.is_viewer_alive():
            # [A] 物理环境步进
            PnPEnv.step_env()

            # [B] 控制循环
            if PnPEnv.env.loop_every(HZ=CONTROL_FREQUENCY):
                process_pending_vla_retrieval(state)
                process_pending_task_decomposition(
                    state,
                    PnPEnv,
                    task_decomposer,
                    rag_navigator=rag_navigator,
                )

                # --- 键位处理 ---
                handle_key_c(state, PnPEnv, arm_policy, arm_runner, base_runner,
                             arm_smoother, base_postproc, trace_manager, arm_orchestrator)
                handle_arrow_keys(state, PnPEnv)
                handle_key_n(state, PnPEnv, arm_policy, arm_runner,
                             arm_smoother, base_runner, base_postproc, trace_manager, arm_orchestrator)
                handle_key_m(state, PnPEnv, arm_runner, arm_smoother,
                             base_runner, base_postproc, trace_manager, arm_orchestrator)
                handle_key_z(state, PnPEnv, arm_policy, arm_runner,
                             base_runner, arm_smoother, base_postproc, trace_manager, arm_orchestrator)
                handle_key_k(state, PnPEnv, base_postproc)
                handle_key_l(state, PnPEnv, arm_policy, arm_runner, arm_smoother, trace_manager, arm_orchestrator)
                handle_key_g(state, PnPEnv)
                handle_key_t(
                    state,
                    PnPEnv,
                    rag_navigator,
                    vla_instruction_rag,
                    vla_instruction_executor,
                    arm_instruction_rag,
                )
                handle_key_p(
                    state,
                    PnPEnv,
                    TASK_SEQUENCE_JSON,
                )
                handle_key_y(
                    state,
                    PnPEnv,
                    arm_runner,
                    arm_smoother,
                    base_runner,
                    base_postproc,
                    task_decomposer,
                    rag_navigator=rag_navigator,
                    trace_manager=trace_manager,
                    arm_orchestrator=arm_orchestrator,
                )
                handle_key_r(
                    state,
                    PnPEnv,
                    base_runner,
                    base_postproc,
                    rag_executor,
                    RAG_TARGET_NODE,
                    RAG_DENSE_OUTPUT_JSON,
                    trace_manager,
                )
                task_sequence_consumed_step = advance_task_sequence(
                    state,
                    PnPEnv,
                    arm_policy,
                    arm_runner,
                    arm_smoother,
                    base_runner,
                    base_postproc,
                    rag_executor,
                    rag_navigator=rag_navigator,
                    vla_instruction_rag=vla_instruction_rag,
                    vla_instruction_executor=vla_instruction_executor,
                    arm_instruction_rag=arm_instruction_rag,
                    arm_instruction_executor=arm_instruction_executor,
                    trace_manager=trace_manager,
                    arm_orchestrator=arm_orchestrator,
                    rag_output_json=RAG_DENSE_OUTPUT_JSON,
                )
                if task_sequence_consumed_step:
                    continue

                # --- 控制逻辑 ---
                if state.control_mode == 'arm':
                    # 自动检测成功 + 超时判定（仅在L键开启后生效）
                    if check_auto_result(state, PnPEnv, arm_policy,
                                         arm_runner, arm_smoother, trace_manager, arm_orchestrator):
                        break

                    if arm_orchestrator.enabled and arm_orchestrator.recovery_active:
                        recovery_done = step_arm_recovery(state, PnPEnv, arm_orchestrator)
                        if recovery_done:
                            arm_orchestrator.finalize_recovery_handoff(
                                PnPEnv,
                                arm_policy,
                                arm_runner,
                                arm_smoother,
                            )
                    elif state.arm_vlm_pause_active and arm_orchestrator.enabled:
                        arm_event = arm_orchestrator.process_pending_check()
                        if arm_event:
                            handle_arm_orchestrator_event(
                                state,
                                PnPEnv,
                                arm_policy,
                                arm_runner,
                                arm_smoother,
                                arm_event,
                                trace_manager=trace_manager,
                                arm_orchestrator=arm_orchestrator,
                            )
                        else:
                            PnPEnv.render(teleop=False, idx=state.step)
                    elif state.auto_mode_arm and arm_policy is not None:
                        arm_event = step_arm_auto(state, PnPEnv, arm_policy, arm_runner,
                                      arm_smoother, IMG_TRANSFORM, device, trace_manager, arm_orchestrator)
                        if arm_event:
                            handle_arm_orchestrator_event(
                                state,
                                PnPEnv,
                                arm_policy,
                                arm_runner,
                                arm_smoother,
                                arm_event,
                                trace_manager=trace_manager,
                                arm_orchestrator=arm_orchestrator,
                            )
                    else:
                        step_arm_manual(state, PnPEnv, teleop)

                else:  # base mode
                    if state.nav_mode_active:
                        step_base_nav(
                            state,
                            PnPEnv,
                            base_runner,
                            base_postproc,
                            trace_manager,
                            vla_instruction_rag,
                        )
                    elif state.rag_vla_handoff_waiting:
                        step_base_wait_vla_handoff(
                            state,
                            PnPEnv,
                            base_runner,
                            base_postproc,
                            trace_manager,
                            vla_instruction_rag,
                        )
                    elif state.auto_mode_base and base_runner:
                        step_base_auto(state, PnPEnv, base_runner, base_postproc, trace_manager)
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
        trace_manager.close()
        arm_orchestrator.on_auto_stop('session_close')
        if arm_runner and arm_runner.running:
            arm_runner.stop()
        if base_runner and base_runner.running:
            base_runner.stop()
        if vla_instruction_executor is not None:
            vla_instruction_executor.shutdown(wait=False, cancel_futures=True)
        if arm_instruction_executor is not None:
            arm_instruction_executor.shutdown(wait=False, cancel_futures=True)
        if arm_policy is not None and hasattr(arm_policy, "close"):
            arm_policy.close()
        if base_policy is not None and hasattr(base_policy, "close"):
            base_policy.close()
        if PnPEnv.env.viewer:
            PnPEnv.env.close_viewer()
        print("🛑 Environment closed.")


if __name__ == "__main__":
    main()
