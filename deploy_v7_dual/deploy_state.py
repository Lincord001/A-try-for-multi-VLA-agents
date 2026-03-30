"""
deploy_state.py
---------------
DeployState: 持有主循环中所有共享可变状态的数据类。

将原先散落在 main() 中通过 nonlocal 共享的局部变量收拢到一个对象，
使得 key_handlers.py 和 control_loop.py 可以直接读写同一个 state 实例，
不再依赖闭包或 nonlocal。
"""

import time
import dataclasses
from typing import Any, Optional

import numpy as np

from orchestration.execution_tracker import BASE_VLA_PROFILE, ExecutionTracker

from .control_helpers import (
    clear_runtime_state as _clear_runtime_state,
    deactivate_arm_auto as _deactivate_arm_auto,
    deactivate_base_auto as _deactivate_base_auto,
    activate_arm_auto as _activate_arm_auto,
)
from .config import (
    RANDOM_INIT_ENABLED,
    RANDOM_INIT_GRIPPER_OPEN,
    TB3_X_GAUSSIAN_ENABLED,
    TB3_X_CENTER,
    TB3_X_OFFSET_STD,
    TB3_X_OFFSET_MIN,
    TB3_X_OFFSET_MAX,
    ARM_SYNC_INFERENCE,
)


@dataclasses.dataclass
class DeployState:
    """部署主循环的全部可变状态。"""

    # ---------- 运行时模式 ----------
    control_mode: str
    instruction_group_indices: dict
    last_instruction_by_mode: dict

    # ---------- 自动 / 手动控制标志 ----------
    auto_mode_arm: bool = False
    auto_mode_base: bool = False

    # ---------- 同步推理 chunk 缓存 ----------
    arm_action_chunk: Optional[np.ndarray] = None
    arm_chunk_step_index: int = 0

    # ---------- 自动检测 ----------
    auto_check_enabled: bool = False
    arm_vlm_pause_active: bool = False
    base_execution_tracker: Optional[ExecutionTracker] = dataclasses.field(
        default_factory=lambda: ExecutionTracker(BASE_VLA_PROFILE)
    )
    base_nudge_active: bool = False
    base_nudge_start_time: float = 0.0
    base_nudge_start_xy: Optional[np.ndarray] = None
    base_nudge_heading_yaw: float = 0.0
    base_nudge_last_progress_time: float = 0.0
    base_nudge_last_progress_dist: float = 0.0

    # ---------- 步计数 ----------
    step: int = 0

    # ---------- 任务统计 ----------
    task_completed_count: int = 0
    task_success_count: int = 0
    task_fail_count: int = 0

    # ---------- 任务计时 ----------
    task_start_time: float = dataclasses.field(default_factory=time.time)
    task_start_step: int = 0
    task_tb3_init: Optional[np.ndarray] = None

    # ---------- 统计数组（不能用 mutable default，需 field） ----------
    task_stats: dict = dataclasses.field(default_factory=lambda: {
        'red': {'x': [], 'y': [], 'z': []},
        'blue': {'x': [], 'y': [], 'z': []},
        'grasp_center_y': [],
    })
    tb3_init_stats: dict = dataclasses.field(default_factory=lambda: {
        'x': [], 'y': [], 'z': []
    })

    # ---------- RAG 导航执行状态 ----------
    nav_mode_active: bool = False
    nav_target_node: Optional[str] = None
    nav_waypoints: list = dataclasses.field(default_factory=list)
    nav_waypoint_index: int = 0
    rag_query_text: Optional[str] = None
    rag_retrieved_target_node: Optional[str] = None
    rag_retrieval_meta: dict = dataclasses.field(default_factory=dict)
    rag_vla_query_text: Optional[str] = None
    rag_vla_instruction: Optional[str] = None
    rag_vla_instruction_meta: dict = dataclasses.field(default_factory=dict)
    rag_vla_pending_future: Optional[Any] = None
    rag_vla_request_seq: int = 0
    rag_vla_pending_request_id: int = 0
    rag_vla_retrieval_pending: bool = False
    rag_vla_retrieval_error: Optional[str] = None
    rag_vla_handoff_waiting: bool = False

    # ---------- 串行任务队列 ----------
    task_sequence_active: bool = False
    task_sequence_tasks: list = dataclasses.field(default_factory=list)
    task_sequence_index: int = 0
    task_sequence_started: bool = False
    task_sequence_pending_result: Optional[dict] = None
    task_sequence_results: list = dataclasses.field(default_factory=list)
    task_sequence_source_path: Optional[str] = None
    task_sequence_prefetch_started: bool = False
    task_sequence_arm_query_futures: dict = dataclasses.field(default_factory=dict)
    task_sequence_arm_query_results: dict = dataclasses.field(default_factory=dict)
    task_sequence_arm_return_home_active: bool = False
    task_sequence_arm_post_home_result: Optional[dict] = None

    # ---------- 自动重置选项（从 config 读取，主程序初始化后传入） ----------
    auto_reset_options: dict = dataclasses.field(default_factory=lambda: {
        "random_init_enabled": RANDOM_INIT_ENABLED,
        "random_init_gripper_open": RANDOM_INIT_GRIPPER_OPEN,
        "tb3_x_gaussian_enabled": TB3_X_GAUSSIAN_ENABLED,
        "tb3_x_center": TB3_X_CENTER,
        "tb3_x_offset_std": TB3_X_OFFSET_STD,
        "tb3_x_offset_min": TB3_X_OFFSET_MIN,
        "tb3_x_offset_max": TB3_X_OFFSET_MAX,
    })

    # -------------------------------------------------------------------------
    # 辅助方法（替代 main() 中的 nonlocal 闭包）
    # -------------------------------------------------------------------------

    def reset_task_timer(self, env=None):
        """重置任务计时器和起始步数。"""
        from .task_runtime import get_tb3_init_pose
        self.task_start_time = time.time()
        self.task_start_step = self.step
        if env is not None:
            self.task_tb3_init = get_tb3_init_pose(env)

    def clear_runtime_state(self, arm_policy, arm_runner, base_runner, arm_smoother, base_postproc):
        """统一清理策略/runner 缓存和后处理器状态。"""
        self.arm_action_chunk, self.arm_chunk_step_index = _clear_runtime_state(
            arm_policy=arm_policy,
            arm_runner=arm_runner,
            base_runner=base_runner,
            arm_smoother=arm_smoother,
            base_postproc=base_postproc,
        )

    def deactivate_arm_auto(self, arm_runner, arm_smoother,
                            disable_auto_check=False, reset_runner_state=False):
        """停止 ARM 自动控制，可选关闭自动检测与清理 runner 状态。"""
        (self.auto_mode_arm,
         self.auto_check_enabled,
         self.arm_action_chunk,
         self.arm_chunk_step_index) = _deactivate_arm_auto(
            auto_mode_arm=self.auto_mode_arm,
            auto_check_enabled=self.auto_check_enabled,
            arm_runner=arm_runner,
            arm_smoother=arm_smoother,
            arm_sync_inference=ARM_SYNC_INFERENCE,
            disable_auto_check=disable_auto_check,
            reset_runner_state=reset_runner_state,
        )
        self.arm_vlm_pause_active = False

    def deactivate_base_auto(self, base_runner, base_postproc, reset_runner_state=False):
        """停止 BASE 自动控制，可选清理 runner 状态。"""
        self.auto_mode_base = _deactivate_base_auto(
            auto_mode_base=self.auto_mode_base,
            base_runner=base_runner,
            base_postproc=base_postproc,
            reset_runner_state=reset_runner_state,
        )
        self.base_nudge_active = False
        self.base_nudge_start_time = 0.0
        self.base_nudge_start_xy = None
        self.base_nudge_heading_yaw = 0.0
        self.base_nudge_last_progress_time = 0.0
        self.base_nudge_last_progress_dist = 0.0
        if self.base_execution_tracker is not None:
            self.base_execution_tracker.reset()

    def activate_arm_auto(self, arm_policy, arm_runner, arm_smoother,
                          enable_auto_check=False, reset_timer=False, env=None):
        """启动 ARM 自动控制，返回更新后的状态与 chunk 指针。"""
        (self.auto_mode_arm,
         self.auto_check_enabled,
         self.arm_action_chunk,
         self.arm_chunk_step_index) = _activate_arm_auto(
            auto_check_enabled=self.auto_check_enabled,
            arm_policy=arm_policy,
            arm_runner=arm_runner,
            arm_smoother=arm_smoother,
            arm_sync_inference=ARM_SYNC_INFERENCE,
            enable_auto_check=enable_auto_check,
        )
        self.arm_vlm_pause_active = False
        if reset_timer:
            self.reset_task_timer(env)

    def start_navigation(self, waypoints, target_node):
        """开启 RAG 导航执行。"""
        self.nav_waypoints = [list(p) for p in waypoints]
        self.nav_waypoint_index = 0
        self.nav_target_node = str(target_node)
        self.nav_mode_active = len(self.nav_waypoints) > 0

    def stop_navigation(self):
        """停止 RAG 导航执行并清空状态。"""
        self.nav_mode_active = False
        self.nav_waypoints = []
        self.nav_waypoint_index = 0

    def update_retrieval_result(self, query_text, retrieval_result):
        """更新第四阶段检索结果缓存。"""
        self.rag_query_text = str(query_text)
        self.rag_retrieval_meta = dict(retrieval_result)
        target_node = retrieval_result.get("target_node")
        self.rag_retrieved_target_node = str(target_node) if target_node is not None else None

    def clear_retrieval_result(self):
        """清空第四阶段检索结果缓存。"""
        self.rag_query_text = None
        self.rag_retrieved_target_node = None
        self.rag_retrieval_meta = {}
        self.rag_vla_query_text = None
        self.rag_vla_instruction = None
        self.rag_vla_instruction_meta = {}
        self.rag_vla_pending_future = None
        self.rag_vla_pending_request_id = 0
        self.rag_vla_retrieval_pending = False
        self.rag_vla_retrieval_error = None
        self.rag_vla_handoff_waiting = False

    def start_task_sequence(self, tasks, source_path: str):
        """启动串行任务队列。"""
        self.task_sequence_active = True
        self.task_sequence_tasks = [dict(task) for task in tasks]
        self.task_sequence_index = 0
        self.task_sequence_started = False
        self.task_sequence_pending_result = None
        self.task_sequence_results = []
        self.task_sequence_source_path = str(source_path)
        self.task_sequence_prefetch_started = False
        self.task_sequence_arm_query_futures = {}
        self.task_sequence_arm_query_results = {}
        self.task_sequence_arm_return_home_active = False
        self.task_sequence_arm_post_home_result = None

    def stop_task_sequence(self, reason: str):
        """停止串行任务队列。"""
        if self.task_sequence_active:
            print(f"\n⏹️ [TASK QUEUE] Stopped: {reason}")
        self.task_sequence_active = False
        self.task_sequence_tasks = []
        self.task_sequence_index = 0
        self.task_sequence_started = False
        self.task_sequence_pending_result = None
        self.task_sequence_source_path = None
        self.task_sequence_prefetch_started = False
        self.task_sequence_arm_query_futures = {}
        self.task_sequence_arm_query_results = {}
        self.task_sequence_arm_return_home_active = False
        self.task_sequence_arm_post_home_result = None

    def current_task_sequence_task(self):
        """返回当前串行任务；若越界则返回 None。"""
        if not self.task_sequence_active:
            return None
        if self.task_sequence_index < 0 or self.task_sequence_index >= len(self.task_sequence_tasks):
            return None
        return self.task_sequence_tasks[self.task_sequence_index]

    def mark_task_sequence_result(self, status: str, reason: str, details: Optional[dict] = None):
        """记录当前串行任务已完成，等待调度器推进到下一项。"""
        if not self.task_sequence_active or not self.task_sequence_started:
            return
        payload = {
            "status": str(status),
            "reason": str(reason),
        }
        if details:
            payload.update(dict(details))
        self.task_sequence_pending_result = payload

    def begin_task_sequence_arm_return_home(self, status: str, reason: str, details: Optional[dict] = None):
        """对于队列中的 ARM 任务，先回正，再提交任务完成结果。"""
        if not self.task_sequence_active or not self.task_sequence_started:
            return
        payload = {
            "status": str(status),
            "reason": str(reason),
        }
        if details:
            payload.update(dict(details))
        self.task_sequence_arm_return_home_active = True
        self.task_sequence_arm_post_home_result = payload
