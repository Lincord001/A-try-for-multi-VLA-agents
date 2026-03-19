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
from typing import Optional

import numpy as np

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

    def deactivate_base_auto(self, base_runner, base_postproc, reset_runner_state=False):
        """停止 BASE 自动控制，可选清理 runner 状态。"""
        self.auto_mode_base = _deactivate_base_auto(
            auto_mode_base=self.auto_mode_base,
            base_runner=base_runner,
            base_postproc=base_postproc,
            reset_runner_state=reset_runner_state,
        )

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
        if reset_timer:
            self.reset_task_timer(env)
