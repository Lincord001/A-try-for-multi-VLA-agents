"""
recorder_state.py
-----------------
RecorderState: 持有主循环中所有共享可变状态的数据类。

将原先散落在 main() 中通过 nonlocal 共享的局部变量收拢到一个对象，
使得 auto_fsm.py 和 key_handlers.py 可以直接读写同一个 state 实例，
不再依赖闭包或 nonlocal。
"""

import dataclasses
from typing import Optional

from .config import AUTO_STATE_IDLE, MULTI_CONFIG_RECORDING


@dataclasses.dataclass
class RecorderState:
    # ---------- 运行时模式 ----------
    current_mode: str
    episode_ids: dict
    instruction_group_indices: dict
    last_instruction_by_mode: dict

    # ---------- 手动/Y键录制 ----------
    is_recording: bool = False
    current_frames: int = 0

    # ARM 专家录制 post-wait 流程
    arm_post_wait_active: bool = False
    arm_post_wait_counter: int = 0
    arm_auto_save_pending: bool = False
    arm_waiting_for_save: bool = False
    arm_last_queue_display: int = -1
    arm_queue_line_printed: bool = False

    # ---------- 全自动录制状态机 ----------
    auto_state: int = AUTO_STATE_IDLE
    auto_recorded_count: int = 0
    auto_total_recorded_count: int = 0
    auto_wait_counter: int = 0
    auto_reset_retries: int = 0
    auto_shutdown_requested: bool = False
    auto_waiting_for_random_init: bool = False
    auto_mug_selection_reset_count: int = 0
    auto_instruction_prepared: bool = False

    # 多配置支持
    auto_current_config_idx: int = -1
    auto_current_config: Optional[dict] = None

    # ---------- 派生只读属性 ----------
    @property
    def auto_configs(self):
        """返回配置列表（为空则 None，表示单配置模式）"""
        return MULTI_CONFIG_RECORDING if len(MULTI_CONFIG_RECORDING) > 0 else None

    @property
    def auto_use_multi_config(self) -> bool:
        return self.auto_configs is not None

    # ---------- 辅助方法 ----------
    def reset_arm_recording_state(self, env=None, clear_expert_request: bool = False):
        """
        统一清理 ARM 录制流程相关标志，避免多处分支重复赋值。

        Parameters:
            env: PnPEnv 实例（可选），用于同步环境内部录制状态。
            clear_expert_request: 是否同时清除环境的 expert_record_requested 标志。
        """
        self.is_recording = False
        if self.current_mode == 'arm' and env is not None:
            env.is_recording = False
            if clear_expert_request:
                env.expert_record_requested = False
        self.arm_post_wait_active = False
        self.arm_post_wait_counter = 0
        self.arm_auto_save_pending = False
        self.arm_waiting_for_save = False
        self.arm_last_queue_display = -1
        self.arm_queue_line_printed = False
