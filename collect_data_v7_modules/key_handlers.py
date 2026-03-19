"""
key_handlers.py
---------------
所有 GLFW 按键处理函数，以及每帧需要调用的 ARM 专家录制同步逻辑、
Base 模式自动停车保存逻辑和录制超时熔断逻辑。

每个函数接受 RecorderState 和相关依赖，直接修改 state，不使用 nonlocal。

特殊返回值：
  handle_base_auto_stop() 返回 True 表示调用方应 `continue`（跳过本帧剩余逻辑）。
"""

import time
import glfw

from mujoco_env.instruction_utils import (
    INSTRUCTION_GROUPS,
    get_group_info as _get_group_info,
    apply_instruction_from_group as _apply_instruction_from_group,
)

from .config import (
    SELECT_SMALLER_ANGLE_MUG,
    TRAY_INIT_ON_TB3_ENABLED,
    AUTO_RECORD_TARGET_EPISODES,
    MAX_EPISODE_SEC,
    MAX_FRAMES,
    FPS,
    ARM_POST_EXEC_WAIT_FRAMES,
    DATASET_CONFIG,
    AUTO_STATE_IDLE,
    AUTO_STATE_RESETTING,
)
from .recorder_state import RecorderState


# =============================================================================
# 🔥 Y 键 / ARM 专家录制同步逻辑（每帧调用）
# =============================================================================

def sync_arm_expert_recording(
    state: RecorderState,
    env,
    worker,
    dataset,
    current_episode_id: int,
) -> None:
    """
    每帧检查 ARM 专家录制信号，驱动 post-wait 流程和自动保存逻辑。
    仅在 current_mode == 'arm' 且 auto_state == AUTO_STATE_IDLE 时生效。
    """
    if state.current_mode != 'arm' or state.auto_state != AUTO_STATE_IDLE:
        return

    expert_running = bool(env.expert_pending or env.expert_executing)

    # Y键启动后，由环境发出"请求录制"信号，采集脚本负责真正启停与保存流程
    if (
        expert_running
        and getattr(env, 'expert_record_requested', False)
        and not state.is_recording
        and not state.arm_post_wait_active
        and not state.arm_waiting_for_save
    ):
        if worker.saving_in_progress:
            print(f"\n⚠️ Cannot start recording: Save operation in progress. Please wait...")
        else:
            state.is_recording = True
            state.current_frames = 0
            state.arm_post_wait_active = False
            state.arm_post_wait_counter = 0
            state.arm_auto_save_pending = True
            state.arm_waiting_for_save = False
            state.arm_last_queue_display = -1
            state.arm_queue_line_printed = False
            env.expert_record_requested = False
            if worker.clear_queue():
                worker.wait_queue_empty()
                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                    dataset.clear_episode_buffer()
                print(f"🔴 [REC START] [{state.current_mode.upper()}] Recording Episode {current_episode_id} (Auto-start from Expert Policy)...")

    # 专家轨迹结束后进入 post-wait（继续录制几秒缓冲帧）
    if state.is_recording and not expert_running and not state.arm_post_wait_active:
        state.arm_post_wait_active = True
        state.arm_post_wait_counter = ARM_POST_EXEC_WAIT_FRAMES
        print(f"\n⏳ [POST-WAIT] Expert finished. Keep recording for {ARM_POST_EXEC_WAIT_FRAMES / FPS:.1f}s...")

    if state.arm_post_wait_active:
        state.arm_post_wait_counter -= 1
        if state.arm_post_wait_counter <= 0:
            state.arm_post_wait_active = False
            state.is_recording = False
            state.arm_waiting_for_save = True
            state.arm_last_queue_display = -1
            state.arm_queue_line_printed = False
            print(f"\n⏸️ [REC PAUSED] Recording stopped after post-wait period ({state.current_frames} frames buffered)")
            if state.arm_auto_save_pending:
                print(f"   🔄 Auto-saving mode: Waiting for queue to clear...")
            else:
                print(f"   👉 Press [K] to SAVE, or [I] to DISCARD.")

    # 🔥 自动保存逻辑：post-wait 结束且队列清空后自动保存
    if state.arm_auto_save_pending and state.arm_waiting_for_save:
        if not worker.saving_in_progress:
            if worker.qsize() == 0:
                worker.wait_queue_empty()
                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0:
                    try:
                        worker.saving_in_progress = True
                        print(f"\n💾 [AUTO-SAVE] Saving Episode {current_episode_id}...")
                        dataset.save_episode()
                        print(f"✅ [AUTO-SAVED] [{state.current_mode.upper()}] Episode {current_episode_id} saved ({state.current_frames} frames).")
                        state.episode_ids[state.current_mode] += 1
                        state.arm_auto_save_pending = False
                        state.arm_waiting_for_save = False
                        state.arm_last_queue_display = -1
                        state.arm_queue_line_printed = False
                        worker.saving_in_progress = False
                    except Exception as e:
                        print(f"   ❌ Auto-save error: {e}")
                        worker.saving_in_progress = False
                else:
                    print(f"⚠️ [AUTO-SAVE] No data to save. Discarding.")
                    state.arm_auto_save_pending = False
                    state.arm_waiting_for_save = False
                    state.arm_last_queue_display = -1
                    state.arm_queue_line_printed = False

    # 🔥 独立的队列状态显示逻辑（在等待保存/丢弃期间持续更新）
    if state.arm_waiting_for_save:
        queue_remaining = worker.qsize()
        if queue_remaining != state.arm_last_queue_display or not state.arm_queue_line_printed:
            state.arm_last_queue_display = queue_remaining
            if queue_remaining > 0:
                print(f"\r   📊 Queue: {queue_remaining:4d} frames still processing in background...   ", end='', flush=True)
                state.arm_queue_line_printed = True
            else:
                print(f"\r   📊 Queue: All frames processed.                                   ")
                state.arm_queue_line_printed = True


# =============================================================================
# [J] 开始录制
# =============================================================================

def handle_key_j(
    state: RecorderState,
    env,
    worker,
    dataset,
    current_episode_id: int,
) -> None:
    """[J] 开始录制 (全自动模式下禁用)"""
    if not env.env.is_key_pressed_once(glfw.KEY_J):
        return

    if state.auto_state != AUTO_STATE_IDLE:
        print(f"\n⚠️ [J] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
    elif not state.is_recording:
        if worker.saving_in_progress:
            print(f"\n⚠️ [J] Cannot start recording: Save operation in progress. Please wait...")
        else:
            state.is_recording = True
            state.current_frames = 0
            if worker.clear_queue():
                worker.wait_queue_empty()
                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                    dataset.clear_episode_buffer()
                print(f"🔴 [REC START] [{state.current_mode.upper()}] Recording Episode {current_episode_id} ...")


# =============================================================================
# [K] 停止并保存
# =============================================================================

def handle_key_k(
    state: RecorderState,
    env,
    worker,
    dataset,
    current_episode_id: int,
) -> None:
    """[K] 停止并保存 (全自动模式下禁用)"""
    if not env.env.is_key_pressed_once(glfw.KEY_K):
        return

    if state.auto_state != AUTO_STATE_IDLE:
        print(f"\n⚠️ [K] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
        return

    has_buffered = (
        state.current_mode == 'arm'
        and hasattr(dataset, 'episode_buffer')
        and dataset.episode_buffer is not None
        and len(dataset.episode_buffer) > 0
    )
    if not (state.is_recording or has_buffered):
        return

    if worker.saving_in_progress:
        print(f"\n⚠️ [K] Save operation already in progress. Please wait...")
        return

    state.is_recording = False
    if state.current_mode == 'arm':
        state.reset_arm_recording_state(env=env, clear_expert_request=False)

    worker.saving_in_progress = True
    pending_frames = worker.qsize()
    peak = worker.peak_qsize()
    print(f"\n⏳ Saving Episode {current_episode_id}...")
    print(f"   📊 Recorded: {state.current_frames} frames | Queue backlog: {pending_frames} | Peak: {peak}")

    try:
        while worker.qsize() > 0:
            remaining = worker.qsize()
            progress = (pending_frames - remaining) / max(pending_frames, 1) * 100
            print(f"   ⏳ Processing: {pending_frames - remaining}/{pending_frames} ({progress:.0f}%) - {remaining} frames remaining in queue...", end='\r')
            time.sleep(0.5)
        worker.wait_queue_empty()
        print(f"\n   ✅ Queue cleared!                                          ")

        dataset.save_episode()
        print(f"✅ [SAVED] [{state.current_mode.upper()}] Episode {current_episode_id} saved ({state.current_frames} frames).")
        state.episode_ids[state.current_mode] += 1

        # Base 模式：按 K 保存后，自动切换到下一条导航指令
        if state.current_mode == 'base':
            _apply_instruction_from_group(
                env,
                'base',
                state.instruction_group_indices,
                state.last_instruction_by_mode,
                log_prefix="🔄 [BASE]",
            )
    finally:
        worker.saving_in_progress = False


# =============================================================================
# [I] 丢弃录制
# =============================================================================

def handle_key_i(
    state: RecorderState,
    env,
    worker,
    dataset,
    current_episode_id: int,
) -> None:
    """[I] 丢弃录制 (全自动模式下禁用)"""
    if not env.env.is_key_pressed_once(glfw.KEY_I):
        return

    if state.auto_state != AUTO_STATE_IDLE:
        print(f"\n⚠️ [I] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
        return

    has_buffered_data = hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0
    has_queue_data = worker.qsize() > 0
    if not (state.is_recording or has_buffered_data or has_queue_data):
        return

    if worker.saving_in_progress:
        print(f"\n⚠️ [I] Cannot discard: Save operation in progress. Please wait for save to complete.")
        return

    state.is_recording = False
    if state.current_mode == 'arm':
        state.reset_arm_recording_state(env=env, clear_expert_request=True)
    if worker.clear_queue():
        worker.wait_queue_empty()
        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
            dataset.clear_episode_buffer()
        print(f"❌ [DISCARDED] [{state.current_mode.upper()}] Episode {current_episode_id} data cleared. (ID unchanged)")


# =============================================================================
# [P] 全自动录制模式 开启/停止
# =============================================================================

def handle_key_p(
    state: RecorderState,
    env,
    worker,
    dataset,
) -> None:
    """[P] 启动或停止全自动录制模式"""
    if not env.env.is_key_pressed_once(glfw.KEY_P):
        return

    if state.current_mode != 'arm':
        print(f"\n⚠️ [P] Full-Auto Mode only available in ARM mode. Current: {state.current_mode.upper()}")
        return

    auto_configs = state.auto_configs

    if state.auto_state != AUTO_STATE_IDLE:
        # 正在自动录制中，按 P 停止
        print(f"\n🛑 [P] Stopping Full-Auto Mode...")
        if state.auto_use_multi_config and state.auto_current_config is not None:
            target_episodes = state.auto_current_config['target_episodes']
            config_name = state.auto_current_config['name']
            print(f"   Recorded {state.auto_recorded_count}/{target_episodes} episodes in '{config_name}' before stop.")
            print(f"   Total recorded: {state.auto_total_recorded_count} episodes across all configs.")
        else:
            print(f"   Recorded {state.auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES} episodes before stop.")

        state.reset_arm_recording_state(env=env, clear_expert_request=True)
        env.expert_executing = False
        env.expert_pending = False

        worker.clear_queue()
        worker.wait_queue_empty()
        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
            dataset.clear_episode_buffer()

        state.auto_state = AUTO_STATE_IDLE
        state.auto_current_config_idx = -1
        state.auto_current_config = None
        state.auto_recorded_count = 0
        state.auto_total_recorded_count = 0
        print(f"❌ [AUTO-STOPPED] Current recording discarded. Full-Auto Mode DISABLED.")
    else:
        # 启动全自动录制
        if worker.saving_in_progress:
            print(f"\n⚠️ [P] Cannot start Full-Auto Mode: Save operation in progress. Please wait...")
            return

        state.reset_arm_recording_state(env=env, clear_expert_request=False)
        worker.clear_queue()
        worker.wait_queue_empty()
        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
            dataset.clear_episode_buffer()

        if state.auto_use_multi_config:
            state.auto_current_config_idx = 0
            state.auto_current_config = auto_configs[0]
            state.auto_recorded_count = 0
            state.auto_total_recorded_count = 0

            total_target = sum(cfg['target_episodes'] for cfg in auto_configs)

            print(f"\n" + "="*60)
            print(f" 🤖🔄 MULTI-CONFIG FULL-AUTO MODE ACTIVATED 🔄🤖")
            print(f" 📋 Total Configs: {len(auto_configs)}")
            print(f" 🎯 Total Target: {total_target} episodes")
            print(f" 📁 Dataset: {DATASET_CONFIG['arm']['repo_name']}")
            print(f" 🔢 Starting Episode ID: {state.episode_ids['arm']}")
            print(f" ⏱️ Press [P] again to STOP at any time")
            print(f"="*60)
            print(f"\n📋 Config List:")
            for idx, cfg in enumerate(auto_configs):
                marker = "👉" if idx == 0 else "  "
                cfg_select_smaller = cfg.get('select_smaller_angle_mug', SELECT_SMALLER_ANGLE_MUG)
                cfg_tray_init = cfg.get('tray_init_on_tb3_enabled', TRAY_INIT_ON_TB3_ENABLED)
                print(f"   {marker} [{idx+1}] {cfg['name']}: {cfg['target_episodes']} episodes")
                print(f"       Random Init: {cfg['random_init_enabled']}, Gripper: {'Open' if cfg['random_init_gripper_open'] else 'Closed'}, Select Smaller Angle: {'Enabled' if cfg_select_smaller else 'Disabled'}")
                print(f"       Tray Task Routing: {'Enabled' if cfg_tray_init else 'Disabled'}")
            print(f"="*60)
            print(f"\n🔄 [AUTO] Starting Config 1/{len(auto_configs)}: '{state.auto_current_config['name']}'")
            print(f"   Target: {state.auto_current_config['target_episodes']} episodes")
            config_select_smaller_angle = state.auto_current_config.get('select_smaller_angle_mug', SELECT_SMALLER_ANGLE_MUG)
            config_tray_init_on_tb3 = state.auto_current_config.get('tray_init_on_tb3_enabled', TRAY_INIT_ON_TB3_ENABLED)
            print(f"   Random Init: {state.auto_current_config['random_init_enabled']} (Gripper: {'Open' if state.auto_current_config['random_init_gripper_open'] else 'Closed'})")
            print(f"   Select Smaller Angle Mug: {'Enabled' if config_select_smaller_angle else 'Disabled'}")
            print(f"   Tray Task Routing: {'Enabled' if config_tray_init_on_tb3 else 'Disabled'}")
        else:
            state.auto_recorded_count = 0
            state.auto_total_recorded_count = 0
            print(f"\n" + "="*60)
            print(f" 🤖🔄 FULL-AUTO MODE ACTIVATED 🔄🤖")
            print(f" 🎯 Target: {AUTO_RECORD_TARGET_EPISODES} episodes")
            print(f" 📁 Dataset: {DATASET_CONFIG['arm']['repo_name']}")
            print(f" 🔢 Starting Episode ID: {state.episode_ids['arm']}")
            print(f" ⏱️ Press [P] again to STOP at any time")
            print(f"="*60)

        state.auto_state = AUTO_STATE_RESETTING
        state.auto_wait_counter = 0
        state.auto_reset_retries = 0
        state.auto_mug_selection_reset_count = 0

        if state.auto_use_multi_config:
            print(f"\n🔄 [AUTO] Resetting environment (Episode {state.auto_recorded_count + 1}/{state.auto_current_config['target_episodes']}) [{state.auto_current_config['name']}]...")
        else:
            print(f"\n🔄 [AUTO] Resetting environment (Episode {state.auto_recorded_count + 1}/{AUTO_RECORD_TARGET_EPISODES})...")


# =============================================================================
# [C] 热切换 ARM ↔ BASE
# =============================================================================

def handle_key_c(
    state: RecorderState,
    env,
    worker,
    datasets: dict,
) -> None:
    """[C] 热切换模式 (全自动模式下禁用)"""
    if not env.env.is_key_pressed_once(glfw.KEY_C):
        return

    if state.auto_state != AUTO_STATE_IDLE:
        print(f"\n⚠️ [C] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
        return
    if worker.saving_in_progress:
        print(f"\n⚠️ [C] Cannot switch mode: Save operation in progress. Please wait for save to complete.")
        return

    dataset = datasets[state.current_mode]

    if state.is_recording:
        state.is_recording = False
        if state.current_mode == 'arm':
            state.reset_arm_recording_state(env=env, clear_expert_request=False)
        if worker.clear_queue():
            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                dataset.clear_episode_buffer()
            print(f"⚠️ [RECORDING STOPPED] Discarding current recording before switch.")

    worker.wait_queue_empty()

    old_mode = state.current_mode
    state.current_mode = 'arm' if state.current_mode == 'base' else 'base'

    # 清理 base 自动停车状态，避免模式切换后残留触发
    for attr, default in [
        ('base_auto_active', False),
        ('base_auto_stage', 'idle'),
        ('base_auto_wait_counter', 0),
        ('base_auto_stage_steps', 0),
        ('base_auto_record_stop_requested', False),
    ]:
        if hasattr(env, attr):
            setattr(env, attr, default)

    # 清空新模式的缓冲区（防止旧数据混入新录制）
    new_dataset = datasets[state.current_mode]
    if hasattr(new_dataset, 'episode_buffer') and new_dataset.episode_buffer is not None:
        new_dataset.clear_episode_buffer()

    env.control_mode = state.current_mode

    _apply_instruction_from_group(
        env,
        state.current_mode,
        state.instruction_group_indices,
        state.last_instruction_by_mode,
        log_prefix="🔄 [HOT-SWITCH]",
    )

    env.grab_image()

    print(f"\n🔄 [HOT-SWITCH] {old_mode.upper()} → {state.current_mode.upper()}")
    print(f"   📍 Environment state preserved (no reset)")
    print(f"   📁 Now using: {DATASET_CONFIG[state.current_mode]['repo_name']}")
    print(f"   🔢 Next Episode ID: {state.episode_ids[state.current_mode]}\n")


# =============================================================================
# [←]/[→] 切换指令组
# =============================================================================

def handle_arrow_keys(
    state: RecorderState,
    env,
    worker,
) -> None:
    """[←]/[→] 切换当前模式的指令组 (全自动模式下禁用)"""
    left_pressed = env.env.is_key_pressed_once(glfw.KEY_LEFT)
    right_pressed = env.env.is_key_pressed_once(glfw.KEY_RIGHT)
    if not (left_pressed or right_pressed):
        return

    if state.auto_state != AUTO_STATE_IDLE:
        print(f"\n⚠️ [←/→] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
        return
    if worker.saving_in_progress:
        print(f"\n⚠️ [←/→] Cannot switch instruction group: Save operation in progress.")
        return
    if state.is_recording:
        print(f"\n⚠️ [←/→] Cannot switch instruction group while recording. Stop/discard current episode first.")
        return

    groups = INSTRUCTION_GROUPS[state.current_mode]
    if len(groups) <= 1:
        idx, total, group_name, _ = _get_group_info(state.current_mode, state.instruction_group_indices)
        print(f"\nℹ️ [{state.current_mode.upper()}] Only one instruction group: {group_name} ({idx + 1}/{total})")
    else:
        if right_pressed:
            state.instruction_group_indices[state.current_mode] = (state.instruction_group_indices[state.current_mode] + 1) % len(groups)
        else:
            state.instruction_group_indices[state.current_mode] = (state.instruction_group_indices[state.current_mode] - 1) % len(groups)
        _apply_instruction_from_group(
            env,
            state.current_mode,
            state.instruction_group_indices,
            state.last_instruction_by_mode,
            log_prefix="🔁 [GROUP SWITCH]",
            reinitialize_arm=(state.current_mode == 'arm'),
        )


# =============================================================================
# [Z] 手动重置环境 (reset 信号来自 teleop.get_action())
# =============================================================================

def handle_key_z(
    state: RecorderState,
    env,
    worker,
    dataset,
    teleop,
    reset_requested: bool,
    current_episode_id: int,
) -> None:
    """[Z] 手动重置环境 (全自动模式下禁用)"""
    if not reset_requested:
        return

    if state.auto_state != AUTO_STATE_IDLE:
        print(f"\n⚠️ [Z] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
        return
    if worker.saving_in_progress:
        print(f"\n⚠️ [Z] Cannot reset: Save operation in progress. Please wait for save to complete.")
        return

    print("🔄 Environment Reset.")
    if state.is_recording:
        state.is_recording = False
        if state.current_mode == 'arm':
            state.reset_arm_recording_state(env=env, clear_expert_request=False)
        print(f"⚠️ [INTERRUPTED] Recording stopped due to reset. (ID {current_episode_id})")
    # 也需要重置等待状态（即使不在录制中，可能是暂停状态）
    if state.current_mode == 'arm':
        state.reset_arm_recording_state(env=env, clear_expert_request=True)

    env.reset(mode=state.current_mode, preserve_instruction=True)
    teleop.reset()
    if worker.clear_queue():
        worker.wait_queue_empty()
        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
            dataset.clear_episode_buffer()


# =============================================================================
# 🛡️ 录制超时熔断逻辑
# =============================================================================

def check_timeout_circuit_breaker(
    state: RecorderState,
    env,
    worker,
    dataset,
    current_episode_id: int,
) -> None:
    """录制超时时自动丢弃当前录制"""
    if not (state.is_recording and state.current_frames >= MAX_FRAMES):
        return

    state.is_recording = False
    if state.current_mode == 'arm':
        state.reset_arm_recording_state(env=env, clear_expert_request=False)

    if worker.saving_in_progress:
        print(f"\n⚠️ [TIMEOUT] Max duration reached, but save operation in progress. Will discard after save completes.")
        return

    if worker.clear_queue():
        worker.wait_queue_empty()
        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
            dataset.clear_episode_buffer()
        print(f"\n⚠️ [TIMEOUT] Max duration ({MAX_EPISODE_SEC}s) reached!")
        print(f"❌ [AUTO-DISCARDED] [{state.current_mode.upper()}] Episode {current_episode_id} data cleared. (ID unchanged)")


# =============================================================================
# 🚗 Base 模式：H 键自动停车后的录制保存逻辑（每帧检查）
# =============================================================================

def handle_base_auto_stop(
    state: RecorderState,
    env,
    worker,
    dataset,
    current_episode_id: int,
) -> bool:
    """
    Base 模式 H 键自动停车流程结束后，按当前录制状态决定是否自动停录保存。

    Returns:
        True  → 调用方应 `continue`（跳过本帧剩余的数据采集逻辑）
        False → 正常继续
    """
    if state.current_mode != 'base' or not getattr(env, 'base_auto_record_stop_requested', False):
        return False

    h_autosave_just_finished = False

    if worker.saving_in_progress:
        print("\n⚠️ [H] Auto-stop requested, but save operation is in progress. Waiting...")
        return False

    if state.is_recording:
        state.is_recording = False
        worker.saving_in_progress = True
        pending_frames = worker.qsize()
        peak = worker.peak_qsize()
        print(f"\n🚗 [H] Auto parking completed. Auto-stopping recording for Episode {current_episode_id}...")
        print(f"   📊 Recorded: {state.current_frames} frames | Queue backlog: {pending_frames} | Peak: {peak}")
        try:
            while worker.qsize() > 0:
                remaining = worker.qsize()
                progress = (pending_frames - remaining) / max(pending_frames, 1) * 100
                print(
                    f"   ⏳ Processing: {pending_frames - remaining}/{pending_frames} "
                    f"({progress:.0f}%) - {remaining} frames remaining in queue...",
                    end='\r'
                )
                time.sleep(0.5)
            worker.wait_queue_empty()
            print(f"\n   ✅ Queue cleared!                                          ")
            dataset.save_episode()
            print(f"✅ [AUTO-SAVED] [{state.current_mode.upper()}] Episode {current_episode_id} saved ({state.current_frames} frames).")
            state.episode_ids[state.current_mode] += 1
            # H 键自动停车在保存后自动重置环境
            try:
                env.reset(mode=state.current_mode, force_fixed_arm_init=True)
                print("🔄 [H] Environment reset after auto-save. Red/Blue mugs reinitialized, arm reset to fixed home pose.")
            except Exception as reset_err:
                print(f"⚠️ [H] Auto-save succeeded, but post-save reset failed: {reset_err}")
            h_autosave_just_finished = True
        finally:
            worker.saving_in_progress = False
    else:
        print("\n🚗 [H] Auto parking completed. Recording was not active, skip auto-save.")

    # 只消费一次完成信号
    env.base_auto_record_stop_requested = False

    if h_autosave_just_finished:
        state.current_frames = 0
        return True  # caller should `continue`

    return False
