"""
auto_fsm.py
-----------
step_auto_fsm(): 全自动录制状态机，每帧调用一次。

状态转移图：
  IDLE → (P键) → RESETTING → CHECK_CUPS → START_EXPERT → EXECUTING
       → WAIT_QUEUE → SAVING → POST_SAVE → RESETTING (循环)
                            └→ SWITCHING_CONFIG → RESETTING (多配置)
                            └→ IDLE (全部完成)
"""

from mujoco_env.instruction_utils import (
    route_arm_group_for_auto as _route_arm_group_for_auto,
    apply_instruction_from_group as _apply_instruction_from_group,
)

from .config import (
    RANDOM_INIT_ENABLED,
    RANDOM_INIT_GRIPPER_OPEN,
    SELECT_SMALLER_ANGLE_MUG,
    TB3_X_RANDOM_ENABLED,
    TB3_X_MIN,
    TB3_X_MAX,
    TRAY_INIT_ON_TB3_ENABLED,
    AUTO_RESET_WAIT_FRAMES,
    AUTO_CUP_CHECK_TOLERANCE,
    AUTO_POST_SAVE_WAIT_FRAMES,
    AUTO_MAX_RESET_RETRIES,
    AUTO_RECORD_TARGET_EPISODES,
    AUTO_SHUTDOWN_ON_COMPLETE,
    AUTO_STATE_IDLE,
    AUTO_STATE_RESETTING,
    AUTO_STATE_CHECK_CUPS,
    AUTO_STATE_START_EXPERT,
    AUTO_STATE_EXECUTING,
    AUTO_STATE_WAIT_QUEUE,
    AUTO_STATE_SAVING,
    AUTO_STATE_POST_SAVE,
    AUTO_STATE_SWITCHING_CONFIG,
)
from .recorder_state import RecorderState


def step_auto_fsm(
    state: RecorderState,
    env,
    worker,
    dataset,
    teleop,
) -> bool:
    """
    执行一帧的全自动录制状态机逻辑。

    Parameters:
        state   : RecorderState 实例（读写）
        env     : PnPEnv 实例
        worker  : DataSaverWorker 实例
        dataset : 当前模式的 LeRobotDataset
        teleop  : TeleopAgent 实例

    Returns:
        True  → 调用方应跳过本帧后续的数据采集逻辑（continue）
        False → 正常继续
    """
    if state.auto_state == AUTO_STATE_IDLE or state.current_mode != 'arm':
        return False

    # ----- STATE: RESETTING -----
    if state.auto_state == AUTO_STATE_RESETTING:
        # 构建 reset options
        if state.auto_use_multi_config and state.auto_current_config is not None:
            config_random_init = state.auto_current_config['random_init_enabled']
            config_tray_init_on_tb3 = state.auto_current_config.get('tray_init_on_tb3_enabled', TRAY_INIT_ON_TB3_ENABLED)
            reset_opts = {
                'random_init_enabled': config_random_init,
                'random_init_gripper_open': state.auto_current_config['random_init_gripper_open'],
                'select_smaller_angle_mug': state.auto_current_config.get('select_smaller_angle_mug', SELECT_SMALLER_ANGLE_MUG),
                'tb3_x_random_enabled': state.auto_current_config.get('tb3_x_random_enabled', TB3_X_RANDOM_ENABLED),
                'tb3_x_min': state.auto_current_config.get('tb3_x_min', TB3_X_MIN),
                'tb3_x_max': state.auto_current_config.get('tb3_x_max', TB3_X_MAX),
                'force_tray_init_enabled': False,
            }
        else:
            config_random_init = RANDOM_INIT_ENABLED
            config_tray_init_on_tb3 = TRAY_INIT_ON_TB3_ENABLED
            reset_opts = {
                'random_init_enabled': config_random_init,
                'random_init_gripper_open': RANDOM_INIT_GRIPPER_OPEN,
                'select_smaller_angle_mug': SELECT_SMALLER_ANGLE_MUG,
                'tb3_x_random_enabled': TB3_X_RANDOM_ENABLED,
                'tb3_x_min': TB3_X_MIN,
                'tb3_x_max': TB3_X_MAX,
                'force_tray_init_enabled': False,
            }

        routed_group_name = _route_arm_group_for_auto(
            state.instruction_group_indices,
            tray_init_on_tb3_enabled=bool(config_tray_init_on_tb3),
        )
        print(
            f"   🧭 [AUTO-ROUTE] tray_init_on_tb3_enabled={bool(config_tray_init_on_tb3)} "
            f"-> arm group '{routed_group_name}'"
        )

        _apply_instruction_from_group(
            env,
            'arm',
            state.instruction_group_indices,
            state.last_instruction_by_mode,
            log_prefix="[AUTO-PREP]",
        )
        state.auto_instruction_prepared = True

        env.reset(mode='arm', preserve_instruction=True, options=reset_opts)
        teleop.reset()
        # 清空残留数据
        worker.clear_queue()
        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
            dataset.clear_episode_buffer()
        state.auto_wait_counter = AUTO_RESET_WAIT_FRAMES
        state.auto_waiting_for_random_init = False  # 重置标志
        # 🔥 注意：auto_mug_selection_reset_count 不在这里重置，保持状态直到下一轮录制开始
        state.auto_state = AUTO_STATE_CHECK_CUPS
        # 🔥 检查是否启用了随机初始化（使用当前配置的参数）
        if config_random_init != 0 and getattr(env, 'moving_to_random', False):
            print(f"   ⏳ Waiting for random initialization to complete (will take ~3.8s)...")
            state.auto_waiting_for_random_init = True
        else:
            print(f"   ⏳ Waiting {AUTO_RESET_WAIT_FRAMES} frames for physics to stabilize...")

    # ----- STATE: CHECK_CUPS (等待物理稳定 + 等待随机初始化完成 + 检查物体) -----
    elif state.auto_state == AUTO_STATE_CHECK_CUPS:
        # 🔥 首先检查是否正在移动到随机位置（如果启用了随机初始化）
        if getattr(env, 'moving_to_random', False):
            # 还在移动到随机位置，继续等待，不检查物体
            if state.auto_wait_counter % 20 == 0:
                print(f"   ⏳ Waiting for random initialization to complete... (moving_to_random=True)", end='\r')
        else:
            # 随机初始化已完成（或未启用）
            if state.auto_waiting_for_random_init:
                state.auto_wait_counter = AUTO_RESET_WAIT_FRAMES
                state.auto_waiting_for_random_init = False
                print(f"   ✅ Random initialization completed. Waiting {AUTO_RESET_WAIT_FRAMES} frames for physics to stabilize...")

            # 继续等待物理稳定
            state.auto_wait_counter -= 1
            if state.auto_wait_counter <= 0:
                objects_fallen = env.check_objects_fallen(tolerance=AUTO_CUP_CHECK_TOLERANCE)
                objects_ok = not objects_fallen
                if not objects_ok:
                    print(f"   ⚠️ Red mug fallen!")

                if objects_ok:
                    state.auto_reset_retries = 0
                    state.auto_state = AUTO_STATE_START_EXPERT
                    print(f"   ✅ Red mug OK. Starting expert policy...")
                else:
                    state.auto_reset_retries += 1
                    if state.auto_reset_retries >= AUTO_MAX_RESET_RETRIES:
                        print(f"   ❌ Max reset retries ({AUTO_MAX_RESET_RETRIES}) reached! Stopping Full-Auto Mode.")
                        state.auto_state = AUTO_STATE_IDLE
                    else:
                        print(f"   🔄 Retry {state.auto_reset_retries}/{AUTO_MAX_RESET_RETRIES}: Resetting environment again...")
                        state.auto_state = AUTO_STATE_RESETTING

    # ----- STATE: START_EXPERT -----
    elif state.auto_state == AUTO_STATE_START_EXPERT:
        if getattr(env, 'moving_to_random', False):
            pass  # 还在移动，等待
        else:
            # 🔥 确保环境已经选好了目标（在启动专家策略前强制调用一次）
            if not state.auto_instruction_prepared:
                _apply_instruction_from_group(
                    env,
                    'arm',
                    state.instruction_group_indices,
                    state.last_instruction_by_mode,
                )
            state.auto_instruction_prepared = False

            # 🔥 检查杯子选择逻辑
            should_reset_for_mug = False
            if state.auto_use_multi_config and state.auto_current_config is not None:
                config_select_smaller_angle = state.auto_current_config.get('select_smaller_angle_mug', SELECT_SMALLER_ANGLE_MUG)
            else:
                config_select_smaller_angle = SELECT_SMALLER_ANGLE_MUG

            if config_select_smaller_angle and state.auto_mug_selection_reset_count < 2:
                target_color = getattr(env, 'target_color', None)
                if target_color != 'blue':
                    should_reset_for_mug = True
                    state.auto_mug_selection_reset_count += 1
                    print(f"   🔄 Selected mug is not blue (selected: {target_color}). Resetting environment to retry... (Attempt {state.auto_mug_selection_reset_count}/2)")

            if should_reset_for_mug:
                state.auto_state = AUTO_STATE_RESETTING
                state.auto_wait_counter = 0
                return True  # caller should `continue`

            # 清空残留状态
            state.reset_arm_recording_state(env=env, clear_expert_request=True)
            state.current_frames = 0

            # 🔥 在启动专家策略前，再次清空队列和缓冲区（防止残留数据）
            worker.clear_queue()
            worker.wait_queue_empty()
            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                dataset.clear_episode_buffer()

            # 启动专家策略（带录制）
            env.auto_execute_task(record=True)
            state.auto_state = AUTO_STATE_EXECUTING
            print(f"   🤖 Expert policy started. Recording Episode {state.episode_ids['arm']}...")

    # ----- STATE: EXECUTING (专家策略执行中) -----
    elif state.auto_state == AUTO_STATE_EXECUTING:
        if getattr(env, 'moving_to_random', False):
            pass  # 等待随机初始化
        elif not env.expert_pending and not env.expert_executing:
            state.is_recording = False
            state.auto_state = AUTO_STATE_WAIT_QUEUE
            print(f"\n   ✅ Expert execution finished. Waiting for queue to clear...")

    # ----- STATE: WAIT_QUEUE (等待队列清空) -----
    elif state.auto_state == AUTO_STATE_WAIT_QUEUE:
        queue_size = worker.qsize()
        if queue_size > 0:
            print(f"\r   ⏳ Queue: {queue_size} frames remaining...   ", end='', flush=True)
        else:
            print(f"\r   ✅ Queue cleared!                           ")
            state.auto_state = AUTO_STATE_SAVING

    # ----- STATE: SAVING -----
    elif state.auto_state == AUTO_STATE_SAVING:
        auto_configs = state.auto_configs
        try:
            worker.saving_in_progress = True
            worker.wait_queue_empty()
            dataset.save_episode()
            state.episode_ids['arm'] += 1
            state.auto_recorded_count += 1
            state.auto_total_recorded_count += 1

            if state.auto_use_multi_config and state.auto_current_config is not None:
                target_episodes = state.auto_current_config['target_episodes']
                config_name = state.auto_current_config['name']
                print(f"   ✅ [AUTO-SAVED] Episode {state.episode_ids['arm'] - 1} saved! ({state.auto_recorded_count}/{target_episodes}) [{config_name}]")
            else:
                target_episodes = AUTO_RECORD_TARGET_EPISODES
                print(f"   ✅ [AUTO-SAVED] Episode {state.episode_ids['arm'] - 1} saved! ({state.auto_recorded_count}/{target_episodes})")
        except Exception as e:
            print(f"   ❌ Save error: {e}")
        finally:
            worker.saving_in_progress = False

        # 🔥 检查当前配置是否完成
        if state.auto_use_multi_config and state.auto_current_config is not None:
            target_episodes = state.auto_current_config['target_episodes']
            if state.auto_recorded_count >= target_episodes:
                print(f"\n" + "="*60)
                print(f" ✅ Config '{state.auto_current_config['name']}' COMPLETED!")
                print(f" 📊 Recorded: {state.auto_recorded_count}/{target_episodes} episodes")
                print(f" 📊 Total recorded so far: {state.auto_total_recorded_count} episodes")
                print(f"="*60)

                state.auto_current_config_idx += 1
                if state.auto_current_config_idx < len(auto_configs):
                    state.auto_current_config = auto_configs[state.auto_current_config_idx]
                    state.auto_recorded_count = 0
                    state.auto_mug_selection_reset_count = 0
                    config_select_smaller_angle = state.auto_current_config.get('select_smaller_angle_mug', SELECT_SMALLER_ANGLE_MUG)
                    config_tray_init_on_tb3 = state.auto_current_config.get('tray_init_on_tb3_enabled', TRAY_INIT_ON_TB3_ENABLED)
                    print(f"\n🔄 Switching to Config {state.auto_current_config_idx + 1}/{len(auto_configs)}: '{state.auto_current_config['name']}'")
                    print(f"   Target: {state.auto_current_config['target_episodes']} episodes")
                    print(f"   Random Init: {state.auto_current_config['random_init_enabled']} (Gripper: {'Open' if state.auto_current_config['random_init_gripper_open'] else 'Closed'})")
                    print(f"   Select Smaller Angle Mug: {'Enabled' if config_select_smaller_angle else 'Disabled'}")
                    print(f"   Tray Task Routing: {'Enabled' if config_tray_init_on_tb3 else 'Disabled'}")
                    state.auto_state = AUTO_STATE_SWITCHING_CONFIG
                else:
                    print(f"\n" + "="*60)
                    print(f" 🎉 ALL CONFIGS COMPLETED! 🎉")
                    print(f" 📊 Total recorded: {state.auto_total_recorded_count} episodes")
                    print(f" 📁 Final Episode ID: {state.episode_ids['arm']}")
                    print(f"="*60 + "\n")
                    state.auto_state = AUTO_STATE_IDLE
                    if AUTO_SHUTDOWN_ON_COMPLETE:
                        state.auto_shutdown_requested = True
                        print("🔌 Auto-shutdown requested. Closing simulation...")
            else:
                state.auto_mug_selection_reset_count = 0
                state.auto_wait_counter = AUTO_POST_SAVE_WAIT_FRAMES
                state.auto_state = AUTO_STATE_POST_SAVE
        else:
            # 单配置模式
            if state.auto_recorded_count >= AUTO_RECORD_TARGET_EPISODES:
                print(f"\n" + "="*60)
                print(f" 🎉 FULL-AUTO MODE COMPLETED! 🎉")
                print(f" 📊 Total recorded: {state.auto_recorded_count} episodes")
                print(f" 📁 Final Episode ID: {state.episode_ids['arm']}")
                print(f"="*60 + "\n")
                state.auto_state = AUTO_STATE_IDLE
                if AUTO_SHUTDOWN_ON_COMPLETE:
                    state.auto_shutdown_requested = True
                    print("🔌 Auto-shutdown requested. Closing simulation...")
            else:
                state.auto_mug_selection_reset_count = 0
                state.auto_wait_counter = AUTO_POST_SAVE_WAIT_FRAMES
                state.auto_state = AUTO_STATE_POST_SAVE

    # ----- STATE: SWITCHING_CONFIG (配置切换) -----
    elif state.auto_state == AUTO_STATE_SWITCHING_CONFIG:
        state.auto_state = AUTO_STATE_RESETTING

    # ----- STATE: POST_SAVE (保存后等待) -----
    elif state.auto_state == AUTO_STATE_POST_SAVE:
        state.auto_wait_counter -= 1
        if state.auto_wait_counter <= 0:
            if state.auto_use_multi_config and state.auto_current_config is not None:
                target_episodes = state.auto_current_config['target_episodes']
                config_name = state.auto_current_config['name']
                print(f"\n🔄 [AUTO] Resetting for next episode ({state.auto_recorded_count + 1}/{target_episodes}) [{config_name}]...")
            else:
                print(f"\n🔄 [AUTO] Resetting for next episode ({state.auto_recorded_count + 1}/{AUTO_RECORD_TARGET_EPISODES})...")
            state.auto_state = AUTO_STATE_RESETTING

    return False
