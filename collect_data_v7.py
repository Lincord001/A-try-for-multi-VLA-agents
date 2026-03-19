"""
collect_data_v7.py
------------------
数据采集主入口。

所有可调参数请在 collect_data_v7_modules/config.py 中修改。
其余逻辑分别位于：
  - collect_data_v7_modules/data_saver.py      (异步数据保存线程)
  - collect_data_v7_modules/dataset_utils.py   (数据集加载/创建)
  - collect_data_v7_modules/recorder_state.py  (共享状态数据类)
  - collect_data_v7_modules/auto_fsm.py        (全自动录制状态机)
  - collect_data_v7_modules/key_handlers.py    (按键处理逻辑)
"""

import sys
import numpy as np
import os
import shutil
import time

# ── 模块导入（config 必须第一个，因为它设置 CUDA_VISIBLE_DEVICES）──────────────
from collect_data_v7_modules.config import (
    INITIAL_MODE, SEED, XML_PATH, FPS,
    RANDOM_INIT_ENABLED, RANDOM_INIT_GRIPPER_OPEN,
    SELECT_SMALLER_ANGLE_MUG,
    TB3_X_RANDOM_ENABLED, TB3_X_MIN, TB3_X_MAX,
    TRAY_INIT_ON_TB3_ENABLED,
    DATASET_CONFIG, MAX_EPISODE_SEC, MAX_FRAMES,
    AUTO_RECORD_TARGET_EPISODES,
    MULTI_CONFIG_RECORDING,
    AUTO_STATE_IDLE, AUTO_STATE_EXECUTING,
)
from collect_data_v7_modules.data_saver import DataSaverWorker
from collect_data_v7_modules.dataset_utils import load_or_create_dataset
from collect_data_v7_modules.recorder_state import RecorderState
from collect_data_v7_modules.auto_fsm import step_auto_fsm
from collect_data_v7_modules.key_handlers import (
    sync_arm_expert_recording,
    handle_key_j,
    handle_key_k,
    handle_key_i,
    handle_key_p,
    handle_key_c,
    handle_arrow_keys,
    handle_key_z,
    check_timeout_circuit_breaker,
    handle_base_auto_stop,
)

from mujoco_env.y_env7 import SimpleEnv7, TABLE_Z_HEIGHT
from mujoco_env.teleop import TeleopAgent
from mujoco_env.instruction_utils import (
    validate_instruction_groups as _validate_instruction_groups,
    apply_instruction_from_group as _apply_instruction_from_group,
)


def main():
    _validate_instruction_groups()

    # ── 环境初始化 ──────────────────────────────────────────────────────────────
    random_init_mode_names = {0: "Disabled", 1: "V1 (扇形区域)", 2: "V2 (圆形交集)"}
    print(f"Initializing Environment in [{INITIAL_MODE.upper()}] mode...")
    print(f"🔥 V6 environment: Red/Blue mug (plate removed from scene)")
    print(f"🎲 Random Init: {RANDOM_INIT_ENABLED} ({random_init_mode_names.get(RANDOM_INIT_ENABLED, 'Unknown')}) (Gripper: {'Open' if RANDOM_INIT_GRIPPER_OPEN else 'Closed'})")
    print(f"🎯 Select Smaller Angle Mug: {'Enabled' if SELECT_SMALLER_ANGLE_MUG else 'Disabled'}")
    print(f"🚗 TB3 X Randomization: {'Uniform' if TB3_X_RANDOM_ENABLED else 'Fixed'} (range=[{TB3_X_MIN:.3f}, {TB3_X_MAX:.3f}])")
    print(f"🧺 Tray Task Routing: {'Enabled' if TRAY_INIT_ON_TB3_ENABLED else 'Disabled'}")

    PnPEnv = SimpleEnv7(
        XML_PATH, seed=SEED, state_type='joint_angle', 
        random_init_enabled=RANDOM_INIT_ENABLED,
        random_init_gripper_open=RANDOM_INIT_GRIPPER_OPEN,
        select_smaller_angle_mug=SELECT_SMALLER_ANGLE_MUG,
        tb3_x_random_enabled=TB3_X_RANDOM_ENABLED,
        tb3_x_min=TB3_X_MIN,
        tb3_x_max=TB3_X_MAX,
    )
    PnPEnv.reset(mode=INITIAL_MODE, options={'force_tray_init_enabled': False})
    teleop = TeleopAgent(PnPEnv)

    # ── 数据集加载 ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(" 📂 Loading/Creating Datasets for HOT-SWITCH support...")
    print("="*50)
    datasets = {
        'arm': load_or_create_dataset('arm'),
        'base': load_or_create_dataset('base'),
    }
    episode_ids = {
        'arm': datasets['arm'].num_episodes,
        'base': datasets['base'].num_episodes,
    }

    # ── 后台线程 ─────────────────────────────────────────────────────────────────
    worker = DataSaverWorker(datasets)
    worker.start()

    # ── 共享状态 ─────────────────────────────────────────────────────────────────
    state = RecorderState(
        current_mode=INITIAL_MODE,
        episode_ids=episode_ids,
        instruction_group_indices={'arm': 0, 'base': 0},
        last_instruction_by_mode={'arm': None, 'base': None},
    )

    # ── 启动提示 ─────────────────────────────────────────────────────────────────
    use_multi_config = len(MULTI_CONFIG_RECORDING) > 0
    print("\n" + "="*60)
    print(f" 🚀 DATA COLLECTION MODE: {INITIAL_MODE.upper()} (HOT-SWITCH ENABLED)")
    print(f" 📁 ARM  Dataset: {DATASET_CONFIG['arm']['repo_name']} (Episode #{episode_ids['arm']})")
    print(f" 📁 BASE Dataset: {DATASET_CONFIG['base']['repo_name']} (Episode #{episode_ids['base']})")
    print(f" 🧵 ASYNC RECORDER ACTIVE (Using Background Thread)")
    print(f" 🛡️ Safety Limit: Max {MAX_EPISODE_SEC}s ({MAX_FRAMES} frames) per episode")
    print("-"*60)
    print(" 🎮 Control Keys (通用):")
    print("  [J] : Start Recording (开始录制)")
    print("  [K] : Stop & SAVE (停止并保存)")
    print("  [I] : 🔥 DISCARD Recording (丢弃当前录制)")
    print("  [Z] : Reset Environment Only (仅重置环境)")
    print("  [C] : 🔥 HOT-SWITCH Mode (Base ↔ Arm)")
    print("  [←]/[→] : 🧭 Switch Instruction Group (切换当前模式指令组)")
    print("-"*60)
    print("  Base Mode Only (小车模式专用):")
    print("  [H] : 🚗 Base Auto Parking (base模式自动收尾停车，录制中会自动停录保存)")
    print("-"*60)
    print(" 🤖 ARM Mode Only (机械臂模式专用):")
    print("  [T] : Test Mode: Auto Execute (测试模式，不录制)")
    print("  [Y] : 🎥 Record Mode: Auto Execute + Recording + Auto Save")
    print("        → 执行完成后等待3秒，自动保存（无需手动操作）")
    print("  [I] : Discard Recording (丢弃当前录制)")
    print("  [O] : 🏠 Smooth Return Home (平滑归位机械臂)")
    print("-"*60)
    print(" 🤖🔄 FULL-AUTO Recording (全自动录制):")
    if use_multi_config:
        total_target = sum(cfg['target_episodes'] for cfg in MULTI_CONFIG_RECORDING)
        print(f"  [P] : 🔥 Start/Stop Multi-Config Full-Auto Mode")
        print(f"        → Total: {len(MULTI_CONFIG_RECORDING)} configs, {total_target} episodes")
        print(f"        → Configs: {', '.join([cfg['name'] for cfg in MULTI_CONFIG_RECORDING])}")
        print("        → Auto: Reset → Check Cups → Expert → Save → Loop → Switch Config")
        print("        → Press [P] again to STOP (discard current recording)")
    else:
        print(f"  [P] : 🔥 Start/Stop Full-Auto Mode (Target: {AUTO_RECORD_TARGET_EPISODES} episodes)")
        print("        → Auto: Reset → Check Cups → Expert → Save → Loop")
        print("        → Press [P] again to STOP (discard current recording)")
    print("-"*60)
    print(f" Current Mode: {INITIAL_MODE.upper()} | Next Episode ID: {episode_ids[INITIAL_MODE]}")
    print("="*60 + "\n")

    # 使用当前模式的指令组初始化任务文本
    _apply_instruction_from_group(
        PnPEnv,
        state.current_mode,
        state.instruction_group_indices,
        state.last_instruction_by_mode,
        log_prefix="[INIT]",
        reinitialize_arm=(state.current_mode == 'arm'),
    )

    # ── 主循环 ───────────────────────────────────────────────────────────────────
    try:
        while PnPEnv.env.is_viewer_alive() and not state.auto_shutdown_requested:
            PnPEnv.step_env()
            
            if not PnPEnv.env.loop_every(HZ=FPS):
                continue
                
                # 获取当前模式的数据集（方便后续使用）
            dataset = datasets[state.current_mode]
            current_episode_id = state.episode_ids[state.current_mode]

            # ── 全自动录制状态机 ────────────────────────────────────────────────
            if step_auto_fsm(state, PnPEnv, worker, dataset, teleop):
                continue  # FSM 要求跳过本帧（mug 选择重置）

            # ── 按键处理 ────────────────────────────────────────────────────────
            # Y键专家录制同步（每帧检查，非单次按键）
            sync_arm_expert_recording(state, PnPEnv, worker, dataset, current_episode_id)

            # 单次按键
            handle_key_j(state, PnPEnv, worker, dataset, current_episode_id)
            handle_key_k(state, PnPEnv, worker, dataset, current_episode_id)
            handle_key_i(state, PnPEnv, worker, dataset, current_episode_id)
            handle_key_p(state, PnPEnv, worker, dataset)
            handle_key_c(state, PnPEnv, worker, datasets)
            handle_arrow_keys(state, PnPEnv, worker)

            # ── 超时熔断 ────────────────────────────────────────────────────────
            check_timeout_circuit_breaker(state, PnPEnv, worker, dataset, current_episode_id)

            # ── 图像预采集（在 step 之前，与部署环境保持一致）──────────────────
            auto_is_recording = (
                state.auto_state == AUTO_STATE_EXECUTING
                and (PnPEnv.expert_pending or PnPEnv.expert_executing)
            )
            images_dict_safe = None
            if state.is_recording or auto_is_recording:
                images_dict_raw = PnPEnv.grab_image()
                images_dict_safe = {k: v.copy() for k, v in images_dict_raw.items()}

            # ── 机器人控制 / Z键重置 ────────────────────────────────────────────
            if hasattr(PnPEnv, 'base_auto_recording_active'):
                PnPEnv.base_auto_recording_active = bool(state.current_mode == 'base' and state.is_recording)
            action, reset_requested = teleop.get_action(mode=state.current_mode)

            handle_key_z(state, PnPEnv, worker, dataset, teleop, reset_requested, current_episode_id)

            # ── 物理仿真步进 ─────────────────────────────────────────────────────
            current_state = PnPEnv.step(
                action,
                mode=state.current_mode,
                action_type='eef_pose' if state.current_mode == 'arm' else None,
            )

            # ── H键 Base 自动停车保存 ────────────────────────────────────────────
            # 注意：dataset/current_episode_id 此处可能已被 handle_key_c 更新，重新读取
            dataset = datasets[state.current_mode]
            current_episode_id = state.episode_ids[state.current_mode]
            if handle_base_auto_stop(state, PnPEnv, worker, dataset, current_episode_id):
                continue  # H 自动保存完成，跳过本帧剩余逻辑

            # ── 数据记录 ─────────────────────────────────────────────────────────
            if state.current_mode == 'arm':
                action_to_save = PnPEnv.current_arm_q[:7].astype(np.float32)
            else:
                if hasattr(PnPEnv, 'get_base_action_intent'):
                    action_to_save = PnPEnv.get_base_action_intent().astype(np.float32)
                else:
                    action_to_save = action.astype(np.float32)

            if (state.is_recording or auto_is_recording) and images_dict_safe is not None:
                if state.current_mode == 'base':
                    pos = PnPEnv.env.get_p_body('tb3_base')
                    rot = PnPEnv.env.get_R_body('tb3_base')
                    theta = np.arctan2(rot[1, 0], rot[0, 0])
                    base_pose = np.array([pos[0], pos[1], theta], dtype=np.float32)
                    worker.put((state.current_mode, images_dict_safe, current_state, action_to_save,
                                PnPEnv.obj_init_pose, PnPEnv.instruction, base_pose))
                else:
                    worker.put((state.current_mode, images_dict_safe, current_state, action_to_save,
                                PnPEnv.obj_init_pose, PnPEnv.instruction))

                state.current_frames += 1
                if state.current_frames % FPS == 0:
                    q_size = worker.qsize()
                    mem_mb = q_size * 1.5
                    warn = " ⚠️ HIGH MEM!" if q_size > 500 else (" 📈" if q_size > 200 else "")
                    print(f"   [{state.current_mode.upper()}] Recording... {state.current_frames} frames | Queue: {q_size} (~{mem_mb:.0f}MB){warn}    ", end='\r')

            # ── 渲染 ─────────────────────────────────────────────────────────────
            PnPEnv.render(teleop=True, idx=current_episode_id)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
        if state.auto_state != AUTO_STATE_IDLE:
            print(f"   🛑 Full-Auto Mode was active. Progress: {state.auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES} episodes saved.")
            print(f"   ❌ Current recording (if any) will be discarded.")

    finally:
        # ── 清理 ─────────────────────────────────────────────────────────────────
        print("\n🛑 Stopping worker thread...")
        worker.running = False
        worker.wait_queue_empty()
        worker.join(timeout=5.0)
        if worker.is_alive():
            print("⚠️ Warning: Worker thread did not exit in time")
        else:
            print("✅ Worker thread stopped")
        
        PnPEnv.env.close_viewer()
        
        print("⏳ Waiting for image writers to finish...")
        time.sleep(2.0)
        
        for mode_name, ds in datasets.items():
            images_path = ds.root / 'images'
            if os.path.exists(images_path):
                max_retries = 3
                retry_delay = 1.0
                for attempt in range(max_retries):
                    try:
                        shutil.rmtree(images_path)
                        print(f"✅ [{mode_name.upper()}] Cleaned up images directory")
                        break
                    except OSError as e:
                        if attempt < max_retries - 1:
                            print(f"⚠️ [{mode_name.upper()}] Retry {attempt + 1}/{max_retries}: Waiting before retry...")
                            time.sleep(retry_delay)
                        else:
                            print(f"⚠️ [{mode_name.upper()}] Warning: Could not fully remove images directory: {e}")
                            print("   (This is usually harmless - files may still be in use)")
        
        print("\n📊 Final Episode Counts:")
        print(f"   ARM:  {state.episode_ids['arm']} episodes")
        print(f"   BASE: {state.episode_ids['base']} episodes")


if __name__ == "__main__":
    main()
