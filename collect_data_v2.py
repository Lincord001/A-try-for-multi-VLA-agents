import sys
import numpy as np
import os
import shutil
import time
import glfw
import threading
import queue
from PIL import Image
from mujoco_env.y_env2_test import SimpleEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    🎛️ 常用配置（频繁调整区）🎛️                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- 🔥 随机初始化开关 (Random Initialization Switch) ---
RANDOM_INIT_ENABLED = 1            # 0: 关闭, 1: 旧版(扇形区域), 2: 新版(圆形交集)
RANDOM_INIT_GRIPPER_OPEN = True    # True: 初始化时夹爪张开, False: 初始化时夹爪闭合

# --- 🤖 全自动录制 [P键] ---
AUTO_RECORD_TARGET_EPISODES = 150    # 🎯 目标录制条数（达到后自动停止）
AUTO_SHUTDOWN_ON_COMPLETE = True     # 🔌 完成后是否自动关闭仿真环境

# --- 📁 数据集名称与路径 ---
ARM_DATASET_NAME = 'omy_arm_data_v2'       # Arm 模式数据集名称
ARM_DATASET_ROOT = './demo_data_arm_v2'    # Arm 模式数据集保存路径

# --- 🖼️ 图像与录制 ---
IMG_SIZE = 224                       # 图像分辨率 (224=ViT标准, 256=兼容旧数据)
FPS = 20                             # 录制帧率 (Hz)
MAX_EPISODE_SEC = 200                # 单条数据最大时长（秒），超时自动丢弃

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    ⚙️ 高级配置（一般不需要修改）⚙️                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- 全自动录制细节参数 ---
AUTO_RESET_WAIT_FRAMES = 40          # 重置后等待帧数（约2秒，让物理引擎稳定）
AUTO_CUP_CHECK_TOLERANCE = 0.05      # 物体 z 坐标容差（±5cm，超出判定为倒了）
AUTO_RED_MUG_EXPECTED_Z = 0.83       # 🔥 红色杯子正常 z 坐标
AUTO_POST_SAVE_WAIT_FRAMES = 20      # 保存后等待帧数（约1秒）
AUTO_MAX_RESET_RETRIES = 5           # 物体倒了时最大重试次数

# --- 场景配置 ---
SEED = 0 
XML_PATH = './asset/example_scene_y2.xml'  # 🔥 V2环境的XML路径

# --- 派生配置（自动计算，勿手动修改）---
MAX_FRAMES = MAX_EPISODE_SEC * FPS

DATASET_CONFIG = {
    'arm': {
        'repo_name': ARM_DATASET_NAME,
        'root': ARM_DATASET_ROOT,
    }
}

MODE_CONFIG = {
    'arm': {
        'action_shape': (7,),
        'state_shape': (7,),  # [q1, q2, q3, q4, q5, q6, gripper]
    }
}

# --- 自动录制状态机常量（内部使用，勿修改）---
AUTO_STATE_IDLE = 0
AUTO_STATE_RESETTING = 1
AUTO_STATE_CHECK_CUPS = 2
AUTO_STATE_START_EXPERT = 3
AUTO_STATE_EXECUTING = 4
AUTO_STATE_WAIT_QUEUE = 5
AUTO_STATE_SAVING = 6
AUTO_STATE_POST_SAVE = 7


# ================= 🧵 异步处理工作线程 🧵 =================

class DataSaverWorker(threading.Thread):
    """数据保存工作线程 (优化版：无阻塞 + 快速 resize) - V2环境只有arm模式"""
    def __init__(self, datasets):
        """
        Parameters:
            datasets: dict, {'arm': LeRobotDataset}
        """
        super().__init__()
        self.datasets = datasets
        self.queue = queue.Queue(maxsize=0)  # 0 = 无限大小
        self.daemon = True
        self.running = True
        self._peak_qsize = 0
        self.saving_in_progress = False

    def put(self, item):
        """主线程调用：将原始数据放入队列（永不阻塞）"""
        self.queue.put_nowait(item)
        current_size = self.queue.qsize()
        if current_size > self._peak_qsize:
            self._peak_qsize = current_size

    def qsize(self):
        """返回当前队列大小"""
        return self.queue.qsize()
    
    def peak_qsize(self):
        """返回队列峰值大小"""
        return self._peak_qsize

    @staticmethod
    def _fast_resize(img):
        """使用 PIL resize"""
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(pil_img)

    def run(self):
        """子线程循环：后台处理数据"""
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            mode = item[0]
            images_dict = item[1]
            state = item[2]
            action = item[3]
            obj_init = item[4]
            task = item[5]

            dataset = self.datasets.get(mode)
            if dataset is None:
                print(f"Warning: No dataset for mode '{mode}'")
                self.queue.task_done()
                continue

            frame_data = {
                "observation.state": state,
                "action": action,
                "obj_init": obj_init,
            }

            try:
                # V2环境只有arm模式
                frame_data["observation.images.agent"] = self._fast_resize(images_dict['agent'])
                frame_data["observation.images.wrist"] = self._fast_resize(images_dict['wrist'])

                dataset.add_frame(frame_data, task=task)
            except Exception as e:
                print(f"Error in worker thread: {e}")
            finally:
                self.queue.task_done()

    def wait_queue_empty(self):
        """等待所有数据处理完毕（录制结束后调用）"""
        self.queue.join()
    
    def clear_queue(self):
        """🔥 清空队列中所有未处理的数据（丢弃时调用）"""
        if self.saving_in_progress:
            print(f"   ⚠️ Cannot clear queue: Save operation in progress. Please wait...")
            return False
        
        cleared_count = 0
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
                cleared_count += 1
            except queue.Empty:
                break
        if cleared_count > 0:
            print(f"   🗑️ Cleared {cleared_count} frames from queue.")
        return True

# ===========================================

def load_or_create_dataset(mode='arm'):
    """加载或创建arm模式的数据集 - V2环境只有arm模式"""
    config = DATASET_CONFIG['arm']
    mode_cfg = MODE_CONFIG['arm']
    root = config['root']
    repo_name = config['repo_name']
    
    if os.path.exists(root):
        print(f"\n[{mode.upper()}] Dataset found at: {root}")
        print(">> Loading existing dataset (Append Mode)...")
        dataset = LeRobotDataset(repo_name, root=root)
        print(f">> Found {dataset.num_episodes} existing episodes.")
    else:
        print(f"\n[{mode.upper()}] No existing dataset found at: {root}")
        print(">> Creating NEW dataset...")
        
        features = {
            "observation.state": {"dtype": "float32", "shape": mode_cfg['state_shape'], "names": ["state"]},
            "action": {"dtype": "float32", "shape": mode_cfg['action_shape'], "names": ["action"]},
            "obj_init": {"dtype": "float32", "shape": (6,), "names": ["obj_init"]},  # 🔥 V2: 红色杯子(3) + 盘子(3)
        }

        # V2环境只有arm模式
        features["observation.images.agent"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
        features["observation.images.wrist"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}

        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            root=root,
            robot_type="omy",
            fps=FPS,
            features=features,
            image_writer_threads=20,
            image_writer_processes=8,
        )
    
    return dataset

def main():
    # V2环境只有arm模式
    current_mode = 'arm'
    
    print(f"Initializing Environment in ARM mode...")
    print(f"🔥 Simple Mode: Red mug only")
    random_init_mode_names = {0: "Disabled", 1: "V1 (扇形区域)", 2: "V2 (圆形交集)"}
    print(f"🎲 Random Init: {RANDOM_INIT_ENABLED} ({random_init_mode_names.get(RANDOM_INIT_ENABLED, 'Unknown')}) (Gripper: {'Open' if RANDOM_INIT_GRIPPER_OPEN else 'Closed'})")
    PnPEnv = SimpleEnv2(
        XML_PATH, seed=SEED, state_type='joint_angle', 
        random_init_enabled=RANDOM_INIT_ENABLED,
        random_init_gripper_open=RANDOM_INIT_GRIPPER_OPEN
    )
    PnPEnv.reset(mode=current_mode)

    # ---------------------------------------------------------
    # 🔥 1. 加载数据集
    print("\n" + "="*50)
    print(" 📂 Loading/Creating Dataset...")
    print("="*50)
    
    datasets = {
        'arm': load_or_create_dataset('arm'),
    }
    
    episode_ids = {
        'arm': datasets['arm'].num_episodes,
    }

    # 🔥 启动后台工作线程
    worker = DataSaverWorker(datasets)
    worker.start()

    is_recording = False
    current_frames = 0
    
    # 🤖 全自动录制状态变量
    auto_state = AUTO_STATE_IDLE
    auto_recorded_count = 0
    auto_wait_counter = 0
    auto_reset_retries = 0
    auto_shutdown_requested = False
    auto_waiting_for_random_init = False

    print("\n" + "="*60)
    print(f" 🚀 DATA COLLECTION MODE: ARM (V2环境只有arm模式)")
    print(f" 📁 Dataset: {DATASET_CONFIG['arm']['repo_name']} (Episode #{episode_ids['arm']})")
    print(f" 🧵 ASYNC RECORDER ACTIVE (Using Background Thread)")
    print(f" 🛡️ Safety Limit: Max {MAX_EPISODE_SEC}s ({MAX_FRAMES} frames) per episode")
    print("-"*60)
    print(" 🎮 Control Keys:")
    print("  [J] : Start Recording (开始录制)")
    print("  [K] : Stop & SAVE (停止并保存)")
    print("  [I] : 🔥 DISCARD Recording (丢弃当前录制)")
    print("  [Z] : Reset Environment Only (仅重置环境)")
    print("-"*60)
    print(" 🤖 ARM Mode (机械臂模式):")
    print("  [T] : Test Mode: Auto Execute (测试模式，不录制)")
    print("  [Y] : 🎥 Record Mode: Auto Execute + Recording + Auto Save")
    print("  [O] : 🏠 Smooth Return Home (平滑归位机械臂)")
    print("-"*60)
    print(" 🤖🔄 FULL-AUTO Recording (全自动录制):")
    print(f"  [P] : 🔥 Start/Stop Full-Auto Mode (Target: {AUTO_RECORD_TARGET_EPISODES} episodes)")
    print("        → Auto: Reset → Check Cups → Expert → Save → Loop")
    print("        → Press [P] again to STOP (discard current recording)")
    print("-"*60)
    print(f" Next Episode ID: {episode_ids['arm']}")
    print("="*60 + "\n")

    try:
        while PnPEnv.env.is_viewer_alive() and not auto_shutdown_requested:
            PnPEnv.step_env()
            
            if PnPEnv.env.loop_every(HZ=FPS):
                
                dataset = datasets[current_mode]
                current_episode_id = episode_ids[current_mode]
                
                # ================= 🤖 全自动录制状态机 🤖 =================
                if auto_state != AUTO_STATE_IDLE:
                    
                    # ----- STATE: RESETTING -----
                    if auto_state == AUTO_STATE_RESETTING:
                        PnPEnv.reset(mode='arm')
                        worker.clear_queue()
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()
                        auto_wait_counter = AUTO_RESET_WAIT_FRAMES
                        auto_waiting_for_random_init = False
                        auto_state = AUTO_STATE_CHECK_CUPS
                        if RANDOM_INIT_ENABLED != 0 and getattr(PnPEnv, 'moving_to_random', False):
                            print(f"   ⏳ Waiting for random initialization to complete (will take ~3.8s)...")
                            auto_waiting_for_random_init = True
                        else:
                            print(f"   ⏳ Waiting {AUTO_RESET_WAIT_FRAMES} frames for physics to stabilize...")
                    
                    # ----- STATE: CHECK_CUPS -----
                    elif auto_state == AUTO_STATE_CHECK_CUPS:
                        if getattr(PnPEnv, 'moving_to_random', False):
                            if auto_wait_counter % 20 == 0:
                                print(f"   ⏳ Waiting for random initialization to complete... (moving_to_random=True)", end='\r')
                            pass
                        else:
                            if auto_waiting_for_random_init:
                                auto_wait_counter = AUTO_RESET_WAIT_FRAMES
                                auto_waiting_for_random_init = False
                                print(f"   ✅ Random initialization completed. Waiting {AUTO_RESET_WAIT_FRAMES} frames for physics to stabilize...")
                            
                            auto_wait_counter -= 1
                            if auto_wait_counter <= 0:
                                # 🔥 检查红色杯子
                                obj_name = 'body_obj_mug_5'
                                expected_z = AUTO_RED_MUG_EXPECTED_Z
                                
                                objects_ok = True
                                try:
                                    obj_z = PnPEnv.env.get_p_body(obj_name)[2]
                                    if abs(obj_z - expected_z) > AUTO_CUP_CHECK_TOLERANCE:
                                        objects_ok = False
                                        print(f"   ⚠️ Red mug {obj_name} fallen! (z={obj_z:.3f}, expected={expected_z})")
                                except Exception as e:
                                    print(f"   ⚠️ Error checking mug {obj_name}: {e}")
                                    objects_ok = False
                                
                                if objects_ok:
                                    auto_reset_retries = 0
                                    auto_state = AUTO_STATE_START_EXPERT
                                    print(f"   ✅ Red mug OK. Starting expert policy...")
                                else:
                                    auto_reset_retries += 1
                                    if auto_reset_retries >= AUTO_MAX_RESET_RETRIES:
                                        print(f"   ❌ Max reset retries ({AUTO_MAX_RESET_RETRIES}) reached! Stopping Full-Auto Mode.")
                                        auto_state = AUTO_STATE_IDLE
                                    else:
                                        print(f"   🔄 Retry {auto_reset_retries}/{AUTO_MAX_RESET_RETRIES}: Resetting environment again...")
                                        auto_state = AUTO_STATE_RESETTING
                    
                    # ----- STATE: START_EXPERT -----
                    elif auto_state == AUTO_STATE_START_EXPERT:
                        if getattr(PnPEnv, 'moving_to_random', False):
                            pass
                        else:
                            PnPEnv.set_instruction()
                            
                            is_recording = False
                            current_frames = 0
                            PnPEnv.is_recording = False
                            PnPEnv._expert_done_printed = False
                            PnPEnv._waiting_for_save = False
                            
                            worker.clear_queue()
                            worker.wait_queue_empty()
                            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                dataset.clear_episode_buffer()
                            
                            PnPEnv.auto_execute_task(record=True)
                            auto_state = AUTO_STATE_EXECUTING
                            print(f"   🤖 Expert policy started. Recording Episode {episode_ids['arm']}...")
                    
                    # ----- STATE: EXECUTING -----
                    elif auto_state == AUTO_STATE_EXECUTING:
                        if getattr(PnPEnv, 'moving_to_random', False):
                            pass
                        elif not PnPEnv.expert_pending and not PnPEnv.expert_executing and not PnPEnv.expert_waiting_save:
                            is_recording = False
                            auto_state = AUTO_STATE_WAIT_QUEUE
                            print(f"\n   ✅ Expert execution finished. Waiting for queue to clear...")
                    
                    # ----- STATE: WAIT_QUEUE -----
                    elif auto_state == AUTO_STATE_WAIT_QUEUE:
                        queue_size = worker.qsize()
                        if queue_size > 0:
                            print(f"\r   ⏳ Queue: {queue_size} frames remaining...   ", end='', flush=True)
                        else:
                            print(f"\r   ✅ Queue cleared!                           ")
                            auto_state = AUTO_STATE_SAVING
                    
                    # ----- STATE: SAVING -----
                    elif auto_state == AUTO_STATE_SAVING:
                        try:
                            worker.saving_in_progress = True
                            worker.wait_queue_empty()
                            dataset.save_episode()
                            episode_ids['arm'] += 1
                            auto_recorded_count += 1
                            print(f"   ✅ [AUTO-SAVED] Episode {episode_ids['arm'] - 1} saved! ({auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES})")
                        except Exception as e:
                            print(f"   ❌ Save error: {e}")
                        finally:
                            worker.saving_in_progress = False
                        
                        if auto_recorded_count >= AUTO_RECORD_TARGET_EPISODES:
                            print(f"\n" + "="*60)
                            print(f" 🎉 FULL-AUTO MODE COMPLETED! 🎉")
                            print(f" 📊 Total recorded: {auto_recorded_count} episodes")
                            print(f" 📁 Final Episode ID: {episode_ids['arm']}")
                            print(f"="*60 + "\n")
                            auto_state = AUTO_STATE_IDLE
                            if AUTO_SHUTDOWN_ON_COMPLETE:
                                auto_shutdown_requested = True
                                print("🔌 Auto-shutdown requested. Closing simulation...")
                        else:
                            auto_wait_counter = AUTO_POST_SAVE_WAIT_FRAMES
                            auto_state = AUTO_STATE_POST_SAVE
                    
                    # ----- STATE: POST_SAVE -----
                    elif auto_state == AUTO_STATE_POST_SAVE:
                        auto_wait_counter -= 1
                        if auto_wait_counter <= 0:
                            print(f"\n🔄 [AUTO] Resetting for next episode ({auto_recorded_count + 1}/{AUTO_RECORD_TARGET_EPISODES})...")
                            auto_state = AUTO_STATE_RESETTING
                
                # ================= 按键逻辑 =================
                
                if auto_state == AUTO_STATE_IDLE:  # V2环境只有arm模式
                    if PnPEnv.is_recording and not is_recording:
                        if worker.saving_in_progress:
                            print(f"\n⚠️ Cannot start recording: Save operation in progress. Please wait...")
                        else:
                            is_recording = True
                            current_frames = 0
                            PnPEnv._expert_done_printed = False
                            if worker.clear_queue():
                                worker.wait_queue_empty()
                                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                    dataset.clear_episode_buffer()
                                print(f"🔴 [REC START] [ARM] Recording Episode {current_episode_id} (Auto-start from Expert Policy)...")
                    
                    if not PnPEnv.is_recording and not PnPEnv.expert_executing and not PnPEnv.expert_pending and not PnPEnv.expert_waiting_save and is_recording:
                        if not hasattr(PnPEnv, '_expert_done_printed') or not PnPEnv._expert_done_printed:
                            is_recording = False
                            PnPEnv._expert_done_printed = True
                            
                            if getattr(PnPEnv, 'expert_auto_save', False):
                                print(f"\n⏸️ [REC PAUSED] Recording stopped after post-wait period ({current_frames} frames buffered)")
                                print(f"   🔄 Auto-saving mode: Waiting for queue to clear...")
                                PnPEnv._waiting_for_save = True
                                PnPEnv._last_queue_display = -1
                                PnPEnv._queue_line_printed = False
                            else:
                                print(f"\n⏸️ [REC PAUSED] Recording stopped after post-wait period ({current_frames} frames buffered)")
                                print(f"   👉 Press [U] to SAVE, or [I] to DISCARD.")
                                PnPEnv._waiting_for_save = True
                                PnPEnv._last_queue_display = -1
                                PnPEnv._queue_line_printed = False
                    
                    if getattr(PnPEnv, 'expert_auto_save', False) and getattr(PnPEnv, '_waiting_for_save', False):
                        if not worker.saving_in_progress:
                            if worker.qsize() == 0:
                                worker.wait_queue_empty()
                                
                                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0:
                                    try:
                                        worker.saving_in_progress = True
                                        print(f"\n💾 [AUTO-SAVE] Saving Episode {current_episode_id}...")
                                        
                                        dataset.save_episode()
                                        print(f"✅ [AUTO-SAVED] [ARM] Episode {current_episode_id} saved ({current_frames} frames).")
                                        episode_ids['arm'] += 1
                                        
                                        PnPEnv.expert_auto_save = False
                                        PnPEnv._waiting_for_save = False
                                        PnPEnv._expert_done_printed = False
                                        worker.saving_in_progress = False
                                    except Exception as e:
                                        print(f"   ❌ Auto-save error: {e}")
                                        worker.saving_in_progress = False
                                else:
                                    print(f"⚠️ [AUTO-SAVE] No data to save. Discarding.")
                                    PnPEnv.expert_auto_save = False
                                    PnPEnv._waiting_for_save = False
                                    PnPEnv._expert_done_printed = False
                    
                    if getattr(PnPEnv, '_waiting_for_save', False):
                        queue_remaining = worker.qsize()
                        if queue_remaining != getattr(PnPEnv, '_last_queue_display', -1):
                            PnPEnv._last_queue_display = queue_remaining
                            if queue_remaining > 0:
                                print(f"\r   📊 Queue: {queue_remaining:4d} frames still processing in background...   ", end='', flush=True)
                            else:
                                print(f"\r   📊 Queue: All frames processed.                                   ")
                                PnPEnv._queue_line_printed = False
                
                # [J] 开始录制
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_J):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [J] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif not is_recording:
                        if worker.saving_in_progress:
                            print(f"\n⚠️ [J] Cannot start recording: Save operation in progress. Please wait...")
                        else:
                            is_recording = True
                            current_frames = 0
                            if worker.clear_queue():
                                worker.wait_queue_empty()
                                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                    dataset.clear_episode_buffer()
                                print(f"🔴 [REC START] [ARM] Recording Episode {current_episode_id} ...")

                # [K] 停止并保存
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_K):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [K] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif is_recording or (hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0):
                        if worker.saving_in_progress:
                            print(f"\n⚠️ [K] Save operation already in progress. Please wait...")
                        else:
                            is_recording = False
                            # V2环境只有arm模式
                            PnPEnv.is_recording = False
                            PnPEnv._expert_done_printed = False
                            PnPEnv._waiting_for_save = False
                            
                            worker.saving_in_progress = True
                            
                            pending_frames = worker.qsize()
                            peak = worker.peak_qsize()
                            print(f"\n⏳ Saving Episode {current_episode_id}...")
                            print(f"   📊 Recorded: {current_frames} frames | Queue backlog: {pending_frames} | Peak: {peak}")
                            
                            try:
                                while worker.qsize() > 0:
                                    remaining = worker.qsize()
                                    progress = (pending_frames - remaining) / max(pending_frames, 1) * 100
                                    print(f"   ⏳ Processing: {pending_frames - remaining}/{pending_frames} ({progress:.0f}%) - {remaining} frames remaining in queue...", end='\r')
                                    time.sleep(0.5)
                                worker.wait_queue_empty()
                                print(f"\n   ✅ Queue cleared!                                          ")
                                
                                dataset.save_episode()
                                print(f"✅ [SAVED] [ARM] Episode {current_episode_id} saved ({current_frames} frames).")
                                episode_ids['arm'] += 1
                            finally:
                                worker.saving_in_progress = False

                # 🔥 [I] 丢弃录制
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_I):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [I] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    else:
                        has_buffered_data = hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0
                        has_queue_data = worker.qsize() > 0
                        if is_recording or has_buffered_data or has_queue_data:
                            if worker.saving_in_progress:
                                print(f"\n⚠️ [I] Cannot discard: Save operation in progress. Please wait for save to complete.")
                            else:
                                is_recording = False
                                # V2环境只有arm模式
                                PnPEnv.is_recording = False
                                PnPEnv._expert_done_printed = False
                                PnPEnv.expert_waiting_save = False
                                PnPEnv._waiting_for_save = False
                                PnPEnv.expert_auto_save = False
                                if worker.clear_queue():
                                    worker.wait_queue_empty()
                                    if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                        dataset.clear_episode_buffer()
                                    print(f"❌ [DISCARDED] [ARM] Episode {current_episode_id} data cleared. (ID unchanged)")

                # ================= 🤖🔄 [P] 全自动录制模式 🔄🤖 =================
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_P):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n🛑 [P] Stopping Full-Auto Mode...")
                        print(f"   Recorded {auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES} episodes before stop.")
                        
                        is_recording = False
                        PnPEnv.is_recording = False
                        PnPEnv.expert_executing = False
                        PnPEnv.expert_pending = False
                        PnPEnv.expert_waiting_save = False
                        PnPEnv._expert_done_printed = False
                        PnPEnv._waiting_for_save = False
                        PnPEnv.expert_auto_save = False
                        
                        worker.clear_queue()
                        worker.wait_queue_empty()
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()
                        
                        auto_state = AUTO_STATE_IDLE
                        print(f"❌ [AUTO-STOPPED] Current recording discarded. Full-Auto Mode DISABLED.")
                    else:
                        if worker.saving_in_progress:
                            print(f"\n⚠️ [P] Cannot start Full-Auto Mode: Save operation in progress. Please wait...")
                        else:
                            is_recording = False
                            PnPEnv.is_recording = False
                            worker.clear_queue()
                            worker.wait_queue_empty()
                            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                dataset.clear_episode_buffer()
                            
                            auto_recorded_count = 0
                            auto_state = AUTO_STATE_RESETTING
                            auto_wait_counter = 0
                            auto_reset_retries = 0
                            
                            print(f"\n" + "="*60)
                            print(f" 🤖🔄 FULL-AUTO MODE ACTIVATED 🔄🤖")
                            print(f" 🎯 Target: {AUTO_RECORD_TARGET_EPISODES} episodes")
                            print(f" 📁 Dataset: {DATASET_CONFIG['arm']['repo_name']}")
                            print(f" 🔢 Starting Episode ID: {episode_ids['arm']}")
                            print(f" ⏱️ Press [P] again to STOP at any time")
                            print(f"="*60)
                            print(f"\n🔄 [AUTO] Resetting environment (Episode {auto_recorded_count + 1}/{AUTO_RECORD_TARGET_EPISODES})...")

                # ================= 🛡️ 熔断逻辑 =================
                
                if is_recording and current_frames >= MAX_FRAMES:
                    is_recording = False
                    # V2环境只有arm模式
                    PnPEnv.is_recording = False
                    PnPEnv._waiting_for_save = False
                    if worker.saving_in_progress:
                        print(f"\n⚠️ [TIMEOUT] Max duration reached, but save operation in progress. Will discard after save completes.")
                    else:
                        if worker.clear_queue():
                            worker.wait_queue_empty()
                            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                dataset.clear_episode_buffer()
                            print(f"\n⚠️ [TIMEOUT] Max duration ({MAX_EPISODE_SEC}s) reached!")
                            print(f"❌ [AUTO-DISCARDED] [ARM] Episode {current_episode_id} data cleared. (ID unchanged)")
                
                # ===========================================================

                # 🔥 数据收集优化 - 图像在 step 之前获取
                auto_is_recording = (auto_state == AUTO_STATE_EXECUTING and 
                                     (PnPEnv.expert_pending or PnPEnv.expert_executing or PnPEnv.expert_waiting_save))
                if is_recording or auto_is_recording:
                    images_dict_raw = PnPEnv.grab_image()
                    images_dict_safe = {k: v.copy() for k, v in images_dict_raw.items()}
                
                # 机器人控制
                action, reset = PnPEnv.teleop_robot(mode='arm')  # V2环境只有arm模式
                
                # [Z] 手动重置环境
                if reset:
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [Z] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif worker.saving_in_progress:
                        print(f"\n⚠️ [Z] Cannot reset: Save operation in progress. Please wait for save to complete.")
                    else:
                        print("🔄 Environment Reset.")
                        if is_recording:
                            is_recording = False
                            # V2环境只有arm模式
                            PnPEnv.is_recording = False
                            PnPEnv._waiting_for_save = False
                            print(f"⚠️ [INTERRUPTED] Recording stopped due to reset. (ID {current_episode_id})")
                        # V2环境只有arm模式
                        PnPEnv._waiting_for_save = False
                        PnPEnv.expert_auto_save = False
                        PnPEnv.reset(mode='arm')
                        if worker.clear_queue():
                            worker.wait_queue_empty()
                            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                dataset.clear_episode_buffer()

                # 物理执行
                current_state = PnPEnv.step(action, mode='arm')  # V2环境只有arm模式
                
                # V2环境只有arm模式
                action_to_save = PnPEnv.current_arm_q[:7].astype(np.float32)

                # 🔥 数据收集：异步处理模式
                if is_recording or auto_is_recording:
                    worker.put((current_mode, images_dict_safe, current_state, action_to_save, PnPEnv.obj_init_pose, PnPEnv.instruction))
                    current_frames += 1
                    
                    if current_frames % FPS == 0:
                        q_size = worker.qsize()
                        mem_mb = q_size * 1.5
                        warn = ""
                        if q_size > 500:
                            warn = " ⚠️ HIGH MEM!"
                        elif q_size > 200:
                            warn = " 📈"
                        print(f"   [ARM] Recording... {current_frames} frames | Queue: {q_size} (~{mem_mb:.0f}MB){warn}    ", end='\r')
                
                PnPEnv.render(teleop=True, idx=current_episode_id)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
        if auto_state != AUTO_STATE_IDLE:
            print(f"   🛑 Full-Auto Mode was active. Progress: {auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES} episodes saved.")
            print(f"   ❌ Current recording (if any) will be discarded.")
    finally:
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
        
        # V2环境只有arm模式
        ds = datasets['arm']
        images_path = ds.root / 'images'
        if os.path.exists(images_path):
            max_retries = 3
            retry_delay = 1.0
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(images_path)
                    print(f"✅ [ARM] Cleaned up images directory")
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ [ARM] Retry {attempt + 1}/{max_retries}: Waiting before retry...")
                        time.sleep(retry_delay)
                    else:
                        print(f"⚠️ [ARM] Warning: Could not fully remove images directory: {e}")
        
        print("\n📊 Final Episode Counts:")
        print(f"   ARM:  {episode_ids['arm']} episodes")

if __name__ == "__main__":
    main()
