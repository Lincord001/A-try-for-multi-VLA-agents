import sys
import numpy as np
import os
import shutil
import time
import glfw
import threading
import queue
from PIL import Image  # 🔥 V4.1: 改回 PIL，与部署环境保持一致
from mujoco_env.y_env4 import SimpleEnv4 
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    🎛️ 常用配置（频繁调整区）🎛️                               ║
# ║          以下参数是您最可能需要修改的，已按使用频率排序                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- 🤖 全自动录制 [P键] ---
AUTO_RECORD_TARGET_EPISODES = 1    # 🎯 目标录制条数（达到后自动停止）
AUTO_SHUTDOWN_ON_COMPLETE = True     # 🔌 完成后是否自动关闭仿真环境

# --- 📁 数据集名称与路径 ---
ARM_DATASET_NAME = 'omy_arm_data_v4'       # Arm 模式数据集名称
ARM_DATASET_ROOT = './demo_data_arm_v4'    # Arm 模式数据集保存路径
BASE_DATASET_NAME = 'omy_base_data_v4'     # Base 模式数据集名称  
BASE_DATASET_ROOT = './demo_data_base_v4'  # Base 模式数据集保存路径

# --- 🖼️ 图像与录制 ---
IMG_SIZE = 224                       # 图像分辨率 (224=ViT标准, 256=兼容旧数据)
FPS = 20                             # 录制帧率 (Hz)
MAX_EPISODE_SEC = 200                # 单条数据最大时长（秒），超时自动丢弃

# --- 🎮 初始模式 ---
INITIAL_MODE = 'arm'                 # 启动时的模式: 'arm' 或 'base'

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    ⚙️ 高级配置（一般不需要修改）⚙️                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- 全自动录制细节参数 ---
AUTO_RESET_WAIT_FRAMES = 40          # 重置后等待帧数（约2秒，让物理引擎稳定）
AUTO_CUP_CHECK_TOLERANCE = 0.05      # 杯子 z 坐标容差（±5cm，超出判定为倒了）
AUTO_CUP_EXPECTED_Z = 0.345          # 杯子正常 z 坐标
AUTO_POST_SAVE_WAIT_FRAMES = 20      # 保存后等待帧数（约1秒）
AUTO_MAX_RESET_RETRIES = 5           # 杯子倒了时最大重试次数

# --- 场景配置 ---
SEED = 0 
XML_PATH = './asset/example_scene_y4.xml'

# --- 派生配置（自动计算，勿手动修改）---
MAX_FRAMES = MAX_EPISODE_SEC * FPS

DATASET_CONFIG = {
    'arm': {
        'repo_name': ARM_DATASET_NAME,
        'root': ARM_DATASET_ROOT,
    },
    'base': {
        'repo_name': BASE_DATASET_NAME,
        'root': BASE_DATASET_ROOT,
    }
}

MODE_CONFIG = {
    'arm': {
        'action_shape': (7,),
        'state_shape': (6,),
    },
    'base': {
        'action_shape': (2,),
        'state_shape': (2,),
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
    """支持双模式的数据保存工作线程 (优化版：无阻塞 + 快速 resize)"""
    def __init__(self, datasets):
        """
        Parameters:
            datasets: dict, {'arm': LeRobotDataset, 'base': LeRobotDataset}
        """
        super().__init__()
        self.datasets = datasets
        # 🔥 关键改动 1：去掉 maxsize 限制，永不阻塞主线程
        self.queue = queue.Queue(maxsize=0)  # 0 = 无限大小
        self.daemon = True
        self.running = True
        self._peak_qsize = 0  # 记录峰值，用于调试
        self.saving_in_progress = False  # 🔥 保存进行中标志（防止竞态条件）

    def put(self, item):
        """主线程调用：将原始数据放入队列（永不阻塞）"""
        self.queue.put_nowait(item)  # 非阻塞 put
        # 更新峰值记录
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
        """🔥 V4.1: 改回 PIL resize，与部署环境保持一致"""
        # 使用全局配置的 IMG_SIZE
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
            base_pose = item[6] if len(item) > 6 else None

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
                # 🔥 使用快速 resize
                if mode == 'arm':
                    frame_data["observation.images.agent"] = self._fast_resize(images_dict['agent'])
                    frame_data["observation.images.wrist"] = self._fast_resize(images_dict['wrist'])
                    frame_data["observation.images.back"] = self._fast_resize(images_dict['back'])
                elif mode == 'base':
                    frame_data["observation.images.front"] = self._fast_resize(images_dict['front'])
                    frame_data["observation.images.left"] = self._fast_resize(images_dict['left'])
                    frame_data["observation.images.right"] = self._fast_resize(images_dict['right'])
                    if base_pose is not None:
                        frame_data["base_pose"] = base_pose

                # 写入硬盘
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
        # 🔥 防止在保存期间清空队列（竞态条件保护）
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

def load_or_create_dataset(mode):
    """加载或创建指定模式的数据集"""
    config = DATASET_CONFIG[mode]
    mode_cfg = MODE_CONFIG[mode]
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
            "obj_init": {"dtype": "float32", "shape": (12,), "names": ["obj_init"]},
        }

        if mode == 'arm':
            features["observation.images.agent"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
            features["observation.images.wrist"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
            features["observation.images.back"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
        elif mode == 'base':
            for cam in ['front', 'left', 'right']:
                features[f"observation.images.{cam}"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
            features["base_pose"] = {
                "dtype": "float32", 
                "shape": (3,), 
                "names": ["x", "y", "theta"]
            }

        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            root=root,
            robot_type="omy",
            fps=FPS,
            features=features,
            image_writer_threads=20,  # 🔥 增加线程数
            image_writer_processes=8, # 🔥 增加进程数
        )
    
    return dataset

def main():
    # 🔥 运行时模式变量（支持热切换）
    current_mode = INITIAL_MODE
    
    print(f"Initializing Environment in [{current_mode.upper()}] mode...")
    PnPEnv = SimpleEnv4(XML_PATH, seed=SEED, state_type='joint_angle')
    PnPEnv.reset(mode=current_mode)

    # ---------------------------------------------------------
    # 🔥 1. 加载两个数据集（支持热切换）🔥
    print("\n" + "="*50)
    print(" 📂 Loading/Creating Datasets for HOT-SWITCH support...")
    print("="*50)
    
    datasets = {
        'arm': load_or_create_dataset('arm'),
        'base': load_or_create_dataset('base'),
    }
    
    # 各模式的 Episode ID（独立计数）
    episode_ids = {
        'arm': datasets['arm'].num_episodes,
        'base': datasets['base'].num_episodes,
    }

    # 🔥 启动后台工作线程（传入两个数据集）🔥
    worker = DataSaverWorker(datasets)
    worker.start()

    is_recording = False
    current_frames = 0
    
    # 🤖 全自动录制状态变量
    auto_state = AUTO_STATE_IDLE           # 当前自动录制状态
    auto_recorded_count = 0                # 已录制的 episode 数量
    auto_wait_counter = 0                  # 等待计数器（帧数）
    auto_reset_retries = 0                 # 杯子倒了重试次数
    auto_shutdown_requested = False        # 🔥 是否请求自动关闭

    print("\n" + "="*60)
    print(f" 🚀 DATA COLLECTION MODE: {current_mode.upper()} (HOT-SWITCH ENABLED)")
    print(f" 📁 ARM  Dataset: {DATASET_CONFIG['arm']['repo_name']} (Episode #{episode_ids['arm']})")
    print(f" 📁 BASE Dataset: {DATASET_CONFIG['base']['repo_name']} (Episode #{episode_ids['base']})")
    print(f" 🧵 ASYNC RECORDER ACTIVE (Using Background Thread)")
    print(f" 🛡️ Safety Limit: Max {MAX_EPISODE_SEC}s ({MAX_FRAMES} frames) per episode")
    print("-"*60)
    print(" 🎮 Control Keys (通用):")
    print("  [J] : Start Recording (开始录制)")
    print("  [K] : Stop & SAVE (停止并保存)")
    print("  [I] : 🔥 DISCARD Recording (丢弃当前录制) [NEW!]")
    print("  [Z] : Reset Environment Only (仅重置环境)")
    print("  [C] : 🔥 HOT-SWITCH Mode (Base ↔ Arm)")
    print("-"*60)
    print(" 🤖 ARM Mode Only (机械臂模式专用):")
    print("  [T] : Test Mode: Auto Execute (测试模式，不录制)")
    print("  [Y] : 🎥 Record Mode: Auto Execute + Recording")
    print("        → 执行完成后等待3秒，自动暂停录制")
    print("        → 暂停后按 [U] 保存, 或 [I] 丢弃")
    print("  [U] : 🔥 SAVE Paused Recording (保存暂停的录制) [NEW!]")
    print("  [O] : 🏠 Smooth Return Home (平滑归位机械臂)")
    print("-"*60)
    print(" 🤖🔄 FULL-AUTO Recording (全自动录制):")
    print(f"  [P] : 🔥 Start/Stop Full-Auto Mode (Target: {AUTO_RECORD_TARGET_EPISODES} episodes)")
    print("        → Auto: Reset → Check Cups → Expert → Save → Loop")
    print("        → Press [P] again to STOP (discard current recording)")
    print("-"*60)
    print(f" Current Mode: {current_mode.upper()} | Next Episode ID: {episode_ids[current_mode]}")
    print("="*60 + "\n")

    try:
        # 🔥 2. 移除 NUM_DEMO 限制，实现无限录制 🔥
        while PnPEnv.env.is_viewer_alive() and not auto_shutdown_requested:
            PnPEnv.step_env()
            
            if PnPEnv.env.loop_every(HZ=FPS):
                
                # 获取当前模式的数据集（方便后续使用）
                dataset = datasets[current_mode]
                current_episode_id = episode_ids[current_mode]
                
                # ================= 🤖 全自动录制状态机 🤖 =================
                if auto_state != AUTO_STATE_IDLE and current_mode == 'arm':
                    
                    # ----- STATE: RESETTING -----
                    if auto_state == AUTO_STATE_RESETTING:
                        # 执行重置
                        PnPEnv.reset(mode='arm')
                        # 清空残留数据
                        worker.clear_queue()
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()
                        auto_wait_counter = AUTO_RESET_WAIT_FRAMES
                        auto_state = AUTO_STATE_CHECK_CUPS
                        print(f"   ⏳ Waiting {AUTO_RESET_WAIT_FRAMES} frames for physics to stabilize...")
                    
                    # ----- STATE: CHECK_CUPS (等待物理稳定 + 检查杯子) -----
                    elif auto_state == AUTO_STATE_CHECK_CUPS:
                        auto_wait_counter -= 1
                        if auto_wait_counter <= 0:
                            # 检查所有杯子的 z 坐标
                            cups_ok = True
                            cup_names = ['body_obj_mug_5', 'body_obj_mug_6', 'body_obj_mug_7', 'body_obj_mug_8']
                            for cup_name in cup_names:
                                try:
                                    cup_z = PnPEnv.env.get_p_body(cup_name)[2]
                                    if abs(cup_z - AUTO_CUP_EXPECTED_Z) > AUTO_CUP_CHECK_TOLERANCE:
                                        cups_ok = False
                                        print(f"   ⚠️ Cup {cup_name} fallen! (z={cup_z:.3f}, expected={AUTO_CUP_EXPECTED_Z})")
                                        break
                                except Exception as e:
                                    print(f"   ⚠️ Error checking cup {cup_name}: {e}")
                                    cups_ok = False
                                    break
                            
                            if cups_ok:
                                # 杯子正常，启动专家策略
                                auto_reset_retries = 0
                                auto_state = AUTO_STATE_START_EXPERT
                                print(f"   ✅ All cups OK. Starting expert policy...")
                            else:
                                # 杯子倒了，重新重置
                                auto_reset_retries += 1
                                if auto_reset_retries >= AUTO_MAX_RESET_RETRIES:
                                    print(f"   ❌ Max reset retries ({AUTO_MAX_RESET_RETRIES}) reached! Stopping Full-Auto Mode.")
                                    auto_state = AUTO_STATE_IDLE
                                else:
                                    print(f"   🔄 Retry {auto_reset_retries}/{AUTO_MAX_RESET_RETRIES}: Resetting environment again...")
                                    auto_state = AUTO_STATE_RESETTING
                    
                    # ----- STATE: START_EXPERT -----
                    elif auto_state == AUTO_STATE_START_EXPERT:
                        # 清空残留状态
                        is_recording = False
                        current_frames = 0
                        PnPEnv.is_recording = False
                        PnPEnv._expert_done_printed = False
                        PnPEnv._waiting_for_save = False
                        
                        # 启动专家策略（带录制）
                        PnPEnv.auto_execute_task(record=True)
                        auto_state = AUTO_STATE_EXECUTING
                        print(f"   🤖 Expert policy started. Recording Episode {episode_ids['arm']}...")
                    
                    # ----- STATE: EXECUTING (专家策略执行中) -----
                    elif auto_state == AUTO_STATE_EXECUTING:
                        # 检查专家策略是否完成（包括 post-wait）
                        if not PnPEnv.expert_pending and not PnPEnv.expert_executing and not PnPEnv.expert_waiting_save:
                            # 专家策略执行完毕
                            is_recording = False  # 确保本地录制标志关闭
                            auto_state = AUTO_STATE_WAIT_QUEUE
                            print(f"\n   ✅ Expert execution finished. Waiting for queue to clear...")
                    
                    # ----- STATE: WAIT_QUEUE (等待队列清空) -----
                    elif auto_state == AUTO_STATE_WAIT_QUEUE:
                        queue_size = worker.qsize()
                        if queue_size > 0:
                            # 显示队列进度
                            print(f"\r   ⏳ Queue: {queue_size} frames remaining...   ", end='', flush=True)
                        else:
                            # 队列清空，开始保存
                            print(f"\r   ✅ Queue cleared!                           ")
                            auto_state = AUTO_STATE_SAVING
                    
                    # ----- STATE: SAVING -----
                    elif auto_state == AUTO_STATE_SAVING:
                        try:
                            worker.saving_in_progress = True
                            worker.wait_queue_empty()  # 最终确认
                            dataset.save_episode()
                            episode_ids['arm'] += 1
                            auto_recorded_count += 1
                            print(f"   ✅ [AUTO-SAVED] Episode {episode_ids['arm'] - 1} saved! ({auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES})")
                        except Exception as e:
                            print(f"   ❌ Save error: {e}")
                        finally:
                            worker.saving_in_progress = False
                        
                        # 检查是否达到目标
                        if auto_recorded_count >= AUTO_RECORD_TARGET_EPISODES:
                            print(f"\n" + "="*60)
                            print(f" 🎉 FULL-AUTO MODE COMPLETED! 🎉")
                            print(f" 📊 Total recorded: {auto_recorded_count} episodes")
                            print(f" 📁 Final Episode ID: {episode_ids['arm']}")
                            print(f"="*60 + "\n")
                            auto_state = AUTO_STATE_IDLE
                            # 🔥 如果设置了自动关闭，请求关闭仿真环境
                            if AUTO_SHUTDOWN_ON_COMPLETE:
                                auto_shutdown_requested = True
                                print("🔌 Auto-shutdown requested. Closing simulation...")
                        else:
                            # 继续下一轮
                            auto_wait_counter = AUTO_POST_SAVE_WAIT_FRAMES
                            auto_state = AUTO_STATE_POST_SAVE
                    
                    # ----- STATE: POST_SAVE (保存后等待) -----
                    elif auto_state == AUTO_STATE_POST_SAVE:
                        auto_wait_counter -= 1
                        if auto_wait_counter <= 0:
                            print(f"\n🔄 [AUTO] Resetting for next episode ({auto_recorded_count + 1}/{AUTO_RECORD_TARGET_EPISODES})...")
                            auto_state = AUTO_STATE_RESETTING
                
                # ================= 按键逻辑 =================
                
                # 🔥 同步环境的录制状态（用于Y键自动录制）
                # 🔥 注意：全自动模式下跳过手动录制同步逻辑
                if current_mode == 'arm' and auto_state == AUTO_STATE_IDLE:
                    # 如果环境开始录制，但本地还没开始，则同步开始
                    if PnPEnv.is_recording and not is_recording:
                        # 🔥 检查是否正在保存
                        if worker.saving_in_progress:
                            print(f"\n⚠️ Cannot start recording: Save operation in progress. Please wait...")
                        else:
                            is_recording = True
                            current_frames = 0
                            PnPEnv._expert_done_printed = False  # 🔥 重置提示标志
                            # 🔥 先清空队列（防止残留的旧数据混入新录制）
                            if worker.clear_queue():
                                # 🔥 等待正在处理的帧完成
                                worker.wait_queue_empty()
                                # 🔥 最后清空缓冲区（此时保证没有旧数据会再写入）
                                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                    dataset.clear_episode_buffer()
                                print(f"🔴 [REC START] [{current_mode.upper()}] Recording Episode {current_episode_id} (Auto-start from Expert Policy)...")
                    
                    # 🔥 专家策略执行完毕+等待期结束后，自动停止录制（但不保存）
                    # 此时 is_recording=True 但 PnPEnv.is_recording=False
                    if not PnPEnv.is_recording and not PnPEnv.expert_executing and not PnPEnv.expert_pending and not PnPEnv.expert_waiting_save and is_recording:
                        # 只在刚停止时打印一次提示
                        if not hasattr(PnPEnv, '_expert_done_printed') or not PnPEnv._expert_done_printed:
                            is_recording = False  # 🔥 停止本地录制（但数据仍在缓冲区，未保存）
                            print(f"\n⏸️ [REC PAUSED] Recording stopped after post-wait period ({current_frames} frames buffered)")
                            print(f"   👉 Press [U] to SAVE, or [I] to DISCARD.")
                            PnPEnv._expert_done_printed = True
                            PnPEnv._waiting_for_save = True  # 🔥 新增：标记正在等待用户保存/丢弃
                            PnPEnv._last_queue_display = -1  # 🔥 初始化队列显示计数器
                            PnPEnv._queue_line_printed = False  # 🔥 初始化打印标志
                    
                    # 🔥 独立的队列状态显示逻辑（在等待保存/丢弃期间持续更新）
                    if getattr(PnPEnv, '_waiting_for_save', False):
                        queue_remaining = worker.qsize()
                        if queue_remaining != getattr(PnPEnv, '_last_queue_display', -1):
                            PnPEnv._last_queue_display = queue_remaining
                            if queue_remaining > 0:
                                # 🔥 使用回车符覆盖整行，实时更新
                                print(f"\r   📊 Queue: {queue_remaining:4d} frames still processing in background...   ", end='', flush=True)
                            else:
                                # 队列清空，打印最终状态并换行
                                print(f"\r   📊 Queue: All frames processed.                                   ")
                                PnPEnv._queue_line_printed = False  # 重置标志
                
                # [J] 开始录制 (全自动模式下禁用)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_J):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [J] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif not is_recording:
                        # 🔥 检查是否正在保存
                        if worker.saving_in_progress:
                            print(f"\n⚠️ [J] Cannot start recording: Save operation in progress. Please wait...")
                        else:
                            is_recording = True
                            current_frames = 0  # 重置计数器
                            # 🔥 先清空队列（防止残留的旧数据混入新录制）
                            if worker.clear_queue():
                                # 🔥 等待正在处理的帧完成
                                worker.wait_queue_empty()
                                # 🔥 最后清空缓冲区（此时保证没有旧数据会再写入）
                                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                    dataset.clear_episode_buffer()
                                # 🔥 3. 录制开始时打印当前是第几条 🔥
                                print(f"🔴 [REC START] [{current_mode.upper()}] Recording Episode {current_episode_id} ...")

                # [K] 停止并保存 (Success) - Base模式用，Arm模式也可用 (全自动模式下禁用)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_K):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [K] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif is_recording or (current_mode == 'arm' and hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0):
                        # 🔥 检查是否已有保存操作在进行
                        if worker.saving_in_progress:
                            print(f"\n⚠️ [K] Save operation already in progress. Please wait...")
                        else:
                            is_recording = False
                            # 🔥 同步停止环境的录制状态
                            if current_mode == 'arm':
                                PnPEnv.is_recording = False
                                PnPEnv._expert_done_printed = False  # 重置提示标志
                                PnPEnv._waiting_for_save = False     # 🔥 重置队列显示等待标志
                            
                            # 🔥 设置保存标志（防止竞态条件）
                            worker.saving_in_progress = True
                            
                            pending_frames = worker.qsize()
                            peak = worker.peak_qsize()
                            print(f"\n⏳ Saving Episode {current_episode_id}...")
                            print(f"   📊 Recorded: {current_frames} frames | Queue backlog: {pending_frames} | Peak: {peak}")
                            
                            try:
                                # 🔥 带进度的等待（显示队列状态）
                                while worker.qsize() > 0:
                                    remaining = worker.qsize()
                                    progress = (pending_frames - remaining) / max(pending_frames, 1) * 100
                                    print(f"   ⏳ Processing: {pending_frames - remaining}/{pending_frames} ({progress:.0f}%) - {remaining} frames remaining in queue...", end='\r')
                                    time.sleep(0.5)
                                worker.wait_queue_empty()  # 最终确认
                                print(f"\n   ✅ Queue cleared!                                          ")
                                
                                dataset.save_episode()
                                print(f"✅ [SAVED] [{current_mode.upper()}] Episode {current_episode_id} saved ({current_frames} frames).")
                                episode_ids[current_mode] += 1
                            finally:
                                # 🔥 重置保存标志
                                worker.saving_in_progress = False

                # 🔥 [U] 保存已暂停的录制 (Arm模式专用，用于专家策略自动录制后的手动确认保存) (全自动模式下禁用)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_U):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [U] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif current_mode == 'arm':
                        # 🔥 检查：如果录制还在进行中，不允许保存
                        if is_recording:
                            print(f"\n⚠️ [U] Invalid operation: Cannot save while recording is still in progress.")
                            print(f"   Reason: Recording flag is active (is_recording=True). Please wait for recording to finish.")
                            if PnPEnv.expert_executing or PnPEnv.expert_pending or PnPEnv.expert_waiting_save:
                                print(f"   Status: Expert policy is {'executing' if PnPEnv.expert_executing else 'pending' if PnPEnv.expert_pending else 'in post-wait period'}.")
                        # 🔥 检查：如果队列中还有待处理的帧，不允许保存
                        elif worker.qsize() > 0:
                            queue_size = worker.qsize()
                            print(f"\n⚠️ [U] Invalid operation: Cannot save while frames are still being processed.")
                            print(f"   Reason: Queue still has {queue_size} frame(s) waiting to be processed in background.")
                            print(f"   Please wait for queue to clear (watch the queue counter above).")
                        else:
                            # 检查是否有待保存的数据
                            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0:
                                # 🔥 检查是否已有保存操作在进行
                                if worker.saving_in_progress:
                                    print(f"\n⚠️ [U] Invalid operation: Save operation already in progress.")
                                    print(f"   Reason: Another save operation is currently running. Please wait for it to complete.")
                                else:
                                    # 🔥 设置保存标志（防止竞态条件）
                                    worker.saving_in_progress = True
                                    
                                    pending_frames = worker.qsize()
                                    peak = worker.peak_qsize()
                                    print(f"\n⏳ [U] Saving paused recording Episode {current_episode_id}...")
                                    print(f"   📊 Buffered: {current_frames} frames | Queue backlog: {pending_frames} | Peak: {peak}")
                                    
                                    try:
                                        # 🔥 带进度的等待（显示队列状态）
                                        while worker.qsize() > 0:
                                            remaining = worker.qsize()
                                            progress = (pending_frames - remaining) / max(pending_frames, 1) * 100
                                            print(f"   ⏳ Processing: {pending_frames - remaining}/{pending_frames} ({progress:.0f}%) - {remaining} frames remaining in queue...", end='\r')
                                            time.sleep(0.5)
                                        worker.wait_queue_empty()  # 最终确认
                                        print(f"\n   ✅ Queue cleared!                                          ")
                                        
                                        dataset.save_episode()
                                        print(f"✅ [SAVED] [{current_mode.upper()}] Episode {current_episode_id} saved ({current_frames} frames).")
                                        episode_ids[current_mode] += 1
                                        PnPEnv._expert_done_printed = False  # 重置提示标志
                                        PnPEnv._waiting_for_save = False  # 🔥 重置等待标志
                                    finally:
                                        # 🔥 重置保存标志
                                        worker.saving_in_progress = False
                            else:
                                print(f"⚠️ [U] Invalid operation: No buffered data to save.")
                                print(f"   Reason: Episode buffer is empty. No frames have been recorded yet.")
                    else:
                        print(f"⚠️ [U] Invalid operation: Save function only available in ARM mode.")
                        print(f"   Reason: Current mode is {current_mode.upper()}. Use [K] key for BASE mode.")

                # 🔥 [I] 丢弃录制 (两种模式通用) (全自动模式下禁用)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_I):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [I] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    else:
                        # 检查是否正在录制或有待保存的数据
                        has_buffered_data = hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None and len(dataset.episode_buffer) > 0
                        has_queue_data = worker.qsize() > 0
                        if is_recording or has_buffered_data or has_queue_data:
                            # 🔥 检查是否正在保存
                            if worker.saving_in_progress:
                                print(f"\n⚠️ [I] Cannot discard: Save operation in progress. Please wait for save to complete.")
                            else:
                                is_recording = False
                                # 🔥 同步停止环境的录制状态
                                if current_mode == 'arm':
                                    PnPEnv.is_recording = False
                                    PnPEnv._expert_done_printed = False  # 重置提示标志
                                    PnPEnv.expert_waiting_save = False   # 🔥 重置等待状态
                                    PnPEnv._waiting_for_save = False     # 🔥 重置队列显示等待标志
                                # 🔥 先清空队列（防止新数据继续进入）
                                if worker.clear_queue():
                                    # 🔥 等待正在处理的帧完成并写入缓冲区（确保所有数据都被捕获）
                                    worker.wait_queue_empty()
                                    # 🔥 最后清空缓冲区（包括刚才写入的帧）
                                    if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                        dataset.clear_episode_buffer()
                                    print(f"❌ [DISCARDED] [{current_mode.upper()}] Episode {current_episode_id} data cleared. (ID unchanged)")

                # ================= 🤖🔄 [P] 全自动录制模式 🔄🤖 =================
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_P):
                    if current_mode != 'arm':
                        print(f"\n⚠️ [P] Full-Auto Mode only available in ARM mode. Current: {current_mode.upper()}")
                    elif auto_state != AUTO_STATE_IDLE:
                        # 🔥 正在自动录制中，按 P 停止
                        print(f"\n🛑 [P] Stopping Full-Auto Mode...")
                        print(f"   Recorded {auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES} episodes before stop.")
                        
                        # 丢弃当前正在录制的数据
                        is_recording = False
                        PnPEnv.is_recording = False
                        PnPEnv.expert_executing = False
                        PnPEnv.expert_pending = False
                        PnPEnv.expert_waiting_save = False
                        PnPEnv._expert_done_printed = False
                        PnPEnv._waiting_for_save = False
                        
                        # 清空队列和缓冲区
                        worker.clear_queue()
                        worker.wait_queue_empty()
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()
                        
                        auto_state = AUTO_STATE_IDLE
                        print(f"❌ [AUTO-STOPPED] Current recording discarded. Full-Auto Mode DISABLED.")
                    else:
                        # 🔥 启动全自动录制
                        if worker.saving_in_progress:
                            print(f"\n⚠️ [P] Cannot start Full-Auto Mode: Save operation in progress. Please wait...")
                        else:
                            # 先清空任何残留数据
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

                # ================= 🔥 [C] 热切换模式 🔥 ================= (全自动模式下禁用)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_C):
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [C] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif worker.saving_in_progress:
                        print(f"\n⚠️ [C] Cannot switch mode: Save operation in progress. Please wait for save to complete.")
                    else:
                        # 如果正在录制，先停止并丢弃
                        if is_recording:
                            is_recording = False
                            # 🔥 同步停止环境的录制状态
                            if current_mode == 'arm':
                                PnPEnv.is_recording = False
                                PnPEnv._waiting_for_save = False  # 🔥 重置队列显示等待标志
                            # 🔥 先清空队列
                            if worker.clear_queue():
                                # 🔥 再清空缓冲区
                                if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                    dataset.clear_episode_buffer()
                                print(f"⚠️ [RECORDING STOPPED] Discarding current recording before switch.")
                        
                        # 等待队列清空（确保没有残留）
                        worker.wait_queue_empty()
                        
                        # 切换模式
                        old_mode = current_mode
                        current_mode = 'arm' if current_mode == 'base' else 'base'
                        
                        # 🔥 清空新模式的缓冲区（防止旧数据混入新录制）
                        new_dataset = datasets[current_mode]
                        if hasattr(new_dataset, 'episode_buffer') and new_dataset.episode_buffer is not None:
                            new_dataset.clear_episode_buffer()
                        
                        # 🔥 关键：不调用 reset()，只更新环境的 control_mode
                        PnPEnv.control_mode = current_mode
                        
                        # 更新任务指令（根据新模式）
                        PnPEnv.set_instruction()
                        
                        # 刷新图像缓存（切换相机）
                        PnPEnv.grab_image()
                        
                        print(f"\n🔄 [HOT-SWITCH] {old_mode.upper()} → {current_mode.upper()}")
                        print(f"   📍 Environment state preserved (no reset)")
                        print(f"   📁 Now using: {DATASET_CONFIG[current_mode]['repo_name']}")
                        print(f"   🔢 Next Episode ID: {episode_ids[current_mode]}\n")

                # ================= 🛡️ 熔断逻辑 (自动停止) 🛡️ =================
                
                if is_recording and current_frames >= MAX_FRAMES:
                    is_recording = False
                    # 🔥 同步停止环境的录制状态
                    if current_mode == 'arm':
                        PnPEnv.is_recording = False
                        PnPEnv._waiting_for_save = False  # 🔥 重置队列显示等待标志
                    # 🔥 检查是否正在保存
                    if worker.saving_in_progress:
                        print(f"\n⚠️ [TIMEOUT] Max duration reached, but save operation in progress. Will discard after save completes.")
                    else:
                        # 🔥 先清空队列（防止新数据继续进入）
                        if worker.clear_queue():
                            # 🔥 等待正在处理的帧完成并写入缓冲区（确保所有数据都被捕获）
                            worker.wait_queue_empty()
                            # 🔥 最后清空缓冲区（包括刚才写入的帧）
                            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                dataset.clear_episode_buffer()
                            print(f"\n⚠️ [TIMEOUT] Max duration ({MAX_EPISODE_SEC}s) reached!")
                            print(f"❌ [AUTO-DISCARDED] [{current_mode.upper()}] Episode {current_episode_id} data cleared. (ID unchanged)")
                
                # ===========================================================

                # 🔥 V4.1: 数据收集优化 - 图像在 step 之前获取（与原始工程一致）
                # 🔥 全自动模式下：在 EXECUTING 阶段自动录制
                auto_is_recording = (auto_state == AUTO_STATE_EXECUTING and 
                                     (PnPEnv.expert_pending or PnPEnv.expert_executing or PnPEnv.expert_waiting_save))
                # 先获取当前状态的图像（用于数据记录）
                if is_recording or auto_is_recording:
                    # grab_image 这里只做内存拷贝，不做 Resize，速度快很多
                    images_dict_raw = PnPEnv.grab_image()
                    # 必须使用 .copy()，因为 images_dict_raw 可能会在下一帧被覆盖
                    images_dict_safe = {k: v.copy() for k, v in images_dict_raw.items()}
                
                # 机器人控制
                action, reset = PnPEnv.teleop_robot(mode=current_mode)
                
                # [Z] 手动重置环境 (全自动模式下禁用)
                if reset:
                    if auto_state != AUTO_STATE_IDLE:
                        print(f"\n⚠️ [Z] Disabled: Full-Auto Mode is active. Press [P] to stop first.")
                    elif worker.saving_in_progress:
                        print(f"\n⚠️ [Z] Cannot reset: Save operation in progress. Please wait for save to complete.")
                    else:
                        print("🔄 Environment Reset.")
                        # 🔥 先停止录制（如果正在录制）
                        if is_recording:
                            is_recording = False
                            if current_mode == 'arm':
                                PnPEnv.is_recording = False
                                PnPEnv._waiting_for_save = False  # 🔥 重置队列显示等待标志
                            print(f"⚠️ [INTERRUPTED] Recording stopped due to reset. (ID {current_episode_id})")
                        # 🔥 也需要重置等待状态（即使不在录制中，可能是暂停状态）
                        if current_mode == 'arm':
                            PnPEnv._waiting_for_save = False
                        PnPEnv.reset(mode=current_mode)
                        # 🔥 先清空队列（防止新数据继续进入）
                        if worker.clear_queue():
                            # 🔥 等待正在处理的帧完成并写入缓冲区（确保所有数据都被捕获）
                            worker.wait_queue_empty()
                            # 🔥 最后清空缓冲区（包括刚才写入的帧）
                            if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                                dataset.clear_episode_buffer()

                # 物理执行
                current_state = PnPEnv.step(action, mode=current_mode)
                
                if current_mode == 'arm':
                    action_to_save = PnPEnv.current_arm_q[:7].astype(np.float32)
                else:
                    action_to_save = action.astype(np.float32)

                # 🔥 数据收集：异步处理模式 🔥
                if is_recording or auto_is_recording:

                    # 🔥 获取真实位姿 (Ground Truth) - 仅在 base 模式下
                    if current_mode == 'base':
                        pos = PnPEnv.env.get_p_body('tb3_base')  # [x, y, z]
                        rot = PnPEnv.env.get_R_body('tb3_base')  # 旋转矩阵
                        theta = np.arctan2(rot[1, 0], rot[0, 0])  # 简单的 Yaw 角计算
                        base_pose = np.array([pos[0], pos[1], theta], dtype=np.float32)
                        # 打包发送给后台工作线程（包含 mode 和 base_pose）
                        worker.put((current_mode, images_dict_safe, current_state, action_to_save, PnPEnv.obj_init_pose, PnPEnv.instruction, base_pose))
                    else:
                        # 打包发送给后台工作线程（arm 模式，不包含 base_pose）
                        worker.put((current_mode, images_dict_safe, current_state, action_to_save, PnPEnv.obj_init_pose, PnPEnv.instruction))
                    
                    current_frames += 1
                    
                    # 每秒打印进度，显示队列积压情况
                    if current_frames % FPS == 0:
                        q_size = worker.qsize()
                        # 内存警告：队列每帧约 1.5MB (3张 800x600 RGB)
                        mem_mb = q_size * 1.5
                        warn = ""
                        if q_size > 500:
                            warn = " ⚠️ HIGH MEM!"
                        elif q_size > 200:
                            warn = " 📈"
                        print(f"   [{current_mode.upper()}] Recording... {current_frames} frames | Queue: {q_size} (~{mem_mb:.0f}MB){warn}    ", end='\r')
                
                PnPEnv.render(teleop=True, idx=current_episode_id)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C).")
        # 🔥 如果在全自动模式下，显示当前进度
        if auto_state != AUTO_STATE_IDLE:
            print(f"   🛑 Full-Auto Mode was active. Progress: {auto_recorded_count}/{AUTO_RECORD_TARGET_EPISODES} episodes saved.")
            print(f"   ❌ Current recording (if any) will be discarded.")
    finally:
        # 停止后台工作线程
        print("\n🛑 Stopping worker thread...")
        worker.running = False
        # 等待队列中的所有任务处理完毕
        worker.wait_queue_empty()
        # 等待线程真正退出
        worker.join(timeout=5.0)
        if worker.is_alive():
            print("⚠️ Warning: Worker thread did not exit in time")
        else:
            print("✅ Worker thread stopped")
        
        PnPEnv.env.close_viewer()
        
        # 清理逻辑 - 等待所有图像写入进程完成后再删除（两个数据集都要清理）
        print("⏳ Waiting for image writers to finish...")
        time.sleep(2.0)  # 给写入进程时间完成
        
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
        print(f"   ARM:  {episode_ids['arm']} episodes")
        print(f"   BASE: {episode_ids['base']} episodes")

if __name__ == "__main__":
    main()
