import sys
import numpy as np
import os
import shutil
import time
import glfw
import threading
import queue
import cv2  # 🔥 比 PIL 快 5-10 倍
from mujoco_env.y_env4 import SimpleEnv4 
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ================= 🛡️ 安全配置区域 🛡️ =================

# 初始模式选择（可通过 C 键热切换）
INITIAL_MODE = 'arm'  # 'arm' 或 'base'

# ================= 📁 数据集配置 📁 =================
# 两种模式的数据集配置（支持热切换，同时管理两个数据集）

DATASET_CONFIG = {
    'arm': {
        'repo_name': 'omy_arm_data_v4',
        'root': "./demo_data_arm_v4",
    },
    'base': {
        'repo_name': 'omy_base_data_v4',
        'root': "./demo_data_base_v4",
    }
}

# ================= ⚙️ 场景配置 ⚙️ =================
SEED = 0 
XML_PATH = './asset/example_scene_y4.xml'

# 🔥 安全熔断设置 🔥
# 单条数据最大录制时长 (秒)
# 超过这个时间将自动丢弃，防止内存溢出
MAX_EPISODE_SEC = 200  
FPS = 20
MAX_FRAMES = MAX_EPISODE_SEC * FPS

# 🔥 图像分辨率配置（影响数据量和训练速度）
# 256x256: 标准，兼容现有数据
# 224x224: ViT 标准输入，减少 23% 数据量（推荐）
# 196x196: 进一步压缩，减少 41% 数据量
IMG_SIZE = 224  # ← 修改这里来调整分辨率

# 各模式的动作和状态维度配置
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
        """🔥 使用 OpenCV resize，比 PIL 快 5-10 倍"""
        # 使用全局配置的 IMG_SIZE
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

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

    print("\n" + "="*50)
    print(f" 🚀 DATA COLLECTION MODE: {current_mode.upper()} (HOT-SWITCH ENABLED)")
    print(f" 📁 ARM  Dataset: {DATASET_CONFIG['arm']['repo_name']} (Episode #{episode_ids['arm']})")
    print(f" 📁 BASE Dataset: {DATASET_CONFIG['base']['repo_name']} (Episode #{episode_ids['base']})")
    print(f" 🧵 ASYNC RECORDER ACTIVE (Using Background Thread)")
    print(f" 🛡️ Safety Limit: Max {MAX_EPISODE_SEC}s ({MAX_FRAMES} frames) per episode")
    print(" Control Keys:")
    print("  [J] : Start Recording (开始录制)")
    print("  [K] : Stop & SAVE (保存当前条)")
    print("  [Q] : Stop & DISCARD (废弃当前条，重录)")
    print("  [Z] : Reset Environment Only (仅重置环境)")
    print("  [C] : 🔥 HOT-SWITCH Mode (Base ↔ Arm, 不重置环境)")
    print("  [I] : Smooth Return Home (平滑归位机械臂, 仅 Arm 模式)")
    print("  [T] : 🤖 Test Mode: Auto Execute Expert Policy (测试模式，不录制, 仅 Arm 模式)")
    print("  [Y] : 🎥 Record Mode: Auto Execute + Start Recording (录制模式，自动执行并录制, 仅 Arm 模式)")
    print(f" Current Mode: {current_mode.upper()} | Next Episode ID: {episode_ids[current_mode]}")
    print("="*50 + "\n")

    try:
        # 🔥 2. 移除 NUM_DEMO 限制，实现无限录制 🔥
        while PnPEnv.env.is_viewer_alive():
            PnPEnv.step_env()
            
            if PnPEnv.env.loop_every(HZ=FPS):
                
                # 获取当前模式的数据集（方便后续使用）
                dataset = datasets[current_mode]
                current_episode_id = episode_ids[current_mode]
                
                # ================= 按键逻辑 =================
                
                # 🔥 同步环境的录制状态（用于Y键自动录制）
                if current_mode == 'arm':
                    # 如果环境开始录制，但本地还没开始，则同步开始
                    if PnPEnv.is_recording and not is_recording:
                        is_recording = True
                        current_frames = 0
                        PnPEnv._expert_done_printed = False  # 🔥 重置提示标志
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()
                        worker.wait_queue_empty()
                        print(f"🔴 [REC START] [{current_mode.upper()}] Recording Episode {current_episode_id} (Auto-start from Expert Policy)...")
                    # 🔥 专家策略结束后，只打印提示，不自动停止录制
                    # 让用户人工检查后按 K（保存）或 Q（丢弃）
                    elif not PnPEnv.is_recording and not PnPEnv.expert_executing and not PnPEnv.expert_pending and is_recording:
                        # 只在专家策略刚结束时打印一次提示
                        if not hasattr(PnPEnv, '_expert_done_printed') or not PnPEnv._expert_done_printed:
                            print(f"\n✅ Expert Policy finished! Recording continues...")
                            print(f"   👀 Please check the result, then press [K] to SAVE or [Q] to DISCARD.")
                            PnPEnv._expert_done_printed = True
                
                # [J] 开始录制
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_J):
                    if not is_recording:
                        is_recording = True
                        current_frames = 0  # 重置计数器
                        # 🔥 仅在 episode_buffer 存在时清空（作为保险）
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer() # 确保缓冲区干净
                        # 确保之前的队列处理完了
                        worker.wait_queue_empty()
                        # 🔥 3. 录制开始时打印当前是第几条 🔥
                        print(f"🔴 [REC START] [{current_mode.upper()}] Recording Episode {current_episode_id} ...")

                # [K] 停止并保存 (Success)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_K):
                    if is_recording:
                        is_recording = False
                        # 🔥 同步停止环境的录制状态
                        if current_mode == 'arm':
                            PnPEnv.is_recording = False
                            PnPEnv._expert_done_printed = False  # 重置提示标志
                        pending_frames = worker.qsize()
                        peak = worker.peak_qsize()
                        print(f"\n⏳ Saving Episode {current_episode_id}...")
                        print(f"   📊 Recorded: {current_frames} frames | Queue backlog: {pending_frames} | Peak: {peak}")
                        
                        # 🔥 带进度的等待
                        while worker.qsize() > 0:
                            remaining = worker.qsize()
                            progress = (pending_frames - remaining) / max(pending_frames, 1) * 100
                            print(f"   ⏳ Processing: {pending_frames - remaining}/{pending_frames} ({progress:.0f}%) - {remaining} remaining...", end='\r')
                            time.sleep(0.5)
                        worker.wait_queue_empty()  # 最终确认
                        print(f"\n   ✅ Queue cleared!                                          ")
                        
                        dataset.save_episode()
                        print(f"✅ [SAVED] [{current_mode.upper()}] Episode {current_episode_id} saved ({current_frames} frames).")
                        episode_ids[current_mode] += 1

                # [Q] 停止并废弃 (Discard / Fail)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_Q):
                    if is_recording:
                        is_recording = False
                        # 🔥 同步停止环境的录制状态
                        if current_mode == 'arm':
                            PnPEnv.is_recording = False
                            PnPEnv._expert_done_printed = False  # 重置提示标志
                        # 🔥 仅在 episode_buffer 存在时清空（作为保险）
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer() # 清空缓存，不保存
                        print(f"❌ [DISCARDED] [{current_mode.upper()}] Episode {current_episode_id} data cleared. (ID unchanged)")

                # ================= 🔥 [C] 热切换模式 🔥 =================
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_C):
                    # 如果正在录制，先停止并丢弃
                    if is_recording:
                        is_recording = False
                        # 🔥 同步停止环境的录制状态
                        if current_mode == 'arm':
                            PnPEnv.is_recording = False
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer()
                        print(f"⚠️ [RECORDING STOPPED] Discarding current recording before switch.")
                    
                    # 等待队列清空
                    worker.wait_queue_empty()
                    
                    # 切换模式
                    old_mode = current_mode
                    current_mode = 'arm' if current_mode == 'base' else 'base'
                    
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
                    if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                        dataset.clear_episode_buffer()
                    print(f"\n⚠️ [TIMEOUT] Max duration ({MAX_EPISODE_SEC}s) reached!")
                    print(f"❌ [AUTO-DISCARDED] [{current_mode.upper()}] Episode {current_episode_id} data cleared. (ID unchanged)")
                
                # ===========================================================

                # 机器人控制
                action, reset = PnPEnv.teleop_robot(mode=current_mode)
                
                # [Z] 手动重置环境
                if reset:
                    print("🔄 Environment Reset.")
                    # 🔥 先停止录制（如果正在录制）
                    if is_recording:
                        is_recording = False
                        if current_mode == 'arm':
                            PnPEnv.is_recording = False
                        print(f"⚠️ [INTERRUPTED] Recording stopped due to reset. (ID {current_episode_id})")
                    PnPEnv.reset(mode=current_mode)
                    # 🔥 仅在 episode_buffer 存在时清空（作为保险）
                    if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                        dataset.clear_episode_buffer()

                # 物理执行
                current_state = PnPEnv.step(action, mode=current_mode)
                
                if current_mode == 'arm':
                    action_to_save = PnPEnv.current_arm_q[:7].astype(np.float32)
                else:
                    action_to_save = action.astype(np.float32)

                # 🔥 数据收集：异步处理模式 🔥
                if is_recording:
                    # grab_image 这里只做内存拷贝，不做 Resize，速度快很多
                    images_dict_raw = PnPEnv.grab_image()
                    
                    # 必须使用 .copy()，因为 images_dict_raw 可能会在下一帧被覆盖
                    images_dict_safe = {k: v.copy() for k, v in images_dict_raw.items()}

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
        print("Interrupted.")
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
