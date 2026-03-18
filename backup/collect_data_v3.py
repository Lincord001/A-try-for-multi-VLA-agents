import sys
import numpy as np
import os
import shutil
import time
import glfw
import threading
import queue
from PIL import Image
from mujoco_env.y_env3 import SimpleEnv3 
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ================= 🛡️ 安全配置区域 🛡️ =================

# 模式选择
#COLLECT_MODE = 'arm'  
COLLECT_MODE = 'base'

# ================= 📁 数据集配置 📁 =================
# 取消注释你想要使用的数据集配置（REPO_NAME 和 ROOT 必须成对取消注释）

# Base 模式数据集选项（当 COLLECT_MODE = 'base' 时使用）：

# 选项 1: 原始 base 数据集
#REPO_NAME = 'omy_base_data'
#ROOT = "./demo_data_base"

# 选项 2: 清理后的 base 数据集（当前激活）
REPO_NAME = 'omy_base_data_clean'
ROOT = "./demo_data_base_ver_2_clean"

# Arm 模式数据集选项（当 COLLECT_MODE = 'arm' 时使用）：
#REPO_NAME = 'omy_arm_data'
#ROOT = "./demo_data_arm"

# ================= ⚙️ 场景配置 ⚙️ =================
SEED = 0 
XML_PATH = './asset/example_scene_y3.xml'

# 🔥 安全熔断设置 🔥
# 单条数据最大录制时长 (秒)
# 超过这个时间将自动丢弃，防止内存溢出
MAX_EPISODE_SEC = 200  
FPS = 20
MAX_FRAMES = MAX_EPISODE_SEC * FPS

# 根据模式自动设置动作和状态维度
if COLLECT_MODE == 'arm':
    ACTION_SHAPE = (7,) 
    STATE_SHAPE = (6,)
elif COLLECT_MODE == 'base':
    ACTION_SHAPE = (2,)
    STATE_SHAPE = (2,)
else:
    raise ValueError("COLLECT_MODE must be 'arm' or 'base'")

# ================= 🧵 异步处理工作线程 🧵 =================

class DataSaverWorker(threading.Thread):
    def __init__(self, dataset, collect_mode):
        super().__init__()
        self.dataset = dataset
        self.collect_mode = collect_mode
        self.queue = queue.Queue(maxsize=200)
        self.daemon = True # 设置为守护线程，主程序退出时自动退出
        self.running = True

    def put(self, item):
        """主线程调用：将原始数据放入队列"""
        self.queue.put(item)

    def run(self):
        """子线程循环：后台处理数据"""
        while self.running:
            try:
                # 等待数据，超时1秒避免卡死
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # item 包含: (images_dict_raw, current_state, action, obj_init, instruction, base_pose)
            # base_pose 仅在 base 模式下存在
            if self.collect_mode == 'base':
                images_dict, state, action, obj_init, task, base_pose = item
            else:
                images_dict, state, action, obj_init, task = item
                base_pose = None

            # --- 这里是耗时的 CPU 操作，现在在后台运行，不卡主界面 ---
            frame_data = {
                "observation.state": state,
                "action": action,
                "obj_init": obj_init,
            }

            try:
                # 执行 Resize (耗时大户)
                if self.collect_mode == 'arm':
                    frame_data["observation.images.agent"] = np.array(Image.fromarray(images_dict['agent']).resize((256,256)))
                    frame_data["observation.images.wrist"] = np.array(Image.fromarray(images_dict['wrist']).resize((256,256)))
                elif self.collect_mode == 'base':
                    frame_data["observation.images.front"] = np.array(Image.fromarray(images_dict['front']).resize((256,256)))
                    frame_data["observation.images.left"] = np.array(Image.fromarray(images_dict['left']).resize((256,256)))
                    frame_data["observation.images.right"] = np.array(Image.fromarray(images_dict['right']).resize((256,256)))
                    # 🔥 新增：添加 base_pose 到 frame_data（不使用 observation. 前缀）
                    if base_pose is not None:
                        frame_data["base_pose"] = base_pose

                # 写入硬盘
                self.dataset.add_frame(frame_data, task=task)
            except Exception as e:
                print(f"Error in worker thread: {e}")
            finally:
                self.queue.task_done()

    def wait_queue_empty(self):
        """等待所有数据处理完毕"""
        self.queue.join()

# ===========================================

def main():
    print(f"Initializing Environment in [{COLLECT_MODE.upper()}] mode...")
    PnPEnv = SimpleEnv3(XML_PATH, seed=SEED, state_type='joint_angle')
    PnPEnv.reset(mode=COLLECT_MODE)

    # ---------------------------------------------------------
    # 🔥 1. 智能数据集加载逻辑 (Auto-Resume) 🔥
    if os.path.exists(ROOT):
        print(f"\nExample dataset folder found at: {ROOT}")
        print(">> Loading existing dataset (Append Mode)...")
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)
        print(f">> Found {dataset.num_episodes} existing episodes.")
    else:
        print(f"\nNo existing dataset found at: {ROOT}")
        print(">> Creating NEW dataset...")
        
        # 动态定义 Features
        features = {
            "observation.state": {"dtype": "float32", "shape": STATE_SHAPE, "names": ["state"]},
            "action": {"dtype": "float32", "shape": ACTION_SHAPE, "names": ["action"]},
            "obj_init": {"dtype": "float32", "shape": (9,), "names": ["obj_init"]},  # 3个物体 * 3坐标 = 9
        }

        if COLLECT_MODE == 'arm':
            features["observation.images.agent"] = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
            features["observation.images.wrist"] = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
        elif COLLECT_MODE == 'base':
            for cam in ['front', 'left', 'right']:
                features[f"observation.images.{cam}"] = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
            # 🔥 新增：base_pose 字段，存储 (x, y, theta)
            # 注意：不使用 observation. 前缀，避免被模型自动使用
            features["base_pose"] = {
                "dtype": "float32", 
                "shape": (3,), 
                "names": ["x", "y", "theta"]
            }

        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            root=ROOT,
            robot_type="omy",
            fps=FPS,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,
        )

    # 🔥 启动后台工作线程 🔥
    worker = DataSaverWorker(dataset, COLLECT_MODE)
    worker.start()

    # 初始化当前序号 (从已有的最后一条往后数)
    current_episode_id = dataset.num_episodes
    is_recording = False
    # 记录当前条目已录制的帧数
    current_frames = 0

    print("\n" + "="*50)
    print(f" DATA COLLECTION MODE: {COLLECT_MODE.upper()}")
    print(f" 📁 Dataset: {REPO_NAME} @ {ROOT}")
    print(f" 🧵 ASYNC RECORDER ACTIVE (Using Background Thread)")
    print(f" 🛡️ Safety Limit: Max {MAX_EPISODE_SEC}s ({MAX_FRAMES} frames) per episode")
    print(" Control Keys:")
    print("  [J] : Start Recording (开始录制)")
    print("  [K] : Stop & SAVE (保存当前条)")
    print("  [Q] : Stop & DISCARD (废弃当前条，重录)")
    print("  [Z] : Reset Environment Only (仅重置环境)")
    print(f" Next Episode ID: {current_episode_id}")
    print("="*50 + "\n")

    try:
        # 🔥 2. 移除 NUM_DEMO 限制，实现无限录制 🔥
        while PnPEnv.env.is_viewer_alive():
            PnPEnv.step_env()
            
            if PnPEnv.env.loop_every(HZ=FPS):
                
                # ================= 按键逻辑 =================
                
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
                        print(f"🔴 [REC START] Recording Episode {current_episode_id} ...")

                # [K] 停止并保存 (Success)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_K):
                    if is_recording:
                        is_recording = False
                        # 等待 Worker 把队列里的帧都处理完塞进 dataset buffer
                        print("⏳ Saving... (Waiting for worker thread)")
                        worker.wait_queue_empty()
                        dataset.save_episode()
                        print(f"✅ [SAVED] Episode {current_episode_id} saved ({current_frames} frames).")
                        current_episode_id += 1 # 只有保存了才+1                        # PnPEnv.reset(mode=COLLECT_MODE) # 可选：保存后自动重置

                # [Q] 停止并废弃 (Discard / Fail)
                if PnPEnv.env.is_key_pressed_once(glfw.KEY_Q):
                    if is_recording:
                        is_recording = False
                        # 🔥 仅在 episode_buffer 存在时清空（作为保险）
                        if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                            dataset.clear_episode_buffer() # 清空缓存，不保存
                        # 队列里的剩余任务其实已经无所谓了，让它跑完或者不理它
                        print(f"❌ [DISCARDED] Episode {current_episode_id} data cleared. (ID unchanged)")
                        # 下次按 C，ID 还是 current_episode_id

                # ================= 🛡️ 熔断逻辑 (自动停止) 🛡️ =================
                
                if is_recording and current_frames >= MAX_FRAMES:
                    is_recording = False
                    # 🔥 超时自动丢弃数据，防止录制时间过长
                    # 🔥 仅在 episode_buffer 存在时清空（作为保险）
                    if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                        dataset.clear_episode_buffer() # 清空缓存，不保存
                    print(f"\n⚠️ [TIMEOUT] Max duration ({MAX_EPISODE_SEC}s) reached!")
                    print(f"❌ [AUTO-DISCARDED] Episode {current_episode_id} data cleared. (ID unchanged)")
                    # 下次按 J，ID 还是 current_episode_id
                
                # ===========================================================

                # 机器人控制
                action, reset = PnPEnv.teleop_robot(mode=COLLECT_MODE)
                
                # [Z] 手动重置环境
                if reset:
                    print("🔄 Environment Reset.")
                    PnPEnv.reset(mode=COLLECT_MODE)
                    # 🔥 仅在 episode_buffer 存在时清空（作为保险）
                    if hasattr(dataset, 'episode_buffer') and dataset.episode_buffer is not None:
                        dataset.clear_episode_buffer()
                    if is_recording:
                        is_recording = False
                        print(f"⚠️ [INTERRUPTED] Recording stopped due to reset. (ID {current_episode_id})")

                # 物理执行
                current_state = PnPEnv.step(action, mode=COLLECT_MODE)
                
                if COLLECT_MODE == 'arm':
                    action_to_save = PnPEnv.current_arm_q[:7].astype(np.float32)
                else:
                    action_to_save = action.astype(np.float32)

                # 🔥 数据收集：异步处理模式 🔥
                if is_recording:
                    # grab_image 这里只做内存拷贝，不做 Resize，速度快很多
                    images_dict_raw = PnPEnv.grab_image()
                    
                    # 必须使用 .copy()，因为 images_dict_raw 可能会在下一帧被覆盖
                    # 这里拷贝一次内存开销很小，相比 resize 收益巨大
                    images_dict_safe = {k: v.copy() for k, v in images_dict_raw.items()}

                    # 🔥 获取真实位姿 (Ground Truth) - 仅在 base 模式下
                    if COLLECT_MODE == 'base':
                        pos = PnPEnv.env.get_p_body('tb3_base')  # [x, y, z]
                        rot = PnPEnv.env.get_R_body('tb3_base')  # 旋转矩阵
                        theta = np.arctan2(rot[1, 0], rot[0, 0])  # 简单的 Yaw 角计算
                        base_pose = np.array([pos[0], pos[1], theta], dtype=np.float32)
                        # 打包发送给后台工作线程（包含 base_pose）
                        worker.put((images_dict_safe, current_state, action_to_save, PnPEnv.obj_init_pose, PnPEnv.instruction, base_pose))
                    else:
                        # 打包发送给后台工作线程（arm 模式，不包含 base_pose）
                        worker.put((images_dict_safe, current_state, action_to_save, PnPEnv.obj_init_pose, PnPEnv.instruction))
                    
                    current_frames += 1
                    
                    # 可选：在控制台打印进度 (每秒打印一次)，显示队列积压情况
                    if current_frames % FPS == 0:
                        q_size = worker.queue.qsize()
                        print(f"   Recording... {current_frames}/{MAX_FRAMES} frames | Queue Lag: {q_size} frames", end='\r')
                
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
        # 清理逻辑 - 等待所有图像写入进程完成后再删除
        if os.path.exists(dataset.root / 'images'):
            # 等待一段时间，确保 LeRobotDataset 的图像写入进程/线程完成
            print("⏳ Waiting for image writers to finish...")
            time.sleep(2.0)  # 给写入进程时间完成
            
            # 使用重试机制删除
            max_retries = 3
            retry_delay = 1.0
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(dataset.root / 'images')
                    print("✅ Cleaned up images directory")
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ Retry {attempt + 1}/{max_retries}: Waiting before retry...")
                        time.sleep(retry_delay)
                    else:
                        print(f"⚠️ Warning: Could not fully remove images directory: {e}")
                        print("   (This is usually harmless - files may still be in use)")
                        print("   You can manually delete it later if needed.")

if __name__ == "__main__":
    main()
