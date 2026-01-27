import sys
import numpy as np
import torch
import time
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.y_env3 import SimpleEnv3
from mujoco_env.utils import add_title_to_img

# ================= 配置区域 =================
#REPO_NAME = 'omy_base_data'
#ROOT = "./demo_data_base"
REPO_NAME = 'omy_base_data_clean'
ROOT = './demo_data_base_clean'

XML_PATH = './asset/example_scene_y3.xml'
FPS = 20  # 录制时的帧率
LOOP_EPISODE = 51  # 如果设置为非零值（如5），则循环播放该episode；为0时播放所有episodes
# ===========================================

class VisualizerEnv(SimpleEnv3):
    def render(self, idx=0, rel_time=0.0):
        # 1. 3D 背景
        if self.env.viewer is not None:
            self.env.viewer.cam.type = 2 
            try:
                cam_id = self.env.model.camera('tb3_view').id
                self.env.viewer.cam.fixedcamid = cam_id
            except: pass

        # 2. Overlay 图像
        if hasattr(self, 'rec_img_front'):
            img_front = add_title_to_img(self.rec_img_front, text="[REC] Front (Main)", shape=(640, 480))
            self.env.viewer_rgb_overlay(img_front, loc='top left')

        if hasattr(self, 'rec_img_left'):
            img_left = add_title_to_img(self.rec_img_left, text="[REC] Left", shape=(320, 240))
            self.env.viewer_rgb_overlay(img_left, loc='top right')

        if hasattr(self, 'rec_img_right'):
            img_right = add_title_to_img(self.rec_img_right, text="[REC] Right", shape=(320, 240))
            self.env.viewer_rgb_overlay(img_right, loc='bottom right')

        # 3. 显示时间信息 (核心修改)
        # 覆盖原本的 Sim Time，显示当前 Episode 的相对时间
        self.env.viewer_text_overlay(
            text1=f'Ep {idx} | Rel Time: {rel_time:.2f}s', 
            text2=f'Total Sim: {self.env.data.time:.2f}s'
        )
        
        self.env.render()

def main():
    print(f"Loading dataset from {ROOT}...")
    try:
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Initializing Environment...")
    env = VisualizerEnv(XML_PATH, state_type='joint_angle')
    
    print("\nStarting Playback...")
    print("Sync Mode: Wall-Clock Locking (Strict)")
    
    episode_data_index = dataset.episode_data_index
    
    # 确定要播放的episode列表
    if LOOP_EPISODE != 0:
        # 验证episode索引是否有效
        if LOOP_EPISODE < 0 or LOOP_EPISODE >= dataset.num_episodes:
            print(f"Error: LOOP_EPISODE={LOOP_EPISODE} is out of range [0, {dataset.num_episodes-1}]")
            return
        episode_list = [LOOP_EPISODE]
        print(f"Loop Mode: Will loop playback episode {LOOP_EPISODE}")
    else:
        episode_list = list(range(dataset.num_episodes))
        print(f"Normal Mode: Will play all {dataset.num_episodes} episodes")
    
    # 外层循环：如果LOOP_EPISODE不为0，则无限循环播放
    while True:
        for ep_idx in episode_list:
            print(f"\n=== Playing Episode {ep_idx} ===")
            
            from_idx = episode_data_index["from"][ep_idx].item()
            to_idx = episode_data_index["to"][ep_idx].item()
            frame_iterator = iter(range(from_idx, to_idx))
            
            # 重置回原点，但位置连续性由物理引擎保持（如果不reset joint的话）
            # 这里使用 mode='base' 进行标准重置
            env.reset(mode='base')
            
            # 🔥 读取本集第一帧的位姿来初始化小车
            # 注意：set_base_pose 会自动使用 0.05m 的安全高度，让机器人自然掉落而不是从地下弹出来
            try:
                first_frame = dataset[from_idx]
                if 'base_pose' in first_frame:
                    start_pose = first_frame['base_pose'].numpy()
                    start_x, start_y, start_theta = start_pose[0], start_pose[1], start_pose[2]
                    env.set_base_pose(start_x, start_y, start_theta)  # z 默认为 0.05m 安全高度
                    print(f"Reset robot to: x={start_x:.2f}, y={start_y:.2f}, theta={np.degrees(start_theta):.2f}° (z=0.05m safe height)")
            except KeyError:
                print("No base_pose found in dataset, using default origin.")
            except Exception as e:
                print(f"Warning: Could not set initial pose from dataset: {e}") 
            
            # 🔥 记录本集开始时的物理绝对时间 🔥
            start_abs_sim_time = env.env.data.time
            
            # 记录本集开始时的墙上绝对时间
            episode_start_wall_time = time.time()
            played_frames = 0
            
            # 🔥 位置漂移统计变量 🔥
            max_drift = 0.0
            total_drift = 0.0
            drift_count = 0
            
            while env.env.is_viewer_alive():
                # 1. 物理步进 (Sim Time 持续增加)
                env.step_env()
                
                # 2. 帧率门控 (根据 Sim Time 的增量来触发)
                if env.env.loop_every(HZ=FPS):
                    try:
                        frame_idx = next(frame_iterator)
                    except StopIteration:
                        break
                    
                    # --- 加载数据 ---
                    item = dataset[frame_idx]
                    
                    # 1. 这一步是为了让画面更新，特别是机械臂姿态
                    env.step(item['action'].numpy(), mode='base') 
                    
                    # 2. 【关键修改】如果数据里有真实位置，强制把小车按回去！
                    # 这样你就不会看到漂移了，看到的就是你当时真实的完美操作
                    if 'base_pose' in item:
                        recorded_pose = item['base_pose'].numpy()
                        x, y, theta = recorded_pose[0], recorded_pose[1], recorded_pose[2]
                        # 强制覆盖物理引擎的计算结果
                        env.set_base_pose(x, y, theta)

                    # --- 图像更新 ---
                    env.rec_img_front = (item['observation.images.front'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    env.rec_img_left = (item['observation.images.left'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    env.rec_img_right = (item['observation.images.right'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # --- 计算相对 Sim Time ---
                    current_rel_sim_time = env.env.data.time - start_abs_sim_time
                    
                    # --- 渲染 ---
                    env.render(idx=ep_idx, rel_time=current_rel_sim_time)
                    
                    # --- 3. 强力墙上时钟同步 (Wall-Clock Sync) ---
                    played_frames += 1
                    
                    # 理论上当前帧应该在什么时刻播放 (比如第20帧应该是1.0秒)
                    target_wall_time = played_frames * (1.0 / FPS)
                    
                    # 实际上过去了多久
                    current_wall_time = time.time() - episode_start_wall_time
                    
                    # 偏差计算
                    diff = target_wall_time - current_wall_time
                    
                    # 如果跑太快了 (Diff > 0)，就睡过去
                    if diff > 0:
                        time.sleep(diff)
                    
                    # --- 4. 位置漂移检测 🔥 ---
                    pos_error = None
                    if 'base_pose' in item:
                        # 获取当前回放时的实际位置
                        curr_replay_pos = env.env.get_p_body('tb3_base')[:2]  # [x, y]
                        
                        # 获取当初录制时的位置 (Ground Truth)
                        recorded_pose = item['base_pose'].numpy()  # [x, y, theta]
                        recorded_pos = recorded_pose[:2]  # [x, y]
                        
                        # 计算位置误差（欧氏距离）
                        pos_error = np.linalg.norm(curr_replay_pos - recorded_pos)
                        
                        # 更新统计
                        max_drift = max(max_drift, pos_error)
                        total_drift += pos_error
                        drift_count += 1
                    
                    # [调试日志] 每秒打印一次，检查同步情况和位置漂移
                    if played_frames % FPS == 0:
                        if pos_error is not None:
                            avg_drift = total_drift / drift_count if drift_count > 0 else 0.0
                            print(f"Frame: {played_frames} | Sim: {current_rel_sim_time:.2f}s | "
                                  f"Wall: {time.time() - episode_start_wall_time:.2f}s | "
                                  f"Lag: {diff:.4f}s | "
                                  f"Drift: {pos_error*100:.2f}cm (Max: {max_drift*100:.2f}cm, Avg: {avg_drift*100:.2f}cm)", 
                                  end='\r', flush=True)
                        else:
                            print(f"Frame: {played_frames} | Sim: {current_rel_sim_time:.2f}s | "
                                  f"Wall: {time.time() - episode_start_wall_time:.2f}s | "
                                  f"Lag: {diff:.4f}s", 
                                  end='\r', flush=True)

            print(f"\nEpisode {ep_idx} finished.")
            # 显示位置漂移统计信息
            if drift_count > 0:
                avg_drift = total_drift / drift_count
                print(f"  📊 Position Drift Stats: Max={max_drift*100:.2f} cm, Avg={avg_drift*100:.2f} cm, Samples={drift_count}")
            
            # 🔥 关键修复：如果viewer已关闭，退出内层for循环
            if not env.env.is_viewer_alive():
                print("\nViewer closed. Exiting...")
                break
        
        # 🔥 关键修复：检查viewer状态，如果已关闭则退出外层while循环
        if not env.env.is_viewer_alive():
            break
        
        # 如果LOOP_EPISODE为0，播放完所有episodes后退出
        if LOOP_EPISODE == 0:
            break
    
    print("All episodes finished.")
    env.env.close_viewer()

if __name__ == "__main__":
    main()