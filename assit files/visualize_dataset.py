import sys
import numpy as np
import torch
import time
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.y_env4 import SimpleEnv4
from mujoco_env.y_env6 import SimpleEnv6
from mujoco_env.utils import add_title_to_img

# ================= 配置区域 =================
# 🔥 模式选择：'base' 或 'arm'
MODE = 'arm'  # 切换这里来选择审阅底盘数据还是机械臂数据

# 🔥 数据集版本选择（同一模式下在新老数据之间切换）
# 可选：
# - arm 模式:  'v5' / 'v6'
# - base 模式: 'v4'
DATASET_VERSION = 'v6'

# 数据集配置（会根据 MODE + DATASET_VERSION 自动选择）
DATASET_CONFIG = {
    'base': {
        'v6': {
            'repo_name': 'demo_data_base_v6',
            'root': './demo_data_base_v6',
            'image_keys': ['front', 'left', 'right'],  # 底盘模式的相机
            'env_backend': 'y6',
            'xml_path': './asset/example_scene_y6.xml',
        }
    },
    'arm': {
        'v5': {
            'repo_name': 'demo_data_arm_v5',
            'root': './demo_data_arm_v5',
            'image_keys': ['agent', 'wrist'],  # 机械臂模式的相机
            'env_backend': 'y4',
            'xml_path': './asset/example_scene_y4.xml',
        },
        'v6': {
            'repo_name': 'demo_data_arm_v6',
            'root': './demo_data_arm_v6',
            'image_keys': ['agent', 'wrist'],  # 与 v5 保持一致
            'env_backend': 'y6',
            'xml_path': './asset/example_scene_y6.xml',
        },
    }
}

FPS = 20  # 录制时的帧率
LOOP_EPISODE = 0  # 如果设置为非零值（如5），则循环播放该episode；为0时播放所有episodes
START_EPISODE = 198  # 如果设置为非零值（如3），则从该episode开始播放到最后一个；为0时从第一个开始播放
# ===========================================


class VisualizerRenderMixin:
    def __init__(self, mode='base', **kwargs):
        super().__init__(**kwargs)
        self.vis_mode = mode
    
    def render(self, idx=0, rel_time=0.0):
        # =================== 🔥 图像 Overlay 显示逻辑 🔥 ===================
        # 与 y_env4.py 的 render() 布局保持一致
        
        if self.vis_mode == 'arm':
            # === Arm 模式布局 (与 y_env4.py 一致) ===
            # 🔥 修改：arm模式只有2个相机（agent, wrist），匹配 collect_data_v4.py
            # rec_img_0 = agent, rec_img_1 = wrist
            
            # 1. 右上角: Agent 全局视角 (主视角)
            if hasattr(self, 'rec_img_0'):
                img = add_title_to_img(self.rec_img_0, text="[REC] Agent View", shape=(640, 480))
                self.env.viewer_rgb_overlay(img, loc='top right')
            
            # 2. 右下角: Wrist 手腕视角
            if hasattr(self, 'rec_img_1'):
                img = add_title_to_img(self.rec_img_1, text="[REC] Wrist View", shape=(640, 480))
                self.env.viewer_rgb_overlay(img, loc='bottom right')
        
        else:
            # === Base 模式布局 (与 y_env4.py 一致) ===
            # rec_img_0 = front, rec_img_1 = left, rec_img_2 = right
            
            # 1. 右上角: Front 正前方 (主视角)
            if hasattr(self, 'rec_img_0'):
                img = add_title_to_img(self.rec_img_0, text="[REC] Front View", shape=(640, 480))
                self.env.viewer_rgb_overlay(img, loc='top right')
            
            # 2. 左上角: Left 左侧视角
            if hasattr(self, 'rec_img_1'):
                img = add_title_to_img(self.rec_img_1, text="[REC] Left View", shape=(640, 480))
                self.env.viewer_rgb_overlay(img, loc='top left')
            
            # 3. 右下角: Right 右侧视角
            if hasattr(self, 'rec_img_2'):
                img = add_title_to_img(self.rec_img_2, text="[REC] Right View", shape=(640, 480))
                self.env.viewer_rgb_overlay(img, loc='bottom right')

        # =================== 🎥 主屏幕摄像头逻辑 🔥 ===================
        # 与 y_env4.py 的摄像头切换逻辑一致
        
        if self.env.viewer is not None:
            if self.vis_mode == 'base':
                # Base 模式：使用 tb3_chase 跟随摄像头
                self.env.viewer.cam.type = 2  # 固定摄像头模式
                try:
                    cam_id = self.env.model.camera('tb3_chase').id
                    self.env.viewer.cam.fixedcamid = cam_id
                except Exception:
                    # 回退：跟踪模式
                    self.env.viewer.cam.type = 1
                    try:
                        self.env.viewer.cam.trackbodyid = self.env.model.body('tb3_base').id
                    except:
                        pass
            else:
                # Arm 模式：使用自由视角（可鼠标拖动旋转）
                self.env.viewer.cam.type = 0  # 自由摄像头
                self.env.viewer.cam.trackbodyid = -1  # 不跟踪任何物体

        # =================== 显示时间信息 ===================
        mode_label = "🚗 BASE" if self.vis_mode == 'base' else "🦾 ARM"
        self.env.viewer_text_overlay(
            text1=f'{mode_label} | Ep {idx} | Rel Time: {rel_time:.2f}s', 
            text2=f'Total Sim: {self.env.data.time:.2f}s'
        )
        
        self.env.render()


class VisualizerEnv4(VisualizerRenderMixin, SimpleEnv4):
    def __init__(self, xml_path, mode='base', **kwargs):
        super().__init__(xml_path=xml_path, mode=mode, **kwargs)


class VisualizerEnv6(VisualizerRenderMixin, SimpleEnv6):
    def __init__(self, xml_path, mode='base', **kwargs):
        super().__init__(xml_path=xml_path, mode=mode, **kwargs)


def resolve_env_class(env_backend):
    if env_backend == 'y4':
        return VisualizerEnv4
    if env_backend == 'y6':
        return VisualizerEnv6
    raise ValueError(f"Unsupported env backend: {env_backend}")


def main():
    # 根据 MODE + DATASET_VERSION 获取配置
    mode_config = DATASET_CONFIG.get(MODE)
    if mode_config is None:
        print(f"Error: Unsupported MODE='{MODE}'. Available: {list(DATASET_CONFIG.keys())}")
        return

    if DATASET_VERSION not in mode_config:
        print(
            f"Error: Unsupported DATASET_VERSION='{DATASET_VERSION}' for MODE='{MODE}'. "
            f"Available: {list(mode_config.keys())}"
        )
        return

    config = mode_config[DATASET_VERSION]
    repo_name = config['repo_name']
    root = config['root']
    image_keys = config['image_keys']
    env_backend = config['env_backend']
    xml_path = config['xml_path']
    env_class = resolve_env_class(env_backend)
    
    print(f"=" * 50)
    print(f"📺 Visualize Dataset - Mode: {MODE.upper()} | Version: {DATASET_VERSION}")
    print(f"=" * 50)
    print(f"Loading dataset from {root}...")
    
    try:
        dataset = LeRobotDataset(repo_name, root=root)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset loaded: {dataset.num_episodes} episodes, {len(dataset)} frames")
    print(f"Image keys: {image_keys}")
    
    # 🔥 错误检查：LOOP_EPISODE 和 START_EPISODE 不能同时不为0
    if LOOP_EPISODE != 0 and START_EPISODE != 0:
        print(f"Error: LOOP_EPISODE={LOOP_EPISODE} and START_EPISODE={START_EPISODE} cannot both be non-zero!")
        print("Please set one of them to 0.")
        return
    
    print("\nInitializing Environment...")
    # 🔥 关键：action_type 必须是 'joint_angle'，因为数据集保存的是关节角度而非末端位姿
    # 🔥 添加随机初始化参数；环境后端和 XML 会随数据集版本自动切换
    env = env_class(
        xml_path,
        mode=MODE, 
        action_type='joint_angle', 
        state_type='joint_angle',
        random_init_enabled=0,  # 可视化时关闭随机初始化
        random_init_gripper_open=True
    )
    
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
    elif START_EPISODE != 0:
        # 验证episode索引是否有效
        if START_EPISODE < 0 or START_EPISODE >= dataset.num_episodes:
            print(f"Error: START_EPISODE={START_EPISODE} is out of range [0, {dataset.num_episodes-1}]")
            return
        episode_list = list(range(START_EPISODE, dataset.num_episodes))
        print(f"Start Mode: Will play from episode {START_EPISODE} to {dataset.num_episodes-1} ({len(episode_list)} episodes)")
    else:
        episode_list = list(range(dataset.num_episodes))
        print(f"Normal Mode: Will play all {dataset.num_episodes} episodes")
    
    # 外层循环：如果LOOP_EPISODE不为0，则无限循环播放
    while True:
        for ep_idx in episode_list:
            print(f"\n=== Playing Episode {ep_idx} ({MODE.upper()} mode) ===")
            
            from_idx = episode_data_index["from"][ep_idx].item()
            to_idx = episode_data_index["to"][ep_idx].item()
            frame_iterator = iter(range(from_idx, to_idx))
            
            # 重置环境
            env.reset(mode=MODE)
            
            # 🔥 读取本集第一帧的位姿来初始化小车（仅 base 模式有 base_pose）
            if MODE == 'base':
                try:
                    first_frame = dataset[from_idx]
                    if 'base_pose' in first_frame:
                        start_pose = first_frame['base_pose'].numpy()
                        start_x, start_y, start_theta = start_pose[0], start_pose[1], start_pose[2]
                        env.set_base_pose(start_x, start_y, start_theta)
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
            
            # 🔥 位置漂移统计变量（仅 base 模式使用）🔥
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
                    
                    # 执行动作
                    # 🔥 注意：使用的是 action 数据（7维：6个关节角度 + 1个夹爪状态），而不是 state 数据
                    # state 数据（6维）只包含末端位姿，不包含夹爪状态，无法完整复原机械臂动作
                    env.step(item['action'].numpy(), mode=MODE) 
                    
                    # 【base 模式】如果数据里有真实位置，强制把小车按回去
                    if MODE == 'base' and 'base_pose' in item:
                        recorded_pose = item['base_pose'].numpy()
                        x, y, theta = recorded_pose[0], recorded_pose[1], recorded_pose[2]
                        env.set_base_pose(x, y, theta)

                    # --- 图像更新 (根据模式选择不同的 key) ---
                    for i, key in enumerate(image_keys):
                        img_key = f'observation.images.{key}'
                        if img_key in item:
                            img = (item[img_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            setattr(env, f'rec_img_{i}', img)
                    
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
                    
                    # --- 4. 位置漂移检测（仅 base 模式）🔥 ---
                    pos_error = None
                    if MODE == 'base' and 'base_pose' in item:
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
            # 显示位置漂移统计信息（仅 base 模式）
            if MODE == 'base' and drift_count > 0:
                avg_drift = total_drift / drift_count
                print(f"  📊 Position Drift Stats: Max={max_drift*100:.2f} cm, Avg={avg_drift*100:.2f} cm, Samples={drift_count}")
            
            # 🔥 关键修复：如果viewer已关闭，退出内层for循环
            if not env.env.is_viewer_alive():
                print("\nViewer closed. Exiting...")
                break
        
        # 🔥 关键修复：检查viewer状态，如果已关闭则退出外层while循环
        if not env.env.is_viewer_alive():
            break
        
        # 如果LOOP_EPISODE为0，播放完所有episodes后退出（包括START_EPISODE模式）
        if LOOP_EPISODE == 0:
            break
    
    print("All episodes finished.")
    env.env.close_viewer()

if __name__ == "__main__":
    main()
