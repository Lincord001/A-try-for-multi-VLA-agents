import os
import cv2
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def natural_sort_key(s):
    """用于文件名的自然排序 (让 2.jpg 排在 10.jpg 前面)"""
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class DatasetViewer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            print(f"❌ 错误: 找不到文件夹 '{data_dir}'")
            sys.exit(1)

        # 从数据目录提取 repo_name (例如: demo_data_arm_v4 -> demo_data_arm_v4)
        self.repo_name = os.path.basename(os.path.abspath(data_dir))
        
        print(f"📂 加载数据集: {self.repo_name}")
        print(f"📁 数据路径: {data_dir}")
        
        try:
            self.dataset = LeRobotDataset(self.repo_name, root=data_dir)
        except Exception as e:
            print(f"❌ 错误: 无法加载数据集: {e}")
            sys.exit(1)
        
        self.num_episodes = self.dataset.num_episodes
        self.episode_data_index = self.dataset.episode_data_index
        
        # 自动检测可用的图像键
        # 尝试从第一个样本中检测
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            self.image_keys = []
            for key in sample.keys():
                if key.startswith('observation.images.'):
                    cam_name = key.replace('observation.images.', '')
                    self.image_keys.append(cam_name)
            self.image_keys.sort()  # 排序以保持一致性
        else:
            # 如果数据集为空，使用默认值（匹配 collect_data_v4.py 的 arm 模式）
            self.image_keys = ['agent', 'wrist']
        
        print(f"✅ 数据集加载成功: {self.num_episodes} 个 Episodes, {len(self.dataset)} 帧")
        print(f"📷 检测到相机: {self.image_keys}")
        
        self.current_ep_idx = 0
        self.current_frame_idx = 0
        self.load_episode(0)

    def load_episode(self, ep_idx):
        """加载指定 Episode 的帧范围信息"""
        if ep_idx < 0 or ep_idx >= self.num_episodes:
            return
        
        self.current_ep_idx = ep_idx
        
        # 获取该 episode 的帧范围
        from_idx = self.episode_data_index["from"][ep_idx].item()
        to_idx = self.episode_data_index["to"][ep_idx].item()
        
        self.frame_start = from_idx
        self.frame_end = to_idx
        self.frame_count = to_idx - from_idx
        
        print(f"✅ 加载 Episode {ep_idx}: 帧范围 [{from_idx}, {to_idx}), 共 {self.frame_count} 帧")
        
        self.current_frame_idx = 0

    def get_frame_image(self, frame_idx):
        """读取并拼接当前帧的所有视角"""
        if self.frame_count == 0:
            return np.zeros((300, 300, 3), dtype=np.uint8)

        # 计算实际的数据集索引
        actual_idx = self.frame_start + frame_idx
        if actual_idx >= len(self.dataset):
            return np.zeros((300, 300, 3), dtype=np.uint8)

        try:
            item = self.dataset[actual_idx]
        except Exception as e:
            print(f"⚠️ 警告: 无法读取帧 {actual_idx}: {e}")
            return np.zeros((300, 300, 3), dtype=np.uint8)

        imgs = []
        # 遍历所有相机
        for cam_name in self.image_keys:
            img_key = f'observation.images.{cam_name}'
            if img_key in item:
                # 从 tensor 转换为 numpy array
                img_tensor = item[img_key]
                if hasattr(img_tensor, 'numpy'):
                    img = img_tensor.numpy()
                elif hasattr(img_tensor, 'cpu'):
                    img = img_tensor.cpu().numpy()
                else:
                    img = np.array(img_tensor)
                
                # 处理不同的 tensor 格式
                # 可能是 (C, H, W) 或 (H, W, C)
                if len(img.shape) == 3:
                    if img.shape[0] == 3 or img.shape[0] == 1:
                        # (C, H, W) -> (H, W, C)
                        img = img.transpose(1, 2, 0)
                    # 如果是 (H, W, C) 但 C=1，需要扩展
                    if img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=2)
                
                # 确保是 uint8 格式，值域在 [0, 255]
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # 确保是 BGR 格式 (cv2.putText 需要 BGR)
                if img.shape[2] == 3:
                    # 假设输入是 RGB，转换为 BGR 用于 cv2 操作
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                # 如果该相机缺帧
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.putText(img, "Missing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # 在图片上写上相机名字 (使用 BGR 格式)
            cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            imgs.append(img)
        
        # 水平拼接所有相机视图
        if imgs:
            # 统一高度
            h_min = min(im.shape[0] for im in imgs)
            imgs_resized = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min)) for im in imgs]
            combined = np.hstack(imgs_resized)
        else:
            combined = np.zeros((300, 300, 3), dtype=np.uint8)
            
        # 叠加全局信息 (使用 BGR 格式)
        info_text = f"Ep: {self.current_ep_idx} ({self.current_ep_idx+1}/{self.num_episodes}) | Frame: {frame_idx}/{self.frame_count-1} | Total: {actual_idx}"
        cv2.putText(combined, info_text, (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 转换为 RGB 格式供 matplotlib 使用
        if combined.shape[2] == 3:
            combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        return combined

    def run(self):
        print("\n" + "="*50)
        print("🎮 操作说明:")
        print("  [D] 或 [→] : 下一帧")
        print("  [A] 或 [←] : 上一帧")
        print("  [W] 或 [↑] : 下一个 Episode")
        print("  [S] 或 [↓] : 上一个 Episode")
        print("  [Q] 或 [ESC]: 退出")
        print("="*50 + "\n")

        # 设置 matplotlib 交互模式
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title('Dataset Viewer')
        
        # 初始化显示
        image = self.get_frame_image(self.current_frame_idx)
        im = ax.imshow(image)
        ax.axis('off')
        plt.tight_layout()
        
        def update_display():
            """更新显示的图像"""
            image = self.get_frame_image(self.current_frame_idx)
            im.set_data(image)
            # 更新标题显示当前信息
            actual_idx = self.frame_start + self.current_frame_idx
            ax.set_title(f"Episode: {self.current_ep_idx} ({self.current_ep_idx+1}/{self.num_episodes}) | "
                        f"Frame: {self.current_frame_idx}/{self.frame_count-1} | "
                        f"Total Index: {actual_idx}", 
                        fontsize=10)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        
        def on_key_press(event):
            """处理键盘事件"""
            if event.key is None:
                return
            
            key = event.key.lower()
            
            # 退出
            if key == 'q' or key == 'escape':
                plt.close('all')
                return
            
            # 下一帧 (D / Right)
            elif key == 'd' or key == 'right':
                if self.current_frame_idx < self.frame_count - 1:
                    self.current_frame_idx += 1
                    update_display()
            
            # 上一帧 (A / Left)
            elif key == 'a' or key == 'left':
                if self.current_frame_idx > 0:
                    self.current_frame_idx -= 1
                    update_display()
            
            # 下一个 Episode (W / Up)
            elif key == 'w' or key == 'up':
                if self.current_ep_idx < self.num_episodes - 1:
                    self.load_episode(self.current_ep_idx + 1)
                    update_display()
            
            # 上一个 Episode (S / Down)
            elif key == 's' or key == 'down':
                if self.current_ep_idx > 0:
                    self.load_episode(self.current_ep_idx - 1)
                    update_display()
        
        # 绑定键盘事件
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # 初始显示
        update_display()
        
        # 保持窗口打开
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            pass
        finally:
            plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看采集到的机器人数据集 (LeRobot 格式)")
    parser.add_argument('--data_dir', type=str, default='./demo_data_arm_v4', 
                        help="数据集根目录路径 (默认: ./demo_data_arm_v4)")
    args = parser.parse_args()
    
    viewer = DatasetViewer(args.data_dir)
    viewer.run()
