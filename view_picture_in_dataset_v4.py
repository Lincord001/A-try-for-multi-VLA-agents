import os
import cv2
import numpy as np
import argparse
import sys
import textwrap
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# ================= 配置区域 =================
# 支持 mode + version 选择，和 visualize_dataset.py 保持一致风格
DATASET_CONFIG = {
    "base": {
        "v4": {
            "repo_name": "demo_data_base_v4",
            "root": "./demo_data_base_v4",
            "image_keys": ["front", "left", "right"],
        },
        "v6": {
            "repo_name": "omy_base_data_v6",
            "root": "./demo_data_base_v6",
            "image_keys": ["front", "left", "right"],
        },
        "v7": {
            "repo_name": "omy_base_data_v7_5",
            "root": "./demo_data_base_v7_5",
            "image_keys": ["front", "left", "right"],
        },
    },
    "arm": {
        "v5": {
            "repo_name": "demo_data_arm_v5",
            "root": "./demo_data_arm_v5",
            "image_keys": ["agent", "wrist"],
        },
        "v5_3": {
            "repo_name": "omy_arm_data_v5_3",
            "root": "./demo_data_arm_v5_3",
            "image_keys": ["agent", "wrist"],
        },
        "v6": {
            "repo_name": "omy_arm_data_v6_1",
            "root": "./demo_data_arm_v6_1",
            "image_keys": ["agent", "wrist"],
        },
                "v7": {
            "repo_name": "omy_arm_data_v7_1",
            "root": "./demo_data_arm_v7_1",
            "image_keys": ["agent", "wrist"],
        },
    },
}


def resolve_dataset_path(mode, version, data_dir):
    """优先使用 data_dir；否则按 mode/version 从配置中解析路径。"""
    if data_dir:
        return data_dir, None

    mode_cfg = DATASET_CONFIG.get(mode)
    if mode_cfg is None:
        raise ValueError(f"Unsupported mode '{mode}'. Available: {list(DATASET_CONFIG.keys())}")

    ver_cfg = mode_cfg.get(version)
    if ver_cfg is None:
        raise ValueError(
            f"Unsupported version '{version}' for mode '{mode}'. Available: {list(mode_cfg.keys())}"
        )
    return ver_cfg["root"], ver_cfg["repo_name"]


def build_repo_candidates(data_dir, mode=None, version=None, repo_name=None):
    """构建多个候选 repo_name，逐个尝试加载以兼容不同历史数据集。"""
    candidates = []

    if repo_name:
        candidates.append(repo_name)

    base = os.path.basename(os.path.abspath(data_dir))
    if base:
        candidates.append(base)
        if base.startswith("demo_data_"):
            candidates.append(base.replace("demo_data_", "omy_", 1))

    if mode in DATASET_CONFIG and version in DATASET_CONFIG[mode]:
        preferred = DATASET_CONFIG[mode][version]["repo_name"]
        candidates.insert(0, preferred)

    # 去重并保持顺序
    uniq = []
    for name in candidates:
        if name and name not in uniq:
            uniq.append(name)
    return uniq

class DatasetViewer:
    def __init__(self, data_dir, mode="arm", version="v6", repo_name=None):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            print(f"❌ 错误: 找不到文件夹 '{data_dir}'")
            sys.exit(1)
        
        repo_candidates = build_repo_candidates(
            data_dir=data_dir, mode=mode, version=version, repo_name=repo_name
        )

        self.repo_name = None
        print(f"📁 数据路径: {data_dir}")
        print(f"🔍 尝试 repo_name 候选: {repo_candidates}")

        load_err = None
        self.dataset = None
        for candidate in repo_candidates:
            try:
                self.dataset = LeRobotDataset(candidate, root=data_dir)
                self.repo_name = candidate
                break
            except Exception as e:
                load_err = e

        if self.dataset is None:
            print(f"❌ 错误: 无法加载数据集。最后一次错误: {load_err}")
            print("   你可以手动指定 --repo_name 再试一次。")
            sys.exit(1)

        print(f"📂 加载数据集: {self.repo_name}")
        
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
            # 如果数据集为空，回退到配置里的默认相机
            mode_cfg = DATASET_CONFIG.get(mode, {})
            ver_cfg = mode_cfg.get(version, {})
            self.image_keys = ver_cfg.get("image_keys", ['agent', 'wrist'])
        
        print(f"✅ 数据集加载成功: {self.num_episodes} 个 Episodes, {len(self.dataset)} 帧")
        print(f"📷 检测到相机: {self.image_keys}")
        
        self.current_ep_idx = 0
        self.current_frame_idx = 0
        self.load_episode(0)

    @staticmethod
    def _to_display_text(value):
        """将数据项中的文本字段安全转换为可显示字符串。"""
        if value is None:
            return ""

        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return str(value.item())
            return " ".join(str(x) for x in value.tolist())

        if isinstance(value, (list, tuple)):
            return " ".join(str(x) for x in value)

        return str(value)

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

    def get_frame_item(self, frame_idx):
        """读取当前帧对应的数据项。"""
        if self.frame_count == 0:
            return None

        actual_idx = self.frame_start + frame_idx
        if actual_idx >= len(self.dataset):
            return None

        try:
            return self.dataset[actual_idx]
        except Exception as e:
            print(f"⚠️ 警告: 无法读取帧 {actual_idx}: {e}")
            return None

    def get_frame_instruction(self, frame_idx):
        """提取当前帧对应的任务/指令文本。"""
        item = self.get_frame_item(frame_idx)
        if item is None:
            return "N/A"

        candidate_keys = [
            "task",
            "instruction",
            "instructions",
            "language_instruction",
            "text",
            "prompt",
        ]
        for key in candidate_keys:
            if key in item:
                text = self._to_display_text(item[key]).strip()
                if text:
                    return text

        return "N/A"

    def get_frame_image(self, frame_idx):
        """读取并拼接当前帧的所有视角"""
        if self.frame_count == 0:
            return np.zeros((300, 300, 3), dtype=np.uint8)

        # 计算实际的数据集索引
        actual_idx = self.frame_start + frame_idx
        if actual_idx >= len(self.dataset):
            return np.zeros((300, 300, 3), dtype=np.uint8)

        item = self.get_frame_item(frame_idx)
        if item is None:
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
        print("  输入框: 输入数字跳转 (例如: '5' 跳转到 Episode 5, '2:10' 跳转到 Episode 2 的第 10 帧)")
        print("="*50 + "\n")

        # 设置 matplotlib 交互模式
        plt.ion()
        fig = plt.figure(figsize=(12, 9))
        fig.canvas.manager.set_window_title('Dataset Viewer')
        fig.subplots_adjust(top=0.86)
        
        # 创建主图像区域（为输入框留出空间）
        ax = plt.subplot2grid((10, 1), (0, 0), rowspan=9)
        
        # 初始化显示
        image = self.get_frame_image(self.current_frame_idx)
        im = ax.imshow(image)
        ax.axis('off')
        instruction_text = fig.text(
            0.5,
            0.97,
            "",
            ha='center',
            va='top',
            fontsize=16,
        )
        
        # 创建输入框区域
        axbox = plt.subplot2grid((10, 1), (9, 0))
        axbox.axis('off')  # 隐藏输入框区域的坐标轴
        text_box = TextBox(axbox, '跳转到: ', initial='', textalignment='left')
        
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
            instruction = self.get_frame_instruction(self.current_frame_idx)
            instruction_text.set_text(textwrap.fill(f"Instruction: {instruction}", width=100))
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
        
        def on_text_submit(text):
            """处理输入框提交"""
            text = text.strip()
            if not text:
                return
            
            try:
                # 检查是否是 "ep:frame" 格式
                if ':' in text:
                    parts = text.split(':')
                    if len(parts) == 2:
                        ep_idx = int(parts[0].strip())
                        frame_idx = int(parts[1].strip())
                        
                        # 验证 episode 索引
                        if ep_idx < 0 or ep_idx >= self.num_episodes:
                            print(f"⚠️ 警告: Episode 索引 {ep_idx} 超出范围 [0, {self.num_episodes-1}]")
                            return
                        
                        # 加载指定的 episode
                        self.load_episode(ep_idx)
                        
                        # 验证 frame 索引
                        if frame_idx < 0 or frame_idx >= self.frame_count:
                            print(f"⚠️ 警告: Frame 索引 {frame_idx} 超出范围 [0, {self.frame_count-1}]")
                            self.current_frame_idx = 0
                        else:
                            self.current_frame_idx = frame_idx
                        
                        update_display()
                        print(f"✅ 跳转到 Episode {ep_idx}, Frame {frame_idx}")
                        text_box.set_val('')  # 清空输入框
                        return
                
                # 否则，当作 episode 索引处理
                ep_idx = int(text)
                
                if ep_idx < 0 or ep_idx >= self.num_episodes:
                    print(f"⚠️ 警告: Episode 索引 {ep_idx} 超出范围 [0, {self.num_episodes-1}]")
                    text_box.set_val('')  # 清空输入框
                    return
                
                # 跳转到指定的 episode，frame 设为 0
                self.load_episode(ep_idx)
                self.current_frame_idx = 0
                update_display()
                print(f"✅ 跳转到 Episode {ep_idx}")
                text_box.set_val('')  # 清空输入框
                
            except ValueError:
                print(f"⚠️ 警告: 无效的输入格式。请输入数字 (例如: '5') 或 'ep:frame' 格式 (例如: '2:10')")
                text_box.set_val('')  # 清空输入框
        
        # 绑定键盘事件
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # 绑定输入框提交事件
        text_box.on_submit(on_text_submit)
        
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
    parser.add_argument(
        "--mode",
        type=str,
        default="arm",
        choices=list(DATASET_CONFIG.keys()),
        help="数据模式: arm 或 base (默认: arm)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v7",
        help="数据集版本 (例如: arm 的 v5/v5_3/v6，base 的 v4/v6)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据集根目录路径；若不填则按 mode/version 自动解析",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help="可选：手动指定 LeRobotDataset 的 repo_name（加载失败时可用）",
    )
    args = parser.parse_args()

    try:
        data_dir, resolved_repo_name = resolve_dataset_path(args.mode, args.version, args.data_dir)
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)

    manual_repo = args.repo_name if args.repo_name else resolved_repo_name
    viewer = DatasetViewer(
        data_dir=data_dir,
        mode=args.mode,
        version=args.version,
        repo_name=manual_repo,
    )
    viewer.run()
