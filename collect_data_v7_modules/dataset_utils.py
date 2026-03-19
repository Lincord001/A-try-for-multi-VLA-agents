"""
dataset_utils.py
----------------
数据集的加载与创建工具函数。
支持：新建、追加、空壳检测、损坏兜底。
"""

import os
import json
import shutil
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from .config import IMG_SIZE, FPS, DATASET_CONFIG, MODE_CONFIG


# ===========================================

def _dataset_data_file_count(root):
    """统计数据集根目录下 data/videos 的文件数（用于判断是否为空壳目录）。"""
    data_count = 0
    video_count = 0
    data_dir = os.path.join(root, "data")
    videos_dir = os.path.join(root, "videos")

    if os.path.isdir(data_dir):
        for _, _, files in os.walk(data_dir):
            data_count += len(files)
    if os.path.isdir(videos_dir):
        for _, _, files in os.walk(videos_dir):
            video_count += len(files)

    return data_count, video_count


def _is_empty_shell_dataset(root):
    """
    判断是否是"空壳数据集目录"：
    - 目录存在但为空
    - 或仅有 meta/info.json，且 total_episodes=0 且没有 data/videos 文件
    """
    if not os.path.isdir(root):
        return False

    entries = [e for e in os.listdir(root) if not e.startswith(".")]
    if len(entries) == 0:
        return True

    info_path = os.path.join(root, "meta", "info.json")
    if not os.path.isfile(info_path):
        return False

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        # meta 都读不出来，视为异常目录，交给后续加载失败兜底处理
        return False

    total_episodes = int(info.get("total_episodes", 0))
    data_count, video_count = _dataset_data_file_count(root)
    return total_episodes == 0 and data_count == 0 and video_count == 0


def _create_dataset(mode, root, repo_name, mode_cfg):
    """创建新数据集（从 load_or_create_dataset 中抽出来，避免重复代码）。"""
    print(f"\n[{mode.upper()}] Creating NEW dataset at: {root}")

    features = {
        "observation.state": {"dtype": "float32", "shape": mode_cfg['state_shape'], "names": ["state"]},
        "action": {"dtype": "float32", "shape": mode_cfg['action_shape'], "names": ["action"]},
        "obj_init": {"dtype": "float32", "shape": (9,), "names": ["obj_init"]},  # 🔥 红色杯子(3) + 蓝色杯子(3) + 盘子(3, 已移出场景)
    }

    if mode == 'arm':
        features["observation.images.agent"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
        features["observation.images.wrist"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
    elif mode == 'base':
        for cam in ['front', 'left', 'right']:
            features[f"observation.images.{cam}"] = {"dtype": "image", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
        features["base_pose"] = {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "theta"]
        }

    return LeRobotDataset.create(
        repo_id=repo_name,
        root=root,
        robot_type="omy",
        fps=FPS,
        features=features,
        image_writer_threads=20,  # 🔥 增加线程数
        image_writer_processes=8, # 🔥 增加进程数
    )


def load_or_create_dataset(mode):
    """加载或创建指定模式的数据集"""
    config = DATASET_CONFIG[mode]
    mode_cfg = MODE_CONFIG[mode]
    root = config['root']
    repo_name = config['repo_name']

    if os.path.exists(root) and _is_empty_shell_dataset(root):
        print(f"\n[{mode.upper()}] Detected empty shell dataset at: {root}")
        print(">> Removing empty folder and recreating clean dataset...")
        shutil.rmtree(root, ignore_errors=True)

    if not os.path.exists(root):
        return _create_dataset(mode, root, repo_name, mode_cfg)

    print(f"\n[{mode.upper()}] Dataset found at: {root}")
    print(">> Loading existing dataset (Append Mode)...")
    try:
        dataset = LeRobotDataset(repo_name, root=root)
        print(f">> Found {dataset.num_episodes} existing episodes.")
        return dataset
    except Exception as e:
        # 目录存在但结构异常时，避免下次启动继续卡在同一目录
        backup_root = f"{root}_corrupted_{int(time.time())}"
        print(f">> ⚠️ Failed to load dataset: {e}")
        print(f">> Moving broken dataset to: {backup_root}")
        shutil.move(root, backup_root)
        return _create_dataset(mode, root, repo_name, mode_cfg)
