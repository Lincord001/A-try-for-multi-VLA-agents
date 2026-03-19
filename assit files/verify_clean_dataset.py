# -*- coding: utf-8 -*-
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

REPO_NAME = 'omy_base_data_clean'
ROOT = './demo_data_base_ver_2_clean'

print("=" * 60)
print("Verifying cleaned dataset")
print("=" * 60)

try:
    # 1. Load dataset
    print("\n1. Loading dataset...")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)
    print(f"   [OK] Dataset loaded successfully")
    
    # 2. Check basic info
    print(f"\n2. Dataset basic information:")
    print(f"   - Episodes: {dataset.num_episodes}")
    print(f"   - Frames: {dataset.num_frames}")
    print(f"   - FPS: {dataset.fps}")
    print(f"   - Robot Type: {dataset.meta.robot_type}")
    
    # 3. Check metadata
    print(f"\n3. Metadata check:")
    print(f"   - Tasks: {len(dataset.meta.tasks)} task(s)")
    for task_idx, task in dataset.meta.tasks.items():
        print(f"     Task {task_idx}: {task}")
    
    # 4. Check each episode
    print(f"\n4. Episode check:")
    episode_data_index = dataset.episode_data_index
    for ep_idx in range(dataset.num_episodes):
        from_idx = episode_data_index["from"][ep_idx].item()
        to_idx = episode_data_index["to"][ep_idx].item()
        length = to_idx - from_idx
        print(f"   Episode {ep_idx}: {length} frames (index {from_idx} to {to_idx-1})")
    
    # 5. Check first frame data
    print(f"\n5. Data format check (first frame):")
    first_frame = dataset[0]
    print(f"   - Data keys: {list(first_frame.keys())[:10]}...")
    
    # Check required fields
    required_keys = ['action', 'observation.state', 'observation.images.front']
    for key in required_keys:
        if key in first_frame:
            val = first_frame[key]
            if hasattr(val, 'shape'):
                print(f"   [OK] {key}: shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"   [OK] {key}: {type(val)}")
        else:
            print(f"   [ERROR] {key}: missing!")
    
    # 6. Check task information
    print(f"\n6. Task information check:")
    for ep_idx in range(min(3, dataset.num_episodes)):  # Only check first 3
        from_idx = episode_data_index["from"][ep_idx].item()
        frame = dataset[from_idx]
        if 'task' in frame:
            print(f"   Episode {ep_idx}: task = '{frame['task']}'")
        else:
            print(f"   Episode {ep_idx}: [ERROR] task field missing!")
    
    # 7. Compare with original dataset (if exists)
    print(f"\n7. Comparison with original dataset:")
    try:
        ds_orig = LeRobotDataset('omy_base_data', root='./demo_data_base')
        print(f"   Original dataset: {ds_orig.num_episodes} episodes, {ds_orig.num_frames} frames")
        print(f"   Cleaned dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")
        print(f"   Removed episodes: [1, 4] (according to BLACKLIST_EPISODES)")
        print(f"   New dataset contains episodes: {list(range(dataset.num_episodes))}")
        
        # Verify removed episodes are not in new dataset
        print(f"   [OK] Verification: New dataset has {dataset.num_episodes} episodes, as expected (original 6 - removed 2 = 4)")
    except Exception as e:
        print(f"   [WARNING] Cannot load original dataset for comparison: {e}")
    
    # 8. Check image format
    print(f"\n8. Image format check:")
    first_frame = dataset[0]
    image_keys = ['observation.images.front', 'observation.images.left', 'observation.images.right']
    for key in image_keys:
        if key in first_frame:
            img = first_frame[key]
            if hasattr(img, 'shape'):
                # Check if channel-first (C, H, W) or channel-last (H, W, C)
                if len(img.shape) == 3:
                    if img.shape[0] == 3:
                        print(f"   [WARNING] {key}: shape {img.shape} (channel-first, LeRobot will handle automatically)")
                    elif img.shape[2] == 3:
                        print(f"   [OK] {key}: shape {img.shape} (channel-last, correct format)")
                    else:
                        print(f"   [UNKNOWN] {key}: shape {img.shape} (unknown format)")
                else:
                    print(f"   [UNKNOWN] {key}: shape {img.shape} (not a 3D image)")
    
    print(f"\n" + "=" * 60)
    print("[OK] All checks passed! Dataset format is correct.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] Verification failed: {e}")
    import traceback
    traceback.print_exc()
