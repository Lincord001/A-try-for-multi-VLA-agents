#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练数据中杯子位置分布

功能：
1. 读取所有 episode 的 obj_init 数据
2. 仅统计指令要求抓取的杯子位置（红/蓝分别统计）
3. 可视化分布情况（包括对比图）
4. 统计抓取中心（Y + offset）的分布
5. 统计并可视化小车初始位姿（base_pose）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 配置
DATASET_ROOT = './demo_data_arm_v5_2'
EXPERT_Y_GRASP_OFFSET = 0.067  # 从 y_env4.py 中获取

def load_episode_tasks(meta_path):
    """读取 episodes.jsonl 中的任务信息，返回 {episode_index: task}"""
    tasks_map = {}
    if not meta_path.exists():
        print(f"⚠️  Warning: meta file not found: {meta_path}")
        return tasks_map
    try:
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                episode_index = data.get('episode_index')
                tasks = data.get('tasks', [])
                if episode_index is not None and len(tasks) > 0:
                    tasks_map[int(episode_index)] = tasks[0]
    except Exception as e:
        print(f"Error loading tasks from {meta_path}: {e}")
    return tasks_map


def load_episode_data(episode_path):
    """加载单个 episode 的数据"""
    try:
        df = pd.read_parquet(episode_path)
        # obj_init 在第一帧应该是一样的（所有帧的 obj_init 都相同）
        if len(df) > 0:
            obj_init = df['obj_init'].iloc[0]
            task = None
            base_pose = None
            if 'task' in df.columns:
                task = df['task'].iloc[0]
            elif 'instruction' in df.columns:
                task = df['instruction'].iloc[0]
            if 'base_pose' in df.columns:
                base_pose = df['base_pose'].iloc[0]
            return obj_init, task, base_pose
        return None
    except Exception as e:
        print(f"Error loading {episode_path}: {e}")
        return None

def analyze_cup_positions():
    """分析杯子位置分布"""
    print("="*70)
    print("📊 Analyzing Cup Position Distribution in Training Data")
    print("="*70)
    
    # 1. 找到所有 episode 文件
    data_dir = Path(DATASET_ROOT) / 'data' / 'chunk-000'
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        return
    
    episode_files = sorted(data_dir.glob('episode_*.parquet'))
    total_episodes = len(episode_files)
    print(f"\n📁 Found {total_episodes} episodes in {data_dir}")
    
    if total_episodes == 0:
        print("❌ No episode files found!")
        return
    
    # 2. 读取所有 episode 的 obj_init（仅统计任务要求抓取的杯子）
    print("\n📖 Loading episode data...")
    # 读取任务信息（meta/episodes.jsonl）
    meta_path = Path(DATASET_ROOT) / 'meta' / 'episodes.jsonl'
    tasks_map = load_episode_tasks(meta_path)
    if len(tasks_map) == 0:
        print("⚠️  Warning: No tasks found in meta. Will not be able to filter by target cup.")

    cup_positions = {
        'red': {'x': [], 'y': [], 'z': []},
        'blue': {'x': [], 'y': [], 'z': []},
        'yellow': {'x': [], 'y': [], 'z': []},
        'green': {'x': [], 'y': [], 'z': []},
    }
    base_positions = {'x': [], 'y': [], 'yaw': []}
    grasp_center_y = []  # 抓取中心的 Y 坐标（杯子 Y + offset）
    
    failed_count = 0
    red_target_count = 0
    blue_target_count = 0
    unknown_target_count = 0
    for i, ep_file in enumerate(episode_files):
        if (i + 1) % 50 == 0:
            print(f"   Processing {i+1}/{total_episodes}...", end='\r')
        
        result = load_episode_data(ep_file)
        if result is None:
            failed_count += 1
            continue
        obj_init, task_from_data, base_pose = result

        # base_pose 格式通常为 [x, y, yaw]
        if base_pose is not None and len(base_pose) >= 3:
            base_positions['x'].append(base_pose[0])
            base_positions['y'].append(base_pose[1])
            base_positions['yaw'].append(base_pose[2])

        # 从文件名解析 episode_index，例如 episode_000123.parquet
        try:
            episode_index = int(ep_file.stem.split('_')[-1])
        except Exception:
            episode_index = None

        task = tasks_map.get(episode_index)
        if task is None:
            task = task_from_data

        # obj_init 格式: 9维 = 红色杯子(3) + 蓝色杯子(3) + 盘子(3)
        # [mug5_x, mug5_y, mug5_z, mug6_x, mug6_y, mug6_z, plate_x, plate_y, plate_z]

        # 根据任务指令判断目标杯子
        target_cup = None
        if task is not None:
            task_lower = str(task).lower()
            if 'red' in task_lower or 'mug_5' in task_lower:
                target_cup = 'red'
            elif 'blue' in task_lower or 'mug_6' in task_lower:
                target_cup = 'blue'

        if target_cup == 'red' and len(obj_init) >= 3:
            cup_positions['red']['x'].append(obj_init[0])
            cup_positions['red']['y'].append(obj_init[1])
            cup_positions['red']['z'].append(obj_init[2])
            grasp_center_y.append(obj_init[1] + EXPERT_Y_GRASP_OFFSET)
            red_target_count += 1
        elif target_cup == 'blue' and len(obj_init) >= 6:
            cup_positions['blue']['x'].append(obj_init[3])
            cup_positions['blue']['y'].append(obj_init[4])
            cup_positions['blue']['z'].append(obj_init[5])
            grasp_center_y.append(obj_init[4] + EXPERT_Y_GRASP_OFFSET)
            blue_target_count += 1
        else:
            unknown_target_count += 1
            continue

        # 注意：黄色和绿色杯子在当前数据集中不存在（obj_init只有9维）
    
    print(f"\n✅ Loaded {total_episodes - failed_count} episodes successfully")
    if failed_count > 0:
        print(f"⚠️  Failed to load {failed_count} episodes")
    
    # 检查数据加载情况
    red_count = len(cup_positions['red']['x'])
    blue_count = len(cup_positions['blue']['x'])
    total_target_count = red_count + blue_count
    base_count = len(base_positions['x'])
    print(f"\n📊 Data Summary (Target Cups Only):")
    print(f"   🔴 Red cup target count: {red_count}")
    print(f"   🔵 Blue cup target count: {blue_count}")
    print(f"   📈 Total target cups: {total_target_count}")
    print(f"   🚗 Base pose count: {base_count}")
    if red_count != 250 or blue_count != 250:
        print(f"   ⚠️  Expected ~250 per cup. Current: red={red_count}, blue={blue_count}")
    if unknown_target_count > 0:
        print(f"   ⚠️  Skipped {unknown_target_count} episodes (unknown target)")
    if red_count == 0 or blue_count == 0:
        print(f"   ⚠️  Warning: Missing target cup data. Check task format.")
    
    # 3. 统计分析
    print("\n" + "="*70)
    print("📈 Statistical Analysis")
    print("="*70)
    
    # 红色杯子统计
    red_y = np.array(cup_positions['red']['y']) if red_count > 0 else np.array([])
    red_x = np.array(cup_positions['red']['x']) if red_count > 0 else np.array([])
    red_z = np.array(cup_positions['red']['z']) if red_count > 0 else np.array([])
    grasp_y = np.array(grasp_center_y) if len(grasp_center_y) > 0 else np.array([])
    
    # 蓝色杯子统计
    blue_y = np.array(cup_positions['blue']['y']) if blue_count > 0 else np.array([])
    blue_x = np.array(cup_positions['blue']['x']) if blue_count > 0 else np.array([])
    blue_z = np.array(cup_positions['blue']['z']) if blue_count > 0 else np.array([])
    base_x = np.array(base_positions['x']) if base_count > 0 else np.array([])
    base_y = np.array(base_positions['y']) if base_count > 0 else np.array([])
    base_yaw = np.array(base_positions['yaw']) if base_count > 0 else np.array([])
    
    if red_count > 0:
        print(f"\n🔴 Red Cup (mug_5) Position Statistics (Target Only):")
        print(f"   X: mean={red_x.mean():.4f}, std={red_x.std():.4f}, range=[{red_x.min():.4f}, {red_x.max():.4f}]")
        print(f"   Y: mean={red_y.mean():.4f}, std={red_y.std():.4f}, range=[{red_y.min():.4f}, {red_y.max():.4f}]")
        print(f"   Z: mean={red_z.mean():.4f}, std={red_z.std():.4f}, range=[{red_z.min():.4f}, {red_z.max():.4f}]")
    
    if blue_count > 0:
        print(f"\n🔵 Blue Cup (mug_6) Position Statistics (Target Only):")
        print(f"   X: mean={blue_x.mean():.4f}, std={blue_x.std():.4f}, range=[{blue_x.min():.4f}, {blue_x.max():.4f}]")
        print(f"   Y: mean={blue_y.mean():.4f}, std={blue_y.std():.4f}, range=[{blue_y.min():.4f}, {blue_y.max():.4f}]")
        print(f"   Z: mean={blue_z.mean():.4f}, std={blue_z.std():.4f}, range=[{blue_z.min():.4f}, {blue_z.max():.4f}]")
    
    if len(grasp_y) > 0:
        print(f"\n🎯 Grasp Center Y (Target Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m):")
        print(f"   mean={grasp_y.mean():.4f}, std={grasp_y.std():.4f}, range=[{grasp_y.min():.4f}, {grasp_y.max():.4f}]")

    if base_count > 0:
        print(f"\n🚗 Base Initial Pose Statistics:")
        print(f"   X: mean={base_x.mean():.4f}, std={base_x.std():.4f}, range=[{base_x.min():.4f}, {base_x.max():.4f}]")
        print(f"   Y: mean={base_y.mean():.4f}, std={base_y.std():.4f}, range=[{base_y.min():.4f}, {base_y.max():.4f}]")
        print(f"   Yaw: mean={base_yaw.mean():.4f}, std={base_yaw.std():.4f}, range=[{base_yaw.min():.4f}, {base_yaw.max():.4f}]")
    
    # 检查分布是否均匀
    # 计算偏度（skewness）
    from scipy import stats
    y_skew = None
    if red_count > 0:
        print(f"\n📊 Distribution Analysis (Red Cup):")
        y_bins = np.linspace(red_y.min(), red_y.max(), 11)
        hist, _ = np.histogram(red_y, bins=y_bins)
        print(f"   Y coordinate bins: {len(y_bins)-1} bins")
        print(f"   Bin counts: {hist}")
        print(f"   Max bin count: {hist.max()}, Min bin count: {hist.min()}")
        y_skew = stats.skew(red_y)
        print(f"   Y coordinate skewness: {y_skew:.4f} ({'偏右' if y_skew > 0 else '偏左' if y_skew < 0 else '对称'})")
    
    # 初始化蓝色杯子偏度变量
    blue_y_skew = None
    if blue_count > 0:
        print(f"\n📊 Distribution Analysis (Blue Cup):")
        blue_y_bins = np.linspace(blue_y.min(), blue_y.max(), 11)
        blue_hist, _ = np.histogram(blue_y, bins=blue_y_bins)
        print(f"   Y coordinate bins: {len(blue_y_bins)-1} bins")
        print(f"   Bin counts: {blue_hist}")
        print(f"   Max bin count: {blue_hist.max()}, Min bin count: {blue_hist.min()}")
        blue_y_skew = stats.skew(blue_y)
        print(f"   Y coordinate skewness: {blue_y_skew:.4f} ({'偏右' if blue_y_skew > 0 else '偏左' if blue_y_skew < 0 else '对称'})")
    
    # 4. 可视化
    print("\n" + "="*70)
    print("📊 Generating Visualizations...")
    print("="*70)
    
    # 判断是否有红/蓝杯子数据
    has_red_data = red_count > 0
    has_blue_data = blue_count > 0
    
    if has_red_data and has_blue_data:
        # 红蓝都有数据，使用3行3列布局
        fig = plt.figure(figsize=(18, 14))
        
        # ====== 第一行：红色杯子 ======
        # 子图 1: 红色杯子 X 坐标直方图
        ax1 = plt.subplot(3, 3, 1)
        plt.hist(red_x, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_x.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_x.mean():.3f}')
        plt.axvline(np.median(red_x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_x):.3f}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Count')
        plt.title('🔴 Red Cup X Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 2: 红色杯子 Y 坐标直方图
        ax2 = plt.subplot(3, 3, 2)
        plt.hist(red_y, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_y.mean():.3f}')
        plt.axvline(np.median(red_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_y):.3f}')
        plt.xlabel('Y Position (m)')
        plt.ylabel('Count')
        plt.title('🔴 Red Cup Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 3: 红色杯子 Z 坐标直方图
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(red_z, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_z.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_z.mean():.3f}')
        plt.axvline(np.median(red_z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_z):.3f}')
        plt.xlabel('Z Position (m)')
        plt.ylabel('Count')
        plt.title('🔴 Red Cup Z Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ====== 第二行：蓝色杯子 ======
        # 子图 4: 蓝色杯子 X 坐标直方图
        ax4 = plt.subplot(3, 3, 4)
        plt.hist(blue_x, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_x.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_x.mean():.3f}')
        plt.axvline(np.median(blue_x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_x):.3f}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Count')
        plt.title('🔵 Blue Cup X Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 5: 蓝色杯子 Y 坐标直方图
        ax5 = plt.subplot(3, 3, 5)
        plt.hist(blue_y, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_y.mean():.3f}')
        plt.axvline(np.median(blue_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_y):.3f}')
        plt.xlabel('Y Position (m)')
        plt.ylabel('Count')
        plt.title('🔵 Blue Cup Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 6: 蓝色杯子 Z 坐标直方图
        ax6 = plt.subplot(3, 3, 6)
        plt.hist(blue_z, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_z.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_z.mean():.3f}')
        plt.axvline(np.median(blue_z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_z):.3f}')
        plt.xlabel('Z Position (m)')
        plt.ylabel('Count')
        plt.title('🔵 Blue Cup Z Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ====== 第三行：对比图 ======
        # 子图 7: X-Y 散点图（红色和蓝色对比）
        ax7 = plt.subplot(3, 3, 7)
        plt.scatter(red_x, red_y, alpha=0.5, s=20, color='red', label='Red Cup')
        plt.scatter(blue_x, blue_y, alpha=0.5, s=20, color='blue', label='Blue Cup')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('X-Y Position Scatter (Red vs Blue)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 子图 8: 极坐标分布（红色和蓝色对比）
        ax8 = plt.subplot(3, 3, 8, projection='polar')
        ARM_BASE_X = 0.0  # 从 y_env5_2.py 中获取
        ARM_BASE_Y = 0.0
        # 红色杯子极坐标
        dx_red = red_x - ARM_BASE_X
        dy_red = red_y - ARM_BASE_Y
        distances_red = np.sqrt(dx_red**2 + dy_red**2)
        angles_red = np.arctan2(dy_red, dx_red)
        # 蓝色杯子极坐标
        dx_blue = blue_x - ARM_BASE_X
        dy_blue = blue_y - ARM_BASE_Y
        distances_blue = np.sqrt(dx_blue**2 + dy_blue**2)
        angles_blue = np.arctan2(dy_blue, dx_blue)
        plt.scatter(angles_red, distances_red, alpha=0.5, s=20, color='red', label='Red')
        plt.scatter(angles_blue, distances_blue, alpha=0.5, s=20, color='blue', label='Blue')
        plt.xlabel('Angle (rad)')
        plt.title('Polar Position (relative to arm base)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        # 子图 9: 抓取中心 Y 坐标直方图（目标杯子）
        ax9 = plt.subplot(3, 3, 9)
        if len(grasp_y) > 0:
            plt.hist(grasp_y, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(grasp_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {grasp_y.mean():.3f}')
            plt.axvline(np.median(grasp_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(grasp_y):.3f}')
            plt.xlabel(f'Grasp Center Y (Target Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m)')
            plt.ylabel('Count')
            plt.title('Grasp Center Y Position Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
    elif has_red_data:
        # 只有红色杯子数据，使用2行3列布局
        fig = plt.figure(figsize=(16, 10))
        
        # 子图 1: Y 坐标直方图
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(red_y, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_y.mean():.3f}')
        plt.axvline(np.median(red_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_y):.3f}')
        plt.xlabel('Red Cup Y Position (m)')
        plt.ylabel('Count')
        plt.title('Red Cup Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 2: 抓取中心 Y 坐标直方图
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(grasp_y, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(grasp_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {grasp_y.mean():.3f}')
        plt.axvline(np.median(grasp_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(grasp_y):.3f}')
        plt.xlabel(f'Grasp Center Y (Target Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m)')
        plt.ylabel('Count')
        plt.title('Grasp Center Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 3: X-Y 散点图
        ax3 = plt.subplot(2, 3, 3)
        plt.scatter(red_x, red_y, alpha=0.5, s=20, color='red')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Red Cup X-Y Position Scatter')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 子图 4: X 坐标直方图
        ax4 = plt.subplot(2, 3, 4)
        plt.hist(red_x, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_x.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_x.mean():.3f}')
        plt.axvline(np.median(red_x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_x):.3f}')
        plt.xlabel('Red Cup X Position (m)')
        plt.ylabel('Count')
        plt.title('Red Cup X Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 5: Z 坐标直方图
        ax5 = plt.subplot(2, 3, 5)
        plt.hist(red_z, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_z.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_z.mean():.3f}')
        plt.axvline(np.median(red_z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_z):.3f}')
        plt.xlabel('Red Cup Z Position (m)')
        plt.ylabel('Count')
        plt.title('Red Cup Z Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 6: 极坐标分布（相对于机械臂基座）
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        ARM_BASE_X = 0.0
        ARM_BASE_Y = 0.0
        # 计算相对于基座的距离和角度
        dx = red_x - ARM_BASE_X
        dy = red_y - ARM_BASE_Y
        distances = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)
        plt.scatter(angles, distances, alpha=0.5, s=20, color='red')
        plt.xlabel('Angle (rad)')
        plt.title('Red Cup Position (Polar, relative to arm base)')
        plt.grid(True, alpha=0.3)
    
    elif has_blue_data:
        # 只有蓝色杯子数据，使用2行3列布局
        fig = plt.figure(figsize=(16, 10))
        
        # 子图 1: Y 坐标直方图
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(blue_y, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_y.mean():.3f}')
        plt.axvline(np.median(blue_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_y):.3f}')
        plt.xlabel('Blue Cup Y Position (m)')
        plt.ylabel('Count')
        plt.title('Blue Cup Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 2: 抓取中心 Y 坐标直方图
        ax2 = plt.subplot(2, 3, 2)
        if len(grasp_y) > 0:
            plt.hist(grasp_y, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(grasp_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {grasp_y.mean():.3f}')
            plt.axvline(np.median(grasp_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(grasp_y):.3f}')
        plt.xlabel(f'Grasp Center Y (Target Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m)')
        plt.ylabel('Count')
        plt.title('Grasp Center Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 3: X-Y 散点图
        ax3 = plt.subplot(2, 3, 3)
        plt.scatter(blue_x, blue_y, alpha=0.5, s=20, color='blue')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Blue Cup X-Y Position Scatter')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 子图 4: X 坐标直方图
        ax4 = plt.subplot(2, 3, 4)
        plt.hist(blue_x, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_x.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_x.mean():.3f}')
        plt.axvline(np.median(blue_x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_x):.3f}')
        plt.xlabel('Blue Cup X Position (m)')
        plt.ylabel('Count')
        plt.title('Blue Cup X Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 5: Z 坐标直方图
        ax5 = plt.subplot(2, 3, 5)
        plt.hist(blue_z, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_z.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_z.mean():.3f}')
        plt.axvline(np.median(blue_z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_z):.3f}')
        plt.xlabel('Blue Cup Z Position (m)')
        plt.ylabel('Count')
        plt.title('Blue Cup Z Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图 6: 极坐标分布（相对于机械臂基座）
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        ARM_BASE_X = 0.0
        ARM_BASE_Y = 0.0
        # 计算相对于基座的距离和角度
        dx = blue_x - ARM_BASE_X
        dy = blue_y - ARM_BASE_Y
        distances = np.sqrt(dx**2 + dy**2)
        angles = np.arctan2(dy, dx)
        plt.scatter(angles, distances, alpha=0.5, s=20, color='blue')
        plt.xlabel('Angle (rad)')
        plt.title('Blue Cup Position (Polar, relative to arm base)')
        plt.grid(True, alpha=0.3)
    
    else:
        print("⚠️  No target cup data to visualize!")
        return
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = Path('cup_position')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图片
    output_path = output_dir / 'cup_position_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    # 额外保存小车初始坐标可视化
    if base_count > 0:
        fig_base = plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        plt.hist(base_x, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(base_x.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {base_x.mean():.3f}')
        plt.axvline(np.median(base_x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(base_x):.3f}')
        plt.xlabel('Base X Position (m)')
        plt.ylabel('Count')
        plt.title('🚗 Base Initial X Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.hist(base_y, bins=30, alpha=0.7, color='teal', edgecolor='black')
        plt.axvline(base_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {base_y.mean():.3f}')
        plt.axvline(np.median(base_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(base_y):.3f}')
        plt.xlabel('Base Y Position (m)')
        plt.ylabel('Count')
        plt.title('🚗 Base Initial Y Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.hist(base_yaw, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(base_yaw.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {base_yaw.mean():.3f}')
        plt.axvline(np.median(base_yaw), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(base_yaw):.3f}')
        plt.xlabel('Base Yaw (rad)')
        plt.ylabel('Count')
        plt.title('🚗 Base Initial Yaw Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.scatter(base_x, base_y, alpha=0.5, s=20, color='purple')
        plt.xlabel('Base X Position (m)')
        plt.ylabel('Base Y Position (m)')
        plt.title('🚗 Base Initial X-Y Scatter')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        plt.tight_layout()
        base_output_path = output_dir / 'base_position_analysis.png'
        plt.savefig(base_output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Base visualization saved to: {base_output_path}")
    else:
        print("⚠️  base_pose not found in dataset. Skip base visualization.")
    
    # 5. 保存统计结果到 JSON
    stats_dict = {
        'total_episodes': total_episodes - failed_count,
        'red_cup_target_count': red_target_count,
        'blue_cup_target_count': blue_target_count,
        'unknown_target_count': unknown_target_count,
        'base_pose_count': base_count,
    }
    
    if red_count > 0:
        stats_dict['red_cup'] = {
            'x': {
                'mean': float(red_x.mean()),
                'std': float(red_x.std()),
                'min': float(red_x.min()),
                'max': float(red_x.max()),
            },
            'y': {
                'mean': float(red_y.mean()),
                'std': float(red_y.std()),
                'min': float(red_y.min()),
                'max': float(red_y.max()),
            },
            'z': {
                'mean': float(red_z.mean()),
                'std': float(red_z.std()),
                'min': float(red_z.min()),
                'max': float(red_z.max()),
            },
        }
        if y_skew is not None:
            stats_dict['red_cup_y_skewness'] = float(y_skew)
    
    if len(grasp_y) > 0:
        stats_dict['grasp_center_y'] = {
            'mean': float(grasp_y.mean()),
            'std': float(grasp_y.std()),
            'min': float(grasp_y.min()),
            'max': float(grasp_y.max()),
        }

    if base_count > 0:
        stats_dict['base_pose'] = {
            'x': {
                'mean': float(base_x.mean()),
                'std': float(base_x.std()),
                'min': float(base_x.min()),
                'max': float(base_x.max()),
            },
            'y': {
                'mean': float(base_y.mean()),
                'std': float(base_y.std()),
                'min': float(base_y.min()),
                'max': float(base_y.max()),
            },
            'yaw': {
                'mean': float(base_yaw.mean()),
                'std': float(base_yaw.std()),
                'min': float(base_yaw.min()),
                'max': float(base_yaw.max()),
            },
        }
    
    # 如果有蓝色杯子数据，添加到统计中
    if has_blue_data and blue_y_skew is not None:
        stats_dict['blue_cup'] = {
            'x': {
                'mean': float(blue_x.mean()),
                'std': float(blue_x.std()),
                'min': float(blue_x.min()),
                'max': float(blue_x.max()),
            },
            'y': {
                'mean': float(blue_y.mean()),
                'std': float(blue_y.std()),
                'min': float(blue_y.min()),
                'max': float(blue_y.max()),
            },
            'z': {
                'mean': float(blue_z.mean()),
                'std': float(blue_z.std()),
                'min': float(blue_z.min()),
                'max': float(blue_z.max()),
            },
        }
        stats_dict['blue_cup_y_skewness'] = float(blue_y_skew)
    
    stats_path = output_dir / 'cup_position_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"✅ Statistics saved to: {stats_path}")
    
    # 6. 关键发现
    print("\n" + "="*70)
    print("🔍 Key Findings")
    print("="*70)
    
    # 红色杯子相关警告
    if y_skew is not None and abs(y_skew) > 0.5:
        print(f"⚠️  WARNING (Red Cup): Y coordinate distribution is {'right-skewed' if y_skew > 0 else 'left-skewed'} (skewness={y_skew:.3f})")
        print(f"   This means the model may learn a biased action pattern!")
    
    if len(grasp_y) > 0:
        if grasp_y.mean() > 0.05:
            print(f"⚠️  WARNING: Grasp center Y mean ({grasp_y.mean():.3f}m) is significantly positive!")
            print(f"   Model may learn to move in +Y direction first.")
        elif grasp_y.mean() < -0.05:
            print(f"⚠️  WARNING: Grasp center Y mean ({grasp_y.mean():.3f}m) is significantly negative!")
            print(f"   Model may learn to move in -Y direction first.")
    
    if red_count > 0 and red_y.std() < 0.05:
        print(f"⚠️  WARNING (Red Cup): Y coordinate std ({red_y.std():.3f}m) is very small!")
        print(f"   Cup positions may not be diverse enough for generalization.")
    
    # 蓝色杯子相关警告
    if has_blue_data and blue_y_skew is not None:
        if abs(blue_y_skew) > 0.5:
            print(f"⚠️  WARNING (Blue Cup): Y coordinate distribution is {'right-skewed' if blue_y_skew > 0 else 'left-skewed'} (skewness={blue_y_skew:.3f})")
            print(f"   This means the model may learn a biased action pattern for blue cup!")
        
        if blue_y.std() < 0.05:
            print(f"⚠️  WARNING (Blue Cup): Y coordinate std ({blue_y.std():.3f}m) is very small!")
            print(f"   Blue cup positions may not be diverse enough for generalization.")
        
        # 对比红色和蓝色杯子的位置差异
        if red_count > 0:
            red_center = np.array([red_x.mean(), red_y.mean()])
            blue_center = np.array([blue_x.mean(), blue_y.mean()])
            distance_between = np.linalg.norm(red_center - blue_center)
            print(f"\n📊 Red vs Blue Cup Comparison:")
            print(f"   Red cup center: ({red_center[0]:.3f}, {red_center[1]:.3f})")
            print(f"   Blue cup center: ({blue_center[0]:.3f}, {blue_center[1]:.3f})")
            print(f"   Distance between centers: {distance_between:.3f}m")

    if base_count > 0:
        if base_x.std() < 0.05 and base_y.std() < 0.05:
            print("⚠️  WARNING (Base): Base initial XY variance is very small.")
            print("   Mobile base start positions may lack diversity.")
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    
    return stats_dict

if __name__ == "__main__":
    try:
        stats = analyze_cup_positions()
    except ImportError as e:
        if 'scipy' in str(e):
            print("⚠️  scipy not found. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'scipy'])
            stats = analyze_cup_positions()
        else:
            raise