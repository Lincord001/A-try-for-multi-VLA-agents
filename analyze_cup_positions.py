#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练数据中杯子位置分布

功能：
1. 读取所有 episode 的 obj_init 数据
2. 分析红色杯子（mug_5）的 X、Y、Z 坐标分布
3. 可视化分布情况
4. 统计抓取中心（Y + offset）的分布
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 配置
DATASET_ROOT = './demo_data_arm_v4'
EXPERT_Y_GRASP_OFFSET = 0.067  # 从 y_env4.py 中获取

def load_episode_data(episode_path):
    """加载单个 episode 的数据"""
    try:
        df = pd.read_parquet(episode_path)
        # obj_init 在第一帧应该是一样的（所有帧的 obj_init 都相同）
        if len(df) > 0:
            obj_init = df['obj_init'].iloc[0]
            return obj_init
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
    
    # 2. 读取所有 episode 的 obj_init
    print("\n📖 Loading episode data...")
    cup_positions = {
        'red': {'x': [], 'y': [], 'z': []},
        'blue': {'x': [], 'y': [], 'z': []},
        'yellow': {'x': [], 'y': [], 'z': []},
        'green': {'x': [], 'y': [], 'z': []},
    }
    grasp_center_y = []  # 抓取中心的 Y 坐标（杯子 Y + offset）
    
    failed_count = 0
    for i, ep_file in enumerate(episode_files):
        if (i + 1) % 50 == 0:
            print(f"   Processing {i+1}/{total_episodes}...", end='\r')
        
        obj_init = load_episode_data(ep_file)
        if obj_init is None:
            failed_count += 1
            continue
        
        # obj_init 格式: 12维 = 4个杯子 × 3个坐标
        # [mug5_x, mug5_y, mug5_z, mug6_x, mug6_y, mug6_z, mug7_x, mug7_y, mug7_z, mug8_x, mug8_y, mug8_z]
        
        # 红色杯子 (mug_5) - 索引 0, 1, 2
        cup_positions['red']['x'].append(obj_init[0])
        cup_positions['red']['y'].append(obj_init[1])
        cup_positions['red']['z'].append(obj_init[2])
        
        # 计算抓取中心的 Y 坐标（杯子 Y + offset）
        grasp_center_y.append(obj_init[1] + EXPERT_Y_GRASP_OFFSET)
        
        # 其他杯子（如果存在）
        if len(obj_init) >= 12:
            cup_positions['blue']['x'].append(obj_init[3])
            cup_positions['blue']['y'].append(obj_init[4])
            cup_positions['blue']['z'].append(obj_init[5])
            
            cup_positions['yellow']['x'].append(obj_init[6])
            cup_positions['yellow']['y'].append(obj_init[7])
            cup_positions['yellow']['z'].append(obj_init[8])
            
            cup_positions['green']['x'].append(obj_init[9])
            cup_positions['green']['y'].append(obj_init[10])
            cup_positions['green']['z'].append(obj_init[11])
    
    print(f"\n✅ Loaded {total_episodes - failed_count} episodes successfully")
    if failed_count > 0:
        print(f"⚠️  Failed to load {failed_count} episodes")
    
    # 3. 统计分析
    print("\n" + "="*70)
    print("📈 Statistical Analysis")
    print("="*70)
    
    # 红色杯子统计
    red_y = np.array(cup_positions['red']['y'])
    red_x = np.array(cup_positions['red']['x'])
    red_z = np.array(cup_positions['red']['z'])
    grasp_y = np.array(grasp_center_y)
    
    print(f"\n🔴 Red Cup (mug_5) Position Statistics:")
    print(f"   X: mean={red_x.mean():.4f}, std={red_x.std():.4f}, range=[{red_x.min():.4f}, {red_x.max():.4f}]")
    print(f"   Y: mean={red_y.mean():.4f}, std={red_y.std():.4f}, range=[{red_y.min():.4f}, {red_y.max():.4f}]")
    print(f"   Z: mean={red_z.mean():.4f}, std={red_z.std():.4f}, range=[{red_z.min():.4f}, {red_z.max():.4f}]")
    
    print(f"\n🎯 Grasp Center Y (Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m):")
    print(f"   mean={grasp_y.mean():.4f}, std={grasp_y.std():.4f}, range=[{grasp_y.min():.4f}, {grasp_y.max():.4f}]")
    
    # 检查分布是否均匀
    print(f"\n📊 Distribution Analysis:")
    y_bins = np.linspace(red_y.min(), red_y.max(), 11)
    hist, _ = np.histogram(red_y, bins=y_bins)
    print(f"   Y coordinate bins: {len(y_bins)-1} bins")
    print(f"   Bin counts: {hist}")
    print(f"   Max bin count: {hist.max()}, Min bin count: {hist.min()}")
    
    # 计算偏度（skewness）
    from scipy import stats
    y_skew = stats.skew(red_y)
    print(f"   Y coordinate skewness: {y_skew:.4f} ({'偏右' if y_skew > 0 else '偏左' if y_skew < 0 else '对称'})")
    
    # 4. 可视化
    print("\n" + "="*70)
    print("📊 Generating Visualizations...")
    print("="*70)
    
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
    plt.xlabel(f'Grasp Center Y (Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m)')
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
    ARM_BASE_X = -0.4
    ARM_BASE_Y = 0.0
    # 计算相对于基座的距离和角度
    dx = red_x - ARM_BASE_X
    dy = red_y - ARM_BASE_Y
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    # 转换为度数
    angles_deg = np.degrees(angles)
    plt.scatter(angles, distances, alpha=0.5, s=20, color='red')
    plt.xlabel('Angle (rad)')
    plt.title('Red Cup Position (Polar, relative to arm base)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = 'cup_position_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # 5. 保存统计结果到 JSON
    stats_dict = {
        'total_episodes': total_episodes - failed_count,
        'red_cup': {
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
        },
        'grasp_center_y': {
            'mean': float(grasp_y.mean()),
            'std': float(grasp_y.std()),
            'min': float(grasp_y.min()),
            'max': float(grasp_y.max()),
        },
        'y_skewness': float(y_skew),
    }
    
    stats_path = 'cup_position_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"✅ Statistics saved to: {stats_path}")
    
    # 6. 关键发现
    print("\n" + "="*70)
    print("🔍 Key Findings")
    print("="*70)
    
    if abs(y_skew) > 0.5:
        print(f"⚠️  WARNING: Y coordinate distribution is {'right-skewed' if y_skew > 0 else 'left-skewed'} (skewness={y_skew:.3f})")
        print(f"   This means the model may learn a biased action pattern!")
    
    if grasp_y.mean() > 0.05:
        print(f"⚠️  WARNING: Grasp center Y mean ({grasp_y.mean():.3f}m) is significantly positive!")
        print(f"   Model may learn to move in +Y direction first.")
    elif grasp_y.mean() < -0.05:
        print(f"⚠️  WARNING: Grasp center Y mean ({grasp_y.mean():.3f}m) is significantly negative!")
        print(f"   Model may learn to move in -Y direction first.")
    
    if red_y.std() < 0.05:
        print(f"⚠️  WARNING: Y coordinate std ({red_y.std():.3f}m) is very small!")
        print(f"   Cup positions may not be diverse enough for generalization.")
    
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