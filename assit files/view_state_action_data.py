#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 episode_000000.parquet 文件中的 state 数据
"""

import pandas as pd
import numpy as np
import sys
import os

from _script_paths import resolve_repo_path

def view_state_data(parquet_path):
    """
    读取并显示 parquet 文件中的 state 和 action 数据（前30帧）
    
    Args:
        parquet_path: parquet 文件路径
    """
    if not os.path.exists(parquet_path):
        print(f"错误: 文件不存在: {parquet_path}")
        return
    
    # 读取 parquet 文件
    df = pd.read_parquet(parquet_path)
    
    # 提取 state 和 action 数据
    state_array = None
    action_array = None
    
    if 'observation.state' in df.columns:
        state_data = df['observation.state']
        state_array = np.array([s if isinstance(s, np.ndarray) else np.array(s) for s in state_data])
    
    if 'action' in df.columns:
        action_data = df['action']
        action_array = np.array([a if isinstance(a, np.ndarray) else np.array(a) for a in action_data])
    
    # 显示前30帧的 State 数据
    if state_array is not None:
        print("前 30 帧的 State 数据 (x, y, z, roll, pitch, yaw):")
        print("=" * 100)
        print(f"{'Frame':<8} {'x':<12} {'y':<12} {'z':<12} {'roll':<12} {'pitch':<12} {'yaw':<12}")
        print("-" * 100)
        num_frames = min(30, len(state_array))
        for i in range(num_frames):
            state = state_array[i]
            print(f"{i:<8} {state[0]:<12.6f} {state[1]:<12.6f} {state[2]:<12.6f} "
                  f"{state[3]:<12.6f} {state[4]:<12.6f} {state[5]:<12.6f}")
        print()
    
    # 显示前30帧的 Action 数据
    if action_array is not None:
        print("前 30 帧的 Action 数据:")
        print("=" * 100)
        header = f"{'Frame':<8} "
        for i in range(action_array.shape[1]):
            header += f"{'action[' + str(i) + ']':<12} "
        print(header)
        print("-" * 100)
        num_frames = min(30, len(action_array))
        for i in range(num_frames):
            action = action_array[i]
            row = f"{i:<8} "
            for j in range(len(action)):
                row += f"{action[j]:<12.6f} "
            print(row)
    else:
        print("警告: 未找到 'action' 列")


if __name__ == "__main__":
    # 默认路径
    default_path = str(resolve_repo_path("demo_data_arm_v4/data/chunk-000/episode_000451.parquet"))
    
    if len(sys.argv) > 1:
        parquet_path = str(resolve_repo_path(sys.argv[1]))
    else:
        parquet_path = default_path
    
    # 如果使用默认路径，检查是否存在
    if not os.path.exists(parquet_path):
        # 尝试在当前目录下查找
        if os.path.exists(os.path.basename(parquet_path)):
            parquet_path = os.path.basename(parquet_path)
        else:
            print(f"错误: 文件不存在: {parquet_path}")
            print(f"\n用法: python view_state_data.py [parquet_file_path]")
            print(f"示例: python view_state_data.py demo_data_arm_v4/data/chunk-000/episode_000000.parquet")
            sys.exit(1)
    
    view_state_data(parquet_path)
