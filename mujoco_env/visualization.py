"""
Visualization and logging utilities for deployment evaluation.

Extracted from the deploy script to reduce its size. All functions here are
pure utilities with no dependency on runtime policy/env state.
"""

import csv
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_model_version(model_path):
    """
    从模型路径中提取版本号后缀
    
    Args:
        model_path: 模型路径，如 './ckpt/pi0_arm/pretrained_model_arm_v5_2_1'
                   或 './ckpt/pi0_base/pretrained_model_ver_3/pretrained_model'
    
    Returns:
        version_suffix: 版本号后缀，如 'arm_v5_2_1' 或 'ver_3'
    """
    from pathlib import Path
    path = Path(model_path)
    
    # 如果路径以 /pretrained_model 结尾（没有版本号），取父目录名
    if path.name == 'pretrained_model':
        # 取父目录名，如 'pretrained_model_ver_3'
        parent_name = path.parent.name
        # 去掉 'pretrained_model_' 前缀
        if parent_name.startswith('pretrained_model_'):
            return parent_name[len('pretrained_model_'):]
        return parent_name
    else:
        # 取最后一部分，如 'pretrained_model_arm_v5_2_1'
        name = path.name
        # 去掉 'pretrained_model_' 前缀
        if name.startswith('pretrained_model_'):
            return name[len('pretrained_model_'):]
        return name


def ensure_step_log_header(log_path):
    """确保步数日志文件存在并写入表头"""
    log_file = Path(log_path)
    if not log_file.exists():
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'mode', 'result', 'steps',
                'target_color', 'cup_init_x', 'cup_init_y', 'cup_init_z'
            ])


def append_step_log(log_path, mode, result, steps, target_color, cup_init):
    """追加一行任务结果到步数日志"""
    ensure_step_log_header(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            mode, result, steps,
            target_color,
            f"{cup_init[0]:.6f}", f"{cup_init[1]:.6f}", f"{cup_init[2]:.6f}"
        ])


def plot_task_stats(
    stats,
    output_dir,
    success_count=0,
    fail_count=0,
    version_suffix='',
    grasp_y_offset=0.0,
):
    """
    绘制成功任务中目标杯子初始位置分布图
    stats: {'red': {'x': [], 'y': [], 'z': []}, 'blue': {...}}
    success_count: 成功任务数量
    fail_count: 失败任务数量
    version_suffix: 版本号后缀，用于文件名，如 'arm_v5_2_1'
    """
    red_x = np.array(stats['red']['x'])
    red_y = np.array(stats['red']['y'])
    red_z = np.array(stats['red']['z'])
    blue_x = np.array(stats['blue']['x'])
    blue_y = np.array(stats['blue']['y'])
    blue_z = np.array(stats['blue']['z'])
    grasp_y = np.array(stats['grasp_center_y'])
    has_red = len(red_x) > 0
    has_blue = len(blue_x) > 0

    if not has_red and not has_blue:
        print("⚠️ No successful task data to visualize.")
        return

    if has_red and has_blue:
        fig = plt.figure(figsize=(18, 14))
        ax1 = plt.subplot(3, 3, 1)
        plt.hist(red_x, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_x.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_x.mean():.3f}')
        plt.axvline(np.median(red_x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_x):.3f}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Count')
        plt.title('Red Cup X Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 3, 2)
        plt.hist(red_y, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_y.mean():.3f}')
        plt.axvline(np.median(red_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_y):.3f}')
        plt.xlabel('Y Position (m)')
        plt.ylabel('Count')
        plt.title('Red Cup Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax3 = plt.subplot(3, 3, 3)
        plt.hist(red_z, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(red_z.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {red_z.mean():.3f}')
        plt.axvline(np.median(red_z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(red_z):.3f}')
        plt.xlabel('Z Position (m)')
        plt.ylabel('Count')
        plt.title('Red Cup Z Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax4 = plt.subplot(3, 3, 4)
        plt.hist(blue_x, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_x.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_x.mean():.3f}')
        plt.axvline(np.median(blue_x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_x):.3f}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Count')
        plt.title('Blue Cup X Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax5 = plt.subplot(3, 3, 5)
        plt.hist(blue_y, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_y.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_y.mean():.3f}')
        plt.axvline(np.median(blue_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_y):.3f}')
        plt.xlabel('Y Position (m)')
        plt.ylabel('Count')
        plt.title('Blue Cup Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax6 = plt.subplot(3, 3, 6)
        plt.hist(blue_z, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(blue_z.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {blue_z.mean():.3f}')
        plt.axvline(np.median(blue_z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(blue_z):.3f}')
        plt.xlabel('Z Position (m)')
        plt.ylabel('Count')
        plt.title('Blue Cup Z Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax7 = plt.subplot(3, 3, 7)
        plt.scatter(red_x, red_y, alpha=0.5, s=20, color='red', label='Red Cup')
        plt.scatter(blue_x, blue_y, alpha=0.5, s=20, color='blue', label='Blue Cup')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('X-Y Position Scatter (Red vs Blue)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        ax8 = plt.subplot(3, 3, 8, projection='polar')
        arm_base_x = 0.0
        arm_base_y = 0.0
        dx_red = red_x - arm_base_x
        dy_red = red_y - arm_base_y
        dx_blue = blue_x - arm_base_x
        dy_blue = blue_y - arm_base_y
        plt.scatter(np.arctan2(dy_red, dx_red), np.sqrt(dx_red**2 + dy_red**2),
                    alpha=0.5, s=20, color='red', label='Red')
        plt.scatter(np.arctan2(dy_blue, dx_blue), np.sqrt(dx_blue**2 + dy_blue**2),
                    alpha=0.5, s=20, color='blue', label='Blue')
        plt.title('Polar Position (relative to arm base)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        ax9 = plt.subplot(3, 3, 9)
        if len(grasp_y) > 0:
            plt.hist(grasp_y, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(grasp_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {grasp_y.mean():.3f}')
            plt.axvline(np.median(grasp_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(grasp_y):.3f}')
            plt.xlabel(f'Grasp Center Y (Target Cup Y + {grasp_y_offset:.3f}m)')
            plt.ylabel('Count')
            plt.title('Grasp Center Y Position Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

    else:
        fig = plt.figure(figsize=(16, 10))
        if has_red:
            x, y, z, color, title_prefix = red_x, red_y, red_z, 'red', 'Red Cup'
        else:
            x, y, z, color, title_prefix = blue_x, blue_y, blue_z, 'blue', 'Blue Cup'

        ax1 = plt.subplot(2, 3, 1)
        plt.hist(y, bins=30, alpha=0.7, color=color, edgecolor='black')
        plt.axvline(y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.3f}')
        plt.axvline(np.median(y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(y):.3f}')
        plt.xlabel(f'{title_prefix} Y Position (m)')
        plt.ylabel('Count')
        plt.title(f'{title_prefix} Y Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 3, 2)
        if len(grasp_y) > 0:
            plt.hist(grasp_y, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(grasp_y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {grasp_y.mean():.3f}')
            plt.axvline(np.median(grasp_y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(grasp_y):.3f}')
        plt.xlabel(f'Grasp Center Y (Target Cup Y + {grasp_y_offset:.3f}m)')
        plt.ylabel('Count')
        plt.title('Grasp Center Y Position Distribution')
        if len(grasp_y) > 0:
            plt.legend()
        plt.grid(True, alpha=0.3)

        ax3 = plt.subplot(2, 3, 3)
        plt.scatter(x, y, alpha=0.5, s=20, color=color)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'{title_prefix} X-Y Position Scatter')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        ax4 = plt.subplot(2, 3, 4)
        plt.hist(x, bins=30, alpha=0.7, color=color, edgecolor='black')
        plt.axvline(x.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {x.mean():.3f}')
        plt.axvline(np.median(x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(x):.3f}')
        plt.xlabel(f'{title_prefix} X Position (m)')
        plt.ylabel('Count')
        plt.title(f'{title_prefix} X Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax5 = plt.subplot(2, 3, 5)
        plt.hist(z, bins=30, alpha=0.7, color=color, edgecolor='black')
        plt.axvline(z.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {z.mean():.3f}')
        plt.axvline(np.median(z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(z):.3f}')
        plt.xlabel(f'{title_prefix} Z Position (m)')
        plt.ylabel('Count')
        plt.title(f'{title_prefix} Z Position Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        ax6 = plt.subplot(2, 3, 6, projection='polar')
        arm_base_x = 0.0
        arm_base_y = 0.0
        dx = x - arm_base_x
        dy = y - arm_base_y
        plt.scatter(np.arctan2(dy, dx), np.sqrt(dx**2 + dy**2),
                    alpha=0.5, s=20, color=color)
        plt.title(f'{title_prefix} Position (Polar, relative to arm base)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存图片（添加版本号后缀）
    if version_suffix:
        img_filename = f'cup_position_analysis_{version_suffix}.png'
    else:
        img_filename = 'cup_position_analysis.png'
    img_path = out_dir / img_filename
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"✅ Task stats visualization saved to: {img_path}")
    
    # 保存 JSON 统计文件
    import json
    from scipy import stats as scipy_stats
    
    json_stats = {
        'total_episodes': len(red_x) + len(blue_x),
        'red_cup_target_count': len(red_x),
        'blue_cup_target_count': len(blue_x),
        'unknown_target_count': 0,
    }
    
    if has_red:
        json_stats['red_cup'] = {
            'x': {
                'mean': float(red_x.mean()),
                'std': float(red_x.std()),
                'min': float(red_x.min()),
                'max': float(red_x.max())
            },
            'y': {
                'mean': float(red_y.mean()),
                'std': float(red_y.std()),
                'min': float(red_y.min()),
                'max': float(red_y.max())
            },
            'z': {
                'mean': float(red_z.mean()),
                'std': float(red_z.std()),
                'min': float(red_z.min()),
                'max': float(red_z.max())
            }
        }
        json_stats['red_cup_y_skewness'] = float(scipy_stats.skew(red_y))
    
    if has_blue:
        json_stats['blue_cup'] = {
            'x': {
                'mean': float(blue_x.mean()),
                'std': float(blue_x.std()),
                'min': float(blue_x.min()),
                'max': float(blue_x.max())
            },
            'y': {
                'mean': float(blue_y.mean()),
                'std': float(blue_y.std()),
                'min': float(blue_y.min()),
                'max': float(blue_y.max())
            },
            'z': {
                'mean': float(blue_z.mean()),
                'std': float(blue_z.std()),
                'min': float(blue_z.min()),
                'max': float(blue_z.max())
            }
        }
        json_stats['blue_cup_y_skewness'] = float(scipy_stats.skew(blue_y))
    
    if len(grasp_y) > 0:
        json_stats['grasp_center_y'] = {
            'mean': float(grasp_y.mean()),
            'std': float(grasp_y.std()),
            'min': float(grasp_y.min()),
            'max': float(grasp_y.max())
        }
    
    # 🔥 添加成功率统计
    total_tasks = success_count + fail_count
    if total_tasks > 0:
        json_stats['task_statistics'] = {
            'total_tasks': total_tasks,
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': float(success_count / total_tasks * 100)
        }
    
    # 保存 JSON 统计文件（添加版本号后缀）
    if version_suffix:
        json_filename = f'cup_position_stats_{version_suffix}.json'
    else:
        json_filename = 'cup_position_stats.json'
    json_path = out_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"✅ Task stats JSON saved to: {json_path}")
    if total_tasks > 0:
        print(f"   📊 Success Rate: {success_count}/{total_tasks} ({success_count/total_tasks*100:.1f}%)")


def plot_tb3_init_stats(tb3_stats, output_dir, success_count=0, fail_count=0, version_suffix=''):
    """
    绘制成功任务中 TB3 初始坐标分布图
    tb3_stats: {'x': [], 'y': [], 'z': []}
    """
    x = np.array(tb3_stats['x'])
    y = np.array(tb3_stats['y'])
    z = np.array(tb3_stats['z'])

    if len(x) == 0:
        print("⚠️ No successful TB3 init data to visualize.")
        return

    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 3, 1)
    plt.hist(x, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(x.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {x.mean():.3f}')
    plt.axvline(np.median(x), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(x):.3f}')
    plt.xlabel('TB3 Init X Position (m)')
    plt.ylabel('Count')
    plt.title('TB3 Init X Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    plt.hist(y, bins=30, alpha=0.7, color='teal', edgecolor='black')
    plt.axvline(y.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {y.mean():.3f}')
    plt.axvline(np.median(y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(y):.3f}')
    plt.xlabel('TB3 Init Y Position (m)')
    plt.ylabel('Count')
    plt.title('TB3 Init Y Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    plt.hist(z, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(z.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {z.mean():.3f}')
    plt.axvline(np.median(z), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(z):.3f}')
    plt.xlabel('TB3 Init Z Position (m)')
    plt.ylabel('Count')
    plt.title('TB3 Init Z Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    plt.scatter(x, y, alpha=0.5, s=20, color='purple')
    plt.xlabel('TB3 Init X Position (m)')
    plt.ylabel('TB3 Init Y Position (m)')
    plt.title('TB3 Init X-Y Scatter')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    ax5 = plt.subplot(2, 3, 5, projection='polar')
    dx = x
    dy = y
    plt.scatter(np.arctan2(dy, dx), np.sqrt(dx**2 + dy**2), alpha=0.5, s=20, color='purple')
    plt.title('TB3 Init Position (Polar)')
    plt.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    plt.axis('off')
    total_tasks = success_count + fail_count
    summary_lines = [
        f"Samples (success only): {len(x)}",
        f"X mean/std: {x.mean():.3f} / {x.std():.3f}",
        f"Y mean/std: {y.mean():.3f} / {y.std():.3f}",
        f"Z mean/std: {z.mean():.3f} / {z.std():.3f}",
    ]
    if total_tasks > 0:
        summary_lines.append(f"Success rate: {success_count}/{total_tasks} ({success_count / total_tasks * 100:.1f}%)")
    plt.text(0.0, 1.0, "\n".join(summary_lines), fontsize=12, va='top')
    plt.title('TB3 Init Summary')

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if version_suffix:
        img_filename = f'tb3_init_position_analysis_{version_suffix}.png'
    else:
        img_filename = 'tb3_init_position_analysis.png'
    img_path = out_dir / img_filename
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"✅ TB3 init visualization saved to: {img_path}")

    import json
    json_stats = {
        'successful_samples': len(x),
        'tb3_init': {
            'x': {'mean': float(x.mean()), 'std': float(x.std()), 'min': float(x.min()), 'max': float(x.max())},
            'y': {'mean': float(y.mean()), 'std': float(y.std()), 'min': float(y.min()), 'max': float(y.max())},
            'z': {'mean': float(z.mean()), 'std': float(z.std()), 'min': float(z.min()), 'max': float(z.max())},
        },
    }
    if total_tasks > 0:
        json_stats['task_statistics'] = {
            'total_tasks': total_tasks,
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': float(success_count / total_tasks * 100),
        }

    if version_suffix:
        json_filename = f'tb3_init_position_stats_{version_suffix}.json'
    else:
        json_filename = 'tb3_init_position_stats.json'
    json_path = out_dir / json_filename
    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"✅ TB3 init JSON saved to: {json_path}")

