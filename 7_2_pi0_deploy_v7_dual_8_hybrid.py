#!/usr/bin/env python3
"""
V6 环境部署脚本 - 支持 Arm 和 Base 双模式异步推理

功能说明：
- 按 C: 切换 arm/base 控制模式（不重置环境）
- 按 N: 启动当前模式的 pi0 控制（异步推理）
- 按 M: 恢复人类遥控模式
- 按 Z: 重置环境
- 按 Q: 退出

Arm 模式：
- 数据集: demo_data_arm_v4
- 权重: ckpt/pi0_arm/pretrained_model_arm_v4
- 输入: 2个相机(agent/wrist, 224x224) + 6维关节角度
- 输出: 7维 (6关节角度 + 1夹爪状态) - 绝对量

Base 模式：
- 数据集: demo_data_base_ver_3
- 权重: ckpt/pi0_base/pretrained_model_ver_3/pretrained_model
- 输入: 3个相机(front/left/right, 256x256) + 2维轮速度
- 输出: 2维轮速度指令 - 绝对量
"""

import os
import sys
import time
import csv
import random
from pathlib import Path

# ==========================================
# 🔥 环境配置
# ==========================================
# 任务超时判定（秒）：超过该时间未成功则判定失败并自动重置
TASK_TIMEOUT_SEC = 60
# 任务循环次数：执行多少个任务后自动退出（0表示无限循环）
TASK_LOOP_COUNT = 100  # 0 = 无限循环，>0 = 执行指定次数后退出
# 任务步数/结果日志文件
STEP_LOG_PATH = './task_steps_log.csv'
# 成功统计输出目录（与 cup_position 文件夹一致）
TASK_STATS_OUTPUT_DIR = './cup_position'
print("Setting up environment variables for Hugging Face...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'

import threading
import copy
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import glfw

# ==========================================
# 🔧 模型配置
# ==========================================

# Arm 模型配置 (V5.3)
ARM_CONFIG = {
    'model_path': './ckpt/pi0_arm/pretrained_model_arm_v7_1',
    'dataset_repo_id': 'omy_arm_data_v7_1',
    'dataset_root': './demo_data_arm_v7_1',
    'chunk_size': 64,          # 🔥 V5.3: 从 5 改为 64
    'n_action_steps': 64,      # 🔥 V5.3: 从 5 改为 64
    'image_size': 224,  # arm 模型使用 224x224
    'state_dim': 7,            # 🔥 修复：应该是7维 [q1, q2, q3, q4, q5, q6, gripper]，匹配数据采集脚本
    'action_dim': 7,
    'camera_keys': ['agent', 'wrist'],
}

# Base 模型配置 (V3)
BASE_CONFIG = {
    'model_path': './ckpt/pi0_base/pretrained_model_base_v7_6',
    'dataset_repo_id': 'omy_base_data_v7_6',
    'dataset_root': './demo_data_base_v7_6',
    'chunk_size': 32,          # V6: 与 arm 对齐
    'n_action_steps': 32,      # V6: 与 arm 对齐
    'image_size': 224,  # base 模型使用 256x256
    'state_dim': 4,  # [v_left, v_right, sin(yaw), cos(yaw)]
    'action_dim': 2,
    'camera_keys': ['front', 'left', 'right'],
}

# ==========================================
# 🔥 控制频率配置（Hz）
# ==========================================
CONTROL_FREQUENCY = 20  # 控制频率，单位：Hz
CONTROL_DT = 1.0 / CONTROL_FREQUENCY  # 控制周期，单位：秒

# ==========================================
# 🔧 推理参数配置
# ==========================================
# ARM 推理模式总开关（文件开头改这里即可）：
#   - "sync":  同步推理（每帧在主线程推理）
#   - "async": 异步推理（后台线程推理，主线程不阻塞）
ARM_INFERENCE_MODE = "async"
if ARM_INFERENCE_MODE.lower() not in ("sync", "async"):
    raise ValueError("ARM_INFERENCE_MODE must be 'sync' or 'async'")
ARM_SYNC_INFERENCE = ARM_INFERENCE_MODE.lower() == "sync"

# 异步推理参数（仅当 ARM_SYNC_INFERENCE = False 时有效）：
ACTION_HORIZON = 8   # 每个 chunk 实际执行的步数（异步模式）
CHUNK_THRESHOLD = 0  # 当剩余帧数 <= 此值时，开始新推理
# Base 异步推理参数（与 Arm 逻辑保持一致）
BASE_ACTION_HORIZON = 8
BASE_CHUNK_THRESHOLD = 0

# Base 自动推理动作后处理（可选）：
# 仅作用于 auto_mode_base 下模型输出的 (v_left, v_right)。
BASE_POSTPROC_ENABLED = True
BASE_POSTPROC_HEADING_HOLD_ENABLED = True
BASE_POSTPROC_KP_YAW = 64.0
BASE_POSTPROC_MAX_TURN_V = 2.0
BASE_POSTPROC_YAW_DEADBAND = np.deg2rad(0.5)
BASE_POSTPROC_STRAIGHT_DELTA_TH = 1.5
BASE_POSTPROC_MIN_ABS_SPEED = 1.0
BASE_POSTPROC_MAX_WHEEL_ABS = 30.0
# Base 轮速降速：对模型输出的所有运动（前进/后退/转弯）按比例缩放。
BASE_FORWARD_SPEED_SCALE_ENABLED = True
BASE_FORWARD_SPEED_SCALE = 0.5

# 同步推理参数（仅当 ARM_SYNC_INFERENCE = True 时有效）：
# 模型一次推理输出 n_action_steps（当前为64），控制时只执行前 ARM_EXEC_HORIZON 步
ARM_EXEC_HORIZON = 8

# ==========================================
# 🔧 动作平滑配置（防颤抖）
# ==========================================
# EMA 平滑参数：new_action = alpha * raw_action + (1 - alpha) * last_action
# alpha 越小越平滑，但响应越慢；越大越灵敏，但可能颤抖
SMOOTHING_ENABLED = True        # 是否启用动作平滑
SMOOTHING_ALPHA_JOINTS = 0.15    # 关节角度平滑系数 (0.0~1.0)
SMOOTHING_ALPHA_GRIPPER = 0.25   # 夹爪平滑系数 (0.0~1.0)

# 夹爪迟滞控制：防止夹爪在阈值附近频繁切换
GRIPPER_HYSTERESIS_ENABLED = True
GRIPPER_OPEN_THRESH = 0.7       # 超过此值才打开夹爪
GRIPPER_CLOSE_THRESH = 0.25      # 低于此值才关闭夹爪

# ==========================================
# 🔧 模型加载选择
# ==========================================
LOAD_ARM_MODEL = True   # 是否加载 ARM 模型
LOAD_BASE_MODEL = True  # 是否加载 BASE 模型

# ==========================================
# 🔧 Arm 模式难度选择
# ==========================================
# True:  简单模式（Pilot Run Mode）- 只生成红色杯子，位置大范围随机
# False: 困难模式（Normal Mode）- 4个杯子固定位置，小范围随机
ARM_PILOT_RUN_MODE = True  # 🔥 已废弃，保留用于兼容性

# ==========================================
# 🔧 随机初始化配置（匹配 y_env6.py 和 collect_data_v6.py）
# ==========================================
RANDOM_INIT_ENABLED = 0            # 0: 关闭, 1: 旧版(扇形区域), 2: 新版(圆形交集，仅简单模式)
RANDOM_INIT_GRIPPER_OPEN = True    # True: 初始化时夹爪张开, False: 初始化时夹爪闭合

# ==========================================
# 🚗 TB3 托盘 X 坐标随机化配置（匹配 y_env6.py / collect_data_v6.py）
# ==========================================
TB3_X_GAUSSIAN_ENABLED = True   # True: 使用高斯扰动；False: 固定位置
TB3_X_CENTER = 0.20             # 固定位置或扰动中心
TB3_X_OFFSET_STD = 0.04         # 高斯标准差（米）
TB3_X_OFFSET_MIN = -0.10        # 偏移下限（米）
TB3_X_OFFSET_MAX = 0.10         # 偏移上限（米）

# ==========================================
# 🧭 部署侧文本任务指令组（左右方向键切换）
# ==========================================
# 说明：
# - arm/base 各自维护独立的“指令组”
# - 每个组包含多条文本任务
# - 推理输入使用当前组内随机抽样结果（尽量避免与上一条重复）
INSTRUCTION_GROUPS = {
    'arm': [
        {
            'name': 'arm_default',
            'instructions': [
                "Place the red mug on the plate.",
                "Place the blue mug on the plate.",
            ],
        },
                {
            'name': 'arm_01',
            'instructions': [
                "Take the red mug from the tray to the table.",
                "Take the blue mug from the tray to the table.",
            ],
        },
    ],
    'base': [
        {
            'name': 'base_0',
            'instructions': [
                "Go to the workbench.",
                "Drive to the workbench.",
            ],
        },
        {
            'name': 'base_1',
            'instructions': [
                "Move to the kitchen refrigerator.",

            ],
        },
    ],
}


def _validate_instruction_groups():
    for mode in ('arm', 'base'):
        groups = INSTRUCTION_GROUPS.get(mode, [])
        if len(groups) == 0:
            raise ValueError(f"INSTRUCTION_GROUPS['{mode}'] cannot be empty.")
        for group in groups:
            name = group.get('name', '<unnamed>')
            instructions = group.get('instructions', [])
            if len(instructions) == 0:
                raise ValueError(f"Instruction group '{name}' in mode '{mode}' has no instructions.")


def _get_group_info(mode, group_indices):
    groups = INSTRUCTION_GROUPS[mode]
    total = len(groups)
    idx = group_indices[mode] % total
    group = groups[idx]
    group_name = group.get('name', f'{mode}_group_{idx + 1}')
    instructions = list(group.get('instructions', []))
    return idx, total, group_name, instructions


def _pick_instruction_from_active_group(mode, group_indices, last_instruction_by_mode):
    _, _, _, instructions = _get_group_info(mode, group_indices)
    last_inst = last_instruction_by_mode.get(mode)
    candidates = [inst for inst in instructions if inst != last_inst]
    if len(candidates) == 0:
        candidates = instructions
    picked = random.choice(candidates)
    last_instruction_by_mode[mode] = picked
    return picked


def _apply_instruction_from_group(PnPEnv, mode, group_indices, last_instruction_by_mode, log_prefix="", reinitialize_arm=False):
    idx, total, group_name, _ = _get_group_info(mode, group_indices)
    picked_instruction = _pick_instruction_from_active_group(mode, group_indices, last_instruction_by_mode)
    task_type = 'arm' if mode == 'arm' else 'nav'
    PnPEnv.set_instruction(given=picked_instruction, task_type=task_type)
    if mode == 'arm' and reinitialize_arm:
        PnPEnv.reset(mode='arm', preserve_instruction=True)
    extra = f"{log_prefix} " if log_prefix else ""
    print(
        f"{extra}🧭 [{mode.upper()}] Instruction Group: {group_name} "
        f"({idx + 1}/{total}) | Task: {PnPEnv.instruction}"
    )


# 导入 LeRobot 和 MuJoCo 环境
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.types import FeatureType
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from mujoco_env.y_env7 import SimpleEnv7, EXPERT_Y_GRASP_OFFSET
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# ==========================================
# 📊 性能监控工具
# ==========================================
class PerformanceMonitor:
    """性能监控工具，记录各步骤的执行时间"""
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.main_thread_times = {
            'grab_image': [],
            'data_copy': [],
            'get_action': [],
            'step_env': [],
            'render': [],
            'total_loop': []
        }
        self.inference_thread_times = {
            'lock_read': [],
            'img_resize': [],
            'img_totensor': [],
            'img_todevice': [],
            'normalize_inputs': [],
            'prepare_images': [],
            'prepare_state': [],
            'prepare_language': [],
            'sample_actions': [],
            'postprocess': [],
            'total_inference': []
        }
        self.lock = threading.Lock()
        
    def record_main(self, stage, duration):
        """记录主线程各阶段耗时"""
        with self.lock:
            if stage in self.main_thread_times:
                self.main_thread_times[stage].append(duration)
                if len(self.main_thread_times[stage]) > self.window_size:
                    self.main_thread_times[stage].pop(0)
    
    def record_inference(self, stage, duration):
        """记录推理线程各阶段耗时"""
        with self.lock:
            if stage in self.inference_thread_times:
                self.inference_thread_times[stage].append(duration)
                if len(self.inference_thread_times[stage]) > self.window_size:
                    self.inference_thread_times[stage].pop(0)
    
    def get_stats(self, stage, times_list):
        """获取统计信息"""
        if len(times_list) == 0:
            return 0.0, 0.0, 0.0
        arr = np.array(times_list)
        return np.mean(arr), np.min(arr), np.max(arr)
    
    def print_stats(self, step, mode='arm'):
        """打印统计信息"""
        with self.lock:
            print(f"\n{'='*80}")
            print(f"📊 Performance Stats (Step {step}, Mode: {mode.upper()})")
            print(f"{'='*80}")
            
            print("\n[主线程] 时间统计 (ms):")
            for stage, times in self.main_thread_times.items():
                mean, min_val, max_val = self.get_stats(stage, times)
                count = len(times)
                if count > 0:
                    print(f"  {stage:15s}: 平均={mean*1000:6.2f} 最小={min_val*1000:6.2f} 最大={max_val*1000:6.2f} (样本={count})")
            
            print("\n[推理线程] 时间统计 (ms):")
            for stage, times in self.inference_thread_times.items():
                mean, min_val, max_val = self.get_stats(stage, times)
                count = len(times)
                if count > 0:
                    print(f"  {stage:15s}: 平均={mean*1000:6.2f} 最小={min_val*1000:6.2f} 最大={max_val*1000:6.2f} (样本={count})")
            
            total_inference = self.inference_thread_times.get('total_inference', [])
            if len(total_inference) > 0:
                mean_total = np.mean(total_inference)
                print(f"\n🎯 推理线程总耗时: {mean_total*1000:.2f}ms (目标: <250ms)")
            print(f"{'='*80}\n")
    
    def reset(self):
        """重置统计信息"""
        with self.lock:
            for key in self.main_thread_times:
                self.main_thread_times[key] = []
            for key in self.inference_thread_times:
                self.inference_thread_times[key] = []


# ==========================================
# 🔧 动作平滑器（防颤抖）
# ==========================================
class ActionSmoother:
    """
    动作平滑器 - 使用指数移动平均(EMA)减少颤抖
    
    功能：
    1. 关节角度平滑：减少模型输出的高频噪声
    2. 夹爪迟滞控制：防止夹爪在阈值附近频繁切换
    """
    def __init__(self, joint_dim=6, alpha_joints=0.4, alpha_gripper=0.3):
        self.joint_dim = joint_dim
        self.alpha_joints = alpha_joints
        self.alpha_gripper = alpha_gripper
        
        # 历史状态
        self.last_joint_angles = None
        self.last_gripper_cmd = None
        self.gripper_state = False  # True=打开, False=关闭
        
    def reset(self):
        """重置平滑器状态（环境重置时调用）"""
        self.last_joint_angles = None
        self.last_gripper_cmd = None
        self.gripper_state = False
        
    def smooth_action(self, raw_action):
        """
        平滑处理动作
        
        Args:
            raw_action: 原始动作 (7,) - [6关节角度, 1夹爪]
            
        Returns:
            smoothed_action: 平滑后的动作 (7,)
            gripper_state: 夹爪状态 (bool)
        """
        if raw_action is None:
            return None, self.gripper_state
        
        joint_angles = raw_action[:self.joint_dim].copy()
        gripper_cmd = raw_action[self.joint_dim] if len(raw_action) > self.joint_dim else 0.0
        
        # ========== EMA 平滑关节角度 ==========
        if SMOOTHING_ENABLED:
            if self.last_joint_angles is not None:
                joint_angles = (self.alpha_joints * joint_angles + 
                               (1 - self.alpha_joints) * self.last_joint_angles)
            self.last_joint_angles = joint_angles.copy()
        else:
            self.last_joint_angles = joint_angles.copy()
        
        # ========== EMA 平滑夹爪 + 迟滞控制 ==========
        if SMOOTHING_ENABLED:
            if self.last_gripper_cmd is not None:
                gripper_cmd = (self.alpha_gripper * gripper_cmd + 
                              (1 - self.alpha_gripper) * self.last_gripper_cmd)
            self.last_gripper_cmd = gripper_cmd
        else:
            self.last_gripper_cmd = gripper_cmd
        
        # 夹爪迟滞控制
        if GRIPPER_HYSTERESIS_ENABLED:
            if gripper_cmd > GRIPPER_OPEN_THRESH:
                self.gripper_state = True
            elif gripper_cmd < GRIPPER_CLOSE_THRESH:
                self.gripper_state = False
            # 在 CLOSE_THRESH ~ OPEN_THRESH 之间保持不变
        else:
            self.gripper_state = gripper_cmd > 0.5
        
        # 组装平滑后的动作
        smoothed_action = np.concatenate([joint_angles, [gripper_cmd]])
        
        return smoothed_action, self.gripper_state


class BaseActionPostProcessor:
    """Base 模式动作后处理：近似直行时启用航向锁定纠偏。"""

    def __init__(self):
        self.heading_hold_active = False
        self.target_yaw = None
        self.last_sign = 0  # +1 前进, -1 后退

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self.heading_hold_active = False
        self.target_yaw = None
        self.last_sign = 0

    def process(self, raw_action, yaw):
        action = np.array(raw_action, dtype=np.float32).copy()
        if action.shape[0] != 2:
            return action
        if not BASE_POSTPROC_ENABLED:
            return action

        left, right = float(action[0]), float(action[1])
        cmd_sign = 1 if (left + right) >= 0.0 else -1

        # 对所有运动统一降速（包括前进/后退/转弯）。
        if (
            BASE_FORWARD_SPEED_SCALE_ENABLED
            and BASE_FORWARD_SPEED_SCALE >= 0.0
        ):
            left *= BASE_FORWARD_SPEED_SCALE
            right *= BASE_FORWARD_SPEED_SCALE

        same_direction = left * right > 0.0
        delta_small = abs(left - right) <= BASE_POSTPROC_STRAIGHT_DELTA_TH
        speed_enough = max(abs(left), abs(right)) >= BASE_POSTPROC_MIN_ABS_SPEED
        straight_like = same_direction and delta_small and speed_enough

        # 只在“模型意图直行”时做航向纠偏；转向动作不干预。
        if not BASE_POSTPROC_HEADING_HOLD_ENABLED or not straight_like:
            self.reset()
            # 这里返回缩放后的速度，确保转弯/后退同样生效降速。
            return np.clip(
                np.array([left, right], dtype=np.float32),
                -BASE_POSTPROC_MAX_WHEEL_ABS,
                BASE_POSTPROC_MAX_WHEEL_ABS,
            )

        if (not self.heading_hold_active) or (self.last_sign != cmd_sign):
            self.target_yaw = float(yaw)
            self.heading_hold_active = True
            self.last_sign = cmd_sign

        yaw_err = self._wrap_to_pi(self.target_yaw - float(yaw))
        if abs(yaw_err) <= BASE_POSTPROC_YAW_DEADBAND:
            turn = 0.0
        else:
            turn = float(np.clip(BASE_POSTPROC_KP_YAW * yaw_err, -BASE_POSTPROC_MAX_TURN_V, BASE_POSTPROC_MAX_TURN_V))

        v = 0.5 * (left + right)
        corrected = np.array([v - turn, v + turn], dtype=np.float32)
        return np.clip(corrected, -BASE_POSTPROC_MAX_WHEEL_ABS, BASE_POSTPROC_MAX_WHEEL_ABS)


# ==========================================
# 🔥 异步推理运行器 - Arm 模式
# ==========================================
class AsyncArmInferenceRunner:
    """
    Arm 模式的异步推理运行器
    - 输入: 2个相机(agent/wrist) + 6维关节角度
    - 输出: 7维动作 (6关节角度 + 1夹爪状态)
    
    🔥 关键：ARM 模式使用顺序执行而非延迟补偿
    因为 ARM 的动作是绝对关节角度，跳帧会导致位置突变
    """
    def __init__(self, policy, device, img_transform, control_dt, perf_monitor=None):
        self.policy = policy
        self.device = device
        self.img_transform = img_transform
        self.control_dt = control_dt
        self.perf_monitor = perf_monitor
        self.image_size = ARM_CONFIG['image_size']
        
        # 线程同步锁
        self.lock = threading.Lock()
        
        # 共享数据：输入
        self.latest_raw_images = None  # {'agent': np.array, 'wrist': np.array}
        self.latest_state = None       # (6,) 关节角度
        self.latest_task = None        # 任务字符串列表
        self.latest_obs_timestamp = 0
        
        # 共享数据：输出
        self.latest_action_chunk = None  # (n_action_steps, 7)
        self.chunk_start_timestamp = 0
        
        # 🔥 顺序执行计数器（替代延迟补偿）
        self.current_step_index = 0
        self.chunk_id = 0  # 用于检测新 chunk
        
        # 推理频率控制
        self.last_processed_timestamp = 0
        
        # 线程控制
        self.running = False
        self.thread = None

    def start(self):
        # 🔥 防御性检查：如果有旧线程在运行，先停止它
        if self.running and self.thread and self.thread.is_alive():
            print("⚠️ [ARM AsyncRunner] 发现旧线程仍在运行，先停止它")
            self.stop()
        
        # 🔥 重置所有状态，确保每次启动都是干净的
        self.reset_state()
        
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("🤖 [ARM AsyncRunner] 推理线程已启动 (状态已重置)")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        print("🛑 [ARM AsyncRunner] 推理线程已停止")
    
    def reset_state(self):
        """重置所有状态（不停止线程）- 可在环境重置时调用"""
        with self.lock:
            self.latest_raw_images = None
            self.latest_state = None
            self.latest_task = None
            self.latest_obs_timestamp = 0
            self.latest_action_chunk = None
            self.chunk_start_timestamp = 0
            self.current_step_index = 0
            self.chunk_id = 0
            self.last_processed_timestamp = 0

    def update_observation(self, images_dict, state, task, timestamp):
        """主线程调用：更新最新的观测数据"""
        t0 = time.perf_counter()
        
        # 深度拷贝原始数据
        images_copy = {}
        for key, value in images_dict.items():
            if isinstance(value, np.ndarray):
                images_copy[key] = value.copy()
        
        state_copy = np.array(state, copy=True) if state is not None else None
        task_copy = task.copy() if isinstance(task, list) else task
        
        t1 = time.perf_counter()
        if self.perf_monitor:
            self.perf_monitor.record_main('data_copy', t1 - t0)
        
        with self.lock:
            self.latest_raw_images = images_copy
            self.latest_state = state_copy
            self.latest_task = task_copy
            self.latest_obs_timestamp = timestamp

    def get_action_at_time(self, current_time):
        """
        主线程调用：顺序获取下一个动作（不使用延迟补偿）
        
        🔥 ARM 模式特殊处理：
        - 不根据时间差跳帧，而是顺序执行每一帧
        - 只执行 chunk 的前 ACTION_HORIZON 步，然后等待新推理
        - 这样可以减少 chunk 之间的不连续性（颤抖）
        
        返回: (action, status_msg) 其中 action 是 (7,) 数组或 None
        """
        with self.lock:
            if self.latest_action_chunk is None:
                return None, "Wait for init"
            
            chunk = self.latest_action_chunk
            chunk_len = chunk.shape[0]
            step_index = self.current_step_index
            
            # 🔥 使用 ACTION_HORIZON 限制实际执行的步数
            effective_horizon = min(ACTION_HORIZON, chunk_len)
            
            if step_index < effective_horizon:
                # 正常执行：返回当前帧，计数器+1
                action = chunk[step_index]
                self.current_step_index += 1
                return action, f"OK (Step {step_index}/{effective_horizon})"
            else:
                # 已执行 ACTION_HORIZON 步：保持最后一帧的动作，等待新 chunk
                last_idx = effective_horizon - 1 if effective_horizon > 0 else 0
                return chunk[last_idx], f"Hold (Horizon {effective_horizon} reached)"

    def _inference_loop(self):
        """后台线程：执行推理"""
        # 🔥 ARM 模式：只在 chunk 快用完时才开始新推理
        # CHUNK_THRESHOLD 已在文件开头定义
        
        while self.running:
            t_lock_start = time.perf_counter()
            raw_images = None
            state = None
            task = None
            obs_timestamp = 0
            
            with self.lock:
                t_lock_read = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('lock_read', t_lock_read - t_lock_start)
                
                if self.latest_raw_images is not None:
                    # 🔥 检查是否需要新推理：
                    # 1. 首次推理（latest_action_chunk 为 None）
                    # 2. 当前 horizon 快用完了（剩余帧数 <= CHUNK_THRESHOLD）
                    need_inference = False
                    if self.latest_action_chunk is None:
                        need_inference = True
                    else:
                        chunk_len = self.latest_action_chunk.shape[0]
                        effective_horizon = min(ACTION_HORIZON, chunk_len)
                        remaining_steps = effective_horizon - self.current_step_index
                        if remaining_steps <= CHUNK_THRESHOLD:
                            need_inference = True
                    
                    if need_inference and self.latest_obs_timestamp > self.last_processed_timestamp:
                        raw_images = self.latest_raw_images
                        state = self.latest_state
                        task = self.latest_task
                        obs_timestamp = self.latest_obs_timestamp
                        self.last_processed_timestamp = obs_timestamp
                    else:
                        raw_images = None
            
            if raw_images is None:
                time.sleep(0.001)
                continue
            
            t_inference_start = time.perf_counter()
            try:
                # 图像预处理: resize
                t0 = time.perf_counter()
                agent_img = Image.fromarray(raw_images['agent']).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
                wrist_img = Image.fromarray(raw_images['wrist']).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_resize', t1 - t0)
                
                # ToTensor
                t0 = time.perf_counter()
                agent_t = self.img_transform(agent_img).unsqueeze(0)
                wrist_t = self.img_transform(wrist_img).unsqueeze(0)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_totensor', t1 - t0)
                
                # .to(device)
                t0 = time.perf_counter()
                agent_tensor = agent_t.to(self.device)
                wrist_tensor = wrist_t.to(self.device)
                state_tensor = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_todevice', t1 - t0)
                
                # 构建 batch
                batch = {
                    'observation.state': state_tensor,
                    'observation.images.agent': agent_tensor,
                    'observation.images.wrist': wrist_tensor,
                    'task': task
                }
                
                # 执行推理
                with torch.no_grad():
                    t0 = time.perf_counter()
                    batch = self.policy.normalize_inputs(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('normalize_inputs', t1 - t0)
                    
                    t0 = time.perf_counter()
                    images, img_masks = self.policy.prepare_images(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_images', t1 - t0)
                    
                    t0 = time.perf_counter()
                    state_processed = self.policy.prepare_state(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_state', t1 - t0)
                    
                    t0 = time.perf_counter()
                    lang_tokens, lang_masks = self.policy.prepare_language(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_language', t1 - t0)
                    
                    t0 = time.perf_counter()
                    actions = self.policy.model.sample_actions(
                        images, img_masks, lang_tokens, lang_masks, state_processed
                    )
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('sample_actions', t1 - t0)
                    
                    t0 = time.perf_counter()
                    original_action_dim = self.policy.config.action_feature.shape[0]
                    actions = actions[:, :, :original_action_dim]
                    actions = self.policy.unnormalize_outputs({"action": actions})["action"]
                    
                    if self.policy.config.adapt_to_pi_aloha:
                        actions = self.policy._pi_aloha_encode_actions(actions)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('postprocess', t1 - t0)
                
                # 写入输出
                chunk_np = actions[0].cpu().numpy()
                t_inference_end = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('total_inference', t_inference_end - t_inference_start)
                
                with self.lock:
                    self.latest_action_chunk = chunk_np
                    self.chunk_start_timestamp = obs_timestamp
                    # 🔥 新 chunk 准备好时，重置顺序执行计数器
                    self.current_step_index = 0
                    self.chunk_id += 1
                    
            except Exception as e:
                print(f"[ARM] Inference Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)


# ==========================================
# 🔥 异步推理运行器 - Base 模式
# ==========================================
class AsyncBaseInferenceRunner:
    """
    Base 模式的异步推理运行器
    - 输入: 3个相机(front/left/right) + 2维轮速度
    - 输出: 2维轮速度指令
    """
    def __init__(self, policy, device, img_transform, control_dt, perf_monitor=None):
        self.policy = policy
        self.device = device
        self.img_transform = img_transform
        self.control_dt = control_dt
        self.perf_monitor = perf_monitor
        self.image_size = BASE_CONFIG['image_size']
        
        # 线程同步锁
        self.lock = threading.Lock()
        
        # 共享数据：输入
        self.latest_raw_images = None  # {'front': np.array, 'left': np.array, 'right': np.array}
        self.latest_state = None       # (2,) 轮速度
        self.latest_task = None        # 任务字符串列表
        self.latest_obs_timestamp = 0
        
        # 共享数据：输出
        self.latest_action_chunk = None  # (n_action_steps, 2)
        self.chunk_start_timestamp = 0
        
        # 顺序执行计数器（与 ARM 异步逻辑一致）
        self.current_step_index = 0
        self.chunk_id = 0
        
        # 推理频率控制
        self.last_processed_timestamp = 0
        
        # 线程控制
        self.running = False
        self.thread = None

    def start(self):
        # 🔥 防御性检查：如果有旧线程在运行，先停止它
        if self.running and self.thread and self.thread.is_alive():
            print("⚠️ [BASE AsyncRunner] 发现旧线程仍在运行，先停止它")
            self.stop()
        
        # 🔥 重置所有状态，确保每次启动都是干净的
        self.reset_state()
        
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        print("🚗 [BASE AsyncRunner] 推理线程已启动 (状态已重置)")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None
        print("🛑 [BASE AsyncRunner] 推理线程已停止")
    
    def reset_state(self):
        """重置所有状态（不停止线程）- 可在环境重置时调用"""
        with self.lock:
            self.latest_raw_images = None
            self.latest_state = None
            self.latest_task = None
            self.latest_obs_timestamp = 0
            self.latest_action_chunk = None
            self.chunk_start_timestamp = 0
            self.current_step_index = 0
            self.chunk_id = 0
            self.last_processed_timestamp = 0

    def update_observation(self, images_dict, state, task, timestamp):
        """主线程调用：更新最新的观测数据"""
        t0 = time.perf_counter()
        
        images_copy = {}
        for key, value in images_dict.items():
            if isinstance(value, np.ndarray):
                images_copy[key] = value.copy()
        
        state_copy = np.array(state, copy=True) if state is not None else None
        task_copy = task.copy() if isinstance(task, list) else task
        
        t1 = time.perf_counter()
        if self.perf_monitor:
            self.perf_monitor.record_main('data_copy', t1 - t0)
        
        with self.lock:
            self.latest_raw_images = images_copy
            self.latest_state = state_copy
            self.latest_task = task_copy
            self.latest_obs_timestamp = timestamp

    def get_action_at_time(self, current_time):
        """
        主线程调用：顺序获取下一个动作（与 ARM 异步逻辑一致）
        返回: (action, status_msg) 其中 action 是 (2,) 数组或 None
        """
        with self.lock:
            if self.latest_action_chunk is None:
                return None, "Wait for init"
            
            chunk = self.latest_action_chunk
            chunk_len = chunk.shape[0]
            step_index = self.current_step_index
            effective_horizon = min(BASE_ACTION_HORIZON, chunk_len)
            
            if step_index < effective_horizon:
                action = chunk[step_index]
                self.current_step_index += 1
                return action, f"OK (Step {step_index}/{effective_horizon})"
            
            # horizon 用完后保持最后一步，等待新 chunk
            last_idx = effective_horizon - 1 if effective_horizon > 0 else 0
            return chunk[last_idx], f"Hold (Horizon {effective_horizon} reached)"

    def _inference_loop(self):
        """后台线程：执行推理"""
        while self.running:
            t_lock_start = time.perf_counter()
            raw_images = None
            state = None
            task = None
            obs_timestamp = 0
            
            with self.lock:
                t_lock_read = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('lock_read', t_lock_read - t_lock_start)
                
                if self.latest_raw_images is not None:
                    need_inference = False
                    if self.latest_action_chunk is None:
                        need_inference = True
                    else:
                        chunk_len = self.latest_action_chunk.shape[0]
                        effective_horizon = min(BASE_ACTION_HORIZON, chunk_len)
                        remaining_steps = effective_horizon - self.current_step_index
                        if remaining_steps <= BASE_CHUNK_THRESHOLD:
                            need_inference = True
                    
                    if need_inference and self.latest_obs_timestamp > self.last_processed_timestamp:
                        raw_images = self.latest_raw_images
                        state = self.latest_state
                        task = self.latest_task
                        obs_timestamp = self.latest_obs_timestamp
                        self.last_processed_timestamp = obs_timestamp
                    else:
                        raw_images = None
            
            if raw_images is None:
                time.sleep(0.001)
                continue
            
            t_inference_start = time.perf_counter()
            try:
                # 图像预处理: resize
                t0 = time.perf_counter()
                front_img = Image.fromarray(raw_images['front']).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
                left_img = Image.fromarray(raw_images['left']).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
                right_img = Image.fromarray(raw_images['right']).resize((self.image_size, self.image_size), resample=Image.BILINEAR)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_resize', t1 - t0)
                
                # ToTensor
                t0 = time.perf_counter()
                front_t = self.img_transform(front_img).unsqueeze(0)
                left_t = self.img_transform(left_img).unsqueeze(0)
                right_t = self.img_transform(right_img).unsqueeze(0)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_totensor', t1 - t0)
                
                # .to(device)
                t0 = time.perf_counter()
                front_tensor = front_t.to(self.device)
                left_tensor = left_t.to(self.device)
                right_tensor = right_t.to(self.device)
                state_tensor = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_todevice', t1 - t0)
                
                # 构建 batch
                batch = {
                    'observation.state': state_tensor,
                    'observation.images.front': front_tensor,
                    'observation.images.left': left_tensor,
                    'observation.images.right': right_tensor,
                    'task': task
                }
                
                # 执行推理
                with torch.no_grad():
                    t0 = time.perf_counter()
                    batch = self.policy.normalize_inputs(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('normalize_inputs', t1 - t0)
                    
                    t0 = time.perf_counter()
                    images, img_masks = self.policy.prepare_images(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_images', t1 - t0)
                    
                    t0 = time.perf_counter()
                    state_processed = self.policy.prepare_state(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_state', t1 - t0)
                    
                    t0 = time.perf_counter()
                    lang_tokens, lang_masks = self.policy.prepare_language(batch)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('prepare_language', t1 - t0)
                    
                    t0 = time.perf_counter()
                    actions = self.policy.model.sample_actions(
                        images, img_masks, lang_tokens, lang_masks, state_processed
                    )
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('sample_actions', t1 - t0)
                    
                    t0 = time.perf_counter()
                    original_action_dim = self.policy.config.action_feature.shape[0]
                    actions = actions[:, :, :original_action_dim]
                    actions = self.policy.unnormalize_outputs({"action": actions})["action"]
                    
                    if self.policy.config.adapt_to_pi_aloha:
                        actions = self.policy._pi_aloha_encode_actions(actions)
                    t1 = time.perf_counter()
                    if self.perf_monitor:
                        self.perf_monitor.record_inference('postprocess', t1 - t0)
                
                chunk_np = actions[0].cpu().numpy()
                t_inference_end = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('total_inference', t_inference_end - t_inference_start)
                
                with self.lock:
                    self.latest_action_chunk = chunk_np
                    self.chunk_start_timestamp = obs_timestamp
                    self.current_step_index = 0
                    self.chunk_id += 1
                    
            except Exception as e:
                print(f"[BASE] Inference Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)


# ==========================================
# 🔧 辅助函数
# ==========================================
def get_default_transform():
    """返回标准图像变换"""
    return transforms.Compose([transforms.ToTensor()])


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


def plot_task_stats(stats, output_dir, success_count=0, fail_count=0, version_suffix=''):
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
            plt.xlabel(f'Grasp Center Y (Target Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m)')
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
        plt.xlabel(f'Grasp Center Y (Target Cup Y + {EXPERT_Y_GRASP_OFFSET:.3f}m)')
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


def load_arm_policy(device):
    """加载 Arm 模式的 pi0 模型"""
    print("\n" + "="*60)
    print("🤖 Loading ARM Policy...")
    print(f"   Dataset: {ARM_CONFIG['dataset_repo_id']}")
    print(f"   Model: {ARM_CONFIG['model_path']}")
    print("="*60)
    
    try:
        dataset_metadata = LeRobotDatasetMetadata(
            ARM_CONFIG['dataset_repo_id'], 
            root=ARM_CONFIG['dataset_root']
        )
        
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        cfg = PI0Config(
            input_features=input_features, 
            output_features=output_features, 
            chunk_size=ARM_CONFIG['chunk_size'], 
            n_action_steps=ARM_CONFIG['n_action_steps']
        )
        
        policy = PI0Policy.from_pretrained(
            ARM_CONFIG['model_path'], 
            config=cfg, 
            dataset_stats=dataset_metadata.stats
        )
        
        policy.to(device)
        policy.eval()
        print("✅ ARM Policy Loaded Successfully!")
        return policy
        
    except Exception as e:
        print(f"❌ Failed to load ARM policy: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_base_policy(device):
    """加载 Base 模式的 pi0 模型"""
    print("\n" + "="*60)
    print("🚗 Loading BASE Policy...")
    print(f"   Dataset: {BASE_CONFIG['dataset_repo_id']}")
    print(f"   Model: {BASE_CONFIG['model_path']}")
    print("="*60)
    
    try:
        dataset_metadata = LeRobotDatasetMetadata(
            BASE_CONFIG['dataset_repo_id'], 
            root=BASE_CONFIG['dataset_root']
        )
        
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        cfg = PI0Config(
            input_features=input_features, 
            output_features=output_features, 
            chunk_size=BASE_CONFIG['chunk_size'], 
            n_action_steps=BASE_CONFIG['n_action_steps']
        )
        
        policy = PI0Policy.from_pretrained(
            BASE_CONFIG['model_path'], 
            config=cfg, 
            dataset_stats=dataset_metadata.stats
        )
        
        policy.to(device)
        policy.eval()
        print("✅ BASE Policy Loaded Successfully!")
        return policy
        
    except Exception as e:
        print(f"❌ Failed to load BASE policy: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# 主程序
# ==========================================
def main():
    global BASE_FORWARD_SPEED_SCALE_ENABLED
    global BASE_FORWARD_SPEED_SCALE
    _validate_instruction_groups()

    print("\n" + "="*70)
    print("🎮 V6 DUAL-MODE DEPLOYMENT (ARM + BASE)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 打印加载配置
    print(f"\n📋 Model Loading Configuration:")
    print(f"   LOAD_ARM_MODEL: {LOAD_ARM_MODEL}")
    print(f"   LOAD_BASE_MODEL: {LOAD_BASE_MODEL}")
    print(f"   ARM_INFERENCE_MODE: {ARM_INFERENCE_MODE} ({'同步推理' if ARM_SYNC_INFERENCE else '异步推理'})")
    print(f"   CHUNK_THRESHOLD: {CHUNK_THRESHOLD}")
    print(f"   ARM_EXEC_HORIZON: {ARM_EXEC_HORIZON} (SYNC only)")
    print(f"   ACTION_HORIZON: {ACTION_HORIZON} (ASYNC only)")
    print(f"   BASE_ACTION_HORIZON: {BASE_ACTION_HORIZON} (ASYNC only)")
    print(f"   BASE_CHUNK_THRESHOLD: {BASE_CHUNK_THRESHOLD}")
    print(
        f"   BASE_POSTPROC: {'Enabled' if BASE_POSTPROC_ENABLED else 'Disabled'} "
        f"(heading_hold={'On' if BASE_POSTPROC_HEADING_HOLD_ENABLED else 'Off'}, "
        f"kp={BASE_POSTPROC_KP_YAW:.2f}, max_turn={BASE_POSTPROC_MAX_TURN_V:.2f})"
    )
    if BASE_FORWARD_SPEED_SCALE_ENABLED:
        base_scale_status = f"ON ({BASE_FORWARD_SPEED_SCALE:.2f}x)"
    else:
        base_scale_status = "OFF"
    print(f"   BASE_FORWARD_SPEED_SCALE: {base_scale_status}")
    print(f"   ARM_PILOT_RUN_MODE: {ARM_PILOT_RUN_MODE} ({'简单模式' if ARM_PILOT_RUN_MODE else '困难模式'}) [已废弃]")
    print(f"   RANDOM_INIT_ENABLED: {RANDOM_INIT_ENABLED} ({'关闭' if RANDOM_INIT_ENABLED == 0 else 'V1 (扇形区域)' if RANDOM_INIT_ENABLED == 1 else 'V2 (圆形交集)'})")
    print(f"   RANDOM_INIT_GRIPPER_OPEN: {RANDOM_INIT_GRIPPER_OPEN}")
    print(
        f"   TB3_X_GAUSSIAN_ENABLED: {TB3_X_GAUSSIAN_ENABLED} "
        f"(center={TB3_X_CENTER:.3f}, std={TB3_X_OFFSET_STD:.3f}, "
        f"range=[{TB3_X_OFFSET_MIN:+.3f}, {TB3_X_OFFSET_MAX:+.3f}])"
    )
    print(f"\n📋 Task Configuration:")
    print(f"   TASK_TIMEOUT_SEC: {TASK_TIMEOUT_SEC}s")
    print(f"   TASK_LOOP_COUNT: {TASK_LOOP_COUNT} ({'无限循环' if TASK_LOOP_COUNT == 0 else f'执行 {TASK_LOOP_COUNT} 次后退出'})")
    print(f"   TASK_STATS_OUTPUT_DIR: {TASK_STATS_OUTPUT_DIR}")

    # 1. 根据配置加载模型
    arm_policy = None
    base_policy = None
    
    if LOAD_ARM_MODEL:
        arm_policy = load_arm_policy(device)
    else:
        print("\n⏭️  Skipping ARM model loading (LOAD_ARM_MODEL=False)")
    
    if LOAD_BASE_MODEL:
        base_policy = load_base_policy(device)
    else:
        print("\n⏭️  Skipping BASE model loading (LOAD_BASE_MODEL=False)")
    
    if arm_policy is None and base_policy is None:
        print("❌ Both policies failed to load or were disabled. Exiting.")
        return
    
    # 2. 初始化环境 (使用 y7 场景)
    print("\n" + "="*60)
    print("🌍 Initializing MuJoCo Environment (V6)...")
    print("="*60)
    
    xml_path = './asset/example_scene_y7.xml'
    # 🔥 使用 joint_angle 模式，直接支持模型输出的绝对关节角度
    # 手动控制时，teleop_robot 返回 eef_pose 增量，需要临时切换 action_type
    # 🔥 修改：使用 random_init_enabled 和 random_init_gripper_open 参数（匹配 y_env6.py）
    PnPEnv = SimpleEnv7(
        xml_path, 
        action_type='joint_angle',  # 🔥 改为 joint_angle，直接支持绝对关节角度
        state_type='joint_angle',
        random_init_enabled=RANDOM_INIT_ENABLED,
        random_init_gripper_open=RANDOM_INIT_GRIPPER_OPEN,
        tb3_x_gaussian_enabled=TB3_X_GAUSSIAN_ENABLED,
        tb3_x_center=TB3_X_CENTER,
        tb3_x_offset_std=TB3_X_OFFSET_STD,
        tb3_x_offset_min=TB3_X_OFFSET_MIN,
        tb3_x_offset_max=TB3_X_OFFSET_MAX,
    )
    
    # 初始模式设为 arm
    control_mode = 'arm'
    PnPEnv.reset(mode=control_mode)
    instruction_group_indices = {'arm': 0, 'base': 0}
    last_instruction_by_mode = {'arm': None, 'base': None}
    _apply_instruction_from_group(
        PnPEnv,
        control_mode,
        instruction_group_indices,
        last_instruction_by_mode,
        log_prefix="[INIT]",
        reinitialize_arm=(control_mode == 'arm'),
    )

    # 3. 初始化图像预处理
    IMG_TRANSFORM = get_default_transform()
    
    # 4. 初始化推理器
    arm_runner = None
    base_runner = None
    
    if arm_policy is not None:
        if ARM_SYNC_INFERENCE:
            print("🔄 [ARM] Using SYNC inference mode")
        else:
            arm_runner = AsyncArmInferenceRunner(
                arm_policy, device, IMG_TRANSFORM,
                control_dt=CONTROL_DT, perf_monitor=None
            )
            print("🔄 [ARM] Using ASYNC inference mode")
            print(f"   - ACTION_HORIZON: {ACTION_HORIZON}")
            print(f"   - CHUNK_THRESHOLD: {CHUNK_THRESHOLD}")
    
    if base_policy is not None:
        base_runner = AsyncBaseInferenceRunner(
            base_policy, device, IMG_TRANSFORM, 
            control_dt=CONTROL_DT, perf_monitor=None  # Base 模式不使用性能监控
        )
        print("🔄 [BASE] Using ASYNC inference mode")
        print(f"   - BASE_ACTION_HORIZON: {BASE_ACTION_HORIZON}")
        print(f"   - BASE_CHUNK_THRESHOLD: {BASE_CHUNK_THRESHOLD}")
    
    # 6. 初始化动作平滑器（防颤抖）
    arm_smoother = ActionSmoother(
        joint_dim=6, 
        alpha_joints=SMOOTHING_ALPHA_JOINTS, 
        alpha_gripper=SMOOTHING_ALPHA_GRIPPER
    )
    base_postproc = BaseActionPostProcessor()
    print(f"🔧 [ARM] Action Smoother initialized:")
    print(f"   - Smoothing Enabled: {SMOOTHING_ENABLED}")
    print(f"   - Joint Alpha: {SMOOTHING_ALPHA_JOINTS}")
    print(f"   - Gripper Alpha: {SMOOTHING_ALPHA_GRIPPER}")
    print(f"   - Gripper Hysteresis: {GRIPPER_HYSTERESIS_ENABLED} (open>{GRIPPER_OPEN_THRESH}, close<{GRIPPER_CLOSE_THRESH})")
    
    # 控制状态
    auto_mode_arm = False   # arm 模式下是否启用自动控制
    auto_mode_base = False  # base 模式下是否启用自动控制
    # 同步推理 chunk 缓存：每次推理输出64步，仅执行前 ARM_EXEC_HORIZON 步
    arm_action_chunk = None
    arm_chunk_step_index = 0
    auto_check_enabled = False  # 🔥 是否启用自动检测成功（按L键开启）
    step = 0
    # 🔥 任务计数（仅在按下L键开启自动检测后才开始统计）
    task_completed_count = 0  # 已完成的任务计数（成功或失败都算）
    task_success_count = 0    # 成功任务计数
    task_fail_count = 0        # 失败任务计数

    # 任务统计（成功任务的目标杯子初始坐标）
    task_stats = {
        'red': {'x': [], 'y': [], 'z': []},
        'blue': {'x': [], 'y': [], 'z': []},
        'grasp_center_y': [],
    }
    tb3_init_stats = {'x': [], 'y': [], 'z': []}

    def get_target_cup_init(env):
        """获取目标杯子的初始坐标和颜色"""
        target_color = getattr(env, 'target_color', None)
        if target_color is None:
            instruction = getattr(env, 'instruction', '')
            if 'red' in instruction.lower():
                target_color = 'red'
            elif 'blue' in instruction.lower():
                target_color = 'blue'

        if not hasattr(env, 'obj_init_pose') or env.obj_init_pose is None:
            return target_color, np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        if target_color == 'blue':
            cup_init = np.array(env.obj_init_pose[3:6], dtype=np.float32)
        else:
            target_color = 'red'
            cup_init = np.array(env.obj_init_pose[0:3], dtype=np.float32)
        return target_color, cup_init

    def get_tb3_init_pose(env):
        """获取当前任务对应的 TB3 初始坐标"""
        try:
            p_tb3 = env.env.get_p_body('tb3_base')
            return np.array(p_tb3[:3], dtype=np.float32)
        except Exception:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    task_start_time = time.time()
    task_start_step = step
    task_tb3_init = get_tb3_init_pose(PnPEnv)

    def reset_task_timer():
        nonlocal task_start_time, task_start_step, task_tb3_init
        task_start_time = time.time()
        task_start_step = step
        task_tb3_init = get_tb3_init_pose(PnPEnv)
    
    # 打印操作指南
    print("\n" + "="*70)
    print("🎮 V6 DUAL-MODE READY")
    print("="*70)
    print("Controls:")
    print("  [C] Switch between ARM/BASE mode (no env reset)")
    print("  [N] Start PI0 Auto Control (current mode)")
    print("  [M] Switch to Manual Control")
    print("  [Z] Reset Environment")
    print("  [K] Cycle BASE speed scaling: OFF -> 0.75x -> 0.50x")
    print("  [←]/[→] Switch Instruction Group (current mode only)")
    print("  [L] 🔥 Toggle Auto-Control + Auto-Detection (ARM mode only)")
    print("      → Press once to ENABLE: model auto-execute + auto-check success/fail + auto-reset")
    print("      → Press again to DISABLE")
    print("  [Q] Quit")
    print("="*70 + "\n")
    print(f"🎯 Current Mode: {control_mode.upper()}")

    try:
        while PnPEnv.env.is_viewer_alive():
            # [A] 物理环境步进
            PnPEnv.step_env()
            
            # [B] 控制循环
            if PnPEnv.env.loop_every(HZ=CONTROL_FREQUENCY):
                current_time = time.time()
                
                # --- 键位处理 ---
                
                # C 键：切换 arm/base 模式（不重置环境，支持串行接力）
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_C):
                    # 停止当前运行的推理
                    if control_mode == 'arm' and auto_mode_arm:
                        auto_mode_arm = False
                        auto_check_enabled = False  # 🔥 关闭自动检测
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                    elif control_mode == 'base' and auto_mode_base and base_runner:
                        base_runner.stop()
                        auto_mode_base = False
                    
                    # 🔥 重置 runner/policy 状态（清除旧数据）
                    if arm_policy:
                        arm_policy.reset()
                    arm_action_chunk = None
                    arm_chunk_step_index = 0
                    if arm_runner:
                        arm_runner.reset_state()
                    if base_runner:
                        base_runner.reset_state()
                    arm_smoother.reset()  # 🔧 重置平滑器
                    base_postproc.reset()
                    
                    # 切换模式
                    if control_mode == 'arm':
                        control_mode = 'base'
                        auto_check_enabled = False  # 🔥 Base模式不支持自动检测
                    else:
                        control_mode = 'arm'

                    # 同步环境模式并刷新任务文本（不重置环境）
                    PnPEnv.control_mode = control_mode
                    _apply_instruction_from_group(
                        PnPEnv,
                        control_mode,
                        instruction_group_indices,
                        last_instruction_by_mode,
                        log_prefix="🔄 [MODE SWITCH]",
                    )
                    
                    print(f"\n🔄 Mode Switched to: {control_mode.upper()} (env preserved)")
                    print(f"   Task: {PnPEnv.instruction}")

                # 左右键：切换“当前模式”的指令组（arm/base 严格隔离）
                left_pressed = PnPEnv.env.is_key_pressed_once(key=glfw.KEY_LEFT)
                right_pressed = PnPEnv.env.is_key_pressed_once(key=glfw.KEY_RIGHT)
                if left_pressed or right_pressed:
                    groups = INSTRUCTION_GROUPS[control_mode]
                    if len(groups) <= 1:
                        idx, total, group_name, _ = _get_group_info(control_mode, instruction_group_indices)
                        print(f"\nℹ️ [{control_mode.upper()}] Only one instruction group: {group_name} ({idx + 1}/{total})")
                    else:
                        if right_pressed:
                            instruction_group_indices[control_mode] = (instruction_group_indices[control_mode] + 1) % len(groups)
                        else:
                            instruction_group_indices[control_mode] = (instruction_group_indices[control_mode] - 1) % len(groups)
                        _apply_instruction_from_group(
                            PnPEnv,
                            control_mode,
                            instruction_group_indices,
                            last_instruction_by_mode,
                            log_prefix="🔁 [GROUP SWITCH]",
                        )
                
                # N 键：启动自动控制
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_N):
                    if control_mode == 'arm':
                        if arm_policy is not None and not auto_mode_arm:
                            auto_mode_arm = True
                            arm_smoother.reset()  # 🔧 重置平滑器
                            arm_policy.reset()    # 🔥 重置 policy 状态
                            arm_action_chunk = None
                            arm_chunk_step_index = 0
                            if not ARM_SYNC_INFERENCE and arm_runner:
                                arm_runner.start()
                            mode_str = "SYNC" if ARM_SYNC_INFERENCE else "ASYNC"
                            print(f"\n🤖 [ARM] PI0 Auto Control Started! (Mode: {mode_str})")
                        elif arm_policy is None:
                            print("\n⚠️ ARM policy not loaded!")
                    else:  # base mode
                        if base_runner is not None and not auto_mode_base:
                            auto_mode_base = True
                            base_postproc.reset()
                            base_runner.start()
                            print("\n🚗 [BASE] PI0 Auto Control Started!")
                        elif base_runner is None:
                            print("\n⚠️ BASE policy not loaded!")
                
                # M 键：恢复手动控制
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_M):
                    if control_mode == 'arm' and auto_mode_arm:
                        arm_smoother.reset()  # 🔧 重置平滑器
                        auto_mode_arm = False
                        auto_check_enabled = False  # 🔥 手动控制时关闭自动检测
                        arm_action_chunk = None
                        arm_chunk_step_index = 0
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                            arm_runner.reset_state()
                        print("\n👤 [ARM] Switched to Manual Control")
                    elif control_mode == 'base' and auto_mode_base:
                        if base_runner:
                            base_runner.stop()
                            base_runner.reset_state()  # 🔥 清除旧数据
                        base_postproc.reset()
                        auto_mode_base = False
                        print("\n👤 [BASE] Switched to Manual Control")
                
                # Z 键：重置环境
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_Z):
                    # 停止自动控制
                    if auto_mode_arm:
                        auto_mode_arm = False
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                    if auto_mode_base and base_runner:
                        base_runner.stop()
                        auto_mode_base = False
                    
                    # 🔥 关闭自动检测（手动重置时）
                    if auto_check_enabled:
                        auto_check_enabled = False
                    
                    # 🔥 重置 runner/policy 状态（清除旧数据）
                    if arm_policy:
                        arm_policy.reset()
                    arm_action_chunk = None
                    arm_chunk_step_index = 0
                    if arm_runner:
                        arm_runner.reset_state()
                    if base_runner:
                        base_runner.reset_state()
                    arm_smoother.reset()  # 🔧 重置平滑器
                    base_postproc.reset()
                    
                    _apply_instruction_from_group(
                        PnPEnv,
                        control_mode,
                        instruction_group_indices,
                        last_instruction_by_mode,
                        log_prefix="🔄 [RESET]",
                        reinitialize_arm=(control_mode == 'arm'),
                    )
                    if control_mode != 'arm':
                        PnPEnv.reset(mode=control_mode)
                    step = 0
                    reset_task_timer()
                    print(f"\n🔄 Environment Reset. Mode: {control_mode.upper()}")
                    print(f"   Task: {PnPEnv.instruction}")

                # K 键：循环 Base 轮速缩放档位（OFF -> 0.75x -> 0.50x）
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_K):
                    if not BASE_FORWARD_SPEED_SCALE_ENABLED:
                        BASE_FORWARD_SPEED_SCALE_ENABLED = True
                        BASE_FORWARD_SPEED_SCALE = 0.75
                    elif abs(BASE_FORWARD_SPEED_SCALE - 0.75) < 1e-6:
                        BASE_FORWARD_SPEED_SCALE = 0.5
                    else:
                        BASE_FORWARD_SPEED_SCALE_ENABLED = False

                    if BASE_FORWARD_SPEED_SCALE_ENABLED:
                        print(f"\n⚙️ [BASE] Forward speed scaling: ON ({BASE_FORWARD_SPEED_SCALE:.2f}x)")
                    else:
                        print("\n⚙️ [BASE] Forward speed scaling: OFF")

                # L 键：开启/关闭自动检测+自动控制功能（开关）
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_L):
                    if control_mode == 'arm':
                        if not auto_check_enabled:
                            # 开启自动检测 + 自动控制
                            if arm_policy is None:
                                print("\n⚠️ [L] ARM policy not loaded! Cannot start auto control.")
                            else:
                                auto_check_enabled = True
                                auto_mode_arm = True
                                arm_smoother.reset()  # 🔧 重置平滑器
                                arm_policy.reset()    # 🔥 重置 policy 状态
                                arm_action_chunk = None
                                arm_chunk_step_index = 0
                                if not ARM_SYNC_INFERENCE and arm_runner:
                                    arm_runner.start()
                                reset_task_timer()  # 重置任务计时器
                                print("\n✅ [L] Auto-control + Auto-detection ENABLED!")
                                print("   → Model will auto-execute tasks, check success/fail, and auto-reset.")
                        else:
                            # 关闭自动检测 + 自动控制
                            auto_check_enabled = False
                            auto_mode_arm = False
                            arm_smoother.reset()  # 🔧 重置平滑器
                            arm_action_chunk = None
                            arm_chunk_step_index = 0
                            if not ARM_SYNC_INFERENCE and arm_runner:
                                arm_runner.stop()
                                arm_runner.reset_state()
                            print("\n⏸️ [L] Auto-control + Auto-detection DISABLED.")
                    else:
                        print("\n⚠️ [L] Auto-control + Auto-detection only available in ARM mode.")

                # --- 控制逻辑 ---
                
                if control_mode == 'arm':
                    # 🔥 自动检测成功 + 超时判定（仅在L键开启后生效）
                    if auto_check_enabled:
                        elapsed = time.time() - task_start_time
                        # 自动检测成功
                        if PnPEnv.check_success():
                            target_color, cup_init = get_target_cup_init(PnPEnv)
                            append_step_log(STEP_LOG_PATH, control_mode, 'success',
                                            step - task_start_step, target_color, cup_init)
                            task_stats[target_color]['x'].append(float(cup_init[0]))
                            task_stats[target_color]['y'].append(float(cup_init[1]))
                            task_stats[target_color]['z'].append(float(cup_init[2]))
                            task_stats['grasp_center_y'].append(float(cup_init[1] + EXPERT_Y_GRASP_OFFSET))
                            if np.all(np.isfinite(task_tb3_init)):
                                tb3_init_stats['x'].append(float(task_tb3_init[0]))
                                tb3_init_stats['y'].append(float(task_tb3_init[1]))
                                tb3_init_stats['z'].append(float(task_tb3_init[2]))
                            task_completed_count += 1
                            task_success_count += 1
                            success_rate = (task_success_count / task_completed_count * 100) if task_completed_count > 0 else 0.0
                            print(f"\n✅ Task SUCCESS (Auto-detected). Task {task_completed_count}/{TASK_LOOP_COUNT if TASK_LOOP_COUNT > 0 else '∞'} | Success: {task_success_count}/{task_completed_count} ({success_rate:.1f}%). Resetting for next task...")
                            # 🔥 确保环境的随机初始化参数被正确设置（匹配数据采集脚本）
                            PnPEnv.random_init_enabled = RANDOM_INIT_ENABLED
                            PnPEnv.random_init_gripper_open = RANDOM_INIT_GRIPPER_OPEN
                            PnPEnv.tb3_x_gaussian_enabled = TB3_X_GAUSSIAN_ENABLED
                            PnPEnv.tb3_x_center = TB3_X_CENTER
                            PnPEnv.tb3_x_offset_std = TB3_X_OFFSET_STD
                            PnPEnv.tb3_x_offset_min = TB3_X_OFFSET_MIN
                            PnPEnv.tb3_x_offset_max = TB3_X_OFFSET_MAX
                            # 🔥 异步模式下先停推理线程，避免旧观测在 reset 后写回旧动作 chunk
                            if not ARM_SYNC_INFERENCE and arm_runner:
                                arm_runner.stop()
                            # 🔥 重置policy和平滑器状态，确保不会使用旧状态
                            if arm_policy:
                                arm_policy.reset()
                            arm_action_chunk = None
                            arm_chunk_step_index = 0
                            if arm_runner:
                                arm_runner.reset_state()
                            arm_smoother.reset()
                            _apply_instruction_from_group(
                                PnPEnv,
                                control_mode,
                                instruction_group_indices,
                                last_instruction_by_mode,
                                log_prefix="🔄 [AUTO-RESET]",
                                reinitialize_arm=(control_mode == 'arm'),
                            )
                            if control_mode != 'arm':
                                PnPEnv.reset(mode=control_mode)
                            # 🔥 reset 完成后重启异步推理线程，确保只消费新环境观测
                            if not ARM_SYNC_INFERENCE and arm_runner and auto_mode_arm:
                                arm_runner.start()
                            step = 0
                            reset_task_timer()
                            # 🔥 检查是否达到循环次数
                            if TASK_LOOP_COUNT > 0 and task_completed_count >= TASK_LOOP_COUNT:
                                print(f"\n🎯 Reached target task count ({TASK_LOOP_COUNT}). Exiting...")
                                break
                        # 超时判定
                        elif elapsed >= TASK_TIMEOUT_SEC:
                            # 超时失败：只记录失败日志，不记录初始坐标
                            ensure_step_log_header(STEP_LOG_PATH)
                            with open(STEP_LOG_PATH, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    time.strftime('%Y-%m-%d %H:%M:%S'),
                                    control_mode, 'fail', step - task_start_step,
                                    'unknown', 'nan', 'nan', 'nan'  # 不记录初始坐标
                                ])
                            task_completed_count += 1
                            task_fail_count += 1
                            success_rate = (task_success_count / task_completed_count * 100) if task_completed_count > 0 else 0.0
                            print(f"\n⏱️ Task TIMEOUT ({TASK_TIMEOUT_SEC}s). Task {task_completed_count}/{TASK_LOOP_COUNT if TASK_LOOP_COUNT > 0 else '∞'} | Success: {task_success_count}/{task_completed_count} ({success_rate:.1f}%). Resetting for next task...")
                            # 🔥 确保环境的随机初始化参数被正确设置（匹配数据采集脚本）
                            PnPEnv.random_init_enabled = RANDOM_INIT_ENABLED
                            PnPEnv.random_init_gripper_open = RANDOM_INIT_GRIPPER_OPEN
                            PnPEnv.tb3_x_gaussian_enabled = TB3_X_GAUSSIAN_ENABLED
                            PnPEnv.tb3_x_center = TB3_X_CENTER
                            PnPEnv.tb3_x_offset_std = TB3_X_OFFSET_STD
                            PnPEnv.tb3_x_offset_min = TB3_X_OFFSET_MIN
                            PnPEnv.tb3_x_offset_max = TB3_X_OFFSET_MAX
                            # 🔥 异步模式下先停推理线程，避免旧观测在 reset 后写回旧动作 chunk
                            if not ARM_SYNC_INFERENCE and arm_runner:
                                arm_runner.stop()
                            # 🔥 重置policy和平滑器状态，确保不会使用旧状态
                            if arm_policy:
                                arm_policy.reset()
                            arm_action_chunk = None
                            arm_chunk_step_index = 0
                            if arm_runner:
                                arm_runner.reset_state()
                            arm_smoother.reset()
                            _apply_instruction_from_group(
                                PnPEnv,
                                control_mode,
                                instruction_group_indices,
                                last_instruction_by_mode,
                                log_prefix="🔄 [AUTO-RESET]",
                                reinitialize_arm=(control_mode == 'arm'),
                            )
                            if control_mode != 'arm':
                                PnPEnv.reset(mode=control_mode)
                            # 🔥 reset 完成后重启异步推理线程，确保只消费新环境观测
                            if not ARM_SYNC_INFERENCE and arm_runner and auto_mode_arm:
                                arm_runner.start()
                            step = 0
                            reset_task_timer()
                            # 🔥 检查是否达到循环次数
                            if TASK_LOOP_COUNT > 0 and task_completed_count >= TASK_LOOP_COUNT:
                                print(f"\n🎯 Reached target task count ({TASK_LOOP_COUNT}). Exiting...")
                                break

                    if auto_mode_arm and arm_policy is not None:
                        # Arm 自动控制模式
                        # 1. 收集观测数据
                        state = PnPEnv.get_joint_state()  # (7,) 包含夹爪状态
                        images_dict = PnPEnv.grab_image()  # {'agent', 'wrist'}

                        action_step = None
                        if ARM_SYNC_INFERENCE:
                            # ========== 同步推理（按 chunk 执行） ==========
                            # 每次推理得到 n_action_steps（当前64步），仅执行前 ARM_EXEC_HORIZON 步
                            agent_img = Image.fromarray(images_dict['agent']).resize(
                                (ARM_CONFIG['image_size'], ARM_CONFIG['image_size']),
                                resample=Image.BILINEAR
                            )
                            wrist_img = Image.fromarray(images_dict['wrist']).resize(
                                (ARM_CONFIG['image_size'], ARM_CONFIG['image_size']),
                                resample=Image.BILINEAR
                            )
                            agent_tensor = IMG_TRANSFORM(agent_img).unsqueeze(0).to(device)
                            wrist_tensor = IMG_TRANSFORM(wrist_img).unsqueeze(0).to(device)

                            data = {
                                'observation.state': torch.tensor([state], dtype=torch.float32).to(device),
                                'observation.images.agent': agent_tensor,
                                'observation.images.wrist': wrist_tensor,
                                'task': [PnPEnv.instruction],
                            }

                            need_new_chunk = arm_action_chunk is None
                            if not need_new_chunk:
                                current_horizon = min(ARM_EXEC_HORIZON, arm_action_chunk.shape[0])
                                need_new_chunk = arm_chunk_step_index >= current_horizon

                            if need_new_chunk:
                                # 不用 select_action()，避免其内部 action queue 导致“64步全消耗后才重推理”
                                with torch.no_grad():
                                    batch = arm_policy.normalize_inputs(data)
                                    images, img_masks = arm_policy.prepare_images(batch)
                                    state_processed = arm_policy.prepare_state(batch)
                                    lang_tokens, lang_masks = arm_policy.prepare_language(batch)

                                    actions = arm_policy.model.sample_actions(
                                        images, img_masks, lang_tokens, lang_masks, state_processed
                                    )

                                    original_action_dim = arm_policy.config.action_feature.shape[0]
                                    actions = actions[:, :, :original_action_dim]
                                    actions = arm_policy.unnormalize_outputs({"action": actions})["action"]

                                    if arm_policy.config.adapt_to_pi_aloha:
                                        actions = arm_policy._pi_aloha_encode_actions(actions)

                                action_np = actions.detach().cpu().numpy()
                                if action_np.ndim == 3:
                                    chunk_np = action_np[0]
                                elif action_np.ndim == 2:
                                    chunk_np = action_np
                                elif action_np.ndim == 1:
                                    chunk_np = action_np[None, :]
                                else:
                                    raise RuntimeError(f"Unexpected action tensor shape: {action_np.shape}")

                                if chunk_np.shape[0] > 0:
                                    arm_action_chunk = chunk_np[:, :7]
                                    arm_chunk_step_index = 0

                            if arm_action_chunk is not None and arm_action_chunk.shape[0] > 0:
                                current_horizon = min(ARM_EXEC_HORIZON, arm_action_chunk.shape[0])
                                if arm_chunk_step_index < current_horizon:
                                    action_step = arm_action_chunk[arm_chunk_step_index]
                                    arm_chunk_step_index += 1
                        else:
                            # ========== 异步推理（参考 dual_test） ==========
                            obs_capture_time = time.time()
                            arm_runner.update_observation(images_dict, state, [PnPEnv.instruction], obs_capture_time)
                            action_step, status_msg = arm_runner.get_action_at_time(time.time())
                            if action_step is None:
                                # 没有新动作时保持当前位置，避免突然跳变
                                action_step = state.copy()
                        
                        # 5. 执行动作（使用平滑器）
                        if action_step is not None:
                            # 🔧 使用平滑器处理动作
                            smoothed_action, gripper_state = arm_smoother.smooth_action(action_step)
                            
                            # 🔥 直接使用 step 方法，传入绝对关节角度（7维：[6关节角度 + 1夹爪]）
                            PnPEnv.step(smoothed_action, mode='arm')
                            PnPEnv.gripper_state = gripper_state  # 🔧 使用迟滞控制后的夹爪状态
                        
                        # 6. 更新 p0 和 R0（用于保持 eef_pose 状态同步）
                        PnPEnv.p0, PnPEnv.R0 = PnPEnv.env.get_pR_body(body_name='tcp_link')
                        
                        # 7. 渲染
                        PnPEnv.render(teleop=False, idx=step)
                        
                        # 8. 步数递增
                        step += 1
                        
                        # 9. 打印状态
                        if step % 50 == 0:
                            mode_str = "SYNC" if ARM_SYNC_INFERENCE else "ASYNC"
                            print(f"[ARM-{mode_str}] Step {step} | Task: {PnPEnv.instruction}")
                        
                    else:
                        # Arm 手动控制模式
                        # 🔥 teleop_robot 返回的是 eef_pose 增量，需要临时切换 action_type
                        original_action_type = PnPEnv.action_type
                        PnPEnv.action_type = 'eef_pose'  # 临时切换为 eef_pose 模式
                        
                        action, reset = PnPEnv.teleop_robot(mode='arm')
                        if reset:
                            _apply_instruction_from_group(
                                PnPEnv,
                                'arm',
                                instruction_group_indices,
                                last_instruction_by_mode,
                                log_prefix="🔄 [ARM RESET]",
                                reinitialize_arm=True,
                            )
                            step = 0
                            reset_task_timer()
                        else:
                            PnPEnv.step(action, mode='arm')
                        
                        PnPEnv.action_type = original_action_type  # 恢复为 joint_angle 模式
                        PnPEnv.render(teleop=True, idx=step)
                        step += 1
                
                else:  # base mode
                    if auto_mode_base and base_runner:
                        # Base 自动控制模式
                        
                        # 1. 收集观测数据
                        state = PnPEnv.get_base_state()  # (4,) [轮速 + 朝向sin/cos]
                        images_dict = PnPEnv.grab_image()  # {'front', 'left', 'right'}
                        
                        obs_capture_time = time.time()
                        
                        # 2. 更新观测到推理线程
                        base_runner.update_observation(images_dict, state, [PnPEnv.instruction], obs_capture_time)
                        
                        # 3. 获取动作
                        action_step, status_msg = base_runner.get_action_at_time(time.time())
                        
                        # 4. 执行动作
                        if action_step is not None:
                            # action_step 是 (2,) 数组: 左右轮速度
                            yaw = float(np.arctan2(float(state[2]), float(state[3])))
                            action_step = base_postproc.process(action_step, yaw)
                            PnPEnv.step(action_step, mode='base')
                        else:
                            # 没有可用动作，停止
                            base_postproc.reset()
                            PnPEnv.step(np.array([0.0, 0.0]), mode='base')
                        
                        # 5. 渲染
                        PnPEnv.render(teleop=False, idx=step)
                        
                        # 6. 步数递增
                        step += 1
                        
                        # 7. 打印状态
                        if step % 50 == 0:
                            print(f"[BASE] Step {step} | Task: {PnPEnv.instruction}")
                        
                    else:
                        # Base 手动控制模式
                        action, reset = PnPEnv.teleop_robot(mode='base')
                        if reset:
                            _apply_instruction_from_group(
                                PnPEnv,
                                'base',
                                instruction_group_indices,
                                last_instruction_by_mode,
                                log_prefix="🔄 [BASE RESET]",
                            )
                            PnPEnv.reset(mode='base')
                            base_postproc.reset()
                            if base_runner:
                                base_runner.reset_state()
                            step = 0
                            reset_task_timer()
                        else:
                            PnPEnv.step(action, mode='base')
                        PnPEnv.render(teleop=True, idx=step)
                        step += 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        # 🔥 只有当有统计数据时才输出统计图（仅在arm模式的L键自动检测模式下才有数据）
        total_tasks = task_success_count + task_fail_count
        if total_tasks > 0:
            # 输出成功任务统计图（从 ARM 模型路径提取版本号）
            version_suffix = ''
            if LOAD_ARM_MODEL and ARM_CONFIG.get('model_path'):
                version_suffix = extract_model_version(ARM_CONFIG['model_path'])
            plot_task_stats(task_stats, TASK_STATS_OUTPUT_DIR, task_success_count, task_fail_count, version_suffix=version_suffix)
            plot_tb3_init_stats(tb3_init_stats, TASK_STATS_OUTPUT_DIR, task_success_count, task_fail_count, version_suffix=version_suffix)
            # 打印最终统计信息
            print(f"\n📊 Final Statistics:")
            print(f"   Total Tasks: {total_tasks}")
            print(f"   Success: {task_success_count}")
            print(f"   Fail: {task_fail_count}")
            print(f"   Success Rate: {task_success_count/total_tasks*100:.1f}%")
        else:
            print("\n📊 No task statistics to output (L-key auto-detection mode was not used).")
        # 清理
        if arm_runner and arm_runner.running:
            arm_runner.stop()
        if base_runner and base_runner.running:
            base_runner.stop()
        if PnPEnv.env.viewer:
            PnPEnv.env.close_viewer()
        print("🛑 Environment closed.")


if __name__ == "__main__":
    main()
