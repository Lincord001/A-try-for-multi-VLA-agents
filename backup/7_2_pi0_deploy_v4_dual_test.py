#!/usr/bin/env python3
"""
V4 环境部署脚本 - 支持 Arm 和 Base 双模式异步推理

功能说明：
- 按 C: 切换 arm/base 模式
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

# ==========================================
# 🔥 环境配置
# ==========================================
print("Setting up environment variables for Hugging Face...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'

import threading
import copy
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import glfw

# ==========================================
# 🔧 模型配置
# ==========================================

# Arm 模型配置 (V4.1)
ARM_CONFIG = {
    'model_path': './ckpt/pi0_arm/pretrained_model_arm_v4',
    'dataset_repo_id': 'omy_arm_data_v4',
    'dataset_root': './demo_data_arm_v4',
    'chunk_size': 5,           # 🔥 V4.1: 从 20 改为 5
    'n_action_steps': 5,       # 🔥 V4.1: 从 20 改为 5
    'image_size': 224,  # arm 模型使用 224x224
    'state_dim': 7,            # 🔥 修复：应该是7维 [q1, q2, q3, q4, q5, q6, gripper]，匹配数据采集脚本
    'action_dim': 7,
    'camera_keys': ['agent', 'wrist'],
}

# Base 模型配置 (V3)
BASE_CONFIG = {
    'model_path': './ckpt/pi0_base/pretrained_model_ver_3/pretrained_model',
    'dataset_repo_id': 'omy_base_data_ver_3',
    'dataset_root': './demo_data_base_ver_3',
    'chunk_size': 20,
    'n_action_steps': 20,
    'image_size': 256,  # base 模型使用 256x256
    'state_dim': 2,
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
# ==========================================
# 🔧 ARM 模式推理方式选择
# ==========================================
# ARM_SYNC_INFERENCE:
#   - True:  同步推理（每帧直接调用模型，简单稳定，但会阻塞环境）
#   - False: 异步推理（后台线程推理，主线程不阻塞，环境持续运行）
ARM_SYNC_INFERENCE = False # 🔥 可手动切换：True=同步推理, False=异步推理

# 异步推理参数（仅当 ARM_SYNC_INFERENCE = False 时有效）：
ACTION_HORIZON = 5   # 每个 chunk 实际执行的步数
CHUNK_THRESHOLD = 0  # 当剩余帧数 <= 此值时，开始新推理

# ==========================================
# 🔧 动作平滑配置（防颤抖）
# ==========================================
# EMA 平滑参数：new_action = alpha * raw_action + (1 - alpha) * last_action
# alpha 越小越平滑，但响应越慢；越大越灵敏，但可能颤抖
SMOOTHING_ENABLED = True        # 是否启用动作平滑
SMOOTHING_ALPHA_JOINTS = 0.4    # 关节角度平滑系数 (0.0~1.0)
SMOOTHING_ALPHA_GRIPPER = 0.3   # 夹爪平滑系数 (0.0~1.0)

# 夹爪迟滞控制：防止夹爪在阈值附近频繁切换
GRIPPER_HYSTERESIS_ENABLED = True
GRIPPER_OPEN_THRESH = 0.6       # 超过此值才打开夹爪
GRIPPER_CLOSE_THRESH = 0.4      # 低于此值才关闭夹爪

# ==========================================
# 🔧 模型加载选择
# ==========================================
LOAD_ARM_MODEL = True   # 是否加载 ARM 模型
LOAD_BASE_MODEL = False  # 是否加载 BASE 模型

# ==========================================
# 🔧 Arm 模式难度选择
# ==========================================
# True:  简单模式（Pilot Run Mode）- 只生成红色杯子，位置大范围随机
# False: 困难模式（Normal Mode）- 4个杯子固定位置，小范围随机
ARM_PILOT_RUN_MODE = True  # 🔥 已废弃，保留用于兼容性

# ==========================================
# 🔧 随机初始化配置（匹配 y_env4.py 和 collect_data_v4.py）
# ==========================================
RANDOM_INIT_ENABLED = 0            # 0: 关闭, 1: 旧版(扇形区域), 2: 新版(圆形交集，仅简单模式)
RANDOM_INIT_GRIPPER_OPEN = True    # True: 初始化时夹爪张开, False: 初始化时夹爪闭合

# 导入 LeRobot 和 MuJoCo 环境
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.types import FeatureType
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from mujoco_env.y_env4 import SimpleEnv4
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
        主线程调用：根据当前时间获取正确的动作
        返回: (action, status_msg) 其中 action 是 (2,) 数组或 None
        """
        with self.lock:
            if self.latest_action_chunk is None:
                return None, "Wait for init"
            
            if self.chunk_start_timestamp == 0:
                return None, "Wait for init"
            
            chunk = self.latest_action_chunk
            start_time = self.chunk_start_timestamp
            
        time_delta = current_time - start_time
        step_index = int(time_delta / self.control_dt)
        
        chunk_len = chunk.shape[0]
        
        if step_index < 0:
            return chunk[0], "Sync Error (Future)"
            
        if step_index < chunk_len:
            return chunk[step_index], f"OK (Step {step_index})"
        else:
            return None, "Timeout (Stale Chunk)"

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
                    if self.latest_obs_timestamp > self.last_processed_timestamp:
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
    print("\n" + "="*70)
    print("🎮 V4 DUAL-MODE DEPLOYMENT (ARM + BASE)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 打印加载配置
    print(f"\n📋 Model Loading Configuration:")
    print(f"   LOAD_ARM_MODEL: {LOAD_ARM_MODEL}")
    print(f"   LOAD_BASE_MODEL: {LOAD_BASE_MODEL}")
    print(f"   ARM_SYNC_INFERENCE: {ARM_SYNC_INFERENCE} ({'同步推理' if ARM_SYNC_INFERENCE else '异步推理'})")
    if not ARM_SYNC_INFERENCE:
        print(f"   ACTION_HORIZON: {ACTION_HORIZON}")
        print(f"   CHUNK_THRESHOLD: {CHUNK_THRESHOLD}")
    print(f"   ARM_PILOT_RUN_MODE: {ARM_PILOT_RUN_MODE} ({'简单模式' if ARM_PILOT_RUN_MODE else '困难模式'}) [已废弃]")
    print(f"   RANDOM_INIT_ENABLED: {RANDOM_INIT_ENABLED} ({'关闭' if RANDOM_INIT_ENABLED == 0 else 'V1 (扇形区域)' if RANDOM_INIT_ENABLED == 1 else 'V2 (圆形交集)'})")
    print(f"   RANDOM_INIT_GRIPPER_OPEN: {RANDOM_INIT_GRIPPER_OPEN}")

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
    
    # 2. 初始化环境 (使用 y_env4)
    print("\n" + "="*60)
    print("🌍 Initializing MuJoCo Environment (V4)...")
    print("="*60)
    
    xml_path = './asset/example_scene_y4.xml'
    # 🔥 使用 joint_angle 模式，直接支持模型输出的绝对关节角度
    # 手动控制时，teleop_robot 返回 eef_pose 增量，需要临时切换 action_type
    # 🔥 修改：使用 random_init_enabled 和 random_init_gripper_open 参数（匹配 y_env4.py）
    PnPEnv = SimpleEnv4(
        xml_path, 
        action_type='joint_angle',  # 🔥 改为 joint_angle，直接支持绝对关节角度
        state_type='joint_angle',
        random_init_enabled=RANDOM_INIT_ENABLED,
        random_init_gripper_open=RANDOM_INIT_GRIPPER_OPEN
    )
    
    # 初始模式设为 arm
    control_mode = 'arm'
    PnPEnv.reset(mode=control_mode)

    # 3. 初始化图像预处理
    IMG_TRANSFORM = get_default_transform()
    
    # 4. 初始化推理器
    arm_runner = None
    base_runner = None
    
    if arm_policy is not None:
        if ARM_SYNC_INFERENCE:
            print("🔄 [ARM] Using SYNC inference mode (环境会在推理时暂停)")
        else:
            # 初始化异步推理器
            arm_runner = AsyncArmInferenceRunner(
                arm_policy, device, IMG_TRANSFORM,
                control_dt=CONTROL_DT, perf_monitor=None
            )
            print("🔄 [ARM] Using ASYNC inference mode (环境持续运行，不暂停)")
            print(f"   - ACTION_HORIZON: {ACTION_HORIZON}")
            print(f"   - CHUNK_THRESHOLD: {CHUNK_THRESHOLD}")
    
    if base_policy is not None:
        base_runner = AsyncBaseInferenceRunner(
            base_policy, device, IMG_TRANSFORM, 
            control_dt=CONTROL_DT, perf_monitor=None  # Base 模式不使用性能监控
        )
    
    # 6. 初始化动作平滑器（防颤抖）
    arm_smoother = ActionSmoother(
        joint_dim=6, 
        alpha_joints=SMOOTHING_ALPHA_JOINTS, 
        alpha_gripper=SMOOTHING_ALPHA_GRIPPER
    )
    print(f"🔧 [ARM] Action Smoother initialized:")
    print(f"   - Smoothing Enabled: {SMOOTHING_ENABLED}")
    print(f"   - Joint Alpha: {SMOOTHING_ALPHA_JOINTS}")
    print(f"   - Gripper Alpha: {SMOOTHING_ALPHA_GRIPPER}")
    print(f"   - Gripper Hysteresis: {GRIPPER_HYSTERESIS_ENABLED} (open>{GRIPPER_OPEN_THRESH}, close<{GRIPPER_CLOSE_THRESH})")
    
    # 控制状态
    auto_mode_arm = False   # arm 模式下是否启用自动控制
    auto_mode_base = False  # base 模式下是否启用自动控制
    step = 0
    
    # 打印操作指南
    print("\n" + "="*70)
    print("🎮 V4 DUAL-MODE READY")
    print("="*70)
    print("Controls:")
    print("  [C] Switch between ARM/BASE mode")
    print("  [N] Start PI0 Auto Control (current mode)")
    print("  [M] Switch to Manual Control")
    print("  [Z] Reset Environment")
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
                
                # C 键：切换 arm/base 模式
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_C):
                    # 停止当前运行的推理
                    if control_mode == 'arm' and auto_mode_arm:
                        auto_mode_arm = False
                        # 如果是异步模式，停止推理线程
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                    elif control_mode == 'base' and auto_mode_base and base_runner:
                        base_runner.stop()
                        auto_mode_base = False
                    
                    # 🔥 重置 runner/policy 状态（清除旧数据）
                    if arm_policy:
                        arm_policy.reset()
                    if arm_runner:
                        arm_runner.reset_state()
                    if base_runner:
                        base_runner.reset_state()
                    arm_smoother.reset()  # 🔧 重置平滑器
                    
                    # 切换模式
                    if control_mode == 'arm':
                        control_mode = 'base'
                    else:
                        control_mode = 'arm'
                    
                    # 重置环境
                    PnPEnv.reset(mode=control_mode)
                    step = 0
                    print(f"\n🔄 Mode Switched to: {control_mode.upper()}")
                    print(f"   Task: {PnPEnv.instruction}")
                
                # N 键：启动自动控制
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_N):
                    if control_mode == 'arm':
                        if arm_policy is not None and not auto_mode_arm:
                            auto_mode_arm = True
                            arm_smoother.reset()  # 🔧 重置平滑器
                            arm_policy.reset()    # 🔥 重置 policy 状态
                            # 如果是异步模式，启动推理线程
                            if not ARM_SYNC_INFERENCE and arm_runner:
                                arm_runner.start()
                            print(f"\n🤖 [ARM] PI0 Auto Control Started! (Mode: {'SYNC' if ARM_SYNC_INFERENCE else 'ASYNC'})")
                        elif arm_policy is None:
                            print("\n⚠️ ARM policy not loaded!")
                    else:  # base mode
                        if base_runner is not None and not auto_mode_base:
                            auto_mode_base = True
                            base_runner.start()
                            print("\n🚗 [BASE] PI0 Auto Control Started!")
                        elif base_runner is None:
                            print("\n⚠️ BASE policy not loaded!")
                
                # M 键：恢复手动控制
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_M):
                    if control_mode == 'arm' and auto_mode_arm:
                        arm_smoother.reset()  # 🔧 重置平滑器
                        auto_mode_arm = False
                        # 如果是异步模式，停止推理线程
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                            arm_runner.reset_state()  # 🔥 清除旧数据
                        print("\n👤 [ARM] Switched to Manual Control")
                    elif control_mode == 'base' and auto_mode_base:
                        if base_runner:
                            base_runner.stop()
                            base_runner.reset_state()  # 🔥 清除旧数据
                        auto_mode_base = False
                        print("\n👤 [BASE] Switched to Manual Control")
                
                # Z 键：重置环境
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_Z):
                    # 停止自动控制
                    if auto_mode_arm:
                        auto_mode_arm = False
                        # 如果是异步模式，停止推理线程
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                    if auto_mode_base and base_runner:
                        base_runner.stop()
                        auto_mode_base = False
                    
                    # 🔥 重置 runner/policy 状态（清除旧数据）
                    if arm_policy:
                        arm_policy.reset()
                    if arm_runner:
                        arm_runner.reset_state()
                    if base_runner:
                        base_runner.reset_state()
                    arm_smoother.reset()  # 🔧 重置平滑器
                    
                    PnPEnv.reset(mode=control_mode)
                    step = 0
                    print(f"\n🔄 Environment Reset. Mode: {control_mode.upper()}")
                    print(f"   Task: {PnPEnv.instruction}")

                # --- 控制逻辑 ---
                
                if control_mode == 'arm':
                    if auto_mode_arm and arm_policy is not None:
                        # Arm 自动控制模式
                        
                        if ARM_SYNC_INFERENCE:
                            # ========== 同步推理模式 ==========
                            
                            # 1. 收集观测数据
                            # 🔥 使用完整的7维state [q1, q2, q3, q4, q5, q6, gripper]，匹配数据采集脚本
                            state = PnPEnv.get_joint_state()  # (7,) 包含夹爪状态
                            images_dict = PnPEnv.grab_image()  # {'agent', 'wrist'}
                            
                            # 2. 准备图像输入
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
                            
                            # 3. 准备模型输入
                            # 🔥 注意：state 是 7维，包含夹爪状态
                            data = {
                                'observation.state': torch.tensor([state], dtype=torch.float32).to(device),
                                'observation.images.agent': agent_tensor,
                                'observation.images.wrist': wrist_tensor,
                                'task': [PnPEnv.instruction],
                            }
                            
                            # 4. 同步推理（会阻塞主线程）
                            with torch.no_grad():
                                action_tensor = arm_policy.select_action(data)
                            
                            # 5. 取第一个动作并转换为 numpy
                            action_step = action_tensor[0, :7].cpu().detach().numpy()  # (7,)
                            
                        else:
                            # ========== 异步推理模式 ==========
                            
                            # 1. 收集观测数据
                            state = PnPEnv.get_joint_state()  # (7,) 包含夹爪状态
                            images_dict = PnPEnv.grab_image()  # {'agent', 'wrist'}
                            
                            obs_capture_time = time.time()
                            
                            # 2. 更新观测到推理线程
                            arm_runner.update_observation(images_dict, state, [PnPEnv.instruction], obs_capture_time)
                            
                            # 3. 获取动作（从异步推理器获取，不阻塞）
                            action_step, status_msg = arm_runner.get_action_at_time(time.time())
                            
                            # 🔥 如果获取不到动作，使用当前关节角度（保持当前位置）
                            # 这样可以避免机械臂突然移动到零位置
                            if action_step is None:
                                action_step = state.copy()  # 使用当前状态，保持当前位置
                        
                        # 6. 执行动作（使用平滑器）
                        if action_step is not None:
                            # 🔧 使用平滑器处理动作
                            smoothed_action, gripper_state = arm_smoother.smooth_action(action_step)
                            
                            # 🔥 直接使用 step 方法，传入绝对关节角度（7维：[6关节角度 + 1夹爪]）
                            PnPEnv.step(smoothed_action, mode='arm')
                            PnPEnv.gripper_state = gripper_state  # 🔧 使用迟滞控制后的夹爪状态
                        
                        # 7. 更新 p0 和 R0（用于保持 eef_pose 状态同步）
                        PnPEnv.p0, PnPEnv.R0 = PnPEnv.env.get_pR_body(body_name='tcp_link')
                        
                        # 8. 渲染
                        PnPEnv.render(teleop=False, idx=step)
                        
                        # 9. 步数递增
                        step += 1
                        
                        # 10. 打印状态
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
                            PnPEnv.reset(mode='arm')
                            step = 0
                        else:
                            PnPEnv.step(action, mode='arm')
                        
                        PnPEnv.action_type = original_action_type  # 恢复为 joint_angle 模式
                        PnPEnv.render(teleop=True, idx=step)
                        step += 1
                
                else:  # base mode
                    if auto_mode_base and base_runner:
                        # Base 自动控制模式
                        
                        # 1. 收集观测数据
                        state = PnPEnv.get_base_state()  # (2,) 轮速度
                        images_dict = PnPEnv.grab_image()  # {'front', 'left', 'right'}
                        
                        obs_capture_time = time.time()
                        
                        # 2. 更新观测到推理线程
                        base_runner.update_observation(images_dict, state, [PnPEnv.instruction], obs_capture_time)
                        
                        # 3. 获取动作
                        action_step, status_msg = base_runner.get_action_at_time(time.time())
                        
                        # 4. 执行动作
                        if action_step is not None:
                            # action_step 是 (2,) 数组: 左右轮速度
                            PnPEnv.step(action_step, mode='base')
                        else:
                            # 没有可用动作，停止
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
                            PnPEnv.reset(mode='base')
                            if base_runner:
                                base_runner.reset_state()
                            step = 0
                        else:
                            PnPEnv.step(action, mode='base')
                        PnPEnv.render(teleop=True, idx=step)
                        step += 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        # 清理
        if arm_runner and hasattr(arm_runner, 'running') and arm_runner.running:
            arm_runner.stop()
        if base_runner and hasattr(base_runner, 'running') and base_runner.running:
            base_runner.stop()
        if PnPEnv.env.viewer:
            PnPEnv.env.close_viewer()
        print("🛑 Environment closed.")


if __name__ == "__main__":
    main()
