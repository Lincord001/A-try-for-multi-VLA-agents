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
- 输入: 3个相机(agent/wrist/back, 224x224) + 6维关节角度
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
    'state_dim': 6,
    'action_dim': 7,
    'camera_keys': ['agent', 'wrist', 'back'],
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
# ARM 模式推理方式选择：
#   - True: 同步推理（每帧直接调用模型，简单稳定）
#   - False: 异步推理（后台线程推理，主线程不阻塞）
ARM_SYNC_INFERENCE = True  # 🔥 推荐使用同步推理，更稳定

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
LOAD_ARM_MODEL = False   # 是否加载 ARM 模型
LOAD_BASE_MODEL = True  # 是否加载 BASE 模型

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
    3. Chunk 边界插值：平滑过渡新旧 chunk（异步模式）
    """
    def __init__(self, joint_dim=6, alpha_joints=0.4, alpha_gripper=0.3):
        self.joint_dim = joint_dim
        self.alpha_joints = alpha_joints
        self.alpha_gripper = alpha_gripper
        
        # 历史状态
        self.last_joint_angles = None
        self.last_gripper_cmd = None
        self.gripper_state = False  # True=打开, False=关闭
        
        # Chunk 边界插值（异步模式用）
        self.prev_chunk_last_action = None
        self.current_chunk_id = -1
        self.blend_steps = 3  # 新 chunk 开始时的混合步数
        self.blend_counter = 0
        
    def reset(self):
        """重置平滑器状态（环境重置时调用）"""
        self.last_joint_angles = None
        self.last_gripper_cmd = None
        self.gripper_state = False
        self.prev_chunk_last_action = None
        self.current_chunk_id = -1
        self.blend_counter = 0
        
    def smooth_action(self, raw_action, chunk_id=None):
        """
        平滑处理动作
        
        Args:
            raw_action: 原始动作 (7,) - [6关节角度, 1夹爪]
            chunk_id: 当前 chunk ID（异步模式用于检测边界）
            
        Returns:
            smoothed_action: 平滑后的动作 (7,)
            gripper_state: 夹爪状态 (bool)
        """
        if raw_action is None:
            return None, self.gripper_state
        
        joint_angles = raw_action[:self.joint_dim].copy()
        gripper_cmd = raw_action[self.joint_dim] if len(raw_action) > self.joint_dim else 0.0
        
        # ========== 1. 检测 Chunk 边界（异步模式） ==========
        if chunk_id is not None and chunk_id != self.current_chunk_id:
            # 新 chunk 到来
            if self.last_joint_angles is not None:
                self.prev_chunk_last_action = self.last_joint_angles.copy()
                self.blend_counter = self.blend_steps
            self.current_chunk_id = chunk_id
        
        # ========== 2. Chunk 边界插值 ==========
        if self.blend_counter > 0 and self.prev_chunk_last_action is not None:
            # 在新 chunk 开始的几步内，逐渐从旧动作过渡到新动作
            blend_ratio = self.blend_counter / self.blend_steps  # 从 1.0 递减到接近 0
            joint_angles = (1 - blend_ratio) * joint_angles + blend_ratio * self.prev_chunk_last_action
            self.blend_counter -= 1
        
        # ========== 3. EMA 平滑关节角度 ==========
        if SMOOTHING_ENABLED:
            if self.last_joint_angles is not None:
                joint_angles = (self.alpha_joints * joint_angles + 
                               (1 - self.alpha_joints) * self.last_joint_angles)
            self.last_joint_angles = joint_angles.copy()
        else:
            self.last_joint_angles = joint_angles.copy()
        
        # ========== 4. EMA 平滑夹爪 + 迟滞控制 ==========
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
    
    def get_debug_info(self):
        """获取调试信息"""
        return {
            'blend_counter': self.blend_counter,
            'gripper_state': self.gripper_state,
            'last_gripper_cmd': self.last_gripper_cmd,
        }


# ==========================================
# 🔥 异步推理运行器 - Arm 模式
# ==========================================
class AsyncArmInferenceRunner:
    """
    Arm 模式的异步推理运行器
    - 输入: 3个相机(agent/wrist/back) + 6维关节角度
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
        self.latest_raw_images = None  # {'agent': np.array, 'wrist': np.array, 'back': np.array}
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
                agent_img = Image.fromarray(raw_images['agent']).resize((self.image_size, self.image_size))
                wrist_img = Image.fromarray(raw_images['wrist']).resize((self.image_size, self.image_size))
                back_img = Image.fromarray(raw_images['back']).resize((self.image_size, self.image_size))
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_resize', t1 - t0)
                
                # ToTensor
                t0 = time.perf_counter()
                agent_t = self.img_transform(agent_img).unsqueeze(0)
                wrist_t = self.img_transform(wrist_img).unsqueeze(0)
                back_t = self.img_transform(back_img).unsqueeze(0)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_totensor', t1 - t0)
                
                # .to(device)
                t0 = time.perf_counter()
                agent_tensor = agent_t.to(self.device)
                wrist_tensor = wrist_t.to(self.device)
                back_tensor = back_t.to(self.device)
                state_tensor = torch.tensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                t1 = time.perf_counter()
                if self.perf_monitor:
                    self.perf_monitor.record_inference('img_todevice', t1 - t0)
                
                # 构建 batch
                batch = {
                    'observation.state': state_tensor,
                    'observation.images.agent': agent_tensor,
                    'observation.images.wrist': wrist_tensor,
                    'observation.images.back': back_tensor,
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
                front_img = Image.fromarray(raw_images['front']).resize((self.image_size, self.image_size))
                left_img = Image.fromarray(raw_images['left']).resize((self.image_size, self.image_size))
                right_img = Image.fromarray(raw_images['right']).resize((self.image_size, self.image_size))
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
    print(f"   CHUNK_THRESHOLD: {CHUNK_THRESHOLD}")

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
    # V4 环境使用 eef_pose 模式，因为 teleop_robot 返回的是 eef_pose 增量
    # PI0 自动控制时输出绝对关节角度，需要在自动控制逻辑中特殊处理
    PnPEnv = SimpleEnv4(xml_path, action_type='eef_pose', state_type='joint_angle')
    
    # 初始模式设为 arm
    control_mode = 'arm'
    PnPEnv.reset(mode=control_mode)

    # 3. 初始化图像预处理
    IMG_TRANSFORM = get_default_transform()
    
    # 4. 初始化性能监控器
    perf_monitor = PerformanceMonitor(window_size=100)
    
    # 5. 初始化推理器
    arm_runner = None
    base_runner = None
    
    if arm_policy is not None:
        if ARM_SYNC_INFERENCE:
            # 同步模式：不需要 runner，直接使用 policy
            print("🔄 [ARM] Using SYNC inference mode")
        else:
            # 异步模式：创建 runner
            arm_runner = AsyncArmInferenceRunner(
                arm_policy, device, IMG_TRANSFORM, 
                control_dt=CONTROL_DT, perf_monitor=perf_monitor
            )
            print("⚡ [ARM] Using ASYNC inference mode")
    
    if base_policy is not None:
        base_runner = AsyncBaseInferenceRunner(
            base_policy, device, IMG_TRANSFORM, 
            control_dt=CONTROL_DT, perf_monitor=perf_monitor
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
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                        auto_mode_arm = False
                    elif control_mode == 'base' and auto_mode_base and base_runner:
                        base_runner.stop()
                        auto_mode_base = False
                    
                    # 🔥 重置 runner/policy 状态（清除旧数据）
                    if ARM_SYNC_INFERENCE and arm_policy:
                        arm_policy.reset()
                    elif arm_runner:
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
                    perf_monitor.reset()
                    step = 0
                    print(f"\n🔄 Mode Switched to: {control_mode.upper()}")
                    print(f"   Task: {PnPEnv.instruction}")
                
                # N 键：启动自动控制
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_N):
                    if control_mode == 'arm':
                        if arm_policy is not None and not auto_mode_arm:
                            auto_mode_arm = True
                            arm_smoother.reset()  # 🔧 重置平滑器
                            if ARM_SYNC_INFERENCE:
                                # 同步模式：重置 policy
                                arm_policy.reset()
                                print("\n🤖 [ARM] PI0 Auto Control Started (SYNC mode)!")
                            else:
                                # 异步模式：启动 runner
                                if arm_runner:
                                    arm_runner.start()
                                print("\n🤖 [ARM] PI0 Auto Control Started (ASYNC mode)!")
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
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                            arm_runner.reset_state()  # 🔥 清除旧数据
                        arm_smoother.reset()  # 🔧 重置平滑器
                        auto_mode_arm = False
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
                        if not ARM_SYNC_INFERENCE and arm_runner:
                            arm_runner.stop()
                        auto_mode_arm = False
                    if auto_mode_base and base_runner:
                        base_runner.stop()
                        auto_mode_base = False
                    
                    # 🔥 重置 runner/policy 状态（清除旧数据）
                    if ARM_SYNC_INFERENCE and arm_policy:
                        arm_policy.reset()
                    elif arm_runner:
                        arm_runner.reset_state()
                    if base_runner:
                        base_runner.reset_state()
                    arm_smoother.reset()  # 🔧 重置平滑器
                    
                    PnPEnv.reset(mode=control_mode)
                    perf_monitor.reset()
                    step = 0
                    print(f"\n🔄 Environment Reset. Mode: {control_mode.upper()}")
                    print(f"   Task: {PnPEnv.instruction}")

                # --- 控制逻辑 ---
                
                if control_mode == 'arm':
                    if auto_mode_arm and arm_policy is not None:
                        # Arm 自动控制模式
                        t_loop_start = time.perf_counter()
                        
                        # 1. 收集观测数据
                        t0 = time.perf_counter()
                        state = PnPEnv.get_joint_state()[:6]
                        images_dict = PnPEnv.grab_image()  # {'agent', 'wrist', 'back'}
                        t1 = time.perf_counter()
                        perf_monitor.record_main('grab_image', t1 - t0)
                        
                        if ARM_SYNC_INFERENCE:
                            # ==========================================
                            # 🔄 同步推理模式
                            # ==========================================
                            t0 = time.perf_counter()
                            
                            # 准备图像输入
                            agent_img = Image.fromarray(images_dict['agent']).resize((ARM_CONFIG['image_size'], ARM_CONFIG['image_size']))
                            wrist_img = Image.fromarray(images_dict['wrist']).resize((ARM_CONFIG['image_size'], ARM_CONFIG['image_size']))
                            back_img = Image.fromarray(images_dict['back']).resize((ARM_CONFIG['image_size'], ARM_CONFIG['image_size']))
                            
                            agent_tensor = IMG_TRANSFORM(agent_img).unsqueeze(0).to(device)
                            wrist_tensor = IMG_TRANSFORM(wrist_img).unsqueeze(0).to(device)
                            back_tensor = IMG_TRANSFORM(back_img).unsqueeze(0).to(device)
                            
                            # 准备模型输入
                            data = {
                                'observation.state': torch.tensor([state], dtype=torch.float32).to(device),
                                'observation.images.agent': agent_tensor,
                                'observation.images.wrist': wrist_tensor,
                                'observation.images.back': back_tensor,
                                'task': [PnPEnv.instruction],
                            }
                            
                            # 同步推理
                            with torch.no_grad():
                                action_tensor = arm_policy.select_action(data)
                            
                            # 取第一个动作
                            action_step = action_tensor[0, :7].cpu().detach().numpy()
                            status_msg = "SYNC"
                            
                            t1 = time.perf_counter()
                            perf_monitor.record_main('get_action', t1 - t0)
                            
                        else:
                            # ==========================================
                            # ⚡ 异步推理模式
                            # ==========================================
                            obs_capture_time = time.time()
                            arm_runner.update_observation(images_dict, state, [PnPEnv.instruction], obs_capture_time)
                            
                            t0 = time.perf_counter()
                            action_step, status_msg = arm_runner.get_action_at_time(time.time())
                            t1 = time.perf_counter()
                            perf_monitor.record_main('get_action', t1 - t0)
                        
                        # 4. 执行动作（使用平滑器）
                        t0 = time.perf_counter()
                        if action_step is not None:
                            # 🔧 获取 chunk_id（异步模式用于边界检测）
                            chunk_id = None
                            if not ARM_SYNC_INFERENCE and arm_runner:
                                with arm_runner.lock:
                                    chunk_id = arm_runner.chunk_id
                            
                            # 🔧 使用平滑器处理动作
                            smoothed_action, gripper_state = arm_smoother.smooth_action(
                                action_step, chunk_id=chunk_id
                            )
                            
                            joint_angles = smoothed_action[:6]
                            gripper_cmd = smoothed_action[6]
                            
                            gripper_array = np.array([gripper_cmd] * 4)
                            gripper_array[[1, 3]] *= 0.8
                            PnPEnv.current_arm_q = np.concatenate([joint_angles, gripper_array])
                            PnPEnv.gripper_state = gripper_state  # 🔧 使用迟滞控制后的夹爪状态
                        
                        PnPEnv.p0, PnPEnv.R0 = PnPEnv.env.get_pR_body(body_name='tcp_link')
                        t1 = time.perf_counter()
                        perf_monitor.record_main('step_env', t1 - t0)
                        
                        # 5. 渲染
                        t0 = time.perf_counter()
                        PnPEnv.render(teleop=False, idx=step)
                        t1 = time.perf_counter()
                        perf_monitor.record_main('render', t1 - t0)
                        
                        t_loop_end = time.perf_counter()
                        perf_monitor.record_main('total_loop', t_loop_end - t_loop_start)
                        
                        # 6. 检查成功
                        if PnPEnv.check_success():
                            print("\n🎉 ARM Task Success!")
                            if not ARM_SYNC_INFERENCE and arm_runner:
                                arm_runner.stop()
                                arm_runner.reset_state()
                            arm_smoother.reset()  # 🔧 重置平滑器
                            auto_mode_arm = False
                            PnPEnv.reset(mode='arm')
                            step = 0
                            continue
                        
                        # 7. 打印统计
                        if step % 50 == 0:
                            perf_monitor.print_stats(step, mode='arm')
                        elif step % 10 == 0:
                            if ARM_SYNC_INFERENCE:
                                print(f"[ARM SYNC] Step {step}")
                            else:
                                with arm_runner.lock:
                                    chunk_start = arm_runner.chunk_start_timestamp
                                if chunk_start > 0:
                                    lag = (time.time() - chunk_start) / CONTROL_DT
                                else:
                                    lag = float('inf')
                                print(f"[ARM ASYNC] Step {step} | Status: {status_msg} | Lag: {lag:.1f} steps")
                        
                        step += 1
                        
                    else:
                        # Arm 手动控制模式
                        action, reset = PnPEnv.teleop_robot(mode='arm')
                        if reset:
                            PnPEnv.reset(mode='arm')
                            step = 0
                        else:
                            PnPEnv.step(action, mode='arm')
                        PnPEnv.render(teleop=True, idx=step)
                        step += 1
                
                else:  # base mode
                    if auto_mode_base and base_runner:
                        # Base 自动控制模式
                        t_loop_start = time.perf_counter()
                        
                        # 1. 收集观测数据
                        t0 = time.perf_counter()
                        state = PnPEnv.get_base_state()  # (2,) 轮速度
                        images_dict = PnPEnv.grab_image()  # {'front', 'left', 'right'}
                        t1 = time.perf_counter()
                        perf_monitor.record_main('grab_image', t1 - t0)
                        
                        obs_capture_time = time.time()
                        
                        # 2. 更新观测到推理线程
                        base_runner.update_observation(images_dict, state, [PnPEnv.instruction], obs_capture_time)
                        
                        # 3. 获取动作
                        t0 = time.perf_counter()
                        action_step, status_msg = base_runner.get_action_at_time(time.time())
                        t1 = time.perf_counter()
                        perf_monitor.record_main('get_action', t1 - t0)
                        
                        # 4. 执行动作
                        t0 = time.perf_counter()
                        if action_step is not None:
                            # action_step 是 (2,) 数组: 左右轮速度
                            PnPEnv.step(action_step, mode='base')
                        else:
                            # 没有可用动作，停止
                            PnPEnv.step(np.array([0.0, 0.0]), mode='base')
                        t1 = time.perf_counter()
                        perf_monitor.record_main('step_env', t1 - t0)
                        
                        # 5. 渲染
                        t0 = time.perf_counter()
                        PnPEnv.render(teleop=False, idx=step)
                        t1 = time.perf_counter()
                        perf_monitor.record_main('render', t1 - t0)
                        
                        t_loop_end = time.perf_counter()
                        perf_monitor.record_main('total_loop', t_loop_end - t_loop_start)
                        
                        # 6. 打印统计
                        if step % 50 == 0:
                            perf_monitor.print_stats(step, mode='base')
                        elif step % 10 == 0:
                            with base_runner.lock:
                                chunk_start = base_runner.chunk_start_timestamp
                            if chunk_start > 0:
                                lag = (time.time() - chunk_start) / CONTROL_DT
                            else:
                                lag = float('inf')
                            print(f"[BASE] Step {step} | Status: {status_msg} | Lag: {lag:.1f} steps")
                        
                        step += 1
                        
                    else:
                        # Base 手动控制模式
                        action, reset = PnPEnv.teleop_robot(mode='base')
                        if reset:
                            PnPEnv.reset(mode='base')
                            step = 0
                        else:
                            PnPEnv.step(action, mode='base')
                        PnPEnv.render(teleop=True, idx=step)
                        step += 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
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
