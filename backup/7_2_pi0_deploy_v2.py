#!/usr/bin/env python3
"""
V2 环境部署脚本 - PI0 策略部署

功能说明：
- 使用 SimpleEnv2 (V2环境，只有arm模式)
- State输入：7维 [q1, q2, q3, q4, q5, q6, gripper] (包含夹爪状态)
- 图像输入：字典格式 {'agent': ..., 'wrist': ...}
- Action输出：7维 [6关节角度 + 1夹爪状态] - 绝对量
- 按 N: 启动 pi0 控制
- 按 M: 恢复人类遥控模式
- 按 Z: 重置环境
- 按 Q: 退出
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

import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import glfw

# ==========================================
# 🔧 模型配置
# ==========================================

# Arm 模型配置 (V2)
ARM_CONFIG = {
    'model_path': './ckpt/pi0_arm/pretrained_model_arm_v2',  # 🔥 根据实际路径修改
    'dataset_repo_id': 'omy_arm_data_v2',
    'dataset_root': './demo_data_arm_v2',
    'chunk_size': 5,
    'n_action_steps': 5,
    'image_size': 224,  # V2环境使用 224x224
    'state_dim': 7,     # 🔥 V2: 7维 [q1, q2, q3, q4, q5, q6, gripper]
    'action_dim': 7,
    'camera_keys': ['agent', 'wrist'],
}

# ==========================================
# 🔥 控制频率配置（Hz）
# ==========================================
CONTROL_FREQUENCY = 20  # 控制频率，单位：Hz
CONTROL_DT = 1.0 / CONTROL_FREQUENCY  # 控制周期，单位：秒

# ==========================================
# 🔧 动作平滑配置（防颤抖）
# ==========================================
SMOOTHING_ENABLED = True        # 是否启用动作平滑
SMOOTHING_ALPHA_JOINTS = 0.4    # 关节角度平滑系数 (0.0~1.0)
SMOOTHING_ALPHA_GRIPPER = 0.3   # 夹爪平滑系数 (0.0~1.0)

# 夹爪迟滞控制：防止夹爪在阈值附近频繁切换
GRIPPER_HYSTERESIS_ENABLED = True
GRIPPER_OPEN_THRESH = 0.6       # 超过此值才打开夹爪
GRIPPER_CLOSE_THRESH = 0.4      # 低于此值才关闭夹爪

# ==========================================
# 🔧 随机初始化配置（匹配 y_env2_test.py 和 collect_data_v2.py）
# ==========================================
RANDOM_INIT_ENABLED = 0            # 0: 关闭, 1: 旧版(扇形区域), 2: 新版(圆形交集)
RANDOM_INIT_GRIPPER_OPEN = True    # True: 初始化时夹爪张开, False: 初始化时夹爪闭合

# 导入 LeRobot 和 MuJoCo 环境
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.types import FeatureType
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from mujoco_env.y_env2_test import SimpleEnv2
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


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
# 🔧 辅助函数
# ==========================================
def get_default_transform():
    """返回标准图像变换"""
    return transforms.Compose([transforms.ToTensor()])


def load_arm_policy(device):
    """加载 Arm 模式的 pi0 模型"""
    print("\n" + "="*60)
    print("🤖 Loading ARM Policy (V2)...")
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


# ==========================================
# 主程序
# ==========================================
def main():
    print("\n" + "="*70)
    print("🎮 V2 ENVIRONMENT DEPLOYMENT (ARM MODE ONLY)")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 打印配置
    print(f"\n📋 Configuration:")
    print(f"   RANDOM_INIT_ENABLED: {RANDOM_INIT_ENABLED} ({'关闭' if RANDOM_INIT_ENABLED == 0 else 'V1 (扇形区域)' if RANDOM_INIT_ENABLED == 1 else 'V2 (圆形交集)'})")
    print(f"   RANDOM_INIT_GRIPPER_OPEN: {RANDOM_INIT_GRIPPER_OPEN}")
    print(f"   SMOOTHING_ENABLED: {SMOOTHING_ENABLED}")
    print(f"   Joint Alpha: {SMOOTHING_ALPHA_JOINTS}, Gripper Alpha: {SMOOTHING_ALPHA_GRIPPER}")

    # 1. 加载模型
    arm_policy = load_arm_policy(device)
    if arm_policy is None:
        print("❌ Failed to load policy. Exiting.")
        return
    
    # 2. 初始化环境 (使用 SimpleEnv2)
    print("\n" + "="*60)
    print("🌍 Initializing MuJoCo Environment (V2)...")
    print("="*60)
    
    xml_path = './asset/example_scene_y2.xml'
    # 🔥 V2环境使用 joint_angle 模式，直接支持模型输出的绝对关节角度
    PnPEnv = SimpleEnv2(
        xml_path, 
        action_type='joint_angle',  # 🔥 使用 joint_angle，直接支持绝对关节角度
        state_type='joint_angle',
        seed=0,
        random_init_enabled=RANDOM_INIT_ENABLED,
        random_init_gripper_open=RANDOM_INIT_GRIPPER_OPEN
    )
    
    PnPEnv.reset()

    # 3. 初始化图像预处理
    IMG_TRANSFORM = get_default_transform()
    
    # 4. 初始化动作平滑器（防颤抖）
    arm_smoother = ActionSmoother(
        joint_dim=6, 
        alpha_joints=SMOOTHING_ALPHA_JOINTS, 
        alpha_gripper=SMOOTHING_ALPHA_GRIPPER
    )
    print(f"🔧 Action Smoother initialized:")
    print(f"   - Smoothing Enabled: {SMOOTHING_ENABLED}")
    print(f"   - Joint Alpha: {SMOOTHING_ALPHA_JOINTS}")
    print(f"   - Gripper Alpha: {SMOOTHING_ALPHA_GRIPPER}")
    print(f"   - Gripper Hysteresis: {GRIPPER_HYSTERESIS_ENABLED} (open>{GRIPPER_OPEN_THRESH}, close<{GRIPPER_CLOSE_THRESH})")
    
    # 控制状态
    auto_mode = False   # 是否启用自动控制
    step = 0
    
    # 打印操作指南
    print("\n" + "="*70)
    print("🎮 V2 ENVIRONMENT READY")
    print("="*70)
    print("Controls:")
    print("  [N] Start PI0 Auto Control")
    print("  [M] Switch to Manual Control")
    print("  [Z] Reset Environment")
    print("  [Q] Quit")
    print("="*70 + "\n")
    print(f"🎯 Current Mode: {'AUTO' if auto_mode else 'MANUAL'}")
    print(f"📋 Task: {PnPEnv.instruction}")

    try:
        while PnPEnv.env.is_viewer_alive():
            # [A] 物理环境步进
            PnPEnv.step_env()
            
            # [B] 控制循环
            if PnPEnv.env.loop_every(HZ=CONTROL_FREQUENCY):
                
                # --- 键位处理 ---
                
                # N 键：启动自动控制
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_N):
                    if arm_policy is not None and not auto_mode:
                        auto_mode = True
                        arm_smoother.reset()  # 🔧 重置平滑器
                        arm_policy.reset()    # 🔥 重置 policy 状态
                        print("\n🤖 [ARM] PI0 Auto Control Started!")
                    elif arm_policy is None:
                        print("\n⚠️ ARM policy not loaded!")
                
                # M 键：恢复手动控制
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_M):
                    if auto_mode:
                        arm_smoother.reset()  # 🔧 重置平滑器
                        auto_mode = False
                        print("\n👤 [ARM] Switched to Manual Control")
                
                # Z 键：重置环境
                if PnPEnv.env.is_key_pressed_once(key=glfw.KEY_Z):
                    # 停止自动控制
                    if auto_mode:
                        auto_mode = False
                    
                    # 🔥 重置 policy 状态（清除旧数据）
                    arm_policy.reset()
                    arm_smoother.reset()  # 🔧 重置平滑器
                    
                    PnPEnv.reset()
                    step = 0
                    print(f"\n🔄 Environment Reset.")
                    print(f"   Task: {PnPEnv.instruction}")

                # --- 控制逻辑 ---
                
                if auto_mode and arm_policy is not None:
                    # Arm 自动控制模式
                    
                    # 1. 收集观测数据
                    # 🔥 V2环境：state 是 7维 [q1, q2, q3, q4, q5, q6, gripper]，包含夹爪状态
                    state = PnPEnv.get_joint_state()  # (7,) 包含夹爪状态
                    
                    # 🔥 V2环境：grab_image 返回字典格式 {'agent': ..., 'wrist': ...}
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
                    
                    # 4. 同步推理
                    with torch.no_grad():
                        action_tensor = arm_policy.select_action(data)
                    
                    # 5. 取第一个动作并转换为 numpy
                    action_step = action_tensor[0, :7].cpu().detach().numpy()  # (7,)
                    
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
                        print(f"[ARM] Step {step} | Task: {PnPEnv.instruction}")
                
                else:
                    # Arm 手动控制模式
                    # 🔥 teleop_robot 返回的是 eef_pose 增量，需要临时切换 action_type
                    original_action_type = PnPEnv.action_type
                    PnPEnv.action_type = 'eef_pose'  # 临时切换为 eef_pose 模式
                    
                    action, reset = PnPEnv.teleop_robot(mode='arm')
                    if reset:
                        PnPEnv.reset()
                        arm_policy.reset()
                        arm_smoother.reset()
                        step = 0
                    else:
                        PnPEnv.step(action, mode='arm')
                    
                    PnPEnv.action_type = original_action_type  # 恢复为 joint_angle 模式
                    PnPEnv.render(teleop=True, idx=step)
                    step += 1

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        # 清理
        if PnPEnv.env.viewer:
            PnPEnv.env.close_viewer()
        print("🛑 Environment closed.")


if __name__ == "__main__":
    main()
