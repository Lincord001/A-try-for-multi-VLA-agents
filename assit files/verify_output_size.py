import os

from _script_paths import resolve_repo_path

print("Setting up environment variables for Hugging Face...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'
print(f"HF_ENDPOINT set to: {os.environ.get('HF_ENDPOINT')}")

import torch
import numpy as np
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType


def inspect_checkpoint(path_name, policy_path):
    print(f"\n{'='*20} 正在检查路径: {path_name} {'='*20}")
    print(f"📂 路径地址: {policy_path}")
    
    if not os.path.exists(policy_path):
        print("❌ 路径不存在！跳过。")
        return

    try:
        # 准备元数据 (为了加载 stats)
        # 注意：如果您修改了数据集路径，请在这里对应修改
        try:
            dataset_metadata = LeRobotDatasetMetadata("omy_base_data", root=str(resolve_repo_path("./demo_data_base")))
        except:
            print("⚠️ 警告: 找不到本地数据集，尝试使用默认 stats 初始化...")
            dataset_metadata = None

        stats = dataset_metadata.stats if dataset_metadata else None

        # --- 步骤 1: 正常加载 (模拟您之前的代码) ---
        print("\n[Step 1] 尝试默认加载 (不强制注入Config)...")
        policy = PI0Policy.from_pretrained(policy_path, dataset_stats=stats)
        
        # 打印读取到的配置
        print(f"   📋 读到的配置: chunk_size = {policy.config.chunk_size}")
        print(f"   📋 读到的配置: n_action_steps = {policy.config.n_action_steps}")
        
        # --- 步骤 2: 物理权重“验尸” ---
        print("\n[Step 2] 检查模型物理权重层...")
        # PI0 的动作输出层通常叫 action_out_proj 或类似名称
        # 我们遍历模型寻找输出层
        found_layer = False
        for name, module in policy.model.named_modules():
            if 'action_out_proj' in name:
                print(f"   🔍 发现输出层: {name}")
                print(f"   📏 权重形状 (Weight Shape): {module.weight.shape}")
                print(f"   🔢 输出特征数 (Out Features): {module.out_features}")
                
                # 分析
                # Base模式动作维数=2
                if module.out_features == 2:
                    print("   ❌ 结论: 这是一个 [单步预测 (Chunk=1)] 的模型权重。")
                elif module.out_features == 20: # 10 * 2
                    print("   ✅ 结论: 这是一个 [10步预测 (Chunk=10)] 的模型权重。")
                elif module.out_features == 32: # Max dim
                    print("   ❓ 结论: 这是一个固定维度(32)的权重，输出步数取决于 Config。")
                else:
                    print(f"   ℹ️ 结论: 未知维度，可能对应 {module.out_features/2} 步。")
                found_layer = True
                break
        
        if not found_layer:
            print("   ⚠️ 未找到标准的 'action_out_proj' 层，无法直接判断权重。")

        # --- 步骤 3: 实际推理测试 ---
        print("\n[Step 3] 运行一次 select_action 测试...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        policy.to(device)
        policy.eval()
        
        # 构造假数据 (Batch=1)
        # 假设是 Base 模式: state(2), images(3,256,256)
        dummy_state = torch.zeros(1, 2).to(device)
        dummy_img = torch.zeros(1, 3, 256, 256).to(device)
        
        data = {
            'observation.state': dummy_state,
            'observation.images.front': dummy_img,
            'observation.images.left': dummy_img,
            'observation.images.right': dummy_img,
            'task': ["Move forward"],
        }
        
        with torch.no_grad():
            action = policy.select_action(data)
        
        print(f"   🚀 select_action 输出形状: {action.shape}")
        
        if action.shape == (1, 2):
            print("   💀 最终判决: 模型正在以 [单步模式] 运行。")
        elif action.shape == (1, 10, 2):
            print("   🎉 最终判决: 模型正在以 [10步模式] 运行！")
        else:
            print(f"   🤔 最终判决: 输出了意外的形状 {action.shape}")

    except Exception as e:
        print(f"❌ 检查过程中发生错误: {e}")

# ================= 主程序 =================
if __name__ == "__main__":
    print("🔍 开始模型取证分析...")
    
    # 1. 检查您当前使用的路径 (嫌疑最大的路径)
    current_path = str(resolve_repo_path("./ckpt/pi0_base/pretrained_model"))
    inspect_checkpoint("当前使用路径 (pretrained_model)", current_path)
    
    # 2. 检查 Checkpoints 目录 (寻找真正的训练结果)
    # 扫描 ckpt/pi0_base/checkpoints 目录下的所有文件夹
    ckpt_root = str(resolve_repo_path("./ckpt/pi0_base/checkpoints"))
    if os.path.exists(ckpt_root):
        subdirs = sorted([d for d in os.listdir(ckpt_root) if d.isdigit()], key=lambda x: int(x))
        if subdirs:
            print("\n发现以下训练检查点 (Checkpoints):", subdirs)
            # 检查最后一个 (步数最大的)
            last_ckpt = subdirs[-1]
            last_ckpt_path = os.path.join(ckpt_root, last_ckpt, 'pretrained_model')
            inspect_checkpoint(f"训练检查点 (Step {last_ckpt})", last_ckpt_path)
        else:
            print("\n⚠️ checkponts 目录为空，未找到训练过程文件。")
    else:
        print(f"\n⚠️ 找不到 {ckpt_root} 目录。")
