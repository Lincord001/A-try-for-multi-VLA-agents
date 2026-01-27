import os
import sys

# 1. (关键) 设置 Hugging Face 镜像
# 必须在导入 lerobot, huggingface_hub 或 transformers 之前设置
print("Setting up environment variables for Hugging Face...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'
print(f"HF_ENDPOINT set to: {os.environ.get('HF_ENDPOINT')}")

# 2. (可选) 检查依赖
# 确保您在环境中已安装:
# !pip install pytest
# !pip install transformers==4.50.3

# 3. 导入所有必需的库
# (来自 notebook cells `d198aa7b`, `f2152c6f`, `ac83797a`)
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    import numpy as np
    from lerobot.common.datasets.utils import write_json, serialize_dict
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.configs.types import FeatureType
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.datasets.utils import dataset_to_policy_features
    import torch
    from PIL import Image
    import torchvision
    from torchvision import transforms
    from mujoco_env.y_env2 import SimpleEnv2
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保您已经安装了所有依赖 (pip install -r requirements.txt)")
    sys.exit(1)

# 4. 图像转换辅助函数 (来自 notebook cell `ac83797a`)
def get_default_transform(image_size: int = 224):
    """
    返回一个 torchvision 转换:
    将 PIL 图像 [0,255] 转换为 FloatTensor [0.0,1.0], 形状为 C×H×W
    """
    return transforms.Compose([
        transforms.ToTensor(),
    ])

# 5. 主执行函数
def main():
    """
    加载模型并在 MuJoCo 环境中运行部署。
    """
    
    # 5.1. 设置设备 (来自 notebook cell `53895453`)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 5.2. 加载数据集元数据和策略配置 (来自 notebook cell `dee90c0e`)
    print("Loading dataset metadata...")
    try:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./demo_data_language')
    except FileNotFoundError:
        try:
            dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')
        except FileNotFoundError:
            print("Error: 找不到数据集。请先运行 `git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language`")
            print("或者确保 './demo_data_language' 路径正确。")
            return
            
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    cfg = PI0Config(input_features=input_features, output_features=output_features, chunk_size=5, n_action_steps=5)
    
    # 5.3. 加载训练好的策略 (来自 notebook cell `00c5180d`)
    #
    # *** 关键: 在下面两个选项中选择一个来加载模型 ***
    #
    policy_path = None
    policy = None
    
    try:
        # 选项 1: 加载您本地训练的权重
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!! 确保此路径指向您从A100下载的 `pretrained_model` 文件夹 !!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        local_policy_path = './ckpt/pi0_omy/author_data_pretrained_model'
        #local_policy_path = './ckpt/pi0_omy/last_pretrained_model' # 这是您 notebook 中的原始路径
        #local_policy_path = '"Jeongeun/omy_pnp_pi0"' # 这可能是您A100上的路径
        
        print(f"尝试从本地路径加载: {local_policy_path}")
        policy = PI0Policy.from_pretrained(local_policy_path, dataset_stats=dataset_metadata.stats)
        policy_path = local_policy_path

    except Exception as e_local:
        print(f"从本地路径 '{local_policy_path}' 加载失败: {e_local}")
        
        # 选项 2: (备选) 直接从 Hugging Face Hub 加载预训练模型
        # (来自 notebook cell `00c5180d` 的注释行)
        hf_policy_path = "Jeongeun/omy_pnp_pi0"
        print(f"尝试从 Hugging Face Hub 加载: {hf_policy_path}")
        try:
            policy = PI0Policy.from_pretrained(hf_policy_path, config=cfg, dataset_stats=dataset_metadata.stats)
            policy_path = hf_policy_path
        except Exception as e_hf:
            print(f"从 Hugging Face Hub 加载失败: {e_hf}")
            print("错误：本地和Hugging Face Hub的模型都加载失败了。")
            print("请检查：")
            print(f"  1. 本地路径 '{local_policy_path}' 是否正确?")
            print("  2. 您的网络连接和镜像设置是否允许访问 Hugging Face？")
            if "Connection to huggingface.co timed out" in str(e_local) or "Connection to huggingface.co timed out" in str(e_hf):
                print("\n检测到连接超时！请确保您的 `HF_ENDPOINT` 镜像设置正确且网络通畅。")
            return

    # 加载成功
    print(f"\nPolicy a\nPolicy '{policy_path}' 加载成功。")
    policy.to(device)
    policy.eval()

    # 5.4. 初始化 MuJoCo 环境 (来自 notebook cell `f2152c6f`)
    print("Initializing MuJoCo environment...")
    xml_path = './asset/example_scene_y2.xml'
    PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')
    print("Environment initialized.")

    # 5.5. 运行部署/推理循环 (来自 notebook cell `82e9e6fa`)
    step = 0
    PnPEnv.reset(seed=0)
    policy.reset()
    IMG_TRANSFORM = get_default_transform()
    print("Starting deployment loop. 按 'q' 键或关闭窗口退出。")

    try:
        while PnPEnv.env.is_viewer_alive():
            PnPEnv.step_env()
            if PnPEnv.env.loop_every(HZ=20):
                # 检查任务是否成功
                success = PnPEnv.check_success()
                if success:
                    print('Success!')
                    # 重置环境和策略
                    policy.reset()
                    PnPEnv.reset()
                    step = 0

                # 获取当前状态
                state = PnPEnv.get_joint_state()[:6]
                
                # 获取图像
                image, wirst_image = PnPEnv.grab_image()
                image = Image.fromarray(image).resize((256, 256))
                image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(device)
                
                wrist_image = Image.fromarray(wirst_image).resize((256, 256))
                wrist_image_tensor = IMG_TRANSFORM(wrist_image).unsqueeze(0).to(device)
                
                # 准备模型输入
                data = {
                    'observation.state': torch.tensor([state], dtype=torch.float32).to(device),
                    'observation.image': image_tensor,
                    'observation.wrist_image': wrist_image_tensor,
                    'task': [PnPEnv.instruction],
                }
                
                # 选择动作
                with torch.no_grad():
                    action_tensor = policy.select_action(data)
                
                action = action_tensor[0, :7].cpu().detach().numpy()
                print(f"DEBUG: action_tensor.shape = {action_tensor.shape}")
                # 在环境中执行动作
                _ = PnPEnv.step(action)
                PnPEnv.render()
                step += 1

    except KeyboardInterrupt:
        print("\nDeployment loop interrupted by user.")
    finally:
        if PnPEnv.env.viewer:
             PnPEnv.env.close_viewer()
        print("Environment closed.")


if __name__ == "__main__":
    main()
