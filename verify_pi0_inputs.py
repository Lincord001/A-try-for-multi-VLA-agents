from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.configs.types import FeatureType

# 1. 指向你包含 base_pose 的新数据集目录
ROOT = "./demo_data_arm_v4" 
REPO_NAME = "omy_arm_data_v4"

try:
    # 加载数据集元数据
    meta = LeRobotDatasetMetadata(REPO_NAME, root=ROOT)
    print(f"✅ 数据集加载成功。")
    print(f"📦 数据集包含的所有字段: {list(meta.features.keys())}")
    
    # 2. 让 LeRobot 模拟一次"分拣"过程
    policy_features = dataset_to_policy_features(meta.features)
    
    # 显示哪些字段被过滤掉了
    all_keys = set(meta.features.keys())
    policy_keys = set(policy_features.keys())
    filtered_keys = all_keys - policy_keys
    
    if filtered_keys:
        print(f"\n🚫 被过滤掉的字段（不会传给模型）: {list(filtered_keys)}")
    
    # 3. 检查输入特征
    input_features = {
        key: ft for key, ft in policy_features.items() 
        if ft.type is not FeatureType.ACTION
    }
    
    print("\n🧐 === 模型的'眼睛' (Input Features) ===")
    print(f"模型将看到以下输入: {list(input_features.keys())}")
    
    # 显示每个输入特征的类型
    for key, ft in input_features.items():
        print(f"  - {key}: {ft.type.name}")
    
    # 4. 最终判定
    if "base_pose" in input_features or "observation.base_pose" in input_features:
        print("\n❌ 危险！'base_pose' 仍然被模型视作输入！")
        print("   请检查数据集配置，确保 base_pose 不使用 observation. 前缀")
    else:
        print("\n🎉 安全！'base_pose' 已被成功隔离，模型完全看不到它。")
        if "base_pose" in filtered_keys:
            print(f"   ✅ 'base_pose' 在过滤列表中，验证通过！")

except Exception as e:
    print(f"验证出错 (可能是路径不对): {e}")
    import traceback
    traceback.print_exc()