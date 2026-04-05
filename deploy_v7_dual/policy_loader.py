from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import dataset_to_policy_features

from .policy_backends import LocalPolicyBackend


def load_policy(config, device, label, emoji="🤖"):
    """加载 pi0 模型（ARM / BASE 通用）。"""
    print("\n" + "=" * 60)
    print(f"{emoji} Loading {label} Policy...")
    print(f"   Dataset: {config['dataset_repo_id']}")
    print(f"   Model: {config['model_path']}")
    print("=" * 60)

    try:
        dataset_metadata = LeRobotDatasetMetadata(
            config["dataset_repo_id"],
            root=config["dataset_root"],
        )

        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}

        cfg = PI0Config(
            input_features=input_features,
            output_features=output_features,
            chunk_size=config["chunk_size"],
            n_action_steps=config["n_action_steps"],
        )

        policy = PI0Policy.from_pretrained(
            config["model_path"],
            config=cfg,
            dataset_stats=dataset_metadata.stats,
        )

        policy.to(device)
        policy.eval()
        print(f"✅ {label} Policy Loaded Successfully!")
        return LocalPolicyBackend(
            policy=policy,
            config=config,
            device=device,
            label=label,
        )

    except Exception as e:
        print(f"❌ Failed to load {label} policy: {e}")
        import traceback

        traceback.print_exc()
        return None
