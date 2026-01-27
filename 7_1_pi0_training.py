import os

import subprocess

import sys



# 1. (关键) 设置 Hugging Face 镜像和令牌

# 必须在导入 huggingface_hub 或 transformers 之前设置

print("Setting up environment variables for Hugging Face...")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com'

# 保留您在 notebook 中使用的令牌

os.environ['HF_TOKEN'] = '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' 

print(f"HF_ENDPOINT set to: {os.environ.get('HF_ENDPOINT')}")

print(f"HF_TOKEN is set: {'*' * len(os.environ.get('HF_TOKEN', ''))}")





# 2. (可选) 检查依赖

# 确保您在环境中已安装:

# !pip install pytest

# !pip install transformers==4.50.3



# 3. (可选) 检查数据集

# 确保数据集 'omy_pnp_language' 存在于 './demo_data_language' 或 './omy_pnp_language'

# !git clone https://huggingface.co/datasets/Jeongeun/omy_pnp_language



def run_training():

    """

    执行 train_model.py 脚本 并传入 pi0_base.yaml 配置文件。

    注意: 本地权重路径已在 pi0_base.yaml 中的 pretrained_path 字段配置，无需在此处指定。

    """

    config_path = "pi0_base.yaml"

    # 构建在 notebook 中使用的命令

    command = [

        sys.executable,  # 使用当前 Python 解释器

        "train_model.py",

        "--config_path",

        config_path

    ]

    

    print(f"Starting training with config: {config_path}")

    print(f"Running command: {' '.join(command)}")

    

    try:

        # 使用 subprocess.run 来执行命令

        # 我们已经通过 os.environ 设置了环境变量，所以这里不需要再加

        result = subprocess.run(command, check=True, text=True)

        print("Training finished successfully.")

    

    except subprocess.CalledProcessError as e:

        print(f"Training script failed with exit code {e.returncode}")

        print("Error output:\n", e.stderr)

    except FileNotFoundError:

        print(f"Error: 'train_model.py' not found.")

        print("请确保 train_model.py 和 run_training.py 在同一个目录下。")

    except KeyboardInterrupt:

        print("\nTraining interrupted by user.")



if __name__ == "__main__":

    run_training()
