import os
from safetensors.torch import load_file, save_file
from tqdm import tqdm  # 如果没有 tqdm 可以去掉，只是为了好看进度条

# 你的权重目录
model_dir = "./PaliGemmaWeights/paligemma-3b-pt-224"
output_path = os.path.join(model_dir, "model.safetensors")

# 定义分片文件列表
shards = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors"
]

full_state_dict = {}

print(f"📂 正在读取分片文件，目录: {model_dir}")

try:
    # 1. 依次读取每个分片并合并到字典中
    for shard_name in shards:
        shard_path = os.path.join(model_dir, shard_name)
        if not os.path.exists(shard_path):
            print(f"❌ 错误：找不到文件 {shard_path}")
            exit(1)
            
        print(f"   -> 加载 {shard_name} ...")
        # 加载分片权重
        shard_weights = load_file(shard_path)
        # 更新到总字典
        full_state_dict.update(shard_weights)
        
    print(f"✅ 合并完成，共收集到 {len(full_state_dict)} 个参数张量。")

    # 2. 保存为一个完整的 safetensors 文件
    print(f"💾 正在保存为单文件: {output_path} ...")
    save_file(full_state_dict, output_path)
    
    print("🎉 成功！你现在可以运行 train_model.py 了！")

except Exception as e:
    print(f"\n❌ 发生异常: {e}")