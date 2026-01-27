import shutil
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import os
import datasets

# 禁用 HuggingFace datasets 的进度条，避免每个 episode 保存时都输出 "Map" 和 "Creating parquet" 进度条
datasets.disable_progress_bar()

# ================= 配置 =================
SOURCE_REPO = 'omy_base_data'
SOURCE_ROOT = './demo_data_base'

TARGET_REPO = 'omy_base_data_clean'
TARGET_ROOT = './demo_data_base_clean'

# 📝 在这里填入你肉眼检查出的“垃圾数据”ID
BLACKLIST_EPISODES = [51]  # 比如 Ep 1 撞墙了，Ep 4 漂移太大了
# =======================================

def main():
    print(f"Loading source dataset: {SOURCE_ROOT}")
    ds_source = LeRobotDataset(SOURCE_REPO, root=SOURCE_ROOT)
    
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"clean_dataset.py:19","message":"ds_source object created","data":{"has_robot_type":hasattr(ds_source,'robot_type'),"has_fps":hasattr(ds_source,'fps'),"has_features":hasattr(ds_source,'features'),"has_meta":hasattr(ds_source,'meta'),"ds_source_type":str(type(ds_source))},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        if hasattr(ds_source, 'meta'):
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"clean_dataset.py:22","message":"ds_source.meta object check","data":{"meta_type":str(type(ds_source.meta)),"has_robot_type":hasattr(ds_source.meta,'robot_type'),"has_fps":hasattr(ds_source.meta,'fps'),"has_features":hasattr(ds_source.meta,'features'),"has_info":hasattr(ds_source.meta,'info')},"timestamp":int(__import__('time').time()*1000)})+'\n')
        else:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"clean_dataset.py:22","message":"ds_source.meta object check","data":{"meta_exists":False},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        try:
            robot_type_direct = ds_source.robot_type
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"clean_dataset.py:25","message":"Testing ds_source.robot_type","data":{"success":True,"value":robot_type_direct},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except AttributeError as e:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"clean_dataset.py:25","message":"Testing ds_source.robot_type","data":{"success":False,"error":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        try:
            robot_type_meta = ds_source.meta.robot_type
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"clean_dataset.py:28","message":"Testing ds_source.meta.robot_type","data":{"success":True,"value":robot_type_meta},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except Exception as e:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"clean_dataset.py:28","message":"Testing ds_source.meta.robot_type","data":{"success":False,"error":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        try:
            robot_type_info = ds_source.meta.info.get("robot_type") if hasattr(ds_source.meta, 'info') else None
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"clean_dataset.py:31","message":"Testing ds_source.meta.info['robot_type']","data":{"success":True,"value":robot_type_info},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except Exception as e:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"clean_dataset.py:31","message":"Testing ds_source.meta.info['robot_type']","data":{"success":False,"error":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        try:
            fps_direct = ds_source.fps
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"clean_dataset.py:34","message":"Testing ds_source.fps","data":{"success":True,"value":fps_direct},"timestamp":int(__import__('time').time()*1000)})+'\n')
        except Exception as e:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"clean_dataset.py:34","message":"Testing ds_source.fps","data":{"success":False,"error":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    
    # 如果目标文件夹存在，询问是否删除
    if os.path.exists(TARGET_ROOT):
        ans = input(f"Target folder {TARGET_ROOT} exists. Delete and recreate? (y/n): ")
        if ans.lower() == 'y':
            shutil.rmtree(TARGET_ROOT)
        else:
            print("Aborted.")
            return

    # 创建新数据集 (继承旧数据集的 features 配置)
    print(f"Creating target dataset: {TARGET_ROOT}")
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"clean_dataset.py:67","message":"Before LeRobotDataset.create call","data":{"about_to_use_robot_type":"will try ds_source.meta.robot_type"},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    # #region agent log
    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
        import json
        robot_type_val = ds_source.meta.robot_type
        fps_val = ds_source.fps
        features_val = ds_source.features
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"clean_dataset.py:70","message":"Values for LeRobotDataset.create","data":{"robot_type":robot_type_val,"fps":fps_val,"features_keys":list(features_val.keys())[:5] if isinstance(features_val,dict) else str(type(features_val))},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion
    ds_target = LeRobotDataset.create(
        repo_id=TARGET_REPO,
        root=TARGET_ROOT,
        robot_type=ds_source.meta.robot_type,
        fps=ds_source.fps,
        features=ds_source.features,
        image_writer_threads=10, 
        image_writer_processes=5
    )

    print(f"Total episodes: {ds_source.num_episodes}")
    print(f"Removing episodes: {BLACKLIST_EPISODES}")

    # 遍历旧数据，搬运合格的
    kept_count = 0
    for ep_idx in tqdm(range(ds_source.num_episodes)):
        if ep_idx in BLACKLIST_EPISODES:
            continue # 跳过垃圾数据
        
        # 1. 获取这一集的任务指令 (Task Instruction)
        # 从源数据的元数据中获取
        try:
            episode_info = ds_source.meta.episodes[ep_idx]
            # episodes.jsonl 中 tasks 是列表，取第一个
            if 'tasks' in episode_info and len(episode_info['tasks']) > 0:
                task_instruction = episode_info['tasks'][0]
            else:
                task_instruction = "Go to the target location."  # 默认值
        except Exception as e:
            # 如果万一查不到，给个默认值 (Base模式通常是这句话)
            print(f"Warning: Could not get task for episode {ep_idx}: {e}")
            task_instruction = "Go to the target location."

        # 2. 获取这一整集的数据
        # #region agent log
        with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"F","location":"clean_dataset.py:133","message":"Checking episode_data_index","data":{"has_episode_data_index":hasattr(ds_source,'episode_data_index'),"ep_idx":ep_idx},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        # 使用 episode_data_index 获取 episode 的帧范围
        episode_data_index = ds_source.episode_data_index
        from_idx = episode_data_index["from"][ep_idx].item()
        to_idx = episode_data_index["to"][ep_idx].item()
        num_frames = to_idx - from_idx
        
        # #region agent log
        with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
            import json
            f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"G","location":"clean_dataset.py:140","message":"Episode frame range","data":{"ep_idx":ep_idx,"from_idx":from_idx,"to_idx":to_idx,"num_frames":num_frames},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion
        
        # 3. 逐帧搬运 (带上 task!)
        for frame_idx in range(from_idx, to_idx):
            # 获取单帧数据
            frame_item = ds_source[frame_idx]
            
            # #region agent log
            with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
                import json
                if frame_idx == from_idx:  # 只记录第一帧，避免日志过多
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run2","hypothesisId":"H","location":"clean_dataset.py:155","message":"First frame from dataset","data":{"frame_idx":frame_idx,"frame_keys":list(frame_item.keys())[:10],"sample_key_type":str(type(list(frame_item.values())[0])) if frame_item else "empty"},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            # 将 torch.Tensor 转换为 numpy array（add_frame 需要 numpy array）
            # 只传递 features 中定义的实际特征，过滤掉元数据字段
            import numpy as np
            frame_data = {}
            
            # #region agent log
            with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
                import json
                if frame_idx == from_idx:  # 只记录第一帧，避免日志过多
                    frame_item_keys = list(frame_item.keys())
                    features_keys = list(ds_source.features.keys())
                    extra_keys = set(frame_item_keys) - set(features_keys)
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run3","hypothesisId":"I","location":"clean_dataset.py:165","message":"Frame item keys vs features","data":{"frame_item_keys":frame_item_keys[:15],"features_keys":features_keys[:15],"extra_keys":list(extra_keys)},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion
            
            # 只处理 features 中定义的键，排除元数据字段
            excluded_keys = {'frame_index', 'episode_index', 'index', 'task_index', 'timestamp', 'task'}
            for key in ds_source.features:
                if key in frame_item and key not in excluded_keys:
                    val = frame_item[key]
                    # 如果是 torch.Tensor，转换为 numpy
                    if hasattr(val, 'numpy'):
                        val = val.numpy()
                    elif hasattr(val, 'cpu'):
                        val = val.cpu().numpy()
                    elif isinstance(val, (list, tuple)):
                        val = np.array(val)
                    
                    # #region agent log
                    with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
                        import json
                        if frame_idx == from_idx and 'images' in key:  # 只记录第一帧的图像信息
                            val_shape = val.shape if hasattr(val, 'shape') else 'no shape'
                            f.write(json.dumps({"sessionId":"debug-session","runId":"run3","hypothesisId":"J","location":"clean_dataset.py:180","message":"Image shape before conversion","data":{"key":key,"shape":str(val_shape)},"timestamp":int(__import__('time').time()*1000)})+'\n')
                    # #endregion
                    
                    # 如果是图像数据且是 channel-first 格式，转换为 channel-last
                    if 'images' in key and isinstance(val, np.ndarray):
                        if len(val.shape) == 3 and val.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                            val = np.transpose(val, (1, 2, 0))
                            # #region agent log
                            with open('/home/pengguanqi/workspace/.cursor/debug.log', 'a') as f:
                                import json
                                if frame_idx == from_idx:
                                    f.write(json.dumps({"sessionId":"debug-session","runId":"run3","hypothesisId":"J","location":"clean_dataset.py:190","message":"Image shape after conversion","data":{"key":key,"shape":str(val.shape)},"timestamp":int(__import__('time').time()*1000)})+'\n')
                            # #endregion
                    
                    frame_data[key] = val
            
            # 🔥 关键修改：这里要把 task 传进去，这样 tasks.jsonl 才会自动生成
            ds_target.add_frame(frame_data, task=task_instruction)
        
        # 4. 保存这一集 (这里会自动计算并写入 episodes_stats.jsonl)
        ds_target.save_episode()
        kept_count += 1

    print(f"\nDone! Cleaned dataset saved to {TARGET_ROOT}")
    print(f"Original: {ds_source.num_episodes} -> Cleaned: {kept_count} episodes.")
    print("You can now verify the new dataset with visualize_dataset.py")

if __name__ == "__main__":
    main()