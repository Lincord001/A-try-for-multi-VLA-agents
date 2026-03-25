import os
import shutil

import datasets
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Silence noisy HF progress bars.
datasets.disable_progress_bar()

# ================= Config =================
SOURCE_REPO = "omy_base_data_RAG"
SOURCE_ROOT = "./demo_data_base_RAG"

TARGET_REPO = "omy_base_data_RAG_clean"
TARGET_ROOT = "./demo_data_base_RAG_clean"

# Episodes to remove (0-based episode_index).
BLACKLIST_EPISODES = [0]
# ==========================================

TASK_FALLBACK = "Go to the target location."
EXCLUDED_KEYS = {"frame_index", "episode_index", "index", "task_index", "timestamp", "task"}


def _to_numpy(value):
    """Convert dataset values to numpy arrays accepted by add_frame."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return np.asarray(value)


def _get_robot_type(dataset):
    info = getattr(dataset.meta, "info", {})
    if isinstance(info, dict):
        robot_type = info.get("robot_type")
        if robot_type:
            return robot_type

    robot_type = getattr(dataset.meta, "robot_type", None)
    if robot_type:
        return robot_type

    raise RuntimeError("Could not infer robot_type from source dataset metadata.")


def main():
    print(f"Loading source dataset: {SOURCE_ROOT}")
    ds_source = LeRobotDataset(SOURCE_REPO, root=SOURCE_ROOT)

    if os.path.exists(TARGET_ROOT):
        ans = input(f"Target folder {TARGET_ROOT} exists. Delete and recreate? (y/n): ")
        if ans.lower() == "y":
            shutil.rmtree(TARGET_ROOT)
        else:
            print("Aborted.")
            return

    print(f"Creating target dataset: {TARGET_ROOT}")
    ds_target = LeRobotDataset.create(
        repo_id=TARGET_REPO,
        root=TARGET_ROOT,
        robot_type=_get_robot_type(ds_source),
        fps=ds_source.fps,
        features=ds_source.features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    num_episodes = ds_source.num_episodes
    blacklist = set(BLACKLIST_EPISODES)
    invalid_blacklist = sorted(ep for ep in blacklist if ep < 0 or ep >= num_episodes)
    if invalid_blacklist:
        print(f"Warning: Ignoring invalid episode indices: {invalid_blacklist}")
        blacklist = {ep for ep in blacklist if 0 <= ep < num_episodes}

    episode_data_index = ds_source.episode_data_index
    episodes_meta = ds_source.meta.episodes
    copy_keys = [key for key in ds_source.features.keys() if key not in EXCLUDED_KEYS]
    image_key_set = {key for key in copy_keys if "images" in key}

    print(f"Total episodes: {num_episodes}")
    print(f"Removing episodes: {sorted(blacklist)}")

    kept_count = 0
    for ep_idx in tqdm(range(num_episodes), desc="Copy episodes"):
        if ep_idx in blacklist:
            continue

        task_instruction = TASK_FALLBACK
        try:
            tasks = episodes_meta[ep_idx].get("tasks", [])
            if tasks:
                task_instruction = tasks[0]
        except Exception:
            pass

        from_idx = int(episode_data_index["from"][ep_idx].item())
        to_idx = int(episode_data_index["to"][ep_idx].item())

        for frame_idx in range(from_idx, to_idx):
            frame_item = ds_source[frame_idx]
            frame_data = {}

            for key in copy_keys:
                if key not in frame_item:
                    continue
                val = _to_numpy(frame_item[key])
                if key in image_key_set and val.ndim == 3 and val.shape[0] == 3:
                    val = np.transpose(val, (1, 2, 0))
                frame_data[key] = val

            ds_target.add_frame(frame_data, task=task_instruction)

        ds_target.save_episode()
        kept_count += 1

    print(f"\nDone! Cleaned dataset saved to {TARGET_ROOT}")
    print(f"Original: {num_episodes} -> Cleaned: {kept_count} episodes.")
    print("You can now verify the new dataset with visualize_dataset.py")


if __name__ == "__main__":
    main()
