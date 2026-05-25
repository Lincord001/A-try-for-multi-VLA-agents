import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from _script_paths import resolve_repo_path

# ================= Config =================
# Source dataset root. Only read from this folder.
SOURCE_ROOT = str(resolve_repo_path("./demo_data_base_v7_5"))
# Target dataset root. If same as SOURCE_ROOT, edit in-place.
TARGET_ROOT = str(resolve_repo_path("./demo_data_base_v7_5"))

# If TARGET_ROOT exists and SOURCE_ROOT != TARGET_ROOT:
# - True: delete target and recopy source
# - False: abort
RECREATE_TARGET_IF_EXISTS = False

# Preview mode: True means print changes only, no file writes.
DRY_RUN = False

# Episode text edits. Only these episodes will be touched.
EPISODE_TASK_EDITS = [
     {"episode_index": 1029, "task": "Move to the kitchen refrigerator."},
]
# ==========================================


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _parse_episode_edits(edits):
    if not isinstance(edits, list):
        raise ValueError("EPISODE_TASK_EDITS must be a list.")
    mapping = {}
    for item in edits:
        if not isinstance(item, dict):
            raise ValueError("Each edit item in EPISODE_TASK_EDITS must be a dict.")
        if "episode_index" not in item or "task" not in item:
            raise ValueError("Each edit item must contain episode_index and task.")
        ep = int(item["episode_index"])
        task = str(item["task"]).strip()
        if not task:
            raise ValueError(f"Empty task for episode_index={ep}")
        mapping[ep] = task
    return mapping


def _resolve_episode_parquet(dataset_root: Path, episode_index: int, chunks_size: int):
    chunk_idx = episode_index // chunks_size
    return dataset_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"


def _replace_task_index_in_parquet(parquet_path: Path, new_task_index: int, dry_run: bool):
    if dry_run:
        print(f"[DRY-RUN] Would update task_index in: {parquet_path} -> {new_task_index}")
        return

    table = pq.read_table(parquet_path)
    if "task_index" not in table.column_names:
        raise RuntimeError(f"Column task_index not found in {parquet_path}")

    task_col = table.column("task_index")
    task_type = task_col.type
    num_rows = table.num_rows
    new_values = pa.array([new_task_index] * num_rows, type=task_type)

    col_idx = table.schema.get_field_index("task_index")
    table = table.set_column(col_idx, "task_index", new_values)
    pq.write_table(table, parquet_path)


def _prepare_target_dataset(source_root: Path, target_root: Path, dry_run: bool):
    if source_root == target_root:
        print("[MODE] Edit in-place.")
        return

    print("[MODE] Copy source dataset, then edit target dataset.")
    print(f"Source: {source_root}")
    print(f"Target: {target_root}")

    if target_root.exists():
        if not RECREATE_TARGET_IF_EXISTS:
            raise RuntimeError(
                f"Target exists: {target_root}. Set RECREATE_TARGET_IF_EXISTS=True to recreate it."
            )
        if dry_run:
            print(f"[DRY-RUN] Would delete target: {target_root}")
        else:
            shutil.rmtree(target_root)

    if dry_run:
        print(f"[DRY-RUN] Would copy source -> target: {source_root} -> {target_root}")
    else:
        shutil.copytree(source_root, target_root)


def main():
    source_root = Path(SOURCE_ROOT).resolve()
    target_root = Path(TARGET_ROOT).resolve()
    dry_run = DRY_RUN
    edits_map = _parse_episode_edits(EPISODE_TASK_EDITS)

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")
    if not edits_map:
        raise ValueError("EPISODE_TASK_EDITS is empty. Please add at least one edit.")

    _prepare_target_dataset(source_root, target_root, dry_run=dry_run)
    dataset_root = target_root

    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    if not info_path.exists() or not episodes_path.exists() or not tasks_path.exists():
        raise FileNotFoundError("Dataset meta files are missing under dataset_root/meta")

    info = _read_json(info_path)
    episodes = _read_jsonl(episodes_path)
    tasks = _read_jsonl(tasks_path)

    total_episodes = int(info.get("total_episodes", len(episodes)))
    chunks_size = int(info.get("chunks_size", 1000))
    for ep in edits_map:
        if ep < 0 or ep >= total_episodes:
            raise ValueError(f"Invalid episode_index: {ep} (0 <= ep < {total_episodes})")

    task_to_index = {}
    max_task_index = -1
    for row in tasks:
        idx = int(row["task_index"])
        text = str(row["task"])
        task_to_index[text] = idx
        max_task_index = max(max_task_index, idx)

    episode_to_new_task_index = {}
    new_task_entries = []
    for ep, new_task in sorted(edits_map.items()):
        if new_task in task_to_index:
            new_idx = task_to_index[new_task]
        else:
            max_task_index += 1
            new_idx = max_task_index
            task_to_index[new_task] = new_idx
            new_task_entries.append({"task_index": new_idx, "task": new_task})
        episode_to_new_task_index[ep] = new_idx

    if new_task_entries:
        tasks.extend(new_task_entries)
        tasks.sort(key=lambda x: int(x["task_index"]))

    by_episode = {int(row["episode_index"]): row for row in episodes}
    for ep, new_task in edits_map.items():
        row = by_episode.get(ep)
        if row is None:
            raise ValueError(f"Episode {ep} not found in episodes.jsonl")
        row["tasks"] = [new_task]

    print(f"Dataset root: {dataset_root}")
    print(f"Edits count: {len(edits_map)}")
    print(f"Dry-run: {dry_run}")

    touched_parquets = []
    for ep, task_idx in sorted(episode_to_new_task_index.items()):
        parquet_path = _resolve_episode_parquet(dataset_root, ep, chunks_size)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found for episode {ep}: {parquet_path}")
        _replace_task_index_in_parquet(parquet_path, task_idx, dry_run=dry_run)
        touched_parquets.append(parquet_path)

    if not dry_run:
        _write_jsonl(episodes_path, sorted(episodes, key=lambda x: int(x["episode_index"])))
        _write_jsonl(tasks_path, tasks)
        info["total_tasks"] = len(tasks)
        _write_json(info_path, info)

    print("\nDone.")
    print(f"Touched episodes: {sorted(edits_map.keys())}")
    print(f"Touched parquet files: {len(touched_parquets)}")
    if new_task_entries:
        print(f"New task texts appended to tasks.jsonl: {len(new_task_entries)}")


if __name__ == "__main__":
    main()
