#!/usr/bin/env python3
"""Record per-step inputs used by ``execution_tracker.py`` plus wall time.

This module is intentionally standalone so it can be dropped into an existing
control loop with minimal wiring. It records:
- the exact runtime signals consumed by ``ExecutionTracker``
- wall-clock timestamps from both Python and MuJoCo (when available)
- optional tracker outputs/debug snapshots for later post-hoc review

Typical usage inside a control loop:

```python
from orchestration.execution_tracker import ExecutionTracker, ARM_VLA_PROFILE
from record_execution_tracker_trace import ExecutionTrackerTraceRecorder

tracker = ExecutionTracker(ARM_VLA_PROFILE)
tracker.reset(mode="arm_vla", instruction=env.instruction)
recorder = ExecutionTrackerTraceRecorder("./tracker_trace_arm.jsonl")

result = recorder.record_env_step(
    env=env,
    mode="arm_vla",
    tracker=tracker,
    action=smoothed_action,
    step=state.step,
)
```
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np

from orchestration.execution_tracker import (
    MODE_ARM_VLA,
    MODE_BASE_VLA,
    ExecutionTracker,
    TrackerResult,
    TrackerSample,
)


def _r2rpy(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to roll-pitch-yaw in radians."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class ExecutionTrackerTraceRecorder:
    """Append-only JSONL recorder for execution tracker inputs and outputs."""

    def __init__(
        self,
        output_path: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        flush_every: int = 1,
    ):
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self.flush_every = max(1, int(flush_every))
        self._record_count = 0
        self._episode_index = 0
        self._current_episode_id = 0
        self._fp = self.output_path.open("a", encoding="utf-8")

        if self.output_path.stat().st_size == 0:
            self._write_json_line(
                {
                    "record_type": "trace_file_header",
                    "created_at_unix": time.time(),
                    "output_path": str(self.output_path),
                    "metadata": _to_jsonable(self.metadata),
                }
            )

    def close(self) -> None:
        if getattr(self, "_fp", None) is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "ExecutionTrackerTraceRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start_episode(
        self,
        *,
        mode: str,
        instruction: Optional[str] = None,
        tracker: Optional[ExecutionTracker] = None,
        episode_id: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        if episode_id is None:
            episode_id = self._episode_index
            self._episode_index += 1
        self._current_episode_id = int(episode_id)
        payload = {
            "record_type": "episode_start",
            "episode_id": self._current_episode_id,
            "mode": str(mode),
            "instruction": instruction,
            "unix_time": time.time(),
            "tracker_state": None if tracker is None else tracker.to_dict(),
            "extra": _to_jsonable(extra or {}),
        }
        self._write_json_line(payload)
        return self._current_episode_id

    def end_episode(self, *, reason: str = "manual_stop", extra: Optional[Dict[str, Any]] = None) -> None:
        self._write_json_line(
            {
                "record_type": "episode_end",
                "episode_id": self._current_episode_id,
                "unix_time": time.time(),
                "reason": str(reason),
                "extra": _to_jsonable(extra or {}),
            }
        )

    def capture_sample_from_env(
        self,
        env,
        *,
        mode: str,
        action: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> TrackerSample:
        ts = float(time.time() if timestamp is None else timestamp)
        if mode == MODE_BASE_VLA:
            p_tb3, R_tb3 = env.env.get_pR_body("tb3_base")
            yaw = float(np.arctan2(float(R_tb3[1, 0]), float(R_tb3[0, 0])))
            wheel_vel = np.array(
                [
                    env.env.get_qvel_joint("wheel_left_joint")[0],
                    env.env.get_qvel_joint("wheel_right_joint")[0],
                ],
                dtype=np.float64,
            )
            return TrackerSample(
                timestamp=ts,
                base_xy=np.array([float(p_tb3[0]), float(p_tb3[1])], dtype=np.float64),
                base_yaw=yaw,
                wheel_vel=wheel_vel,
                action=None if action is None else np.asarray(action, dtype=np.float64),
            )

        if mode == MODE_ARM_VLA:
            joint_pos = np.asarray(
                env.env.get_qpos_joints(joint_names=env.joint_names),
                dtype=np.float64,
            )
            gripper_raw = env.env.get_qpos_joint("rh_r1")
            if isinstance(gripper_raw, np.ndarray):
                gripper_cmd = float(gripper_raw[0])
            else:
                gripper_cmd = float(gripper_raw)
            p_tcp, R_tcp = env.env.get_pR_body("tcp_link")
            tcp_rpy = _r2rpy(R_tcp).reshape(-1)
            return TrackerSample(
                timestamp=ts,
                joint_pos=joint_pos,
                gripper_cmd=gripper_cmd,
                tcp_pos=np.asarray(p_tcp, dtype=np.float64).reshape(-1),
                tcp_rpy=tcp_rpy,
                action=None if action is None else np.asarray(action, dtype=np.float64),
            )

        raise ValueError("Unsupported mode: %s" % mode)

    def record_env_step(
        self,
        *,
        env,
        mode: str,
        tracker: Optional[ExecutionTracker] = None,
        action: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        step: Optional[int] = None,
        instruction: Optional[str] = None,
        tag: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackerResult]:
        sample = self.capture_sample_from_env(
            env,
            mode=mode,
            action=action,
            timestamp=timestamp,
        )
        return self.record_sample(
            sample=sample,
            env=env,
            mode=mode,
            tracker=tracker,
            step=step,
            instruction=instruction,
            tag=tag,
            extra=extra,
        )

    def record_sample(
        self,
        *,
        sample: TrackerSample,
        env=None,
        mode: str,
        tracker: Optional[ExecutionTracker] = None,
        step: Optional[int] = None,
        instruction: Optional[str] = None,
        tag: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrackerResult]:
        wall_time = self._get_env_wall_time(env)
        unix_time = time.time()
        tracker_result = tracker.update(sample) if tracker is not None else None
        tracker_debug = None if tracker is None else tracker.get_debug_snapshot()

        payload = {
            "record_type": "tracker_step",
            "episode_id": self._current_episode_id,
            "step": None if step is None else int(step),
            "mode": str(mode),
            "instruction": instruction,
            "tag": tag,
            "unix_time": unix_time,
            "sample_timestamp": float(sample.timestamp),
            "env_wall_time": wall_time,
            "sample": _to_jsonable(sample),
            "tracker_result": None if tracker_result is None else _to_jsonable(tracker_result),
            "tracker_debug": None if tracker_debug is None else _to_jsonable(tracker_debug),
            "extra": _to_jsonable(extra or {}),
        }
        self._write_json_line(payload)
        return tracker_result

    def _get_env_wall_time(self, env) -> Optional[float]:
        if env is None:
            return None
        maybe_env = getattr(env, "env", None)
        if maybe_env is None:
            return None
        get_wall_time = getattr(maybe_env, "get_wall_time", None)
        if callable(get_wall_time):
            try:
                return float(get_wall_time())
            except Exception:
                return None
        return None

    def _write_json_line(self, payload: Dict[str, Any]) -> None:
        self._fp.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")
        self._record_count += 1
        if self._record_count % self.flush_every == 0:
            self._fp.flush()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create an empty execution-tracker trace file with metadata."
    )
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--mode", type=str, default="", help="可选：base_vla 或 arm_vla")
    parser.add_argument("--instruction", type=str, default="", help="可选：任务指令")
    args = parser.parse_args()

    with ExecutionTrackerTraceRecorder(args.output) as recorder:
        if args.mode:
            recorder.start_episode(mode=args.mode, instruction=args.instruction or None)
        print("Trace file ready: %s" % recorder.output_path)


if __name__ == "__main__":
    main()
