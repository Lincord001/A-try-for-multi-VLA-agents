"""
execution_trace_manager.py
--------------------------
Execution-tracker trace integration for deployment runtime.

This module keeps trace recording concerns isolated from the control logic:
- owns per-mode recorder files
- owns per-mode tracker instances
- starts/stops episodes with explicit reasons
- records one trace entry per VLA-controlled step
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from orchestration.execution_tracker import (
    ARM_VLA_PROFILE,
    BASE_VLA_PROFILE,
    ExecutionTracker,
)
from record_execution_tracker_trace import ExecutionTrackerTraceRecorder


class ExecutionTraceManager:
    """Manage execution-tracker trace recording for arm/base VLA deployment."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str,
        flush_every: int = 1,
        run_tag: Optional[str] = None,
        evaluate_tracker: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.flush_every = max(1, int(flush_every))
        self.run_tag = str(run_tag or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.evaluate_tracker = bool(evaluate_tracker)
        self.metadata = dict(metadata or {})

        self.arm_recorder: Optional[ExecutionTrackerTraceRecorder] = None
        self.base_recorder: Optional[ExecutionTrackerTraceRecorder] = None
        self.arm_tracker: Optional[ExecutionTracker] = None
        self.base_tracker: Optional[ExecutionTracker] = None
        self.arm_episode_active = False
        self.base_episode_active = False

        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.arm_recorder = ExecutionTrackerTraceRecorder(
            str(self.output_dir / ("arm_vla_trace_%s.jsonl" % self.run_tag)),
            metadata=self._build_metadata(mode="arm_vla"),
            flush_every=self.flush_every,
        )
        self.base_recorder = ExecutionTrackerTraceRecorder(
            str(self.output_dir / ("base_vla_trace_%s.jsonl" % self.run_tag)),
            metadata=self._build_metadata(mode="base_vla"),
            flush_every=self.flush_every,
        )
        self.arm_tracker = ExecutionTracker(ARM_VLA_PROFILE) if self.evaluate_tracker else None
        self.base_tracker = ExecutionTracker(BASE_VLA_PROFILE) if self.evaluate_tracker else None

    def start_arm_episode(self, env, *, step: Optional[int] = None, reason: str = "auto_start") -> None:
        if not self.enabled or self.arm_recorder is None:
            return
        if self.arm_episode_active:
            self.stop_arm_episode(reason="restart_before_new_episode", extra={"next_reason": reason})
        if self.arm_tracker is not None:
            self.arm_tracker.reset(mode="arm_vla", instruction=getattr(env, "instruction", None))
        self.arm_recorder.start_episode(
            mode="arm_vla",
            instruction=getattr(env, "instruction", None),
            tracker=self.arm_tracker,
            extra={
                "reason": reason,
                "step": step,
            },
        )
        self.arm_episode_active = True

    def start_base_episode(self, env, *, step: Optional[int] = None, reason: str = "auto_start") -> None:
        if not self.enabled or self.base_recorder is None:
            return
        if self.base_episode_active:
            self.stop_base_episode(reason="restart_before_new_episode", extra={"next_reason": reason})
        if self.base_tracker is not None:
            self.base_tracker.reset(mode="base_vla", instruction=getattr(env, "instruction", None))
        self.base_recorder.start_episode(
            mode="base_vla",
            instruction=getattr(env, "instruction", None),
            tracker=self.base_tracker,
            extra={
                "reason": reason,
                "step": step,
            },
        )
        self.base_episode_active = True

    def stop_arm_episode(self, *, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or not self.arm_episode_active or self.arm_recorder is None:
            return
        self.arm_recorder.end_episode(reason=reason, extra=extra)
        self.arm_episode_active = False

    def stop_base_episode(self, *, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled or not self.base_episode_active or self.base_recorder is None:
            return
        self.base_recorder.end_episode(reason=reason, extra=extra)
        self.base_episode_active = False

    def stop_current_episode_for_mode(
        self,
        mode: str,
        *,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if mode == "arm":
            self.stop_arm_episode(reason=reason, extra=extra)
        elif mode == "base":
            self.stop_base_episode(reason=reason, extra=extra)

    def record_arm_step(self, env, action, *, step: Optional[int] = None, tag: Optional[str] = None) -> None:
        if not self.enabled or not self.arm_episode_active:
            return
        if self.arm_recorder is None:
            return
        self.arm_recorder.record_env_step(
            env=env,
            mode="arm_vla",
            tracker=self.arm_tracker,
            action=action,
            step=step,
            instruction=getattr(env, "instruction", None),
            tag=tag,
        )

    def record_base_step(self, env, action, *, step: Optional[int] = None, tag: Optional[str] = None) -> None:
        if not self.enabled or not self.base_episode_active:
            return
        if self.base_recorder is None:
            return
        self.base_recorder.record_env_step(
            env=env,
            mode="base_vla",
            tracker=self.base_tracker,
            action=action,
            step=step,
            instruction=getattr(env, "instruction", None),
            tag=tag,
        )

    def close(self) -> None:
        if self.enabled:
            self.stop_arm_episode(reason="session_close")
            self.stop_base_episode(reason="session_close")
        if self.arm_recorder is not None:
            self.arm_recorder.close()
            self.arm_recorder = None
        if self.base_recorder is not None:
            self.base_recorder.close()
            self.base_recorder = None

    def _build_metadata(self, *, mode: str) -> Dict[str, Any]:
        payload = {
            "mode": mode,
            "run_tag": self.run_tag,
            "evaluate_tracker": self.evaluate_tracker,
            "tracker_profile": (
                asdict(ARM_VLA_PROFILE if mode == "arm_vla" else BASE_VLA_PROFILE)
                if self.evaluate_tracker else None
            ),
        }
        payload.update(self.metadata)
        return payload
