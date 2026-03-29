"""Runtime execution monitoring for VLA-controlled tasks.

The tracker is designed for VLA-controlled terminal phases:
- ``base_vla``: fine alignment near a navigation goal.
- ``arm_vla``: arm-only manipulation monitoring for tail-event / timeout checks.

It intentionally does not rely on task ground-truth signals.

Behavior by mode:
- ``base_vla`` keeps the original terminal oscillation completion heuristic.
- ``arm_vla`` no longer directly decides task success. Instead it detects:
  - tail-event candidates: ``closed grasp -> low place point -> release -> lift``
  - timeout checks when no tail event appears for too long
  - whether recovery budget is still available after a VLM rejection
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
import math
from typing import Deque, Dict, List, Optional

import numpy as np


MODE_BASE_VLA = "base_vla"
MODE_ARM_VLA = "arm_vla"

STATE_SEEKING = "SEEKING"
STATE_ACTIVE = "ACTIVE"
STATE_SETTLING = "SETTLING"
STATE_COMPLETED = "COMPLETED"
STATE_RUNNING = "RUNNING"
STATE_TAIL_EVENT_PENDING_VLM = "TAIL_EVENT_PENDING_VLM"
STATE_TIMEOUT_PENDING_VLM = "TIMEOUT_PENDING_VLM"
STATE_FAILED = "FAILED"


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _as_np(value: Optional[np.ndarray], expected_dim: Optional[int] = None) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError("Unexpected array dimension: expected %d, got %d" % (expected_dim, arr.shape[0]))
    return arr


@dataclass
class TrackerSample:
    timestamp: float
    base_xy: Optional[np.ndarray] = None
    base_yaw: Optional[float] = None
    wheel_vel: Optional[np.ndarray] = None
    joint_pos: Optional[np.ndarray] = None
    gripper_cmd: Optional[float] = None
    tcp_pos: Optional[np.ndarray] = None
    tcp_rpy: Optional[np.ndarray] = None
    action: Optional[np.ndarray] = None

    def normalized(self) -> "TrackerSample":
        return TrackerSample(
            timestamp=float(self.timestamp),
            base_xy=_as_np(self.base_xy, expected_dim=2),
            base_yaw=None if self.base_yaw is None else float(self.base_yaw),
            wheel_vel=_as_np(self.wheel_vel, expected_dim=2),
            joint_pos=_as_np(self.joint_pos),
            gripper_cmd=None if self.gripper_cmd is None else float(self.gripper_cmd),
            tcp_pos=_as_np(self.tcp_pos, expected_dim=3),
            tcp_rpy=_as_np(self.tcp_rpy),
            action=_as_np(self.action),
        )


@dataclass
class TrackerResult:
    completed: bool
    confidence: float
    reason: str
    should_pause_vla: bool = False
    should_run_vlm: bool = False
    tail_event_detected: bool = False
    timeout_triggered: bool = False
    recommend_recovery: bool = False
    recovery_budget_remaining: int = 0
    event_time: Optional[float] = None
    debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class ExecutionTrackerConfig:
    history_seconds: float = 6.0
    min_active_seconds: float = 1.5
    min_progress_distance: float = 0.25
    stable_window_seconds: float = 1.2
    completion_hold_seconds: float = 1.0
    progress_epsilon: float = 0.02
    quiet_ratio_threshold: float = 0.75
    min_oscillation_crossings: int = 2

    base_stationary_radius: float = 0.08
    base_stationary_yaw_delta: float = math.radians(12.0)
    base_oscillation_radius: float = 0.03
    base_wheel_vel_deadband: float = 1.2
    base_action_deadband: float = 1.0

    arm_tcp_stationary_radius: float = 0.035
    arm_tcp_stationary_angle: float = math.radians(10.0)
    arm_joint_vel_deadband: float = 0.08
    arm_gripper_stable_delta: float = 0.08
    arm_action_deadband: float = 0.06
    arm_closed_threshold: float = 0.8
    arm_open_threshold: float = 0.1
    arm_tail_pre_window_seconds: float = 2.0
    arm_tail_post_window_seconds: float = 6.5
    arm_tail_lift_threshold: float = 0.09
    arm_tail_descend_threshold: float = 0.04
    arm_timeout_seconds: float = 45.0
    arm_retrigger_cooldown_seconds: float = 8.0
    arm_recovery_cooldown_seconds: float = 4.0
    max_recovery_attempts: int = 2

    def for_mode(self, mode: str) -> "ExecutionTrackerConfig":
        if mode not in (MODE_BASE_VLA, MODE_ARM_VLA):
            raise ValueError("Unsupported tracker mode: %s" % mode)
        return self


BASE_VLA_PROFILE = ExecutionTrackerConfig(
    history_seconds=6.0,
    min_active_seconds=1.0,
    min_progress_distance=0.35,
    stable_window_seconds=1.0,
    completion_hold_seconds=0.9,
    progress_epsilon=0.025,
    quiet_ratio_threshold=0.75,
    min_oscillation_crossings=2,
    base_stationary_radius=0.09,
    base_stationary_yaw_delta=math.radians(10.0),
    base_oscillation_radius=0.035,
    base_wheel_vel_deadband=1.5,
    base_action_deadband=1.4,
)

ARM_VLA_PROFILE = ExecutionTrackerConfig(
    history_seconds=8.0,
    min_active_seconds=1.5,
    min_progress_distance=0.18,
    stable_window_seconds=1.2,
    completion_hold_seconds=1.1,
    progress_epsilon=0.015,
    quiet_ratio_threshold=0.8,
    min_oscillation_crossings=2,
    arm_tcp_stationary_radius=0.03,
    arm_tcp_stationary_angle=math.radians(8.0),
    arm_joint_vel_deadband=0.07,
    arm_gripper_stable_delta=0.05,
    arm_action_deadband=0.05,
    arm_closed_threshold=0.8,
    arm_open_threshold=0.1,
    arm_tail_pre_window_seconds=2.0,
    arm_tail_post_window_seconds=6.8,
    arm_tail_lift_threshold=0.04,
    arm_tail_descend_threshold=0.04,
    arm_timeout_seconds=60.0,
    arm_retrigger_cooldown_seconds=8.0,
    arm_recovery_cooldown_seconds=4.0,
    max_recovery_attempts=2,
)


class ExecutionTracker:
    """Online completion detector for VLA-controlled execution phases."""

    def __init__(self, config: Optional[ExecutionTrackerConfig] = None):
        self.config = replace(config or ExecutionTrackerConfig())
        self.reset()

    def reset(
        self,
        mode: Optional[str] = None,
        instruction: Optional[str] = None,
        preserve_recovery_attempts: bool = False,
    ) -> None:
        recovery_attempts = self.recovery_attempts if preserve_recovery_attempts else 0
        self.mode = mode
        self.instruction = instruction
        self.state = STATE_RUNNING if mode == MODE_ARM_VLA else STATE_SEEKING
        self.samples: Deque[TrackerSample] = deque()
        self.start_time: Optional[float] = None
        self.progress_origin_base_xy: Optional[np.ndarray] = None
        self.progress_origin_tcp_pos: Optional[np.ndarray] = None
        self.settling_since: Optional[float] = None
        self.completed_once = False
        self.failed_once = False
        self.recovery_attempts = recovery_attempts
        self._pending_release: Optional[Dict[str, object]] = None
        self._tail_event_time: Optional[float] = None
        self._timeout_event_time: Optional[float] = None
        self._last_arm_event_time: Optional[float] = None
        self._arm_resume_blocked_until: Optional[float] = None
        self._last_debug: Dict[str, object] = {
            "state": self.state,
            "reason": "reset",
            "mode": self.mode,
            "instruction": self.instruction,
        }

    def update(self, sample: TrackerSample) -> TrackerResult:
        if self.mode is None:
            raise ValueError("Tracker mode is not set. Call reset(mode=...) before update().")

        cfg = self.config.for_mode(self.mode)
        normalized = sample.normalized()
        self._validate_sample(normalized)
        self._append_sample(normalized, cfg.history_seconds)

        if self.mode == MODE_ARM_VLA:
            return self._update_arm(normalized, cfg)

        metrics = self._compute_metrics(cfg)
        completed = self.completed_once
        reason = str(metrics.get("reason", "tracking"))

        if self.completed_once:
            self.state = STATE_COMPLETED
            metrics["state"] = self.state
            return self._result(True, 1.0, "already_completed", metrics)

        if self.state == STATE_SEEKING:
            if self._has_sufficient_activation(metrics, cfg):
                self.state = STATE_ACTIVE
                reason = "activated_after_progress"
            else:
                reason = str(metrics.get("reason", "insufficient_progress"))
        elif self.state == STATE_ACTIVE:
            if self._is_settling_candidate(metrics, cfg):
                self.state = STATE_SETTLING
                self.settling_since = normalized.timestamp
                reason = "entered_settling"
        elif self.state == STATE_SETTLING:
            if not self._is_settling_candidate(metrics, cfg):
                self.state = STATE_ACTIVE
                self.settling_since = None
                reason = "settling_broken_by_motion"
            else:
                settling_elapsed = normalized.timestamp - float(self.settling_since or normalized.timestamp)
                metrics["settling_elapsed"] = settling_elapsed
                if settling_elapsed >= cfg.completion_hold_seconds:
                    self.state = STATE_COMPLETED
                    self.completed_once = True
                    completed = True
                    reason = "completion_hold_satisfied"
        else:
            completed = True
            reason = "already_completed"

        confidence = self._estimate_confidence(metrics, cfg, completed)
        metrics["state"] = self.state
        metrics["reason"] = reason
        self._last_debug = metrics
        return self._result(completed, confidence, reason, metrics)

    def get_debug_snapshot(self) -> Dict[str, object]:
        return dict(self._last_debug)

    def mark_vlm_result(self, *, success: bool, recoverable: bool = False) -> None:
        """Feed back a VLM decision after a tail event or timeout check."""
        if self.mode != MODE_ARM_VLA:
            return
        if success:
            self.state = STATE_COMPLETED
            self.completed_once = True
            return
        if recoverable and self.recovery_attempts < self.config.max_recovery_attempts:
            self.recovery_attempts += 1
            now = self.samples[-1].timestamp if self.samples else 0.0
            self._arm_resume_blocked_until = now + self.config.arm_recovery_cooldown_seconds
            self.state = STATE_RUNNING
            self._pending_release = None
            self._tail_event_time = None
            self._timeout_event_time = None
            return
        self.state = STATE_FAILED
        self.failed_once = True

    def _result(
        self,
        completed: bool,
        confidence: float,
        reason: str,
        debug: Dict[str, object],
        *,
        should_pause_vla: bool = False,
        should_run_vlm: bool = False,
        tail_event_detected: bool = False,
        timeout_triggered: bool = False,
        recommend_recovery: bool = False,
        event_time: Optional[float] = None,
    ) -> TrackerResult:
        debug = dict(debug)
        debug["completed"] = bool(completed)
        debug["confidence"] = float(confidence)
        debug["reason"] = str(reason)
        debug["should_pause_vla"] = bool(should_pause_vla)
        debug["should_run_vlm"] = bool(should_run_vlm)
        debug["tail_event_detected"] = bool(tail_event_detected)
        debug["timeout_triggered"] = bool(timeout_triggered)
        debug["recommend_recovery"] = bool(recommend_recovery)
        debug["recovery_budget_remaining"] = int(max(self.config.max_recovery_attempts - self.recovery_attempts, 0))
        debug["event_time"] = None if event_time is None else float(event_time)
        self._last_debug = debug
        return TrackerResult(
            completed=bool(completed),
            confidence=float(confidence),
            reason=str(reason),
            should_pause_vla=bool(should_pause_vla),
            should_run_vlm=bool(should_run_vlm),
            tail_event_detected=bool(tail_event_detected),
            timeout_triggered=bool(timeout_triggered),
            recommend_recovery=bool(recommend_recovery),
            recovery_budget_remaining=int(max(self.config.max_recovery_attempts - self.recovery_attempts, 0)),
            event_time=None if event_time is None else float(event_time),
            debug=debug,
        )

    def _validate_sample(self, sample: TrackerSample) -> None:
        if self.mode == MODE_BASE_VLA:
            if sample.base_xy is None or sample.base_yaw is None or sample.wheel_vel is None:
                raise ValueError("base_vla sample requires base_xy, base_yaw, wheel_vel")
        elif self.mode == MODE_ARM_VLA:
            if sample.tcp_pos is None or sample.joint_pos is None:
                raise ValueError("arm_vla sample requires tcp_pos and joint_pos")
        else:
            raise ValueError("Unsupported tracker mode: %s" % self.mode)

    def _append_sample(self, sample: TrackerSample, history_seconds: float) -> None:
        if self.start_time is None:
            self.start_time = sample.timestamp
        if self.progress_origin_base_xy is None and sample.base_xy is not None:
            self.progress_origin_base_xy = sample.base_xy.copy()
        if self.progress_origin_tcp_pos is None and sample.tcp_pos is not None:
            self.progress_origin_tcp_pos = sample.tcp_pos.copy()
        self.samples.append(sample)
        cutoff = sample.timestamp - history_seconds
        while len(self.samples) > 1 and self.samples[0].timestamp < cutoff:
            self.samples.popleft()

    def _compute_metrics(self, cfg: ExecutionTrackerConfig) -> Dict[str, object]:
        samples = list(self.samples)
        latest = samples[-1]
        first = samples[0]
        now = latest.timestamp
        active_elapsed = now - float(self.start_time if self.start_time is not None else now)
        window_cutoff = now - cfg.stable_window_seconds
        recent = [s for s in samples if s.timestamp >= window_cutoff]
        if not recent:
            recent = [latest]

        metrics: Dict[str, object] = {
            "mode": self.mode,
            "instruction": self.instruction,
            "sample_count": len(samples),
            "history_seconds": now - first.timestamp,
            "active_elapsed": active_elapsed,
            "settling_elapsed": 0.0 if self.settling_since is None else now - self.settling_since,
        }

        if self.mode == MODE_BASE_VLA:
            metrics.update(self._compute_base_metrics(samples, recent, cfg))
        else:
            metrics.update(self._compute_arm_metrics(samples, recent, cfg))

        return metrics

    def _compute_base_metrics(
        self,
        samples: List[TrackerSample],
        recent: List[TrackerSample],
        cfg: ExecutionTrackerConfig,
    ) -> Dict[str, object]:
        positions = np.stack([s.base_xy for s in samples], axis=0)
        recent_positions = np.stack([s.base_xy for s in recent], axis=0)
        yaws = np.asarray([float(s.base_yaw) for s in samples], dtype=np.float64)
        recent_yaws = np.asarray([float(s.base_yaw) for s in recent], dtype=np.float64)
        wheel_vel = np.stack([s.wheel_vel for s in recent], axis=0)

        anchor = self.progress_origin_base_xy if self.progress_origin_base_xy is not None else positions[0]
        net_progress = float(np.linalg.norm(positions[-1] - anchor))
        recent_progress = float(np.linalg.norm(recent_positions[-1] - recent_positions[0]))
        center = np.mean(recent_positions, axis=0)
        radial_distances = np.linalg.norm(recent_positions - center, axis=1)
        recent_span = float(np.max(radial_distances)) if len(radial_distances) > 0 else 0.0
        recent_path_length = float(np.sum(np.linalg.norm(np.diff(recent_positions, axis=0), axis=1))) if len(recent_positions) > 1 else 0.0
        yaw_deltas = np.asarray([_wrap_to_pi(y - recent_yaws[0]) for y in recent_yaws], dtype=np.float64)
        recent_yaw_span = float(np.max(np.abs(yaw_deltas))) if len(yaw_deltas) > 0 else 0.0
        wheel_speed = np.linalg.norm(wheel_vel, axis=1)
        quiet_ratio = float(np.mean(wheel_speed <= cfg.base_wheel_vel_deadband)) if len(wheel_speed) > 0 else 1.0

        action_norms = self._recent_action_norms(recent)
        if action_norms is None:
            action_quiet_ratio = None
        else:
            action_quiet_ratio = float(np.mean(action_norms <= cfg.base_action_deadband))

        principal_axis = self._principal_axis(recent_positions)
        projections = (recent_positions - center) @ principal_axis
        oscillation_crossings = self._count_zero_crossings(projections, cfg.base_oscillation_radius * 0.25)
        path_to_span_ratio = recent_path_length / max(recent_span, 1e-6)
        progress_stalled = recent_progress <= cfg.progress_epsilon
        oscillating = (
            recent_span <= cfg.base_stationary_radius
            and recent_path_length >= cfg.base_oscillation_radius
            and (
                oscillation_crossings >= cfg.min_oscillation_crossings
                or path_to_span_ratio >= 4.0
            )
        )

        reason = "tracking_base_motion"
        if net_progress < cfg.min_progress_distance:
            reason = "base_progress_below_threshold"
        elif recent_span > cfg.base_stationary_radius:
            reason = "base_recent_span_too_large"
        elif recent_yaw_span > cfg.base_stationary_yaw_delta:
            reason = "base_recent_yaw_span_too_large"
        elif quiet_ratio < cfg.quiet_ratio_threshold:
            reason = "base_wheel_velocity_not_quiet"
        elif not progress_stalled:
            reason = "base_recent_progress_not_stalled"
        elif not oscillating:
            reason = "base_terminal_oscillation_not_detected"

        return {
            "net_progress": net_progress,
            "recent_progress": recent_progress,
            "recent_span": recent_span,
            "recent_path_length": recent_path_length,
            "path_to_span_ratio": path_to_span_ratio,
            "recent_yaw_span": recent_yaw_span,
            "quiet_ratio": quiet_ratio,
            "action_quiet_ratio": action_quiet_ratio,
            "oscillation_crossings": oscillation_crossings,
            "progress_stalled": progress_stalled,
            "oscillating": oscillating,
            "reason": reason,
        }

    def _compute_arm_metrics(
        self,
        samples: List[TrackerSample],
        recent: List[TrackerSample],
        cfg: ExecutionTrackerConfig,
    ) -> Dict[str, object]:
        tcp_positions = np.stack([s.tcp_pos for s in samples], axis=0)
        recent_tcp_positions = np.stack([s.tcp_pos for s in recent], axis=0)
        joint_pos = np.stack([s.joint_pos for s in recent], axis=0)

        anchor = self.progress_origin_tcp_pos if self.progress_origin_tcp_pos is not None else tcp_positions[0]
        net_progress = float(np.linalg.norm(tcp_positions[-1] - anchor))
        recent_progress = float(np.linalg.norm(recent_tcp_positions[-1] - recent_tcp_positions[0]))
        center = np.mean(recent_tcp_positions, axis=0)
        radial_distances = np.linalg.norm(recent_tcp_positions - center, axis=1)
        recent_span = float(np.max(radial_distances)) if len(radial_distances) > 0 else 0.0
        recent_path_length = float(np.sum(np.linalg.norm(np.diff(recent_tcp_positions, axis=0), axis=1))) if len(recent_tcp_positions) > 1 else 0.0

        joint_vel_norms = self._joint_velocity_norms(recent, joint_pos)
        quiet_ratio = float(np.mean(joint_vel_norms <= cfg.arm_joint_vel_deadband)) if len(joint_vel_norms) > 0 else 1.0

        gripper_cmds = [s.gripper_cmd for s in recent if s.gripper_cmd is not None]
        gripper_delta = 0.0 if len(gripper_cmds) <= 1 else float(max(gripper_cmds) - min(gripper_cmds))

        recent_angle_span = 0.0
        if all(s.tcp_rpy is not None for s in recent):
            rpy = np.stack([s.tcp_rpy for s in recent], axis=0)
            diffs = np.stack([[_wrap_to_pi(v) for v in row - rpy[0]] for row in rpy], axis=0)
            recent_angle_span = float(np.max(np.linalg.norm(diffs, axis=1))) if len(diffs) > 0 else 0.0

        action_norms = self._recent_action_norms(recent)
        if action_norms is None:
            action_quiet_ratio = None
        else:
            action_quiet_ratio = float(np.mean(action_norms <= cfg.arm_action_deadband))

        principal_axis = self._principal_axis(recent_tcp_positions)
        projections = (recent_tcp_positions - center) @ principal_axis
        oscillation_crossings = self._count_zero_crossings(projections, cfg.arm_tcp_stationary_radius * 0.25)
        path_to_span_ratio = recent_path_length / max(recent_span, 1e-6)
        progress_stalled = recent_progress <= cfg.progress_epsilon
        oscillating = (
            recent_span <= cfg.arm_tcp_stationary_radius
            and recent_path_length >= cfg.arm_tcp_stationary_radius
            and (
                oscillation_crossings >= cfg.min_oscillation_crossings
                or path_to_span_ratio >= 4.0
            )
        )

        reason = "tracking_arm_motion"
        if net_progress < cfg.min_progress_distance:
            reason = "arm_progress_below_threshold"
        elif recent_span > cfg.arm_tcp_stationary_radius:
            reason = "arm_recent_tcp_span_too_large"
        elif recent_angle_span > cfg.arm_tcp_stationary_angle:
            reason = "arm_recent_tcp_angle_span_too_large"
        elif quiet_ratio < cfg.quiet_ratio_threshold:
            reason = "arm_joint_velocity_not_quiet"
        elif gripper_delta > cfg.arm_gripper_stable_delta:
            reason = "arm_gripper_not_stable"
        elif not progress_stalled:
            reason = "arm_recent_progress_not_stalled"
        elif not oscillating:
            reason = "arm_terminal_oscillation_not_detected"

        return {
            "net_progress": net_progress,
            "recent_progress": recent_progress,
            "recent_span": recent_span,
            "recent_path_length": recent_path_length,
            "path_to_span_ratio": path_to_span_ratio,
            "recent_angle_span": recent_angle_span,
            "quiet_ratio": quiet_ratio,
            "action_quiet_ratio": action_quiet_ratio,
            "oscillation_crossings": oscillation_crossings,
            "gripper_delta": gripper_delta,
            "progress_stalled": progress_stalled,
            "oscillating": oscillating,
            "reason": reason,
        }

    def _update_arm(self, sample: TrackerSample, cfg: ExecutionTrackerConfig) -> TrackerResult:
        metrics = self._compute_metrics(cfg)
        now = sample.timestamp
        active_elapsed = float(metrics.get("active_elapsed", 0.0))
        quiet_ratio = float(metrics.get("quiet_ratio", 0.0))
        recommend_recovery = self.recovery_attempts < cfg.max_recovery_attempts
        metrics["recovery_attempts"] = self.recovery_attempts
        metrics["pending_release"] = None if self._pending_release is None else dict(self._pending_release)
        metrics["tail_event_time"] = self._tail_event_time
        metrics["timeout_event_time"] = self._timeout_event_time
        metrics["resume_blocked_until"] = self._arm_resume_blocked_until

        if self.completed_once:
            self.state = STATE_COMPLETED
            metrics["state"] = self.state
            metrics["reason"] = "vlm_confirmed_success"
            return self._result(
                True,
                1.0,
                "vlm_confirmed_success",
                metrics,
                event_time=self._tail_event_time,
            )

        if self.failed_once or self.state == STATE_FAILED:
            self.state = STATE_FAILED
            metrics["state"] = self.state
            metrics["reason"] = "vlm_confirmed_failure"
            return self._result(
                False,
                1.0,
                "vlm_confirmed_failure",
                metrics,
                recommend_recovery=False,
                event_time=self._timeout_event_time or self._tail_event_time,
            )

        if self._arm_resume_blocked_until is not None and now < self._arm_resume_blocked_until:
            self.state = STATE_RUNNING
            metrics["state"] = self.state
            metrics["reason"] = "recovery_cooldown"
            return self._result(
                False,
                0.2,
                "recovery_cooldown",
                metrics,
                recommend_recovery=recommend_recovery,
            )
        self._arm_resume_blocked_until = None

        if self.state == STATE_TAIL_EVENT_PENDING_VLM:
            metrics["state"] = self.state
            metrics["reason"] = "tail_event_pending_vlm"
            return self._result(
                False,
                0.95,
                "tail_event_pending_vlm",
                metrics,
                should_pause_vla=True,
                should_run_vlm=True,
                tail_event_detected=True,
                recommend_recovery=recommend_recovery,
                event_time=self._tail_event_time,
            )

        if self.state == STATE_TIMEOUT_PENDING_VLM:
            metrics["state"] = self.state
            metrics["reason"] = "timeout_pending_vlm"
            return self._result(
                False,
                0.9,
                "timeout_pending_vlm",
                metrics,
                should_pause_vla=True,
                should_run_vlm=True,
                timeout_triggered=True,
                recommend_recovery=recommend_recovery,
                event_time=self._timeout_event_time,
            )

        self.state = STATE_RUNNING
        self._update_arm_release_candidate(sample, cfg)
        tail_event = self._maybe_complete_arm_tail_event(sample, cfg)
        if tail_event is not None:
            self.state = STATE_TAIL_EVENT_PENDING_VLM
            self._tail_event_time = float(tail_event["event_time"])
            self._last_arm_event_time = self._tail_event_time
            metrics.update(tail_event)
            metrics["state"] = self.state
            metrics["reason"] = "arm_tail_event_detected"
            return self._result(
                False,
                self._estimate_arm_event_confidence(metrics, cfg, timeout_triggered=False),
                "arm_tail_event_detected",
                metrics,
                should_pause_vla=True,
                should_run_vlm=True,
                tail_event_detected=True,
                recommend_recovery=recommend_recovery,
                event_time=self._tail_event_time,
            )

        if (
            active_elapsed >= cfg.arm_timeout_seconds
            and self._timeout_event_time is None
            and self._tail_event_time is None
        ):
            self.state = STATE_TIMEOUT_PENDING_VLM
            self._timeout_event_time = now
            self._last_arm_event_time = now
            metrics["state"] = self.state
            metrics["reason"] = "arm_timeout_without_tail_event"
            return self._result(
                False,
                self._estimate_arm_event_confidence(metrics, cfg, timeout_triggered=True),
                "arm_timeout_without_tail_event",
                metrics,
                should_pause_vla=True,
                should_run_vlm=True,
                timeout_triggered=True,
                recommend_recovery=recommend_recovery,
                event_time=now,
            )

        metrics["state"] = self.state
        metrics["reason"] = "arm_running_waiting_for_tail_event"
        return self._result(
            False,
            float(np.clip(0.2 + 0.5 * quiet_ratio, 0.0, 0.8)),
            "arm_running_waiting_for_tail_event",
            metrics,
            recommend_recovery=recommend_recovery,
        )

    def _update_arm_release_candidate(self, sample: TrackerSample, cfg: ExecutionTrackerConfig) -> None:
        samples = list(self.samples)
        if len(samples) < 3:
            return
        now = sample.timestamp
        if self._pending_release is not None:
            age = now - float(self._pending_release["release_time"])
            if age > cfg.arm_tail_post_window_seconds:
                self._pending_release = None
            else:
                return
        if self._last_arm_event_time is not None and now - self._last_arm_event_time < cfg.arm_retrigger_cooldown_seconds:
            return
        pre_window = self._recent_window(now - cfg.arm_tail_pre_window_seconds, now)
        if len(pre_window) < 3:
            return
        closed_pre = max(float(s.gripper_cmd or 0.0) for s in pre_window)
        open_now = float(sample.gripper_cmd or 0.0)
        z_vals = [float(s.tcp_pos[2]) for s in pre_window if s.tcp_pos is not None]
        if len(z_vals) < 2:
            return
        descend = max(z_vals) - min(z_vals)
        release_detected = closed_pre >= cfg.arm_closed_threshold and open_now <= cfg.arm_open_threshold
        if release_detected and descend >= cfg.arm_tail_descend_threshold:
            self._pending_release = {
                "release_time": now,
                "release_z": float(sample.tcp_pos[2]),
                "closed_pre": closed_pre,
                "open_at_release": open_now,
                "descend_before_release": descend,
                "pre_z_min": min(z_vals),
                "pre_z_max": max(z_vals),
            }

    def _maybe_complete_arm_tail_event(
        self,
        sample: TrackerSample,
        cfg: ExecutionTrackerConfig,
    ) -> Optional[Dict[str, object]]:
        if self._pending_release is None:
            return None
        now = sample.timestamp
        age = now - float(self._pending_release["release_time"])
        if age > cfg.arm_tail_post_window_seconds:
            self._pending_release = None
            return None

        post_window = self._recent_window(float(self._pending_release["release_time"]), now)
        if len(post_window) < 3:
            return None
        post_z = [float(s.tcp_pos[2]) for s in post_window if s.tcp_pos is not None]
        post_g = [float(s.gripper_cmd or 0.0) for s in post_window]
        if not post_z:
            return None

        lift_after_release = max(post_z) - float(self._pending_release["release_z"])
        open_after_release_min = min(post_g)
        open_after_release_max = max(post_g)
        stable_open_ratio = float(np.mean(np.asarray(post_g) <= cfg.arm_open_threshold))

        if lift_after_release < cfg.arm_tail_lift_threshold:
            return None
        if stable_open_ratio < 0.6:
            return None

        event = {
            "event_time": float(self._pending_release["release_time"]),
            "closed_before_release_max": float(self._pending_release["closed_pre"]),
            "open_after_release_min": open_after_release_min,
            "open_after_release_max": open_after_release_max,
            "z_release": float(self._pending_release["release_z"]),
            "z_high_after_release": max(post_z),
            "lift_after_release": lift_after_release,
            "descend_before_release": float(self._pending_release["descend_before_release"]),
            "tail_event_age": age,
            "stable_open_ratio": stable_open_ratio,
        }
        self._pending_release = None
        return event

    def _has_sufficient_activation(self, metrics: Dict[str, object], cfg: ExecutionTrackerConfig) -> bool:
        return (
            float(metrics.get("net_progress", 0.0)) >= cfg.min_progress_distance
            or float(metrics.get("active_elapsed", 0.0)) >= cfg.min_active_seconds
        )

    def _is_settling_candidate(self, metrics: Dict[str, object], cfg: ExecutionTrackerConfig) -> bool:
        if float(metrics.get("active_elapsed", 0.0)) < cfg.min_active_seconds:
            return False
        if float(metrics.get("net_progress", 0.0)) < cfg.min_progress_distance:
            return False
        if not bool(metrics.get("progress_stalled", False)):
            return False
        if not bool(metrics.get("oscillating", False)):
            return False
        if float(metrics.get("quiet_ratio", 0.0)) < cfg.quiet_ratio_threshold:
            return False
        action_quiet_ratio = metrics.get("action_quiet_ratio")
        if action_quiet_ratio is not None and float(action_quiet_ratio) < 0.5:
            return False

        if self.mode == MODE_BASE_VLA:
            if float(metrics.get("recent_span", 1.0)) > cfg.base_stationary_radius:
                return False
            if float(metrics.get("recent_yaw_span", 1.0)) > cfg.base_stationary_yaw_delta:
                return False
        else:
            if float(metrics.get("recent_span", 1.0)) > cfg.arm_tcp_stationary_radius:
                return False
            if float(metrics.get("recent_angle_span", 0.0)) > cfg.arm_tcp_stationary_angle:
                return False
            if float(metrics.get("gripper_delta", 1.0)) > cfg.arm_gripper_stable_delta:
                return False
        return True

    def _estimate_confidence(
        self,
        metrics: Dict[str, object],
        cfg: ExecutionTrackerConfig,
        completed: bool,
    ) -> float:
        quiet_ratio = float(metrics.get("quiet_ratio", 0.0))
        crossings = float(metrics.get("oscillation_crossings", 0.0))
        progress = float(metrics.get("net_progress", 0.0))
        hold_ratio = min(1.0, float(metrics.get("settling_elapsed", 0.0)) / max(cfg.completion_hold_seconds, 1e-6))

        progress_score = min(1.0, progress / max(cfg.min_progress_distance, 1e-6))
        crossing_score = min(1.0, crossings / max(float(cfg.min_oscillation_crossings), 1.0))
        stalled_score = 1.0 if bool(metrics.get("progress_stalled", False)) else 0.0
        oscillating_score = 1.0 if bool(metrics.get("oscillating", False)) else 0.0

        score = (
            0.25 * progress_score
            + 0.20 * quiet_ratio
            + 0.20 * crossing_score
            + 0.20 * stalled_score
            + 0.15 * oscillating_score
        )
        if completed:
            score = max(score, 0.95 * hold_ratio + 0.05)
        return float(np.clip(score, 0.0, 1.0))

    def _estimate_arm_event_confidence(
        self,
        metrics: Dict[str, object],
        cfg: ExecutionTrackerConfig,
        *,
        timeout_triggered: bool,
    ) -> float:
        if timeout_triggered:
            elapsed = float(metrics.get("active_elapsed", 0.0))
            timeout_ratio = min(1.0, elapsed / max(cfg.arm_timeout_seconds, 1e-6))
            progress = float(metrics.get("net_progress", 0.0))
            progress_score = min(1.0, progress / max(cfg.min_progress_distance, 1e-6))
            return float(np.clip(0.55 + 0.25 * timeout_ratio + 0.15 * progress_score, 0.0, 0.95))

        closed_score = min(1.0, float(metrics.get("closed_before_release_max", 0.0)) / max(cfg.arm_closed_threshold, 1e-6))
        open_score = 1.0 if float(metrics.get("open_after_release_min", 1.0)) <= cfg.arm_open_threshold else 0.0
        lift_score = min(1.0, float(metrics.get("lift_after_release", 0.0)) / max(cfg.arm_tail_lift_threshold, 1e-6))
        descend_score = min(1.0, float(metrics.get("descend_before_release", 0.0)) / max(cfg.arm_tail_descend_threshold, 1e-6))
        stable_open_score = float(metrics.get("stable_open_ratio", 0.0))
        score = (
            0.25 * closed_score
            + 0.15 * open_score
            + 0.30 * lift_score
            + 0.20 * descend_score
            + 0.10 * stable_open_score
        )
        return float(np.clip(score, 0.0, 0.98))

    def _recent_window(self, start_ts: float, end_ts: float) -> List[TrackerSample]:
        return [s for s in self.samples if start_ts <= s.timestamp <= end_ts]

    def _recent_action_norms(self, recent: List[TrackerSample]) -> Optional[np.ndarray]:
        action_vectors = [s.action for s in recent if s.action is not None]
        if len(action_vectors) != len(recent):
            return None
        return np.asarray([float(np.sqrt(np.mean(np.square(v)))) for v in action_vectors], dtype=np.float64)

    def _joint_velocity_norms(self, recent: List[TrackerSample], joint_pos: np.ndarray) -> np.ndarray:
        if len(recent) <= 1:
            return np.zeros((1,), dtype=np.float64)
        values = []
        for idx in range(1, len(recent)):
            dt = max(recent[idx].timestamp - recent[idx - 1].timestamp, 1e-6)
            joint_speed = (joint_pos[idx] - joint_pos[idx - 1]) / dt
            values.append(float(np.sqrt(np.mean(np.square(joint_speed)))))
        return np.asarray(values, dtype=np.float64)

    def _principal_axis(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            axis = np.zeros((points.shape[1],), dtype=np.float64)
            axis[0] = 1.0
            return axis
        centered = points - np.mean(points, axis=0, keepdims=True)
        if np.allclose(centered, 0.0):
            axis = np.zeros((points.shape[1],), dtype=np.float64)
            axis[0] = 1.0
            return axis
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        norm = np.linalg.norm(axis)
        if norm <= 1e-9:
            axis = np.zeros((points.shape[1],), dtype=np.float64)
            axis[0] = 1.0
            return axis
        return axis / norm

    def _count_zero_crossings(self, values: np.ndarray, deadband: float) -> int:
        signs: List[int] = []
        for raw in values:
            value = float(raw)
            if abs(value) <= deadband:
                continue
            signs.append(1 if value > 0.0 else -1)
        crossings = 0
        for idx in range(1, len(signs)):
            if signs[idx] != signs[idx - 1]:
                crossings += 1
        return crossings

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "instruction": self.instruction,
            "state": self.state,
            "config": asdict(self.config),
            "debug": self.get_debug_snapshot(),
        }
