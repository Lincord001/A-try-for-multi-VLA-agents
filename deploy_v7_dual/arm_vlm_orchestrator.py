"""
arm_vlm_orchestrator.py
-----------------------
Runtime orchestration for arm VLA:
- monitors arm execution via ExecutionTracker
- triggers VLM checks on tail-event / timeout
- optionally runs a smooth-return-home recovery before resuming VLA
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from orchestration.execution_tracker import ExecutionTracker, ARM_VLA_PROFILE, TrackerSample
from orchestration.vlm_verifier import VLMVerifier, VLMVerifierConfig
from .config import (
    ARM_VLM_HANDOFF_MAX_JOINT_DELTA,
    ARM_VLM_HANDOFF_WARMUP_STEPS,
)


LOGGER = logging.getLogger("arm_vlm_orchestrator")


class ArmVLMOrchestrator:
    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str,
        model: str,
    ):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.model = str(model)
        self.run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.check_index = 0
        self.task_index = -1
        self.current_task_dir: Optional[Path] = None
        self.tracker: Optional[ExecutionTracker] = None
        self.verifier: Optional[VLMVerifier] = None
        self.recovery_active = False
        self.pending_check: Optional[Dict[str, Any]] = None
        self.pending_future: Optional[Future] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.resume_warmup_steps_remaining = 0
        self.resume_max_joint_delta = float(ARM_VLM_HANDOFF_MAX_JOINT_DELTA)
        self.last_verdict: Dict[str, Any] = {}

        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.run_tag).mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arm_vlm")
        self.tracker = ExecutionTracker(ARM_VLA_PROFILE)
        try:
            self.verifier = VLMVerifier(VLMVerifierConfig(model=self.model))
        except Exception as exc:
            LOGGER.warning("Arm VLM verifier unavailable, tracker remains active: %s", exc)
            self.verifier = None

    def on_auto_start(self, env) -> None:
        if not self.enabled or self.tracker is None:
            return
        self.recovery_active = False
        self.pending_check = None
        self.pending_future = None
        self.resume_warmup_steps_remaining = 0
        self.tracker.reset(mode="arm_vla", instruction=getattr(env, "instruction", None))
        self.task_index += 1
        self.current_task_dir = self._build_task_dir(env)
        self._write_task_manifest(
            self.current_task_dir,
            {
                "status": "started",
                "instruction": getattr(env, "instruction", None),
                "task_index": self.task_index,
            },
        )

    def on_auto_stop(self, reason: str) -> None:
        if not self.enabled:
            return
        self.recovery_active = False
        self.pending_check = None
        self.pending_future = None
        self.resume_warmup_steps_remaining = 0
        self.last_verdict = {"status": "stopped", "reason": reason}
        if self.current_task_dir is not None:
            self._write_task_manifest(
                self.current_task_dir,
                {
                    "status": "stopped",
                    "reason": reason,
                    "task_index": self.task_index,
                },
            )

    def after_arm_step(self, env, action, *, step: int) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.tracker is None:
            return None
        if self.recovery_active or self.pending_check is not None:
            return None

        result = self.tracker.update(self._capture_arm_sample(env, action))
        if not result.should_run_vlm:
            return None

        check_dir = self._build_check_dir(trigger_reason=result.reason)
        self.check_index += 1
        check_dir.mkdir(parents=True, exist_ok=True)
        image_paths = self._save_arm_views(env, check_dir)
        self.pending_check = {
            "tracker_result": result,
            "step": int(step),
            "check_dir": check_dir,
            "image_paths": image_paths,
            "instruction": str(getattr(env, "instruction", "")),
            "queued_at": time.perf_counter(),
        }
        self._log_trigger(result, step=step, check_dir=check_dir)
        self._write_task_manifest(
            self.current_task_dir,
            {
                "status": "pending_vlm_check",
                "reason": result.reason,
                "task_index": self.task_index,
                "last_check_dir": str(check_dir),
                "tracker_confidence": result.confidence,
                "event_time": result.event_time,
            },
        )
        return {
            "status": "pause_for_vlm_check",
            "reason": result.reason,
            "check_dir": str(check_dir),
            "event_time": result.event_time,
        }

    def process_pending_check(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.tracker is None or self.pending_check is None:
            return None

        pending = self.pending_check
        tracker_result = pending["tracker_result"]
        check_dir = pending["check_dir"]

        if self.verifier is None:
            self.last_verdict = {
                "status": "verification_unavailable",
                "reason": tracker_result.reason,
                "check_dir": str(check_dir),
            }
            self._write_task_manifest(
                self.current_task_dir,
                {
                    "status": "verification_unavailable",
                    "reason": tracker_result.reason,
                    "task_index": self.task_index,
                    "last_check_dir": str(check_dir),
                },
            )
            self.pending_check = None
            return self.last_verdict

        if self.pending_future is None:
            if self.executor is None:
                raise RuntimeError("ArmVLMOrchestrator executor not initialized")
            queued_for = time.perf_counter() - float(pending.get("queued_at", time.perf_counter()))
            print(
                "[ARM-VLM][timing] submit_check "
                f"task={self.task_index} step={pending['step']} reason={tracker_result.reason} "
                f"queued_for={queued_for:.3f}s check_dir={check_dir}"
            )
            pending["submitted_at"] = time.perf_counter()
            self.pending_future = self.executor.submit(
                self._run_pending_check,
                pending,
            )
            return None

        if not self.pending_future.done():
            return None

        try:
            verify_result = self.pending_future.result()
        except Exception as exc:
            self.pending_future = None
            self.pending_check = None
            self.last_verdict = {
                "status": "verification_unavailable",
                "reason": "%s: %s" % (tracker_result.reason, exc),
                "check_dir": str(check_dir),
            }
            LOGGER.warning("Arm VLM verification failed: %s", exc)
            self._write_task_manifest(
                self.current_task_dir,
                {
                    "status": "verification_unavailable",
                    "reason": "%s: %s" % (tracker_result.reason, exc),
                    "task_index": self.task_index,
                    "last_check_dir": str(check_dir),
                },
            )
            return self.last_verdict

        self.pending_future = None
        submitted_at = float(pending.get("submitted_at", time.perf_counter()))
        future_dt = time.perf_counter() - submitted_at
        print(
            "[ARM-VLM][timing] future_resolved "
            f"task={self.task_index} step={pending['step']} reason={tracker_result.reason} "
            f"wait_after_submit={future_dt:.3f}s check_dir={check_dir}"
        )
        self._save_check_artifacts(check_dir, tracker_result, verify_result, pending["step"])
        self.pending_check = None

        if verify_result.verdict == "success":
            self.tracker.mark_vlm_result(success=True)
            self.last_verdict = {
                "status": "success",
                "reason": tracker_result.reason,
                "vlm": verify_result,
                "check_dir": str(check_dir),
            }
            self._write_task_manifest(
                self.current_task_dir,
                {
                    "status": "success",
                    "reason": tracker_result.reason,
                    "task_index": self.task_index,
                    "last_check_dir": str(check_dir),
                    "vlm_verdict": verify_result.verdict,
                    "vlm_confidence": verify_result.confidence,
                    "vlm_rationale": verify_result.rationale,
                },
            )
            return self.last_verdict

        if verify_result.recoverable:
            self.tracker.mark_vlm_result(success=False, recoverable=True)
            self.recovery_active = True
            self.last_verdict = {
                "status": "recoverable",
                "reason": tracker_result.reason,
                "vlm": verify_result,
                "check_dir": str(check_dir),
            }
            self._write_task_manifest(
                self.current_task_dir,
                {
                    "status": "recoverable_failure",
                    "reason": tracker_result.reason,
                    "task_index": self.task_index,
                    "last_check_dir": str(check_dir),
                    "vlm_verdict": verify_result.verdict,
                    "vlm_confidence": verify_result.confidence,
                    "vlm_rationale": verify_result.rationale,
                },
            )
            return self.last_verdict

        self.tracker.mark_vlm_result(success=False, recoverable=False)
        self.last_verdict = {
            "status": "fail",
            "reason": tracker_result.reason,
            "vlm": verify_result,
            "check_dir": str(check_dir),
        }
        self._write_task_manifest(
            self.current_task_dir,
            {
                "status": "irrecoverable_failure",
                "reason": tracker_result.reason,
                "task_index": self.task_index,
                "last_check_dir": str(check_dir),
                "vlm_verdict": verify_result.verdict,
                "vlm_confidence": verify_result.confidence,
                "vlm_rationale": verify_result.rationale,
            },
        )
        return self.last_verdict

    def _run_pending_check(self, pending: Dict[str, Any]):
        tracker_result = pending["tracker_result"]
        return self.verifier.verify_arm_task(
            agent_image_path=str(pending["image_paths"]["agent"]),
            wrist_image_path=str(pending["image_paths"]["wrist"]),
            instruction=str(pending["instruction"]),
            trigger_reason=str(tracker_result.reason),
            extra_context={
                "step": pending["step"],
                "event_time": tracker_result.event_time,
                "tracker_reason": tracker_result.reason,
                "tracker_confidence": tracker_result.confidence,
            },
        )

    def step_recovery(self, env) -> bool:
        if not self.enabled or not self.recovery_active or self.tracker is None:
            return False
        env.smooth_return_home()
        zero_action = np.concatenate([np.zeros(6, dtype=np.float32), [float(env.gripper_state)]], dtype=np.float32)
        env.step(zero_action, mode='arm', action_type='eef_pose')
        env.p0, env.R0 = env.env.get_pR_body(body_name='tcp_link')
        total_steps = int(max(getattr(env, "home_total_steps", 0), 1))
        interp_steps = int(getattr(env, "home_interp_steps", 0))
        if interp_steps == 1 or interp_steps % 10 == 0:
            print(
                "\n[ARM-VLM] Recovery in progress: "
                f"{interp_steps}/{total_steps} step(s) of smooth return-home."
            )
        if not getattr(env, "returning_home", False):
            self.recovery_active = False
            print("\n[ARM-VLM] Recovery completed: smooth return-home finished.")
            if self.current_task_dir is not None:
                self._write_task_manifest(
                    self.current_task_dir,
                    {
                        "status": "recovery_completed",
                        "task_index": self.task_index,
                        "recovery_attempts": self.tracker.recovery_attempts,
                    },
                )
            self.tracker.reset(
                mode="arm_vla",
                instruction=getattr(env, "instruction", None),
                preserve_recovery_attempts=True,
            )
            self.pending_check = None
            return True
        return False

    def finalize_recovery_handoff(self, env, arm_policy, arm_runner, arm_smoother) -> None:
        if arm_policy is not None:
            arm_policy.reset()
        if arm_runner is not None:
            if hasattr(arm_runner, "running") and not bool(getattr(arm_runner, "running", False)):
                arm_runner.start()
            else:
                arm_runner.reset_state()
        current_state = np.asarray(env.get_joint_state(), dtype=np.float64).reshape(-1)
        arm_smoother.reset()
        arm_smoother.prime_from_state(current_state)
        self.resume_warmup_steps_remaining = int(max(ARM_VLM_HANDOFF_WARMUP_STEPS, 0))
        runner_status = (
            arm_runner.debug_status()
            if arm_runner is not None and hasattr(arm_runner, "debug_status")
            else {
                "running": bool(getattr(arm_runner, "running", False)) if arm_runner is not None else False,
                "has_chunk": False,
                "chunk_len": 0,
                "current_step_index": 0,
            }
        )
        print(
            "\n[ARM-VLM] Recovery handoff complete: "
            "ARM policy/runner reset, auto control will resume on the next control tick."
        )
        print(
            "[ARM-VLM] Post-recovery runner state: "
            f"running={runner_status.get('running')} "
            f"has_chunk={runner_status.get('has_chunk')} "
            f"chunk_len={runner_status.get('chunk_len')} "
            f"step_index={runner_status.get('current_step_index')}"
        )

    def limit_resume_action(self, action_step, robot_state):
        if action_step is None or self.resume_warmup_steps_remaining <= 0:
            return action_step
        ref = np.asarray(robot_state, dtype=np.float64).reshape(-1)
        action = np.asarray(action_step, dtype=np.float64).reshape(-1).copy()
        if action.shape[0] >= 6 and ref.shape[0] >= 6:
            low = ref[:6] - self.resume_max_joint_delta
            high = ref[:6] + self.resume_max_joint_delta
            action[:6] = np.clip(action[:6], low, high)
        self.resume_warmup_steps_remaining -= 1
        return action

    def _capture_arm_sample(self, env, action) -> TrackerSample:
        joint_pos = np.asarray(env.env.get_qpos_joints(joint_names=env.joint_names), dtype=np.float64)
        gripper_raw = env.env.get_qpos_joint("rh_r1")
        gripper_cmd = float(gripper_raw[0]) if isinstance(gripper_raw, np.ndarray) else float(gripper_raw)
        p_tcp, R_tcp = env.env.get_pR_body("tcp_link")
        tcp_rpy = self._r2rpy(R_tcp)
        return TrackerSample(
            timestamp=float(env.env.get_wall_time()),
            joint_pos=joint_pos,
            gripper_cmd=gripper_cmd,
            tcp_pos=np.asarray(p_tcp, dtype=np.float64).reshape(-1),
            tcp_rpy=tcp_rpy,
            action=None if action is None else np.asarray(action, dtype=np.float64),
        )

    def _save_arm_views(self, env, check_dir: Path) -> Dict[str, Path]:
        images = env.grab_image()
        agent = Image.fromarray(images["agent"])
        wrist = Image.fromarray(images["wrist"])
        agent_path = check_dir / "agent.png"
        wrist_path = check_dir / "wrist.png"
        agent.save(agent_path)
        wrist.save(wrist_path)
        return {"agent": agent_path, "wrist": wrist_path}

    def _save_check_artifacts(self, check_dir: Path, tracker_result, verify_result, step: int) -> None:
        payload = {
            "run_tag": self.run_tag,
            "task_index": self.task_index,
            "step": int(step),
            "tracker_result": {
                "completed": tracker_result.completed,
                "confidence": tracker_result.confidence,
                "reason": tracker_result.reason,
                "should_pause_vla": tracker_result.should_pause_vla,
                "should_run_vlm": tracker_result.should_run_vlm,
                "tail_event_detected": tracker_result.tail_event_detected,
                "timeout_triggered": tracker_result.timeout_triggered,
                "recommend_recovery": tracker_result.recommend_recovery,
                "recovery_budget_remaining": tracker_result.recovery_budget_remaining,
                "event_time": tracker_result.event_time,
                "debug": tracker_result.debug,
            },
            "vlm_result": {
                "verdict": verify_result.verdict,
                "confidence": verify_result.confidence,
                "rationale": verify_result.rationale,
                "recoverable": verify_result.recoverable,
                "target_identified": verify_result.target_identified,
                "raw_text": verify_result.raw_text,
            },
        }
        with (check_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        summary = {
            "task_index": self.task_index,
            "check_index": self.check_index,
            "trigger_reason": tracker_result.reason,
            "event_time": tracker_result.event_time,
            "verdict": verify_result.verdict,
            "confidence": verify_result.confidence,
            "recoverable": verify_result.recoverable,
            "target_identified": verify_result.target_identified,
            "rationale": verify_result.rationale,
        }
        with (check_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def _build_task_dir(self, env) -> Path:
        safe_instruction = self._slugify(str(getattr(env, "instruction", "task")))
        task_dir = self.output_dir / self.run_tag / ("task_%03d_%s" % (self.task_index, safe_instruction))
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _build_check_dir(self, *, trigger_reason: str) -> Path:
        base_dir = self.current_task_dir or (self.output_dir / self.run_tag / ("task_%03d_unknown" % max(self.task_index, 0)))
        check_dir = base_dir / ("check_%03d_%s" % (self.check_index, self._slugify(trigger_reason)))
        return check_dir

    def _write_task_manifest(self, task_dir: Optional[Path], payload: Dict[str, Any]) -> None:
        if task_dir is None:
            return
        manifest_path = task_dir / "task_manifest.json"
        existing: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(payload)
        existing.setdefault("run_tag", self.run_tag)
        existing.setdefault("task_dir", str(task_dir))
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    def _log_trigger(self, tracker_result, *, step: int, check_dir: Path) -> None:
        debug = tracker_result.debug or {}
        reason = str(tracker_result.reason)
        event_time = tracker_result.event_time
        event_time_str = "n/a" if event_time is None else f"{float(event_time):.3f}"
        if reason == "arm_tail_event_detected":
            print(
                "\n🧭 [ARM-VLM] Tail event detected | "
                f"step={step} | event_time={event_time_str} | "
                f"release_z={debug.get('z_release', float('nan')):.4f} | "
                f"descend={debug.get('descend_before_release', float('nan')):.4f} | "
                f"lift={debug.get('lift_after_release', float('nan')):.4f} | "
                f"stable_open_ratio={debug.get('stable_open_ratio', float('nan')):.2f}"
            )
            print(f"   check_dir: {check_dir}")
            return
        if reason == "arm_timeout_without_tail_event":
            print(
                "\n⏱️ [ARM-VLM] Timeout trigger | "
                f"step={step} | event_time={event_time_str} | "
                f"active_elapsed={debug.get('active_elapsed', float('nan')):.2f}s | "
                f"net_progress={debug.get('net_progress', float('nan')):.4f}"
            )
            print(f"   check_dir: {check_dir}")

    @staticmethod
    def _slugify(text: str) -> str:
        lowered = str(text).strip().lower()
        chars = []
        for ch in lowered:
            if ch.isalnum():
                chars.append(ch)
            elif ch in {" ", "-", "_"}:
                chars.append("_")
        slug = "".join(chars).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug or "item"

    @staticmethod
    def _r2rpy(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0.0
        return np.array([roll, pitch, yaw], dtype=np.float64)
