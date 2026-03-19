import time
import numpy as np

from .action_utils import drive_with_heading_hold
from .env_constants import (
    BASE_AUTO_FINAL_POS_TOL,
    BASE_AUTO_FINAL_X_TOL,
    BASE_AUTO_FINAL_YAW_TOL,
    BASE_AUTO_KP_ANG,
    BASE_AUTO_KP_LIN,
    BASE_AUTO_MAX_FWD_V,
    BASE_AUTO_MAX_TURN_V,
    BASE_AUTO_MIN_FWD_CMD,
    BASE_AUTO_MIN_TURN_CMD,
    BASE_AUTO_POS_TOL,
    BASE_AUTO_PUSH_FWD_V,
    BASE_AUTO_PUSH_SEC,
    BASE_AUTO_ROTATE_IN_PLACE_TH,
    BASE_AUTO_STAGE_TIMEOUT_SEC,
    BASE_AUTO_STAGE1_TARGET_XY,
    BASE_AUTO_STAGE1_X_TOL,
    BASE_AUTO_STAGE1_Y_TOL,
    BASE_AUTO_STAGE2_TARGET_XY,
    BASE_AUTO_STAGE2_Y_NEAR_MARGIN,
    BASE_AUTO_WAIT_FRAMES,
    BASE_AUTO_YAW_TOL,
    BASE_STRAIGHT_ASSIST_ENABLED,
    BASE_STRAIGHT_KP_YAW,
    BASE_STRAIGHT_MAX_TURN_V,
    BASE_STRAIGHT_YAW_DEADBAND,
)


class BaseAutoParkingAgent:
    """State machine for multi-stage base auto parking."""

    def __init__(self, env):
        self.env = env
        self.active = False
        self.stage = "idle"
        self.wait_counter = 0
        self.wait_deadline = None
        self.stage_steps = 0
        self.recording_active = False
        self.record_stop_requested = False
        self.push_target_yaw = None

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self.active = False
        self.stage = "idle"
        self.wait_counter = 0
        self.wait_deadline = None
        self.stage_steps = 0
        self.recording_active = False
        self.record_stop_requested = False
        self.push_target_yaw = None

    def start(self):
        self.active = True
        self.stage = "goto_stage1"
        self.wait_counter = 0
        self.wait_deadline = None
        self.stage_steps = 0
        self.record_stop_requested = False
        self.push_target_yaw = None
        print(
            "\n🚗 [H] Base auto parking started: "
            "goto (0.2,-0.8) -> goto (0.2,-0.25) -> push forward -> wait 3s -> request stop recording"
        )

    def stop(self, reason):
        self.active = False
        self.stage = "idle"
        self.wait_counter = 0
        self.wait_deadline = None
        self.stage_steps = 0
        self.record_stop_requested = False
        self.push_target_yaw = None
        print(f"🛑 Base auto parking stopped: {reason}")

    def _nav_to_target(
        self,
        target_xy,
        desired_yaw=None,
        pos_tol=BASE_AUTO_POS_TOL,
        yaw_tol=BASE_AUTO_YAW_TOL,
        allow_stop=True,
    ):
        xy, yaw = self.env._get_tb3_pose_xy_yaw()
        delta = target_xy - xy
        dist = np.linalg.norm(delta)

        if desired_yaw is None:
            yaw_ref = np.arctan2(delta[1], delta[0]) if dist > 1e-9 else yaw
        else:
            yaw_ref = desired_yaw

        yaw_err = self._wrap_to_pi(yaw_ref - yaw)

        if abs(yaw_err) > BASE_AUTO_ROTATE_IN_PLACE_TH:
            turn_mag = max(BASE_AUTO_MIN_TURN_CMD, abs(BASE_AUTO_KP_ANG * yaw_err))
            turn = float(
                np.clip(
                    np.sign(yaw_err) * turn_mag,
                    -BASE_AUTO_MAX_TURN_V,
                    BASE_AUTO_MAX_TURN_V,
                )
            )
            return np.array([-turn, turn], dtype=np.float32), False

        if allow_stop and dist <= pos_tol and abs(yaw_err) <= yaw_tol:
            return np.zeros(2, dtype=np.float32), True

        fwd = float(
            np.clip(
                BASE_AUTO_KP_LIN * dist,
                BASE_AUTO_MIN_FWD_CMD,
                BASE_AUTO_MAX_FWD_V,
            )
        )

        turn_raw = BASE_AUTO_KP_ANG * yaw_err
        if abs(yaw_err) > yaw_tol:
            turn_mag = max(BASE_AUTO_MIN_TURN_CMD, abs(turn_raw))
            turn = float(
                np.clip(
                    np.sign(yaw_err) * turn_mag,
                    -BASE_AUTO_MAX_TURN_V,
                    BASE_AUTO_MAX_TURN_V,
                )
            )
        else:
            turn = float(np.clip(turn_raw, -BASE_AUTO_MAX_TURN_V, BASE_AUTO_MAX_TURN_V))
        return np.array([fwd - turn, fwd + turn], dtype=np.float32), False

    def _push_forward_cmd(self):
        if not BASE_STRAIGHT_ASSIST_ENABLED:
            return np.array([BASE_AUTO_PUSH_FWD_V, BASE_AUTO_PUSH_FWD_V], dtype=np.float32)
        if self.push_target_yaw is None:
            _, yaw_now = self.env._get_tb3_pose_xy_yaw()
            self.push_target_yaw = yaw_now
        _, yaw_now = self.env._get_tb3_pose_xy_yaw()
        return drive_with_heading_hold(
            fwd_v=BASE_AUTO_PUSH_FWD_V,
            target_yaw=self.push_target_yaw,
            current_yaw=yaw_now,
            kp_yaw=BASE_STRAIGHT_KP_YAW,
            max_turn_v=BASE_STRAIGHT_MAX_TURN_V,
            yaw_deadband=BASE_STRAIGHT_YAW_DEADBAND,
        )

    def run(self):
        if not self.active:
            self.env._set_base_action_intent([0.0, 0.0])
            return np.zeros(2, dtype=np.float32)

        timeout_frames = int(BASE_AUTO_STAGE_TIMEOUT_SEC * 20)

        if self.stage == "goto_stage1":
            self.stage_steps += 1
            if self.stage_steps > timeout_frames:
                self.stop("stage1 timeout")
                self.env._set_base_action_intent([0.0, 0.0])
                return np.zeros(2, dtype=np.float32)
            cmd, _ = self._nav_to_target(
                BASE_AUTO_STAGE1_TARGET_XY,
                desired_yaw=None,
                pos_tol=max(BASE_AUTO_STAGE1_X_TOL, BASE_AUTO_STAGE1_Y_TOL),
                yaw_tol=np.pi,
                allow_stop=False,
            )
            xy, _ = self.env._get_tb3_pose_xy_yaw()
            x_ok = abs(float(xy[0]) - float(BASE_AUTO_STAGE1_TARGET_XY[0])) <= BASE_AUTO_STAGE1_X_TOL
            y_ok = abs(float(xy[1]) - float(BASE_AUTO_STAGE1_TARGET_XY[1])) <= BASE_AUTO_STAGE1_Y_TOL
            if x_ok and y_ok:
                self.stage = "goto_stage2"
                self.stage_steps = 0
                print("   ✅ Stage 1 passed near: (0.2, -0.8), switching to stage 2 target tracking...")
            self.env._set_base_action_intent(cmd)
            return cmd

        if self.stage == "goto_stage2":
            self.stage_steps += 1
            if self.stage_steps > timeout_frames:
                self.stop("stage2 timeout")
                self.env._set_base_action_intent([0.0, 0.0])
                return np.zeros(2, dtype=np.float32)
            cmd, _ = self._nav_to_target(
                BASE_AUTO_STAGE2_TARGET_XY,
                desired_yaw=None,
                pos_tol=BASE_AUTO_FINAL_POS_TOL,
                yaw_tol=BASE_AUTO_FINAL_YAW_TOL,
                allow_stop=False,
            )
            xy, _ = self.env._get_tb3_pose_xy_yaw()
            x_ok = abs(float(xy[0]) - float(BASE_AUTO_STAGE2_TARGET_XY[0])) <= BASE_AUTO_FINAL_X_TOL
            y_near = float(xy[1]) >= float(BASE_AUTO_STAGE2_TARGET_XY[1]) - BASE_AUTO_STAGE2_Y_NEAR_MARGIN
            if x_ok and y_near:
                self.stage = "push_forward"
                self.wait_deadline = time.monotonic() + BASE_AUTO_PUSH_SEC
                self.stage_steps = 0
                _, yaw_now = self.env._get_tb3_pose_xy_yaw()
                self.push_target_yaw = yaw_now
                print(
                    f"   ✅ Stage 2 near target. Pushing forward for {BASE_AUTO_PUSH_SEC:.1f}s "
                    f"to settle against table edge..."
                )
                self.env._set_base_action_intent([BASE_AUTO_PUSH_FWD_V, BASE_AUTO_PUSH_FWD_V])
                return self._push_forward_cmd()
            self.env._set_base_action_intent(cmd)
            return cmd

        if self.stage == "push_forward":
            if self.wait_deadline is None:
                self.wait_deadline = time.monotonic() + BASE_AUTO_PUSH_SEC
            if time.monotonic() >= self.wait_deadline:
                self.wait_deadline = None
                self.push_target_yaw = None
                if self.recording_active:
                    self.stage = "wait_done"
                    self.wait_counter = BASE_AUTO_WAIT_FRAMES
                    self.stage_steps = 0
                    print("   ✅ Push-forward done. Waiting 60 frames before finish (recording active)...")
                    self.env._set_base_action_intent([0.0, 0.0])
                    return np.zeros(2, dtype=np.float32)
                self.active = False
                self.stage = "idle"
                self.stage_steps = 0
                self.record_stop_requested = True
                print("   ✅ Push-forward done. Recording inactive, skip 3s wait.")
                print("   🛑 Base auto parking finished. Stop-recording requested.")
                self.env._set_base_action_intent([0.0, 0.0])
                return np.zeros(2, dtype=np.float32)
            self.env._set_base_action_intent([BASE_AUTO_PUSH_FWD_V, BASE_AUTO_PUSH_FWD_V])
            return self._push_forward_cmd()

        if self.stage == "wait_done":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.active = False
                self.stage = "idle"
                self.stage_steps = 0
                self.record_stop_requested = True
                print("   🛑 Base auto parking finished. Stop-recording requested.")
            self.env._set_base_action_intent([0.0, 0.0])
            return np.zeros(2, dtype=np.float32)

        self.active = False
        self.stage = "idle"
        self.wait_deadline = None
        self.env._set_base_action_intent([0.0, 0.0])
        return np.zeros(2, dtype=np.float32)
