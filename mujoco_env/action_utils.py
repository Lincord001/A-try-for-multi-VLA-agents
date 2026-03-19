"""
Shared action post-processing utilities.

BaseActionPostProcessor  -- model-output post-processing for Base mode
                            (speed scaling + heading-hold yaw correction).
"""

import numpy as np


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def drive_with_heading_hold(
    *,
    fwd_v: float,
    target_yaw: float,
    current_yaw: float,
    kp_yaw: float,
    max_turn_v: float,
    yaw_deadband: float,
):
    """Keep straight motion using yaw closed-loop correction."""
    yaw_err = _wrap_to_pi(float(target_yaw) - float(current_yaw))
    if abs(yaw_err) <= float(yaw_deadband):
        turn = 0.0
    else:
        turn = float(
            np.clip(
                float(kp_yaw) * yaw_err,
                -float(max_turn_v),
                float(max_turn_v),
            )
        )
    fwd_v = float(fwd_v)
    return np.array([fwd_v - turn, fwd_v + turn], dtype=np.float32)


class BaseActionPostProcessor:
    """Post-process Base mode model outputs: optional speed scaling and
    heading-hold yaw correction for near-straight commands.

    All tuning knobs are passed via the constructor so callers can keep
    their own config section and feed the values in.
    """

    def __init__(
        self,
        *,
        postproc_enabled: bool = True,
        heading_hold_enabled: bool = True,
        kp_yaw: float = 64.0,
        max_turn_v: float = 2.0,
        yaw_deadband: float = np.deg2rad(0.5),
        straight_delta_th: float = 1.5,
        min_abs_speed: float = 1.0,
        max_wheel_abs: float = 30.0,
        speed_scale_enabled: bool = True,
        speed_scale: float = 0.5,
    ):
        self.postproc_enabled = postproc_enabled
        self.heading_hold_enabled = heading_hold_enabled
        self.kp_yaw = kp_yaw
        self.max_turn_v = max_turn_v
        self.yaw_deadband = yaw_deadband
        self.straight_delta_th = straight_delta_th
        self.min_abs_speed = min_abs_speed
        self.max_wheel_abs = max_wheel_abs
        self.speed_scale_enabled = speed_scale_enabled
        self.speed_scale = speed_scale

        self._heading_hold_active = False
        self._target_yaw = None
        self._last_sign = 0

    def reset(self):
        self._heading_hold_active = False
        self._target_yaw = None
        self._last_sign = 0

    def process(self, raw_action, yaw):
        """Apply speed-scaling and heading-hold correction.

        Parameters
        ----------
        raw_action : array-like, shape (2,)
            ``[v_left, v_right]`` from the model.
        yaw : float
            Current chassis yaw in radians.

        Returns
        -------
        np.ndarray, shape (2,)
            Corrected wheel velocities.
        """
        action = np.array(raw_action, dtype=np.float32).copy()
        if action.shape[0] != 2:
            return action
        if not self.postproc_enabled:
            return action

        left, right = float(action[0]), float(action[1])
        cmd_sign = 1 if (left + right) >= 0.0 else -1

        if self.speed_scale_enabled and self.speed_scale >= 0.0:
            left *= self.speed_scale
            right *= self.speed_scale

        same_direction = left * right > 0.0
        delta_small = abs(left - right) <= self.straight_delta_th
        speed_enough = max(abs(left), abs(right)) >= self.min_abs_speed
        straight_like = same_direction and delta_small and speed_enough

        if not self.heading_hold_enabled or not straight_like:
            self.reset()
            return np.clip(
                np.array([left, right], dtype=np.float32),
                -self.max_wheel_abs,
                self.max_wheel_abs,
            )

        if (not self._heading_hold_active) or (self._last_sign != cmd_sign):
            self._target_yaw = float(yaw)
            self._heading_hold_active = True
            self._last_sign = cmd_sign

        yaw_err = _wrap_to_pi(self._target_yaw - float(yaw))
        if abs(yaw_err) <= self.yaw_deadband:
            turn = 0.0
        else:
            turn = float(
                np.clip(
                    self.kp_yaw * yaw_err,
                    -self.max_turn_v,
                    self.max_turn_v,
                )
            )

        v = 0.5 * (left + right)
        corrected = np.array([v - turn, v + turn], dtype=np.float32)
        return np.clip(corrected, -self.max_wheel_abs, self.max_wheel_abs)
