"""
TeleopAgent: keyboard-driven control extracted from SimpleEnv7.teleop_robot.

Usage:
    from mujoco_env.teleop import TeleopAgent

    teleop = TeleopAgent(env)
    ...
    action, reset = teleop.get_action(mode='arm')
    env.step(action, mode='arm', action_type='eef_pose')
"""

import numpy as np
import glfw

from mujoco_env.utils import rotation_matrix
from mujoco_env.transforms import r2rpy
from mujoco_env.action_utils import drive_with_heading_hold
from mujoco_env.env_constants import (
    BASE_STRAIGHT_ASSIST_ENABLED,
    BASE_STRAIGHT_KP_YAW,
    BASE_STRAIGHT_MAX_TURN_V,
    BASE_STRAIGHT_YAW_DEADBAND,
    EXPERT_START_DELAY,
)


class TeleopAgent:
    """Keyboard-driven control agent for both arm and base modes.

    Reads glfw key states from the viewer managed by *env* and returns
    ``(action, reset_requested)`` each tick.  For arm mode the action is
    an ``eef_pose`` delta ``(7,)``; for base mode the action is wheel
    velocities ``(2,)``.
    """

    def __init__(self, env):
        self.env = env
        self._reset_heading_hold()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear heading-hold state (call after env.reset)."""
        self._reset_heading_hold()

    def get_action(self, mode='arm'):
        """Process keyboard input and return *(action, reset_requested)*.

        Parameters
        ----------
        mode : str
            ``'arm'`` or ``'base'``.

        Returns
        -------
        action : np.ndarray
            For arm: ``(7,)`` eef-pose delta ``[dx,dy,dz, drx,dry,drz, gripper]``.
            For base: ``(2,)`` wheel velocities ``[v_left, v_right]``.
        reset_requested : bool
            ``True`` when ``[Z]`` was pressed.
        """
        env = self.env
        viewer = env.env
        reset = False

        if viewer.is_key_pressed_once(key=glfw.KEY_Z):
            reset = True
        if viewer.is_key_pressed_once(key=glfw.KEY_SPACE):
            env.gripper_state = not env.gripper_state

        # --- Arm-only shortcut keys ---
        if mode == 'arm':
            if viewer.is_key_pressed_once(key=glfw.KEY_O):
                if not env.returning_home:
                    print("🏠 [O] Smooth Return Home: Moving arm to initial position...")
                    env.smooth_return_home()
                else:
                    print("⚠️ Arm is already returning home.")

            if viewer.is_key_pressed_once(key=glfw.KEY_T):
                if env.moving_to_random:
                    print("⚠️ Currently moving to random position. Please wait.")
                elif not env.expert_executing and not env.expert_pending:
                    print("🤖 [T] Test Mode: Auto Execute Expert Policy (No Recording)")
                    env.auto_execute_task(record=False)
                else:
                    print("⚠️ Expert policy already running. Press Z to reset.")

            if viewer.is_key_pressed_once(key=glfw.KEY_Y):
                if env.moving_to_random:
                    print("⚠️ Currently moving to random position. Please wait.")
                elif not env.expert_executing and not env.expert_pending:
                    print("🎥 [Y] Record Mode: Auto Execute Expert + Recording")
                    env.auto_execute_task(record=True)
                else:
                    print("⚠️ Expert policy already running. Press Z to reset.")

        if mode == 'arm':
            return self._arm_action(reset)
        return self._base_action(reset)

    # ------------------------------------------------------------------
    # Arm mode
    # ------------------------------------------------------------------

    def _arm_action(self, reset):
        env = self.env
        viewer = env.env

        if env.moving_to_random:
            env.smooth_move_to_random()
            return self._arm_zero_action(), reset

        if env.expert_pending or env.expert_executing:
            action = env.get_expert_action()
            if action is not None:
                if env.expert_pending:
                    print(
                        f"   ⏳ Buffer: {env.expert_countdown}/{EXPERT_START_DELAY} steps...",
                        end='\r',
                    )
                elif len(env.expert_trajectory) > 0:
                    pct = env.expert_trajectory_idx / len(env.expert_trajectory) * 100
                    print(
                        f"   🤖 Expert: {env.expert_trajectory_idx}"
                        f"/{len(env.expert_trajectory)} ({pct:.1f}%)",
                        end='\r',
                    )
                return action, reset

        if env.returning_home:
            env.smooth_return_home()
            return self._arm_zero_action(), reset

        speed = 0.007
        rot_speed = 0.03
        dpos = np.zeros(3)
        drot = np.eye(3)

        if viewer.is_key_pressed_repeat(key=glfw.KEY_W): dpos[0] = -speed
        if viewer.is_key_pressed_repeat(key=glfw.KEY_S): dpos[0] = speed
        if viewer.is_key_pressed_repeat(key=glfw.KEY_A): dpos[1] = -speed
        if viewer.is_key_pressed_repeat(key=glfw.KEY_D): dpos[1] = speed
        if viewer.is_key_pressed_repeat(key=glfw.KEY_R): dpos[2] = speed
        if viewer.is_key_pressed_repeat(key=glfw.KEY_F): dpos[2] = -speed
        if viewer.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=rot_speed, direction=[0, 0, 1])[:3, :3]
        if viewer.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-rot_speed, direction=[0, 0, 1])[:3, :3]

        drot_rpy = r2rpy(drot)
        action = np.concatenate([dpos, drot_rpy, [float(env.gripper_state)]], dtype=np.float32)
        return action, reset

    def _arm_zero_action(self):
        return np.concatenate(
            [np.zeros(6), [float(self.env.gripper_state)]], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Base mode
    # ------------------------------------------------------------------

    def _base_action(self, reset):
        env = self.env
        viewer = env.env

        if viewer.is_key_pressed_once(key=glfw.KEY_H):
            if not env.base_auto_active:
                env._start_base_auto_parking()
            else:
                env._stop_base_auto_parking("manual cancel by [H]")

        if env.base_auto_active:
            return env._run_base_auto_parking(), reset

        v_move = 15.0
        v_turn = 6.0
        w = viewer.is_key_pressed_repeat(key=glfw.KEY_W)
        s = viewer.is_key_pressed_repeat(key=glfw.KEY_S)
        a = viewer.is_key_pressed_repeat(key=glfw.KEY_A)
        d = viewer.is_key_pressed_repeat(key=glfw.KEY_D)

        if a and not d:
            self._reset_heading_hold()
            cmd = np.array([-v_turn, v_turn], dtype=np.float32)
            env._set_base_action_intent(cmd)
            return cmd, reset
        if d and not a:
            self._reset_heading_hold()
            cmd = np.array([v_turn, -v_turn], dtype=np.float32)
            env._set_base_action_intent(cmd)
            return cmd, reset

        if w and not s:
            env._set_base_action_intent([v_move, v_move])
            if (
                not BASE_STRAIGHT_ASSIST_ENABLED
                or not self._heading_active
                or self._heading_sign != 1
            ):
                _, yaw = env._get_tb3_pose_xy_yaw()
                self._heading_target = yaw
                self._heading_active = True
                self._heading_sign = 1
            if BASE_STRAIGHT_ASSIST_ENABLED:
                return self._drive_with_heading_hold(v_move), reset
            return np.array([v_move, v_move], dtype=np.float32), reset

        if s and not w:
            env._set_base_action_intent([-6.0, -6.0])
            if (
                not BASE_STRAIGHT_ASSIST_ENABLED
                or not self._heading_active
                or self._heading_sign != -1
            ):
                _, yaw = env._get_tb3_pose_xy_yaw()
                self._heading_target = yaw
                self._heading_active = True
                self._heading_sign = -1
            if BASE_STRAIGHT_ASSIST_ENABLED:
                return self._drive_with_heading_hold(-6.0), reset
            return np.array([-6.0, -6.0], dtype=np.float32), reset

        self._reset_heading_hold()
        cmd = np.zeros(2, dtype=np.float32)
        env._set_base_action_intent(cmd)
        return cmd, reset

    # ------------------------------------------------------------------
    # Heading-hold helpers (moved from SimpleEnv7)
    # ------------------------------------------------------------------

    def _reset_heading_hold(self):
        self._heading_active = False
        self._heading_target = None
        self._heading_sign = 0

    def _drive_with_heading_hold(self, fwd_v):
        """Apply yaw-lock correction during straight driving."""
        _, yaw = self.env._get_tb3_pose_xy_yaw()
        return drive_with_heading_hold(
            fwd_v=fwd_v,
            target_yaw=self._heading_target,
            current_yaw=yaw,
            kp_yaw=BASE_STRAIGHT_KP_YAW,
            max_turn_v=BASE_STRAIGHT_MAX_TURN_V,
            yaw_deadband=BASE_STRAIGHT_YAW_DEADBAND,
        )
