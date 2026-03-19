import numpy as np

from .env_constants import *


class ExpertPolicyAgent:
    """Expert pick-and-place trajectory generator/executor for SimpleEnv7."""

    _STATE_FIELDS = {
        "expert_trajectory",
        "expert_trajectory_idx",
        "expert_executing",
        "expert_pending",
        "expert_countdown",
        "expert_lifting_to_mid",
        "expert_mid_z_target",
        "expert_lift_target_z",
        "expert_lift_start_pos",
        "expert_lift_interp_steps",
        "expert_lift_total_steps",
        "expert_lift_tremor_prev",
        "expert_record_requested",
        "expert_trajectory_start_pos",
    }

    def __init__(self, host_env):
        object.__setattr__(self, "host_env", host_env)
        self.reset()

    def __getattr__(self, name):
        return getattr(self.host_env, name)

    def __setattr__(self, name, value):
        if name in self._STATE_FIELDS or name == "host_env":
            object.__setattr__(self, name, value)
            return
        setattr(self.host_env, name, value)

    def reset(self):
        self.expert_trajectory = []
        self.expert_trajectory_idx = 0
        self.expert_executing = False
        self.expert_pending = False
        self.expert_countdown = 0
        self.expert_lifting_to_mid = False
        self.expert_mid_z_target = None
        self.expert_lift_target_z = None
        self.expert_lift_start_pos = None
        self.expert_lift_interp_steps = 0
        self.expert_lift_total_steps = 50
        self.expert_lift_tremor_prev = np.zeros(2)
        self.expert_record_requested = False
        self.expert_trajectory_start_pos = None

    def interpolate_move(self, start_pos, end_pos, steps, gripper_state, add_tremor=False):
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps
            pos = start_pos + t * (end_pos - start_pos)
            waypoints.append((pos.copy(), gripper_state))

        if add_tremor and EXPERT_TREMOR_ENABLED:
            waypoints = self.add_orthogonal_tremor(
                waypoints,
                start_pos,
                end_pos,
                EXPERT_TREMOR_AMPLITUDE,
                EXPERT_TREMOR_SMOOTHNESS,
            )

        return waypoints

    def bezier_move(self, p0, p1, p2, steps, gripper_state, add_tremor=False):
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps
            pos = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            waypoints.append((pos.copy(), gripper_state))

        if add_tremor and EXPERT_TREMOR_ENABLED:
            waypoints = self.add_orthogonal_tremor(
                waypoints,
                p0,
                p2,
                EXPERT_TREMOR_AMPLITUDE,
                EXPERT_TREMOR_SMOOTHNESS,
            )

        return waypoints

    def create_gripper_action(self, pos, gripper_state, steps):
        return [(pos.copy(), gripper_state) for _ in range(steps)]

    def distance_based_steps(self, start, end, speed_per_step, min_steps=None, max_steps=None):
        if min_steps is None:
            min_steps = EXPERT_MIN_STEPS
        if max_steps is None:
            max_steps = EXPERT_MAX_STEPS

        dist = np.linalg.norm(end - start)
        noise = np.random.uniform(-EXPERT_SPEED_NOISE_RATIO, EXPERT_SPEED_NOISE_RATIO)
        actual_speed = speed_per_step * (1.0 + noise)
        steps = int(dist / actual_speed)
        steps = np.clip(steps, min_steps, max_steps)
        return steps

    def rand_wait_steps(self, base_steps, noise_range):
        noise = np.random.randint(-noise_range, noise_range + 1)
        return max(3, base_steps + noise)

    def bezier_curve_length(self, p0, p1, p2, num_samples=20):
        length = 0.0
        prev_point = p0
        for i in range(1, num_samples + 1):
            t = i / num_samples
            point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            length += np.linalg.norm(point - prev_point)
            prev_point = point
        return length

    def sample_point_in_circle(self, center_xy, radius):
        r = np.sqrt(np.random.uniform(0, 1)) * radius
        theta = np.random.uniform(0, 2 * np.pi)
        return center_xy + np.array([r * np.cos(theta), r * np.sin(theta)])

    def add_orthogonal_tremor(self, trajectory, start_pos, end_pos, amplitude, smoothness=0.7):
        if len(trajectory) == 0:
            return trajectory

        move_dir = end_pos - start_pos
        move_len = np.linalg.norm(move_dir)
        if move_len < 1e-6:
            return trajectory

        move_dir = move_dir / move_len
        if abs(move_dir[2]) < 0.9:
            ortho1 = np.cross(move_dir, np.array([0, 0, 1]))
            ortho1 = ortho1 / np.linalg.norm(ortho1)
        else:
            ortho1 = np.cross(move_dir, np.array([1, 0, 0]))
            ortho1 = ortho1 / np.linalg.norm(ortho1)

        ortho2 = np.cross(move_dir, ortho1)
        ortho2 = ortho2 / np.linalg.norm(ortho2)

        trajectory_with_tremor = []
        prev_tremor = np.zeros(2)
        for pos, gripper_state in trajectory:
            random_tremor = np.random.uniform(-1, 1, size=2)
            tremor_2d = smoothness * prev_tremor + (1 - smoothness) * random_tremor
            prev_tremor = tremor_2d.copy()

            tremor_magnitude = np.linalg.norm(tremor_2d)
            if tremor_magnitude > 1e-6:
                tremor_2d = tremor_2d / tremor_magnitude * amplitude * np.random.uniform(0.5, 1.5)
            else:
                tremor_2d = np.zeros(2)

            tremor_3d = tremor_2d[0] * ortho1 + tremor_2d[1] * ortho2
            pos_with_tremor = pos + tremor_3d
            trajectory_with_tremor.append((pos_with_tremor.copy(), gripper_state))

        return trajectory_with_tremor

    def compute_current_place_endpoint(self, add_noise=True):
        use_table_endpoint = self._is_target_initialized_on_tray()
        z_noise = np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE) if add_noise else 0.0

        if use_table_endpoint:
            target_color = getattr(self, "target_color", None)
            table_ref = getattr(self, "table_reference_positions", {}).get(target_color)
            if table_ref is not None:
                x_target = float(np.clip(table_ref[0], TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05))
                y_target = float(np.clip(table_ref[1] + EXPERT_Y_PLACE_OFFSET, TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05))
                return np.array([x_target, y_target, EXPERT_Z_PLACE_BASE + z_noise], dtype=np.float32)

        try:
            tb3_pos = self.env.get_p_body("tb3_base")
        except Exception:
            print("⚠️ Cannot find tb3_base body.")
            return None
        y_place_offset = 0.25 * EXPERT_Y_PLACE_OFFSET
        return np.array([tb3_pos[0], tb3_pos[1] + y_place_offset, EXPERT_Z_PLACE_BASE + z_noise], dtype=np.float32)

    def _compute_expert_trajectory(self, current_pos):
        if not hasattr(self, "obj_target"):
            print("⚠️ No target object set. Call set_instruction() first.")
            return []
        obj_pos = self.env.get_p_body(self.obj_target)
        target_on_tray = self._is_target_initialized_on_tray()
        grasp_anchor_pos = obj_pos
        if target_on_tray:
            print(
                f"   🎯 Tray-init target detected ({self.target_color}). "
                f"Using mug pose as grasp anchor: "
                f"({grasp_anchor_pos[0]:.3f}, {grasp_anchor_pos[1]:.3f}, {grasp_anchor_pos[2]:.3f})"
            )

        mid_z = EXPERT_FUNNEL_MID_Z
        mid_z_tolerance = 0.005
        lift_target_z = None
        lift_start_pos = None

        if current_pos[2] < mid_z - mid_z_tolerance:
            lift_target_z = mid_z + np.random.uniform(-mid_z_tolerance, mid_z_tolerance)
            lift_start_pos = current_pos.copy()
            lift_end_pos = np.array([current_pos[0], current_pos[1], lift_target_z])
            print(
                f"   ⬆️ Current Z ({current_pos[2]:.3f}m) below mid Z ({mid_z:.3f}m). "
                f"Will add smooth lift to {lift_target_z:.3f}m in trajectory."
            )
        else:
            lift_end_pos = current_pos.copy()

        z_travel = EXPERT_Z_TRAVEL_BASE + np.random.uniform(-EXPERT_Z_TRAVEL_NOISE, EXPERT_Z_TRAVEL_NOISE)
        adjusted_hover_z = lift_end_pos[2]

        y_grasp_offset = EXPERT_Y_GRASP_OFFSET
        grasp_center_xy = np.array([grasp_anchor_pos[0], grasp_anchor_pos[1] + y_grasp_offset])

        grasp_pos = np.array(
            [
                grasp_anchor_pos[0] + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
                grasp_anchor_pos[1] + y_grasp_offset + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
                EXPERT_Z_GRASP_BASE + np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE),
            ]
        )

        hover_xy = self.sample_point_in_circle(grasp_center_xy, radius=EXPERT_FUNNEL_HOVER_RADIUS)
        hover_pos = np.array([hover_xy[0], hover_xy[1], adjusted_hover_z])

        mid_xy = self.sample_point_in_circle(grasp_center_xy, radius=EXPERT_FUNNEL_MID_RADIUS)
        mid_pos = np.array([mid_xy[0], mid_xy[1], EXPERT_FUNNEL_MID_Z])

        place_pos = self.compute_current_place_endpoint(add_noise=True)
        if place_pos is None:
            return []

        place_hover_pos = np.array([place_pos[0], place_pos[1], z_travel])

        lift_pos = np.array([grasp_pos[0], grasp_pos[1], z_travel])
        retract_pos = np.array([place_pos[0], place_pos[1], EXPERT_RETRACT_HEIGHT])

        approach_speed = EXPERT_SPEED_APPROACH
        try:
            red_mug_pos = self.env.get_p_body("body_obj_mug_5")
            blue_mug_pos = self.env.get_p_body("body_obj_mug_6")

            red_rel_pos = red_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
            blue_rel_pos = blue_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])

            red_angle = np.arctan2(red_rel_pos[1], red_rel_pos[0])
            blue_angle = np.arctan2(blue_rel_pos[1], blue_rel_pos[0])

            red_angle_abs = abs(red_angle)
            blue_angle_abs = abs(blue_angle)
            smaller_angle_mug = "red" if red_angle_abs <= blue_angle_abs else "blue"

            if hasattr(self, "target_color") and self.target_color == smaller_angle_mug:
                approach_speed = EXPERT_SPEED_APPROACH_SMALLER_ANGLE
                print(
                    f"   ⚡ Selected mug ({self.target_color}) is the smaller angle mug. "
                    f"Using faster approach speed: {approach_speed:.3f} m/step"
                )
        except Exception as e:
            print(f"   ⚠️ Warning: Failed to check mug angles for speed adjustment: {e}. Using default approach speed.")

        approach_steps = self.distance_based_steps(lift_end_pos, hover_pos, approach_speed)
        descend_steps_1 = self.distance_based_steps(hover_pos, mid_pos, EXPERT_SPEED_DESCEND)
        descend_steps_2 = self.distance_based_steps(mid_pos, grasp_pos, EXPERT_SPEED_DESCEND)

        grasp_wait_steps = self.rand_wait_steps(EXPERT_GRASP_WAIT_BASE, EXPERT_GRASP_WAIT_NOISE)
        lift_steps = self.distance_based_steps(grasp_pos, lift_pos, EXPERT_SPEED_LIFT)
        lower_steps = self.distance_based_steps(place_hover_pos, place_pos, EXPERT_SPEED_LOWER)
        place_wait_steps = self.rand_wait_steps(EXPERT_PLACE_WAIT_BASE, EXPERT_PLACE_WAIT_NOISE)
        retract_steps = self.distance_based_steps(place_pos, retract_pos, EXPERT_SPEED_RETRACT)

        trajectory = []
        if self.gripper_state:
            trajectory.extend(self.create_gripper_action(current_pos, gripper_state=1.0, steps=1))
            open_wait_steps = self.rand_wait_steps(EXPERT_OPEN_WAIT_BASE, EXPERT_OPEN_WAIT_NOISE)
            trajectory.extend(self.create_gripper_action(current_pos, gripper_state=0.0, steps=open_wait_steps))
            print(f"   🔓 Opening gripper first (was closed): 1 frame wait + {open_wait_steps} steps to open")

        if lift_target_z is not None:
            lift_smooth_steps = self.distance_based_steps(lift_start_pos, lift_end_pos, EXPERT_SPEED_LIFT)
            trajectory.extend(
                self.interpolate_move(
                    lift_start_pos,
                    lift_end_pos,
                    lift_smooth_steps,
                    gripper_state=0.0,
                    add_tremor=True,
                )
            )
            print(f"   ⬆️ Smooth lift: {lift_start_pos[2]:.3f}m -> {lift_target_z:.3f}m ({lift_smooth_steps} steps)")

        trajectory.extend(self.interpolate_move(lift_end_pos, hover_pos, approach_steps, gripper_state=0.0, add_tremor=True))
        trajectory.extend(self.interpolate_move(hover_pos, mid_pos, descend_steps_1, gripper_state=0.0, add_tremor=True))
        trajectory.extend(self.interpolate_move(mid_pos, grasp_pos, descend_steps_2, gripper_state=0.0, add_tremor=True))
        trajectory.extend(self.create_gripper_action(grasp_pos, gripper_state=1.0, steps=grasp_wait_steps))
        trajectory.extend(self.interpolate_move(grasp_pos, lift_pos, lift_steps, gripper_state=1.0, add_tremor=True))

        arm_center = np.array([ARM_BASE_X, ARM_BASE_Y])
        line_start = lift_pos[:2]
        line_end = place_hover_pos[:2]
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq < 1e-10:
            mid_point_xy = line_start
            perp_dir = np.array([1.0, 0.0])
            offset_dist = EXPERT_BEZIER_SMALL_CURVE
        else:
            t_closest = np.dot(arm_center - line_start, line_vec) / line_len_sq
            t_closest = np.clip(t_closest, 0.0, 1.0)
            closest_point = line_start + t_closest * line_vec
            dist_to_line = np.linalg.norm(arm_center - closest_point)

            line_dir = line_vec / np.sqrt(line_len_sq)
            perp_dir = np.array([-line_dir[1], line_dir[0]])

            mid_point_xy = (line_start + line_end) / 2
            vec_to_center = arm_center - mid_point_xy
            if np.dot(perp_dir, vec_to_center) > 0:
                perp_dir = -perp_dir

            if dist_to_line > EXPERT_BEZIER_AVOID_RADIUS:
                offset_dist = EXPERT_BEZIER_SMALL_CURVE
            else:
                required_curve_offset = EXPERT_BEZIER_AVOID_RADIUS - dist_to_line + EXPERT_BEZIER_TANGENT_MARGIN
                offset_dist = 2.0 * required_curve_offset

        offset_dist += np.random.uniform(0.0, EXPERT_BEZIER_XY_OFFSET)
        control_point_xy = mid_point_xy + perp_dir * offset_dist
        control_point_z = z_travel + np.random.uniform(EXPERT_BEZIER_Z_OFFSET_MIN, EXPERT_BEZIER_Z_OFFSET_MAX)
        control_point = np.array([control_point_xy[0], control_point_xy[1], control_point_z])

        bezier_length = self.bezier_curve_length(lift_pos, control_point, place_hover_pos)
        noise = np.random.uniform(-EXPERT_SPEED_NOISE_RATIO, EXPERT_SPEED_NOISE_RATIO)
        actual_transport_speed = EXPERT_SPEED_TRANSPORT * (1.0 + noise)
        transport_steps = int(bezier_length / actual_transport_speed)
        transport_steps = np.clip(transport_steps, EXPERT_MIN_STEPS, EXPERT_MAX_STEPS)

        trajectory.extend(
            self.bezier_move(
                lift_pos,
                control_point,
                place_hover_pos,
                transport_steps,
                gripper_state=1.0,
                add_tremor=True,
            )
        )
        trajectory.extend(self.interpolate_move(place_hover_pos, place_pos, lower_steps, gripper_state=1.0, add_tremor=True))
        trajectory.extend(self.create_gripper_action(place_pos, gripper_state=0.0, steps=place_wait_steps))
        trajectory.extend(self.interpolate_move(place_pos, retract_pos, retract_steps, gripper_state=0.0))

        hover_z_display = (
            f"{hover_pos[2]:.2f}" if EXPERT_FUNNEL_HOVER_Z is not None else f"z_travel({z_travel:.2f})"
        )

        print(f"🤖 Expert trajectory generated: {len(trajectory)} steps")
        color_name = self.target_color if hasattr(self, "target_color") else "unknown"
        print(f"   Target object: {self.obj_target} ({color_name} mug) -> Plate")
        print(f"   🔥 Y-axis grasp offset: {y_grasp_offset*1000:.1f}mm (用于抓取杯子把手)")
        print(
            f"   🔥 Funnel Approach: hover (grasp center, z={hover_z_display}, r={EXPERT_FUNNEL_HOVER_RADIUS*100:.1f}cm) "
            f"-> mid (grasp center, z={EXPERT_FUNNEL_MID_Z:.2f}, r={EXPERT_FUNNEL_MID_RADIUS*100:.1f}cm) -> grasp"
        )
        print(f"   Grasp center: ({grasp_center_xy[0]:.3f}, {grasp_center_xy[1]:.3f}) [Y偏移: {y_grasp_offset*1000:.1f}mm]")
        print(f"   Hover pos: ({hover_pos[0]:.3f}, {hover_pos[1]:.3f}, {hover_pos[2]:.3f})")
        print(f"   Mid pos: ({mid_pos[0]:.3f}, {mid_pos[1]:.3f}, {mid_pos[2]:.3f})")
        print(f"   Grasp pos: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        print(f"   Place pos: ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f})")
        print(
            f"   🔥 Dynamic Steps: approach={approach_steps}, descend1={descend_steps_1}, "
            f"descend2={descend_steps_2}, grasp_wait={grasp_wait_steps}"
        )
        print(f"                    lift={lift_steps}, transport={transport_steps}, lower={lower_steps}")
        print(f"                    place_wait={place_wait_steps}, retract={retract_steps}")
        print(f"   📏 Bezier arc length: {bezier_length:.3f}m")
        return trajectory

    def auto_execute_task(self, record=False):
        current_pos = self.p0.copy()
        if not hasattr(self, "obj_target"):
            print("⚠️ No target object set. Call set_instruction() first.")
            return

        trajectory = self._compute_expert_trajectory(current_pos)
        if len(trajectory) == 0:
            print("⚠️ Failed to compute trajectory. Stopping expert policy.")
            return

        self.expert_trajectory = trajectory
        self.expert_trajectory_idx = 0
        self.expert_pending = False
        self.expert_countdown = 0
        self.expert_executing = True
        self.expert_record_requested = bool(record)
        print(f"🤖 Expert policy initialized. Trajectory computed: {len(trajectory)} steps.")

    def get_expert_action(self):
        if self.expert_pending:
            self.expert_countdown -= 1
            if self.expert_countdown <= 0:
                self.expert_pending = False
                self.expert_executing = True
                print("🚀 Motion Start!")
            else:
                return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)

        if len(self.expert_trajectory) == 0:
            self.expert_executing = False
            return None

        if self.expert_trajectory_idx >= len(self.expert_trajectory):
            self.expert_executing = False
            print(f"\n✅ Expert trajectory finished ({len(self.expert_trajectory)} steps).")
            return None

        target_pos, gripper_state = self.expert_trajectory[self.expert_trajectory_idx]
        self.expert_trajectory_idx += 1
        delta_pos = target_pos - self.p0
        delta_rot = np.zeros(3)
        action = np.concatenate([delta_pos, delta_rot, [gripper_state]], dtype=np.float32)
        self.gripper_state = bool(gripper_state)
        return action
