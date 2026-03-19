import numpy as np

from .env_constants import (
    TABLE_Z_HEIGHT,
    EXPERT_RETRACT_HEIGHT,
    SUCCESS_TCP_Z_MARGIN,
)


class StateObserverMixin:
    """Mixin providing state readout, success detection, and metadata helpers."""

    def get_joint_state(self):
        """
        获取机械臂关节状态 + 夹爪指令
        返回 shape: (7,) -> [q1, q2, q3, q4, q5, q6, gripper]
        """
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]], dtype=np.float32)

    def get_base_state(self):
        """
        [Visual Navigation 专用]
        返回本体真实速度 + 朝向编码 (sin/cos)：
        - 速度保留 Proprioception 闭环信息
        - 朝向使用 sin/cos，避免角度在 +/-pi 处不连续
        - 不返回 x, y 绝对坐标，避免过拟合环境位置
        
        返回 shape: (4,) -> [v_left_real, v_right_real, sin(yaw), cos(yaw)]
        """
        # 1. 获取真实的关节速度 (Sim2Real 关键)
        # 即使你发出的指令是 10，实际电机可能因为摩擦只转到了 8
        # 记录真实速度能让 Policy 学会闭环控制
        try:
            v_left_real = self.env.get_qvel_joint('wheel_left_joint')[0]
            v_right_real = self.env.get_qvel_joint('wheel_right_joint')[0]
        except:
            # Fallback (万一读取失败，返回指令速度)
            v_left_real = self.current_wheel_vel[0]
            v_right_real = self.current_wheel_vel[1]

        # 2. 读取底盘朝向并编码为 sin/cos，避免角度边界跳变
        try:
            R_tb3 = self.env.get_R_body('tb3_base')
            yaw = np.arctan2(R_tb3[1, 0], R_tb3[0, 0])
        except Exception:
            yaw = 0.0
        yaw_sin = np.sin(yaw)
        yaw_cos = np.cos(yaw)

        # 3. 返回低维状态（速度 + 朝向编码）
        return np.array([v_left_real, v_right_real, yaw_sin, yaw_cos], dtype=np.float32)

    def check_success(self):
        """
        成功判定：目标杯子（红色或蓝色）放到小车目标点上
        判定条件：
        1. 杯子的XY坐标到小车目标点的距离足够近（XY距离 < 0.1m）
        2. 夹爪已松开（gripper < 0.1）
        3. 机械臂已撤离（TCP Z > EXPERT_RETRACT_HEIGHT - SUCCESS_TCP_Z_MARGIN）
        """
        if not hasattr(self, 'obj_target'):
            return False
            
        p_mug = self.env.get_p_body(self.obj_target)
        
        # 判断目标杯子是否在小车目标点上
        try:
            place_pos = self.compute_current_place_endpoint(add_noise=False)
            if place_pos is None:
                return False
            target_xy = place_pos[:2]
            xy_dist = np.linalg.norm(p_mug[:2] - target_xy)
            if xy_dist < 0.1:
                # 夹爪已松开
                gripper = self.env.get_qpos_joint('rh_r1')
                if gripper < 0.1:
                    # 机械臂已撤离（支持统一高度偏移）
                    tcp_z = self.env.get_p_body('tcp_link')[2]
                    success_tcp_z_min = EXPERT_RETRACT_HEIGHT - SUCCESS_TCP_Z_MARGIN
                    if tcp_z > success_tcp_z_min:
                        return True
        except Exception as e:
            print(f"Warning: check_success error: {e}")
        return False

    def check_objects_fallen(self, tolerance=0.05):
        """Check whether the red mug has fallen over.

        Parameters
        ----------
        tolerance : float
            Allowed deviation from the expected Z height (metres).

        Returns
        -------
        bool
            ``True`` if the red mug's Z coordinate deviates from
            ``TABLE_Z_HEIGHT`` by more than *tolerance*.
        """
        try:
            obj_z = self.env.get_p_body('body_obj_mug_5')[2]
            return abs(obj_z - TABLE_Z_HEIGHT) > tolerance
        except Exception as e:
            print(f"Warning: check_objects_fallen error: {e}")
            return True

    def get_task_metadata(self):
        """Return a dict describing the current task configuration.

        Keys:
            target_color : str or None  ('red' / 'blue')
            obj_init_pose : np.ndarray or None  (9,) red(3)+blue(3)+plate(3)
            instruction : str or None
        """
        return {
            'target_color': getattr(self, 'target_color', None),
            'obj_init_pose': getattr(self, 'obj_init_pose', None),
            'instruction': getattr(self, 'instruction', None),
        }

    def get_obj_pose(self):
        """
        获取物体位置（返回红色杯子、蓝色杯子和盘子的位置）
        Returns:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug
            p_plate: np.array, position of the plate
        """
        p_mug_red = self.env.get_p_body('body_obj_mug_5')
        p_mug_blue = self.env.get_p_body('body_obj_mug_6')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_mug_red, p_mug_blue, p_plate
