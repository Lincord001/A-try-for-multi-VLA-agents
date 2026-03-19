import numpy as np

from .transforms import rpy2r
from .env_constants import RANDOM_INIT_MOVE_STEPS


class MotionMixin:
    """Mixin providing base-pose utilities and smooth arm motion helpers."""

    def set_base_pose(self, x, y, theta, z=None):
        """
        设置小车底盘的位姿 (x, y, theta)
        
        Parameters:
            x (float): x 坐标
            y (float): y 坐标
            theta (float): 朝向角度 (弧度)，绕 z 轴旋转
            z (float, optional): z 坐标（高度）。如果为 None，默认使用 0.05m 作为安全高度，
                                 让机器人自然掉落而不是从地下弹出来，提高回放一致性。
        """
        try:
            # 将 theta 转换为旋转矩阵 (绕 z 轴旋转)
            R = rpy2r(np.array([0, 0, theta]))
            # 如果未指定 z，使用安全高度 0.05m，让物理引擎自然掉落
            if z is None:
                z = 0.02
            self.env.set_pR_base_body(
                body_name='tb3_base',
                p=np.array([x, y, z]),
                R=R
            )
        except Exception as e:
            print(f"Warning: Could not set TB3 base pose: {e}")

    def _get_tb3_pose_xy_yaw(self):
        """返回底盘当前 (x, y, yaw)。"""
        p_tb3 = self.env.get_p_body('tb3_base')
        R_tb3 = self.env.get_R_body('tb3_base')
        yaw = np.arctan2(R_tb3[1, 0], R_tb3[0, 0])
        return p_tb3[:2], yaw

    def _set_base_action_intent(self, cmd):
        """记录 base 模式用于写数据集的意图动作（未经过二次纠偏）。"""
        self.base_action_intent = np.array(cmd, dtype=np.float32).copy()

    def get_base_action_intent(self):
        """获取当前 base 模式的意图动作（用于采集脚本存盘）。"""
        return self.base_action_intent.copy()

    def teleport_base_and_cups(self, x, y, z, yaw_deg):
        """
        传送小车到指定位置。如果杯子在托盘上（已锁定），则连同杯子一起传送。
        """
        yaw_rad = np.deg2rad(yaw_deg)
        new_p = np.array([x, y, z], dtype=np.float32)
        new_R = rpy2r(np.array([0, 0, yaw_rad]))
        
        # 如果杯子被锁定，先传送杯子
        if hasattr(self, 'locked_cup_info'):
            for mug_name, info in self.locked_cup_info.items():
                target_p = new_p + new_R @ info['rel_p']
                target_R = new_R @ info['rel_R']
                self.env.set_p_base_body(body_name=mug_name, p=target_p)
                self.env.set_R_base_body(body_name=mug_name, R=target_R)
                try:
                    joint_id = self.env.model.body(mug_name).jntadr[0]
                    qvel_adr = self.env.model.jnt_dofadr[joint_id]
                    self.env.data.qvel[qvel_adr:qvel_adr+6] = 0.0
                except:
                    pass
                    
        # 传送小车
        self.env.set_pR_base_body(body_name='tb3_base', p=new_p, R=new_R)
        try:
            joint_id = self.env.model.body('tb3_base').jntadr[0]
            qvel_adr = self.env.model.jnt_dofadr[joint_id]
            self.env.data.qvel[qvel_adr:qvel_adr+6] = 0.0
        except:
            pass
    
    def smooth_return_home(self):
        """
        平滑地将机械臂移动到初始位置
        使用线性插值，逐步移动关节角度
        """
        if self.arm_home_q is None:
            return False
        
        # 如果正在归位中，继续插值
        if self.returning_home:
            self.home_interp_steps += 1
            alpha = min(self.home_interp_steps / self.home_total_steps, 1.0)
            
            # 使用平滑插值（ease-in-out，更自然的运动）
            alpha_smooth = alpha * alpha * (3 - 2 * alpha)
            
            # 从起始位置插值到目标位置
            target_q = self.home_start_q * (1 - alpha_smooth) + self.arm_home_q * alpha_smooth
            
            # 更新机械臂状态（直接设置关节角度）
            gripper_cmd = self.current_arm_q[6:10]  # 保持当前夹爪状态
            self.current_arm_q = np.concatenate([target_q, gripper_cmd])
            
            # 更新 p0 和 R0（用于 eef_pose 模式）
            # 先设置关节角度，然后获取TCP位置
            self.env.forward(q=target_q, joint_names=self.joint_names, increase_tick=False)
            p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
            self.p0 = p_current
            self.R0 = R_current
            
            # 检查是否完成
            if self.home_interp_steps >= self.home_total_steps:
                self.returning_home = False
                self.home_interp_steps = 0
                self.home_start_q = None
                # 确保最终位置精确
                self.current_arm_q = np.concatenate([self.arm_home_q, gripper_cmd])
                self.env.forward(q=self.arm_home_q, joint_names=self.joint_names, increase_tick=False)
                return True
            
            return True
        else:
            # 开始归位：保存当前关节角度作为起始位置
            self.home_start_q = self.env.get_qpos_joints(joint_names=self.joint_names).copy()
            self.returning_home = True
            self.home_interp_steps = 0
            return True
    
    def smooth_move_to_random(self):
        """
        🔥 平滑地将机械臂移动到随机初始化位置
        使用线性插值，逐步移动关节角度
        移动完成后，等待用户按Y键开启专家策略并录制
        """
        if not self.moving_to_random or self.random_target_q is None:
            return False
        
        # 如果正在移动中，继续插值
        self.random_interp_steps += 1
        alpha = min(self.random_interp_steps / RANDOM_INIT_MOVE_STEPS, 1.0)
        
        # 使用平滑插值（ease-in-out，更自然的运动）
        alpha_smooth = alpha * alpha * (3 - 2 * alpha)
        
        # 从起始位置插值到目标位置
        target_q = self.random_start_q * (1 - alpha_smooth) + self.random_target_q * alpha_smooth
        
        # 更新机械臂状态（直接设置关节角度）
        gripper_cmd = self.current_arm_q[6:10]  # 保持当前夹爪状态
        self.current_arm_q = np.concatenate([target_q, gripper_cmd])
        
        # 更新 p0 和 R0（用于 eef_pose 模式）
        self.env.forward(q=target_q, joint_names=self.joint_names, increase_tick=False)
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        self.p0 = p_current
        self.R0 = R_current
        
        # 检查是否完成
        if self.random_interp_steps >= RANDOM_INIT_MOVE_STEPS:
            # 移动完成，确保最终位置精确
            self.current_arm_q = np.concatenate([self.random_target_q, gripper_cmd])
            self.env.forward(q=self.random_target_q, joint_names=self.joint_names, increase_tick=False)
            p_final, R_final = self.env.get_pR_body(body_name='tcp_link')
            self.p0 = p_final
            self.R0 = R_final
            
            # 🔥 移动完成，等待用户按Y键开启专家策略并录制
            self.moving_to_random = False
            print(f"\n✅ Reached random position!")
            print(f"   → Press [Y] to start expert policy with recording")
            # 不自动开启，等待用户按Y键
            return True
        
        return True
