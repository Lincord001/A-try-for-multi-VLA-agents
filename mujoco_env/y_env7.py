import sys
import random
import numpy as np
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

from .env_constants import *
from .xml_helpers import _build_offset_xml_bundle, _sample_tb3_x_uniform
from .base_auto_parking import BaseAutoParkingAgent
from .expert_policy import ExpertPolicyAgent
from .renderer_mixin import RendererMixin
from .state_observer_mixin import StateObserverMixin
from .motion_mixin import MotionMixin
from .instruction_mixin import InstructionMixin
from .scene_init_mixin import SceneInitMixin


class SimpleEnv7(SceneInitMixin, InstructionMixin, MotionMixin, StateObserverMixin, RendererMixin):
    def __init__(self, xml_path, action_type='eef_pose', state_type='joint_angle', seed=None, 
                 random_init_enabled=False, random_init_gripper_open=True, select_smaller_angle_mug=False,
                 tb3_x_gaussian_enabled=True, tb3_x_center=TB3_X_CENTER,
                 tb3_x_offset_std=TB3_X_OFFSET_STD, tb3_x_offset_min=TB3_X_OFFSET_MIN,
                 tb3_x_offset_max=TB3_X_OFFSET_MAX,
                 tb3_x_random_enabled=None, tb3_x_min=TB3_X_MIN, tb3_x_max=TB3_X_MAX):
        self.xml_path_input = xml_path
        self.xml_path_runtime = _build_offset_xml_bundle(xml_path, V6_Z_OFFSET)
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=self.xml_path_runtime)
        self.action_type = action_type
        self.state_type = state_type
        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        
        # 🔥 随机初始化开关（由外部传入）
        self.random_init_enabled = random_init_enabled
        self.random_init_gripper_open = random_init_gripper_open
        
        # 🔥 杯子选择模式开关（由外部传入）
        self.select_smaller_angle_mug = select_smaller_angle_mug
        # 🔥 托盘初始化模式开关（可由采集脚本在运行时覆盖）
        self.force_tray_init_enabled = False
        # 托盘初始化颜色（单次 reset 消费）：避免跨回合状态泄漏
        self.pending_tray_init_color = None
        self._active_tray_init_color = None
        self._suppress_pending_tray_init_update = False

        # 🔥 TB3 初始 X 轴随机化开关与参数（由外部传入）
        # 兼容旧参数：若未传入新开关，沿用旧开关语义；并从旧参数推导新区间。
        if tb3_x_random_enabled is None:
            tb3_x_random_enabled = tb3_x_gaussian_enabled
        if tb3_x_min is None or tb3_x_max is None:
            tb3_x_min = tb3_x_center + tb3_x_offset_min
            tb3_x_max = tb3_x_center + tb3_x_offset_max
        self.tb3_x_random_enabled = bool(tb3_x_random_enabled)
        self.tb3_x_min = float(min(tb3_x_min, tb3_x_max))
        self.tb3_x_max = float(max(tb3_x_min, tb3_x_max))
        # 旧字段保留，避免外部脚本直接访问时报错。
        self.tb3_x_gaussian_enabled = self.tb3_x_random_enabled
        self.tb3_x_center = tb3_x_center
        self.tb3_x_offset_std = tb3_x_offset_std
        self.tb3_x_offset_min = tb3_x_offset_min
        self.tb3_x_offset_max = tb3_x_offset_max
        
        # 默认模式
        self.control_mode = 'arm' 
        
        # 内部状态
        self.current_arm_q = np.zeros(10) 
        self.current_wheel_vel = np.zeros(2)
        
        # 记录当前导航指令（用于避免重复选择）
        self.current_nav_instruction = None
        
        # 机械臂平滑归位相关
        self.arm_home_q = None  # 初始关节角度（将在 reset 中设置）
        self.returning_home = False  # 是否正在归位中
        self.home_start_q = None  # 归位起始关节角度
        self.home_interp_steps = 0  # 归位插值步数
        self.home_total_steps = 100  # 归位总步数（约5秒，20Hz）
        
        # 🔥 随机初始化移动相关
        self.moving_to_random = False  # 是否正在移动到随机位置
        self.random_target_q = None  # 随机目标位置的关节角度
        self.random_start_q = None  # 移动到随机位置的起始关节角度
        self.random_interp_steps = 0  # 移动到随机位置的插值步数
        self.random_target_pos = None  # 随机目标位置（用于打印）
        
        # 🤖 Expert Policy Agent（从环境逻辑中解耦）
        self.expert_policy = ExpertPolicyAgent(self)

        # 以下属性仅供外部脚本（collect_data_v7.py 等）读写，
        # 环境内部逻辑不再依赖它们。后续应由外部脚本自行管理。
        self.is_recording = False

        # 🚗 Base 模式自动停车 Agent（状态机从环境中解耦）
        self.base_auto_parking = BaseAutoParkingAgent(self)
        self.base_action_intent = np.zeros(2, dtype=np.float32)  # 记录用于数据集的“未纠偏意图动作”

        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        self.env.reset()
        self.env.init_viewer(
            distance=2.0, elevation=-30, transparent=False, black_sky=True,
            use_rgb_overlay=False, loc_rgb_overlay='top right',
        )

    # 兼容旧调用接口：转发到 BaseAutoParkingAgent
    def _start_base_auto_parking(self):
        self.base_auto_parking.start()

    def _stop_base_auto_parking(self, reason):
        self.base_auto_parking.stop(reason)

    def _run_base_auto_parking(self):
        return self.base_auto_parking.run()

    # 兼容旧字段访问：collect_data_v7.py 仍按属性读写这些状态
    @property
    def base_auto_active(self):
        return self.base_auto_parking.active

    @base_auto_active.setter
    def base_auto_active(self, value):
        self.base_auto_parking.active = bool(value)

    @property
    def base_auto_stage(self):
        return self.base_auto_parking.stage

    @base_auto_stage.setter
    def base_auto_stage(self, value):
        self.base_auto_parking.stage = str(value)

    @property
    def base_auto_wait_counter(self):
        return self.base_auto_parking.wait_counter

    @base_auto_wait_counter.setter
    def base_auto_wait_counter(self, value):
        self.base_auto_parking.wait_counter = int(value)

    @property
    def base_auto_wait_deadline(self):
        return self.base_auto_parking.wait_deadline

    @base_auto_wait_deadline.setter
    def base_auto_wait_deadline(self, value):
        self.base_auto_parking.wait_deadline = value

    @property
    def base_auto_stage_steps(self):
        return self.base_auto_parking.stage_steps

    @base_auto_stage_steps.setter
    def base_auto_stage_steps(self, value):
        self.base_auto_parking.stage_steps = int(value)

    @property
    def base_auto_recording_active(self):
        return self.base_auto_parking.recording_active

    @base_auto_recording_active.setter
    def base_auto_recording_active(self, value):
        self.base_auto_parking.recording_active = bool(value)

    @property
    def base_auto_record_stop_requested(self):
        return self.base_auto_parking.record_stop_requested

    @base_auto_record_stop_requested.setter
    def base_auto_record_stop_requested(self, value):
        self.base_auto_parking.record_stop_requested = bool(value)

    @property
    def base_auto_push_target_yaw(self):
        return self.base_auto_parking.push_target_yaw

    @base_auto_push_target_yaw.setter
    def base_auto_push_target_yaw(self, value):
        self.base_auto_parking.push_target_yaw = value

    def reset(self, seed=None, mode=None, force_fixed_arm_init=False, preserve_instruction=False, options=None):
        """Reset the environment.

        Parameters
        ----------
        options : dict or None
            Key/value pairs that temporarily (and persistently) override
            instance attributes for this reset.  Supported keys::

                random_init_enabled, random_init_gripper_open,
                select_smaller_angle_mug, force_tray_init_enabled,
                tb3_x_random_enabled, tb3_x_min, tb3_x_max,
                tb3_x_gaussian_enabled, tb3_x_center,
                tb3_x_offset_std, tb3_x_offset_min, tb3_x_offset_max
        """
        if options:
            for key, val in options.items():
                if hasattr(self, key):
                    setattr(self, key, val)

        if seed is not None: np.random.seed(seed)
        
        if mode is not None:
            self.control_mode = mode

        # 单次消费 pending 托盘初始化颜色，避免状态跨回合泄漏。
        self._active_tray_init_color = self.pending_tray_init_color
        self.pending_tray_init_color = None
        
        # 🔥 机械臂初始化：总是先初始化到标准位置（与V2一致）
        q_init = np.deg2rad([0,0,0,0,0,0])
        # 固定初始化：机械臂归位 (与V2一致，机械臂基座在原点 (0, 0))
        # 目标: 基座前方 0.3m -> [0.3, 0.0, 1.0] (桌面0.83 + 安全高度0.17)
        q_zero, _, _ = solve_ik(
            self.env,
            self.joint_names,
            'tcp_link',
            q_init,
            np.array([0.3, 0.0, 0.48 + V6_Z_OFFSET]),
            rpy2r(np.deg2rad([90, -0., 90]))
        )
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)
        
        # 可选：强制机械臂固定初始化（忽略随机初始化开关）
        effective_random_init_enabled = 0 if force_fixed_arm_init else self.random_init_enabled

        # 🔥 如果启用了随机初始化，生成随机目标位置并准备平滑移动
        if effective_random_init_enabled == 1:
            # ====== 旧版随机初始化：扇形区域 ======
            x_target, y_target, z_target = self._sample_random_init_v1()
            
        elif effective_random_init_enabled == 2:
            # ====== 新版随机初始化：环形交集（仅简单模式）======
            result = self._sample_random_init_v2()
            if result is None:
                # V2 失败，回退到 V1
                print("   → Falling back to V1 random initialization.")
                x_target, y_target, z_target = self._sample_random_init_v1()
            else:
                x_target, y_target, z_target = result
        
        # 如果启用了随机初始化，设置目标位置并准备平滑移动
        if effective_random_init_enabled in [1, 2]:
            # 使用IK求解目标位置的关节角度
            target_pos = np.array([x_target, y_target, z_target])
            self.random_target_q, _, _ = solve_ik(
                self.env, self.joint_names, 'tcp_link', q_zero, 
                target_pos, rpy2r(np.deg2rad([90, -0., 90]))
            )
            self.random_target_pos = target_pos  # 保存用于打印
            
            # 准备平滑移动
            self.moving_to_random = True
            self.random_start_q = q_zero.copy()
            self.random_interp_steps = 0
            
            print(f"   → Will smoothly move from standard position to random position")
            print(f"   → After reaching: Press [Y] to start expert policy with recording")
        else:
            # 未启用随机初始化，重置相关状态
            self.moving_to_random = False
            self.random_target_q = None
            self.random_start_q = None
            self.random_interp_steps = 0
            self.random_target_pos = None

        try:
            # 🔥 小车初始位置：可配置模式
            # - 开启时：区间均匀随机 x ~ U([tb3_x_min, tb3_x_max])
            # - 关闭时：固定 x 为区间中点
            if self.tb3_x_random_enabled:
                x_init = _sample_tb3_x_uniform(self.tb3_x_min, self.tb3_x_max)
            else:
                x_init = 0.5 * (self.tb3_x_min + self.tb3_x_max)
            y_init = -0.25
            z_init = 0.0
            yaw_init = np.deg2rad(90)  # +90度，朝向 y 轴正方向
            
            # 将 yaw 转换为旋转矩阵 (绕 z 轴旋转)
            R_init = rpy2r(np.array([0, 0, yaw_init]))
            
            self.env.set_pR_base_body(
                body_name='tb3_base',
                p=np.array([x_init, y_init, z_init]),
                R=R_init
            )
        except Exception as e:
            print(f"Warning: Could not reset TB3 base pose: {e}")
        
        # 状态重置
        self.last_q = copy.deepcopy(q_zero)
        self.tray_initialized_color = None
        self.tray_initialized_body = None
        self.table_reference_positions = {}
        # 🔥 根据开关设置初始夹爪状态
        initial_gripper_state = 0.0 if self.random_init_gripper_open else 1.0
        self.current_arm_q = np.concatenate([q_zero, np.array([initial_gripper_state]*4)]) 
        self.current_wheel_vel = np.zeros(2)
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        self.gripper_state = bool(initial_gripper_state)  # 🔥 同步更新 gripper_state
        
        # 保存初始关节角度（用于平滑归位）
        self.arm_home_q = copy.deepcopy(q_zero)
        self.returning_home = False
        self.home_start_q = None
        self.home_interp_steps = 0
        
        # 🤖 重置专家策略状态
        self.expert_policy.reset()
        # 外部脚本使用的录制标志（环境内部不依赖）
        self.is_recording = False
        self.base_auto_parking.reset()
        self.base_action_intent = np.zeros(2, dtype=np.float32)
        
        # 🔥 注意：moving_to_random 和 random_target_q 在 reset 中根据开关设置，这里不重置

        # 物体初始化
        self._init_objects_demo()
        
        # 获取物体位置
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        # 🔥 保存红色杯子、蓝色杯子和盘子的位置
        self.obj_init_pose = np.concatenate([mug_red_init_pose, mug_blue_init_pose, plate_init_pose], dtype=np.float32)

        for _ in range(100):
            self.step_env()
            
        self._suppress_pending_tray_init_update = True
        try:
            if preserve_instruction and getattr(self, 'instruction', None):
                current_task_type = getattr(self, 'task_type', 'arm' if self.control_mode == 'arm' else 'nav')
                self.set_instruction(given=self.instruction, task_type=current_task_type)
            else:
                self.set_instruction()  # 现在会根据 self.control_mode 自动设置正确的任务文本
        finally:
            self._suppress_pending_tray_init_update = False
        # 🔥 gripper_state 已在上面根据 random_init_gripper_open 设置，这里不再重置
        
        # 重置时刷新一次图像缓存
        self.grab_image()

    def step(self, action, mode='arm', action_type=None):
        """Execute one control step.

        Parameters
        ----------
        action : np.ndarray
            For arm eef_pose: ``(7,)`` ``[dx,dy,dz, drx,dry,drz, gripper]``.
            For arm joint_angle: ``(7,)`` ``[q1..q6, gripper]``.
            For base: ``(2,)`` ``[v_left, v_right]``.
        mode : str
            ``'arm'`` or ``'base'``.
        action_type : str or None
            ``'eef_pose'`` or ``'joint_angle'``.  When ``None`` falls
            back to ``self.action_type``.
        """
        if action_type is None:
            action_type = self.action_type
        self.control_mode = mode
        
        if mode == 'arm':
            self.current_wheel_vel = np.zeros(2)
            if action_type == 'eef_pose':
                q = self.env.get_qpos_joints(joint_names=self.joint_names)
                self.p0 += action[:3]
                self.R0 = self.R0.dot(rpy2r(action[3:6]))
                q, _, _ = solve_ik(
                    env                = self.env,
                    joint_names_for_ik = self.joint_names,
                    body_name_trgt     = 'tcp_link',
                    q_init             = q,
                    p_trgt             = self.p0,
                    R_trgt             = self.R0,
                    max_ik_tick        = 50,       # <--- 关键参数：限制计算次数
                    ik_stepsize        = 1.0,
                    ik_eps             = 1e-2,     # <--- 关键参数：允许 1cm 误差
                    ik_th              = np.radians(5.0),
                    render             = False,
                    verbose_warning    = False     # <--- 关闭报错日志
                )
            elif action_type == 'joint_angle':
                q = action[:-1]
            
            gripper_cmd = np.array([action[-1]]*4)
            gripper_cmd[[1,3]] *= 0.8
            self.current_arm_q = np.concatenate([q, gripper_cmd])
            
            # 返回机械臂状态 (7,) -> [q1, q2, q3, q4, q5, q6, gripper]
            return self.get_joint_state()
            
        elif mode == 'base':
            # Base 模式下，接收 2维 速度指令
            self.current_wheel_vel = action 
            
            # 返回底盘状态 (5,)
            return self.get_base_state()

    def step_env(self):
        full_ctrl = np.concatenate([self.current_arm_q, self.current_wheel_vel])
        self.env.step(full_ctrl)
        self._enforce_cup_tray_lock()
        
    def _enforce_cup_tray_lock(self):
        """
        当杯子在小车的推盘上时，锁定它的相对位置，防止倒下或滑落。
        """
        try:
            p_tb3, R_tb3 = self.env.get_pR_body('tb3_base')
            p_tcp = self.env.get_p_body('tcp_link')
            
            if not hasattr(self, 'locked_cup_info'):
                self.locked_cup_info = {}
                
            for mug_name in ['body_obj_mug_5', 'body_obj_mug_6']:
                p_mug, R_mug = self.env.get_pR_body(mug_name)
                
                xy_dist = np.linalg.norm(p_mug[:2] - p_tb3[:2])
                z_diff = p_mug[2] - p_tb3[2]
                tcp_dist = np.linalg.norm(p_mug - p_tcp)
                
                # 判断杯子是否在托盘上且机械臂不在抓取
                # 托盘高度大约 0.49，杯子中心高度可能在 0.49 左右
                on_tray = (xy_dist < 0.15) and (0.45 < z_diff < 0.55)
                arm_away = (tcp_dist > 0.1)
                
                if on_tray and arm_away:
                    if mug_name not in self.locked_cup_info:
                        # 刚放到托盘上，记录相对位置和姿态
                        rel_p = R_tb3.T @ (p_mug - p_tb3)
                        rel_R = R_tb3.T @ R_mug
                        self.locked_cup_info[mug_name] = {'rel_p': rel_p, 'rel_R': rel_R}
                    else:
                        # 已经锁定，强制更新位置
                        info = self.locked_cup_info[mug_name]
                        target_p = p_tb3 + R_tb3 @ info['rel_p']
                        target_R = R_tb3 @ info['rel_R']
                        
                        self.env.set_p_base_body(body_name=mug_name, p=target_p)
                        self.env.set_R_base_body(body_name=mug_name, R=target_R)
                        
                        # 尝试将速度清零以防物理引擎积分出大速度
                        try:
                            joint_id = self.env.model.body(mug_name).jntadr[0]
                            qvel_adr = self.env.model.jnt_dofadr[joint_id]
                            self.env.data.qvel[qvel_adr:qvel_adr+6] = 0.0
                        except:
                            pass
                else:
                    if mug_name in self.locked_cup_info:
                        # 离开托盘或机械臂靠近，解除锁定
                        del self.locked_cup_info[mug_name]
        except Exception as e:
            pass


    # ====== 🤖 Expert Policy Delegation (专家策略委托到独立 Agent) ======
    def interpolate_move(self, start_pos, end_pos, steps, gripper_state, add_tremor=False):
        return self.expert_policy.interpolate_move(start_pos, end_pos, steps, gripper_state, add_tremor)

    def bezier_move(self, p0, p1, p2, steps, gripper_state, add_tremor=False):
        return self.expert_policy.bezier_move(p0, p1, p2, steps, gripper_state, add_tremor)

    def create_gripper_action(self, pos, gripper_state, steps):
        return self.expert_policy.create_gripper_action(pos, gripper_state, steps)

    def distance_based_steps(self, start, end, speed_per_step, min_steps=None, max_steps=None):
        return self.expert_policy.distance_based_steps(start, end, speed_per_step, min_steps, max_steps)

    def rand_wait_steps(self, base_steps, noise_range):
        return self.expert_policy.rand_wait_steps(base_steps, noise_range)

    def bezier_curve_length(self, p0, p1, p2, num_samples=20):
        return self.expert_policy.bezier_curve_length(p0, p1, p2, num_samples)

    def sample_point_in_circle(self, center_xy, radius):
        return self.expert_policy.sample_point_in_circle(center_xy, radius)

    def add_orthogonal_tremor(self, trajectory, start_pos, end_pos, amplitude, smoothness=0.7):
        return self.expert_policy.add_orthogonal_tremor(trajectory, start_pos, end_pos, amplitude, smoothness)

    def compute_current_place_endpoint(self, add_noise=True):
        return self.expert_policy.compute_current_place_endpoint(add_noise)

    def _compute_expert_trajectory(self, current_pos):
        return self.expert_policy._compute_expert_trajectory(current_pos)

    def auto_execute_task(self, record=False):
        return self.expert_policy.auto_execute_task(record)

    def get_expert_action(self):
        return self.expert_policy.get_expert_action()

    # ----- Backward-compatible expert state accessors -----
    @property
    def expert_trajectory(self):
        return self.expert_policy.expert_trajectory

    @expert_trajectory.setter
    def expert_trajectory(self, value):
        self.expert_policy.expert_trajectory = value

    @property
    def expert_trajectory_idx(self):
        return self.expert_policy.expert_trajectory_idx

    @expert_trajectory_idx.setter
    def expert_trajectory_idx(self, value):
        self.expert_policy.expert_trajectory_idx = value

    @property
    def expert_executing(self):
        return self.expert_policy.expert_executing

    @expert_executing.setter
    def expert_executing(self, value):
        self.expert_policy.expert_executing = bool(value)

    @property
    def expert_pending(self):
        return self.expert_policy.expert_pending

    @expert_pending.setter
    def expert_pending(self, value):
        self.expert_policy.expert_pending = bool(value)

    @property
    def expert_countdown(self):
        return self.expert_policy.expert_countdown

    @expert_countdown.setter
    def expert_countdown(self, value):
        self.expert_policy.expert_countdown = int(value)

    @property
    def expert_lifting_to_mid(self):
        return self.expert_policy.expert_lifting_to_mid

    @expert_lifting_to_mid.setter
    def expert_lifting_to_mid(self, value):
        self.expert_policy.expert_lifting_to_mid = bool(value)

    @property
    def expert_mid_z_target(self):
        return self.expert_policy.expert_mid_z_target

    @expert_mid_z_target.setter
    def expert_mid_z_target(self, value):
        self.expert_policy.expert_mid_z_target = value

    @property
    def expert_lift_target_z(self):
        return self.expert_policy.expert_lift_target_z

    @expert_lift_target_z.setter
    def expert_lift_target_z(self, value):
        self.expert_policy.expert_lift_target_z = value

    @property
    def expert_lift_start_pos(self):
        return self.expert_policy.expert_lift_start_pos

    @expert_lift_start_pos.setter
    def expert_lift_start_pos(self, value):
        self.expert_policy.expert_lift_start_pos = value

    @property
    def expert_lift_interp_steps(self):
        return self.expert_policy.expert_lift_interp_steps

    @expert_lift_interp_steps.setter
    def expert_lift_interp_steps(self, value):
        self.expert_policy.expert_lift_interp_steps = int(value)

    @property
    def expert_lift_total_steps(self):
        return self.expert_policy.expert_lift_total_steps

    @expert_lift_total_steps.setter
    def expert_lift_total_steps(self, value):
        self.expert_policy.expert_lift_total_steps = int(value)

    @property
    def expert_lift_tremor_prev(self):
        return self.expert_policy.expert_lift_tremor_prev

    @expert_lift_tremor_prev.setter
    def expert_lift_tremor_prev(self, value):
        self.expert_policy.expert_lift_tremor_prev = value

    @property
    def expert_record_requested(self):
        return self.expert_policy.expert_record_requested

    @expert_record_requested.setter
    def expert_record_requested(self, value):
        self.expert_policy.expert_record_requested = bool(value)

    @property
    def expert_trajectory_start_pos(self):
        return self.expert_policy.expert_trajectory_start_pos

    @expert_trajectory_start_pos.setter
    def expert_trajectory_start_pos(self, value):
        self.expert_policy.expert_trajectory_start_pos = value

