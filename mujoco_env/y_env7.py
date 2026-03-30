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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .env_constants import *
from .xml_helpers import _build_offset_xml_bundle
from .base_auto_parking import BaseAutoParkingAgent
from .expert_policy import ExpertPolicyAgent
from .renderer_mixin import RendererMixin
from .state_observer_mixin import StateObserverMixin
from .motion_mixin import MotionMixin
from .instruction_mixin import InstructionMixin
from .scene_init_mixin import SceneInitMixin


@dataclass
class ResetConfig:
    seed: Optional[int]
    mode: Optional[str]
    force_fixed_arm_init: bool
    preserve_instruction: bool
    random_init_enabled: int
    random_init_gripper_open: bool
    select_smaller_angle_mug: bool
    force_tray_init_enabled: bool
    tb3_x_random_enabled: bool
    tb3_x_min: float
    tb3_x_max: float
    tb3_x_gaussian_enabled: bool
    tb3_x_center: float
    tb3_x_offset_std: float
    tb3_x_offset_min: float
    tb3_x_offset_max: float
    strict_options: bool = False


@dataclass
class ResetPlan:
    effective_random_init_enabled: int = 0
    random_target_pos: Optional[np.ndarray] = None
    tb3_init_pose: Optional[np.ndarray] = None  # [x, y, z, yaw]
    fallback_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ResetReport:
    config: Optional[ResetConfig] = None
    sampled: Dict[str, Any] = field(default_factory=dict)
    fallback_flags: Dict[str, bool] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def add_warning(self, message: str):
        self.warnings.append(str(message))

    def add_error(self, message: str):
        self.errors.append(str(message))


class SimpleEnv7(SceneInitMixin, InstructionMixin, MotionMixin, StateObserverMixin, RendererMixin):
    RESET_OPTION_KEYS = {
        'random_init_enabled',
        'random_init_gripper_open',
        'select_smaller_angle_mug',
        'force_tray_init_enabled',
        'tb3_x_random_enabled',
        'tb3_x_min',
        'tb3_x_max',
        'tb3_x_gaussian_enabled',
        'tb3_x_center',
        'tb3_x_offset_std',
        'tb3_x_offset_min',
        'tb3_x_offset_max',
        'strict_options',
    }

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
        self.tray_support_hidden_pos = np.array([0.0, 0.0, -0.20], dtype=np.float64)
        self.tray_support_active = {'body_obj_mug_5': False, 'body_obj_mug_6': False}
        self.tray_support_body_names = {
            'body_obj_mug_5': 'tray_support_red',
            'body_obj_mug_6': 'tray_support_blue',
        }
        # 🔥 托盘初始化模式开关（可由采集脚本在运行时覆盖）
        self.force_tray_init_enabled = False
        # 托盘初始化颜色（单次 reset 消费）：避免跨回合状态泄漏
        self.pending_tray_init_color = None
        self._active_tray_init_color = None
        self._suppress_pending_tray_init_update = False
        self.strict_reset_options = False
        self.debug_reset = False

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

    def _call_with_seed(self, seed, fn):
        if seed is None:
            return fn()
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            return fn()
        finally:
            np.random.set_state(state)

    def _resolve_reset_config(self, seed=None, mode=None, force_fixed_arm_init=False, preserve_instruction=False, options=None):
        cfg = ResetConfig(
            seed=seed,
            mode=mode,
            force_fixed_arm_init=bool(force_fixed_arm_init),
            preserve_instruction=bool(preserve_instruction),
            random_init_enabled=int(self.random_init_enabled),
            random_init_gripper_open=bool(self.random_init_gripper_open),
            select_smaller_angle_mug=bool(self.select_smaller_angle_mug),
            force_tray_init_enabled=bool(self.force_tray_init_enabled),
            tb3_x_random_enabled=bool(self.tb3_x_random_enabled),
            tb3_x_min=float(self.tb3_x_min),
            tb3_x_max=float(self.tb3_x_max),
            tb3_x_gaussian_enabled=bool(self.tb3_x_gaussian_enabled),
            tb3_x_center=float(self.tb3_x_center),
            tb3_x_offset_std=float(self.tb3_x_offset_std),
            tb3_x_offset_min=float(self.tb3_x_offset_min),
            tb3_x_offset_max=float(self.tb3_x_offset_max),
            strict_options=bool(getattr(self, 'strict_reset_options', False)),
        )
        warnings = []

        if options:
            for key, val in options.items():
                if key not in self.RESET_OPTION_KEYS:
                    msg = f"Unknown reset option '{key}'"
                    if cfg.strict_options:
                        raise KeyError(msg)
                    warnings.append(msg)
                    continue
                if key == 'strict_options':
                    cfg.strict_options = bool(val)
                    continue
                setattr(cfg, key, val)

        if cfg.random_init_enabled not in (0, 1, 2):
            msg = f"Invalid random_init_enabled={cfg.random_init_enabled}, fallback to 0"
            if cfg.strict_options:
                raise ValueError(msg)
            warnings.append(msg)
            cfg.random_init_enabled = 0

        cfg.tb3_x_min = float(cfg.tb3_x_min)
        cfg.tb3_x_max = float(cfg.tb3_x_max)
        if cfg.tb3_x_min > cfg.tb3_x_max:
            msg = f"tb3_x_min ({cfg.tb3_x_min}) > tb3_x_max ({cfg.tb3_x_max}), values swapped"
            if cfg.strict_options:
                raise ValueError(msg)
            warnings.append(msg)
            cfg.tb3_x_min, cfg.tb3_x_max = cfg.tb3_x_max, cfg.tb3_x_min

        cfg.tb3_x_random_enabled = bool(cfg.tb3_x_random_enabled)
        cfg.random_init_gripper_open = bool(cfg.random_init_gripper_open)
        cfg.select_smaller_angle_mug = bool(cfg.select_smaller_angle_mug)
        cfg.force_tray_init_enabled = bool(cfg.force_tray_init_enabled)
        cfg.tb3_x_gaussian_enabled = bool(cfg.tb3_x_gaussian_enabled)

        return cfg, warnings

    def _apply_reset_config(self, cfg: ResetConfig):
        self.random_init_enabled = int(cfg.random_init_enabled)
        self.random_init_gripper_open = bool(cfg.random_init_gripper_open)
        self.select_smaller_angle_mug = bool(cfg.select_smaller_angle_mug)
        self.force_tray_init_enabled = bool(cfg.force_tray_init_enabled)
        self.tb3_x_random_enabled = bool(cfg.tb3_x_random_enabled)
        self.tb3_x_min = float(cfg.tb3_x_min)
        self.tb3_x_max = float(cfg.tb3_x_max)
        self.tb3_x_gaussian_enabled = bool(cfg.tb3_x_gaussian_enabled)
        self.tb3_x_center = float(cfg.tb3_x_center)
        self.tb3_x_offset_std = float(cfg.tb3_x_offset_std)
        self.tb3_x_offset_min = float(cfg.tb3_x_offset_min)
        self.tb3_x_offset_max = float(cfg.tb3_x_offset_max)
        self.strict_reset_options = bool(cfg.strict_options)
        if cfg.mode is not None:
            self.control_mode = cfg.mode

    def _plan_reset(self, cfg: ResetConfig, report: ResetReport):
        plan = ResetPlan()
        if cfg.seed is not None:
            report.sampled['seed'] = int(cfg.seed)

        plan.effective_random_init_enabled = 0 if cfg.force_fixed_arm_init else int(cfg.random_init_enabled)
        report.sampled['effective_random_init_enabled'] = int(plan.effective_random_init_enabled)

        if plan.effective_random_init_enabled == 1:
            xyz = self._call_with_seed(cfg.seed, self._sample_random_init_v1)
            plan.random_target_pos = np.array(xyz, dtype=np.float64)
        elif plan.effective_random_init_enabled == 2:
            result = self._call_with_seed(cfg.seed, self._sample_random_init_v2)
            if result is None:
                plan.fallback_flags['random_init_v2_to_v1'] = True
                report.add_warning("V2 random init failed, falling back to V1.")
                xyz = self._call_with_seed(cfg.seed, self._sample_random_init_v1)
                plan.random_target_pos = np.array(xyz, dtype=np.float64)
            else:
                plan.random_target_pos = np.array(result, dtype=np.float64)

        rng = np.random.default_rng(cfg.seed)
        if cfg.tb3_x_random_enabled:
            x_init = float(rng.uniform(cfg.tb3_x_min, cfg.tb3_x_max))
        else:
            x_init = float(0.5 * (cfg.tb3_x_min + cfg.tb3_x_max))
        y_init = -0.25
        z_init = 0.0
        yaw_init = float(np.deg2rad(90))
        plan.tb3_init_pose = np.array([x_init, y_init, z_init, yaw_init], dtype=np.float64)
        report.sampled['tb3_init_pose'] = plan.tb3_init_pose.tolist()
        if plan.random_target_pos is not None:
            report.sampled['random_target_pos'] = plan.random_target_pos.tolist()
        return plan

    def _apply_reset_plan(self, cfg: ResetConfig, plan: ResetPlan, report: ResetReport):
        # 单次消费 pending 托盘初始化颜色，避免状态跨回合泄漏。
        self._active_tray_init_color = self.pending_tray_init_color
        self.pending_tray_init_color = None

        # 机械臂总是先初始化到标准位置
        q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
        q_zero, _, _ = solve_ik(
            self.env,
            self.joint_names,
            'tcp_link',
            q_init,
            np.array([0.3, 0.0, 0.48 + V6_Z_OFFSET]),
            rpy2r(np.deg2rad([90, -0.0, 90])),
        )
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)

        if plan.random_target_pos is not None:
            self.random_target_q, _, _ = solve_ik(
                self.env,
                self.joint_names,
                'tcp_link',
                q_zero,
                plan.random_target_pos,
                rpy2r(np.deg2rad([90, -0.0, 90])),
            )
            self.random_target_pos = plan.random_target_pos
            self.moving_to_random = True
            self.random_start_q = q_zero.copy()
            self.random_interp_steps = 0
            print("   → Will smoothly move from standard position to random position")
            print("   → After reaching: Press [Y] to start expert policy with recording")
        else:
            self.moving_to_random = False
            self.random_target_q = None
            self.random_start_q = None
            self.random_interp_steps = 0
            self.random_target_pos = None

        try:
            x_init, y_init, z_init, yaw_init = plan.tb3_init_pose
            R_init = rpy2r(np.array([0, 0, yaw_init]))
            self.env.set_pR_base_body(
                body_name='tb3_base',
                p=np.array([x_init, y_init, z_init]),
                R=R_init,
            )
        except Exception as e:
            report.add_warning(f"Could not reset TB3 base pose: {e}")

        self.last_q = copy.deepcopy(q_zero)
        self.tray_initialized_color = None
        self.tray_initialized_body = None
        self.table_reference_positions = {}

        initial_gripper_state = 0.0 if cfg.random_init_gripper_open else 1.0
        self.current_arm_q = np.concatenate([q_zero, np.array([initial_gripper_state] * 4)])
        self.current_wheel_vel = np.zeros(2)
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        self.gripper_state = bool(initial_gripper_state)

        self.arm_home_q = copy.deepcopy(q_zero)
        self.returning_home = False
        self.home_start_q = None
        self.home_interp_steps = 0

        self.expert_policy.reset()
        self.is_recording = False
        self.base_auto_parking.reset()
        self.base_action_intent = np.zeros(2, dtype=np.float32)
        self._deactivate_all_tray_supports()

        self._init_objects_demo()
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate(
            [mug_red_init_pose, mug_blue_init_pose, plate_init_pose],
            dtype=np.float32,
        )

        for _ in range(100):
            self.step_env()

    def _finalize_reset(self, cfg: ResetConfig, report: ResetReport):
        self._suppress_pending_tray_init_update = True
        try:
            if cfg.preserve_instruction and getattr(self, 'instruction', None):
                current_task_type = getattr(self, 'task_type', 'arm' if self.control_mode == 'arm' else 'nav')
                self.set_instruction(given=self.instruction, task_type=current_task_type)
            else:
                self.set_instruction()
        finally:
            self._suppress_pending_tray_init_update = False

        try:
            self.grab_image()
        except Exception as e:
            report.add_warning(f"grab_image failed during reset: {e}")

    def reset(self, seed=None, mode=None, force_fixed_arm_init=False, preserve_instruction=False, options=None):
        """Reset the environment and return a structured reset report."""
        report = ResetReport()
        cfg, cfg_warnings = self._resolve_reset_config(
            seed=seed,
            mode=mode,
            force_fixed_arm_init=force_fixed_arm_init,
            preserve_instruction=preserve_instruction,
            options=options,
        )
        report.config = cfg
        for warning in cfg_warnings:
            report.add_warning(warning)

        self._apply_reset_config(cfg)
        plan = self._plan_reset(cfg, report)
        report.fallback_flags.update(plan.fallback_flags)
        self._apply_reset_plan(cfg, plan, report)
        self._finalize_reset(cfg, report)

        if self.debug_reset and (report.warnings or report.errors):
            print(
                f"[RESET] warnings={len(report.warnings)} errors={len(report.errors)} "
                f"fallbacks={report.fallback_flags}"
            )
        return report

    def reinitialize_arm_only(self):
        """Reinitialize only the arm state without moving the base or scene objects."""
        q_init = np.deg2rad([0, 0, 0, 0, 0, 0])
        q_zero, _, _ = solve_ik(
            self.env,
            self.joint_names,
            'tcp_link',
            q_init,
            np.array([0.3, 0.0, 0.48 + V6_Z_OFFSET]),
            rpy2r(np.deg2rad([90, -0.0, 90])),
        )
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)

        self.last_q = copy.deepcopy(q_zero)
        initial_gripper_state = 0.0 if self.random_init_gripper_open else 1.0
        self.current_arm_q = np.concatenate([q_zero, np.array([initial_gripper_state] * 4)])
        self.current_wheel_vel = np.zeros(2)
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        self.gripper_state = bool(initial_gripper_state)

        self.arm_home_q = copy.deepcopy(q_zero)
        self.returning_home = False
        self.home_start_q = None
        self.home_interp_steps = 0
        self.moving_to_random = False
        self.random_target_q = None
        self.random_start_q = None
        self.random_interp_steps = 0
        self.random_target_pos = None

        self.expert_policy.reset()
        self._deactivate_all_tray_supports()

        try:
            self.grab_image()
        except Exception:
            pass

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
        self._update_tray_supports()

    def _set_tray_support_local_pos(self, support_body_name, local_pos, forward=False):
        self.env.set_p_body(
            body_name=support_body_name,
            p=np.array(local_pos, dtype=np.float64),
            forward=forward,
        )

    def _deactivate_all_tray_supports(self):
        for mug_name, support_body_name in self.tray_support_body_names.items():
            self._set_tray_support_local_pos(
                support_body_name,
                self.tray_support_hidden_pos,
                forward=False,
            )
            self.tray_support_active[mug_name] = False
        self.env.forward(increase_tick=False)

    def _update_tray_supports(self):
        """
        杯子在托盘上稳定放置后，激活当前位置下方的隐藏杯座。
        杯座作为托盘子 body，后续会自然跟随底盘，无需持续搬动。
        """
        try:
            p_tray, R_tray = self.env.get_pR_body('tb3_tray')
            p_tcp = self.env.get_p_body('tcp_link')
            support_changed = False

            for mug_name, support_body_name in self.tray_support_body_names.items():
                p_mug = self.env.get_p_body(mug_name)
                rel_p = R_tray.T @ (p_mug - p_tray)
                tcp_dist = np.linalg.norm(p_mug - p_tcp)
                on_tray = (
                    abs(rel_p[0]) < 0.085
                    and abs(rel_p[1]) < 0.085
                    and 0.035 < rel_p[2] < 0.11
                )
                arm_away = tcp_dist > 0.11

                if on_tray and arm_away:
                    if not self.tray_support_active[mug_name]:
                        support_local_pos = np.array([rel_p[0], rel_p[1], 0.0], dtype=np.float64)
                        self._set_tray_support_local_pos(
                            support_body_name,
                            support_local_pos,
                            forward=False,
                        )
                        self.tray_support_active[mug_name] = True
                        support_changed = True
                else:
                    if self.tray_support_active[mug_name]:
                        self._set_tray_support_local_pos(
                            support_body_name,
                            self.tray_support_hidden_pos,
                            forward=False,
                        )
                        self.tray_support_active[mug_name] = False
                        support_changed = True

            if support_changed:
                self.env.forward(increase_tick=False)
        except Exception:
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
