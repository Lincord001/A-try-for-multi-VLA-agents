import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

# ====== 🤖 Expert Policy Parameters (专家策略参数) ======
# 这些参数用于自动化录制数据的专家策略，可根据需要调整

# --- 高度参数 ---
# 🔥 V2环境调整：桌面0.83m，盘子0.82m
EXPERT_Z_TRAVEL_BASE = 1          # 巡航基础高度 (Transport cruise height) = 桌面0.83 + 0.2
EXPERT_Z_TRAVEL_NOISE = 0.03         # 巡航高度随机扰动范围 ± (Cruise height noise)
EXPERT_Z_GRASP_BASE = 0.84           # 抓取高度基础值 (Grasp height) = 桌面高度
EXPERT_Z_PLACE_BASE = 0.87          # 放置高度基础值 (Place height) = 盘子高度
EXPERT_RETRACT_HEIGHT = 1.0         # 撤离安全高度 (Retract safe height) = 巡航高度附近

# --- 位置偏移与噪声 ---
EXPERT_XY_NOISE_SCALE = 0.01        # 端点随机噪声范围 ±3mm (Endpoint noise for robustness)
EXPERT_Y_GRASP_OFFSET = 0.067        # 抓取Y轴固定偏移（用于抓取杯子把手）(Y offset for cup handle)
EXPERT_Y_PLACE_OFFSET = 0.03         # 放置时的 Y 轴固定偏移
EXPERT_HOVER_NOISE = 0.01            # 悬停点误差 (Hover point noise)
EXPERT_Z_NOISE = 0.005               # Z轴高度微小随机噪声 (Z height noise)

# --- 🔥 漏斗移动逻辑参数 (Funnel Approach Parameters) ---
EXPERT_FUNNEL_HOVER_RADIUS = 0.03   # 悬停点圆半径（米）(Hover point circle radius, 3cm)
EXPERT_FUNNEL_MID_RADIUS = 0.01     # 中间点圆半径（米）(Mid point circle radius, 1cm)
EXPERT_FUNNEL_HOVER_Z = None        # 悬停点Z坐标（米），None=使用巡航高度z_travel
EXPERT_FUNNEL_MID_Z = 0.92          # 中间点Z坐标（米）(Mid point Z coordinate) = 桌面0.83 + 0.09

# --- 🔥 人类抖动模拟参数 (Human Tremor Simulation Parameters) ---
EXPERT_TREMOR_ENABLED = True           # 是否启用抖动 (Enable tremor)
EXPERT_TREMOR_AMPLITUDE = 0.002       # 抖动幅度（米）(Tremor amplitude, 2mm)
EXPERT_TREMOR_SMOOTHNESS = 0.7        # 抖动平滑度 (0-1，越大越平滑) (Tremor smoothness)

# --- 🔥 动态步数参数（基于距离计算，提高数据多样性）---
EXPERT_SPEED_APPROACH = 0.008        # 接近阶段速度 (Approach speed, m/step)
EXPERT_SPEED_DESCEND = 0.005         # 下降阶段速度 (Descend speed, m/step)
EXPERT_SPEED_LIFT = 0.006            # 提升阶段速度 (Lift speed, m/step)
EXPERT_SPEED_TRANSPORT = 0.008       # 运输阶段速度 (Transport speed, m/step)
EXPERT_SPEED_LOWER = 0.006           # 下降到放置点速度 (Lower speed, m/step)
EXPERT_SPEED_RETRACT = 0.006         # 撤离阶段速度 (Retract speed, m/step)

# 最小/最大步数限制（防止极端情况）
EXPERT_MIN_STEPS = 8                 # 任何阶段的最小步数 (Min steps for any phase)
EXPERT_MAX_STEPS = 200                # 任何阶段的最大步数 (Max steps for any phase)

# 等待阶段步数（带随机扰动范围）
EXPERT_OPEN_WAIT_BASE = 4            # 🔥 初始张开夹爪等待基础步数
EXPERT_OPEN_WAIT_NOISE = 1           # 🔥 初始张开夹爪等待随机扰动 ±
EXPERT_GRASP_WAIT_BASE = 8           # 抓取等待基础步数 (Grasp wait base steps)
EXPERT_GRASP_WAIT_NOISE = 2          # 抓取等待随机扰动 ± (Grasp wait noise)
EXPERT_PLACE_WAIT_BASE = 8           # 放置等待基础步数 (Place wait base steps)
EXPERT_PLACE_WAIT_NOISE = 2          # 放置等待随机扰动 ± (Place wait noise)

# 速度随机扰动范围（增加多样性）
EXPERT_SPEED_NOISE_RATIO = 0.2       # 速度随机扰动比例 ±20% (Speed noise ratio)

# --- 贝塞尔曲线控制点参数 ---
EXPERT_BEZIER_XY_OFFSET = 0.02       # 贝塞尔控制点XY随机偏移范围 (Control point XY offset)
EXPERT_BEZIER_Z_OFFSET_MIN = 0.0     # 贝塞尔控制点Z轴最小偏移 (Control point Z min offset)
EXPERT_BEZIER_Z_OFFSET_MAX = 0.05    # 贝塞尔控制点Z轴最大偏移 (Control point Z max offset)

# --- 🔥 新增：基于机械臂基座避障的贝塞尔曲线参数 ---
EXPERT_BEZIER_AVOID_RADIUS = 0.2     # 机械臂基座避障半径（米）(Arm base avoidance radius)
EXPERT_BEZIER_TANGENT_MARGIN = 0.05  # 与圆相切时的安全余量（米）(Safety margin when tangent to circle)
EXPERT_BEZIER_SMALL_CURVE = 0.08     # 直线不穿过圆时的小弧线偏移（米）(Small curve offset when line doesn't cross circle)

# --- 录制缓冲参数 ---
EXPERT_START_DELAY = 0               # 🔥 启动缓冲期步数（已禁用，立即开始录制）
EXPERT_POST_WAIT = 60                # 🔥 执行完成后等待步数（3秒，20Hz下）用于人工确认

# --- 🔥 夹爪随机初始化参数 (Random Initialization Parameters) ---
RANDOM_INIT_ENABLED = 0              # 🔥 随机初始化模式: 0=关闭, 1=旧版(扇形区域), 2=新版(环形交集)
RANDOM_INIT_CIRCLE_INNER_RADIUS = 0.02  # 🔥 新版随机初始化(V2)：以红色杯子为中心的环形区域内半径（米）
RANDOM_INIT_CIRCLE_OUTER_RADIUS = 0.05  # 🔥 新版随机初始化(V2)：以红色杯子为中心的环形区域外半径（米）
RANDOM_INIT_ANGLE_MIN = 0            # 角度范围最小值（度）(Min angle in degrees)
RANDOM_INIT_ANGLE_MAX = 45           # 角度范围最大值（度）(Max angle in degrees)
RANDOM_INIT_RADIUS_MIN = 0.3         # 径向距离最小值（米）(Min radial distance in meters)
RANDOM_INIT_RADIUS_MAX = 0.4         # 径向距离最大值（米）(Max radial distance in meters)
RANDOM_INIT_Z_MIN = 0.9              # Z坐标最小值（米）(Min Z coordinate in meters) - 旧版使用 = 桌面0.83 + 0.07
RANDOM_INIT_Z_MAX = 1.0              # Z坐标最大值（米）(Max Z coordinate in meters) - 旧版使用 = 巡航高度附近
RANDOM_INIT_Z_MIN_V2 = 0.88         # 🔥 新版随机初始化Z坐标最小值（米） = 中间点高度
RANDOM_INIT_Z_MAX_V2 = 1.0          # 🔥 新版随机初始化Z坐标最大值（米） = 巡航高度附近
RANDOM_INIT_GRIPPER_OPEN = True      # 🔥 初始化时夹爪是否张开 (True=张开, False=闭合)
RANDOM_INIT_MOVE_STEPS = 75         # 🔥 平滑移动到随机位置的步数（约7.5秒，20Hz）

# ====== 🎲 Object Initialization Parameters (物体初始化参数) ======
# 这些参数用于控制红色杯子的随机初始化，可根据需要调整

# --- 机械臂基座位置 ---
ARM_BASE_X = 0.0                     # 机械臂基座X坐标 (Arm base X position)
ARM_BASE_Y = 0.0                     # 机械臂基座Y坐标 (Arm base Y position)

# --- 桌面参数 ---
TABLE_Z_HEIGHT = 0.83                # 桌面高度 (Table height)

# --- 桌面范围限制 ---
TABLE_X_MIN = 0.25                   # 桌面X轴最小值 (Table X min boundary)
TABLE_X_MAX = 0.35                   # 桌面X轴最大值 (Table X max boundary)
TABLE_Y_MIN = -0.05                  # 桌面Y轴最小值 (Table Y min boundary)
TABLE_Y_MAX = 0.25                   # 桌面Y轴最大值 (Table Y max boundary)

# --- 🔥 扇形区域初始化参数（红色杯子）---
MUG_MIN_DIST = 0.30                  # 离机械臂基座最近距离（米）(Min distance from arm base)
MUG_MAX_DIST = 0.40                  # 离机械臂基座最远距离（米）(Max distance from arm base)
MUG_MIN_ANGLE = 0.0                  # 左偏角度（度）(Min angle in degrees, 0° = 正前方)
MUG_MAX_ANGLE = 45.0                 # 右偏角度（度）(Max angle in degrees, 45° = 右偏45度)

class SimpleEnv2:
    def __init__(self, 
                 xml_path,
                action_type='eef_pose', 
                state_type='joint_angle',
                seed = None,
                random_init_enabled=False, 
                random_init_gripper_open=True):
        """
        args:
            xml_path: str, path to the xml file
            action_type: str, type of action space, 'eef_pose','delta_joint_angle' or 'joint_angle'
            state_type: str, type of state space, 'joint_angle' or 'ee_pose'
            seed: int, seed for random number generator
            random_init_enabled: int, 0=关闭, 1=旧版(扇形区域), 2=新版(环形交集)
            random_init_gripper_open: bool, 初始化时夹爪是否张开
        """
        # Load the xml file
        self.env = MuJoCoParserClass(name='Tabletop',rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        self.joint_names = ['joint1',
                    'joint2',
                    'joint3',
                    'joint4',
                    'joint5',
                    'joint6',]
        
        # 🔥 随机初始化开关（由外部传入）
        self.random_init_enabled = random_init_enabled
        self.random_init_gripper_open = random_init_gripper_open
        
        # V2环境只有arm模式
        self.control_mode = 'arm' 
        
        # 内部状态
        self.current_arm_q = np.zeros(10)
        
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
        
        # 🤖 专家策略状态变量
        self.expert_trajectory = []       # 专家轨迹列表 [(pos, gripper_state), ...]
        self.expert_trajectory_idx = 0    # 当前执行到的轨迹索引
        self.expert_executing = False     # 是否正在执行专家策略
        self.expert_pending = False       # 是否处于缓冲期（等待启动）
        self.expert_countdown = 0         # 缓冲期倒计时
        self.is_recording = False         # 录制标志位（用于外部检测是否需要保存图像）
        self.expert_waiting_save = False  # 🔥 是否处于等待保存状态（执行完毕后的等待期）
        self.expert_lift_target_z = None   # 🔥 升高的目标Z坐标（固定值，避免每次重新计算）
        self.expert_lift_start_pos = None  # 🔥 平滑上升的起始位置 (3,)
        self.expert_lift_interp_steps = 0  # 🔥 平滑上升的插值步数
        self.expert_lift_total_steps = 50  # 🔥 平滑上升的总步数（约2.5秒，20Hz）
        self.expert_lift_tremor_prev = np.zeros(2)  # 🔥 平滑上升过程中的抖动状态（XY平面，用于平滑）
        self.expert_post_countdown = 0    # 🔥 执行完毕后的等待倒计时
        self.expert_auto_save = False     # 🔥 是否需要在等待期结束后自动保存（Y键触发）
        
        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        Initialize the viewer
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )
    
    def _sample_random_init_v1(self):
        """
        🔥 旧版随机初始化：在扇形区域内采样
        
        Returns:
            tuple: (x_target, y_target, z_target) 目标位置坐标
        """
        # 1. 生成随机角度和径向距离
        angle_deg = random.uniform(RANDOM_INIT_ANGLE_MIN, RANDOM_INIT_ANGLE_MAX)
        angle_rad = np.deg2rad(angle_deg)
        radius = random.uniform(RANDOM_INIT_RADIUS_MIN, RANDOM_INIT_RADIUS_MAX)
        
        # 2. 转换为笛卡尔坐标（相对于机械臂基座 ARM_BASE_X, ARM_BASE_Y）
        x_target = ARM_BASE_X + radius * np.cos(angle_rad)
        y_target = ARM_BASE_Y + radius * np.sin(angle_rad)
        z_target = random.uniform(RANDOM_INIT_Z_MIN, RANDOM_INIT_Z_MAX)
        
        print(f"🎲 Random Init (V1): pos=({x_target:.3f}, {y_target:.3f}, {z_target:.3f}), angle={angle_deg:.1f}°, radius={radius:.3f}m")
        
        return x_target, y_target, z_target
    
    def _sample_random_init_v2(self):
        """
        🔥 新版随机初始化：在环形区域与扇形区域的交集中采样
        
        Returns:
            tuple: (x_target, y_target, z_target) 目标位置坐标，如果失败则返回 None
        """
        # 🔥 获取红色杯子的位置
        try:
            red_mug_pos = self.env.get_p_body('body_obj_mug_5')
            mug_center_xy = red_mug_pos[:2]  # [x, y]
        except Exception as e:
            print(f"⚠️ Cannot get red mug position: {e}. Falling back to V1.")
            return None
        
        # 🔥 计算交集区域：环形（以红色杯子为中心）∩ 扇形区域（从基座出发）
        
        # ====== 🔥 改进1：交集预检测 ======
        rel_pos_center = mug_center_xy - np.array([ARM_BASE_X, ARM_BASE_Y])
        rel_dist_center = np.linalg.norm(rel_pos_center)
        rel_angle_rad_center = np.arctan2(rel_pos_center[1], rel_pos_center[0])
        rel_angle_deg_center = np.rad2deg(rel_angle_rad_center)
        if rel_angle_deg_center > 180:
            rel_angle_deg_center -= 360
        elif rel_angle_deg_center < -180:
            rel_angle_deg_center += 360
        
        # 预检测：判断交集是否可能为空
        min_possible_dist = rel_dist_center - RANDOM_INIT_CIRCLE_OUTER_RADIUS
        max_possible_dist = rel_dist_center + RANDOM_INIT_CIRCLE_OUTER_RADIUS
        intersection_possible = (max_possible_dist >= RANDOM_INIT_RADIUS_MIN and 
                               min_possible_dist <= RANDOM_INIT_RADIUS_MAX and
                               RANDOM_INIT_ANGLE_MIN <= rel_angle_deg_center <= RANDOM_INIT_ANGLE_MAX)
        
        max_attempts = 1000
        x_target = None
        y_target = None
        
        if not intersection_possible:
            print(f"   ⚠️ Warning: Intersection may be empty. Will try boundary search.")
        
        for attempt in range(max_attempts):
            # 策略：交替使用两种采样方法
            if attempt % 2 == 0:
                # 方法1：在环形区域内采样
                r_inner_sq = RANDOM_INIT_CIRCLE_INNER_RADIUS ** 2
                r_outer_sq = RANDOM_INIT_CIRCLE_OUTER_RADIUS ** 2
                r_sq = np.random.uniform(r_inner_sq, r_outer_sq)
                r_annulus = np.sqrt(r_sq)
                theta_circle = np.random.uniform(0, 2 * np.pi)
                candidate_xy = mug_center_xy + np.array([r_annulus * np.cos(theta_circle), r_annulus * np.sin(theta_circle)])
            else:
                # 方法2：在扇形区域内采样
                angle_deg = random.uniform(RANDOM_INIT_ANGLE_MIN, RANDOM_INIT_ANGLE_MAX)
                angle_rad = np.deg2rad(angle_deg)
                radius = random.uniform(RANDOM_INIT_RADIUS_MIN, RANDOM_INIT_RADIUS_MAX)
                candidate_xy = np.array([ARM_BASE_X, ARM_BASE_Y]) + radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            # 🔥 交集判断：必须同时满足两个条件
            dist_to_mug = np.linalg.norm(candidate_xy - mug_center_xy)
            in_annulus = (RANDOM_INIT_CIRCLE_INNER_RADIUS <= dist_to_mug <= RANDOM_INIT_CIRCLE_OUTER_RADIUS)
            
            # 条件2：在扇形区域内
            rel_pos = candidate_xy - np.array([ARM_BASE_X, ARM_BASE_Y])
            rel_dist = np.linalg.norm(rel_pos)
            rel_angle_rad = np.arctan2(rel_pos[1], rel_pos[0])
            rel_angle_deg = np.rad2deg(rel_angle_rad)
            
            if rel_angle_deg > 180:
                rel_angle_deg -= 360
            elif rel_angle_deg < -180:
                rel_angle_deg += 360
            
            in_sector = (RANDOM_INIT_RADIUS_MIN <= rel_dist <= RANDOM_INIT_RADIUS_MAX and 
                        RANDOM_INIT_ANGLE_MIN <= rel_angle_deg <= RANDOM_INIT_ANGLE_MAX)
            
            if in_annulus and in_sector:
                x_target = candidate_xy[0]
                y_target = candidate_xy[1]
                break
        
        # ====== 🔥 改进2和4：如果采样失败，使用边界点搜索和智能回退 ======
        if x_target is None or y_target is None:
            print(f"   🔍 Random sampling failed ({max_attempts} attempts). Trying boundary search...")
            
            boundary_search_success = False
            
            for theta in np.linspace(0, 2 * np.pi, 36):
                for r_annulus in [RANDOM_INIT_CIRCLE_INNER_RADIUS, RANDOM_INIT_CIRCLE_OUTER_RADIUS]:
                    candidate_xy = mug_center_xy + r_annulus * np.array([np.cos(theta), np.sin(theta)])
                    
                    rel_pos = candidate_xy - np.array([ARM_BASE_X, ARM_BASE_Y])
                    rel_dist = np.linalg.norm(rel_pos)
                    rel_angle_rad = np.arctan2(rel_pos[1], rel_pos[0])
                    rel_angle_deg = np.rad2deg(rel_angle_rad)
                    if rel_angle_deg > 180:
                        rel_angle_deg -= 360
                    elif rel_angle_deg < -180:
                        rel_angle_deg += 360
                    
                    if (RANDOM_INIT_RADIUS_MIN <= rel_dist <= RANDOM_INIT_RADIUS_MAX and 
                        RANDOM_INIT_ANGLE_MIN <= rel_angle_deg <= RANDOM_INIT_ANGLE_MAX):
                        x_target = candidate_xy[0]
                        y_target = candidate_xy[1]
                        boundary_search_success = True
                        print(f"   ✅ Found intersection point via boundary search!")
                        break
                
                if boundary_search_success:
                    break
            
            # 🔥 改进4：如果边界搜索也失败，使用智能回退策略
            if not boundary_search_success:
                center_in_sector = (RANDOM_INIT_RADIUS_MIN <= rel_dist_center <= RANDOM_INIT_RADIUS_MAX and 
                                   RANDOM_INIT_ANGLE_MIN <= rel_angle_deg_center <= RANDOM_INIT_ANGLE_MAX)
                
                if center_in_sector:
                    for test_radius in [RANDOM_INIT_CIRCLE_INNER_RADIUS, 
                                       (RANDOM_INIT_CIRCLE_INNER_RADIUS + RANDOM_INIT_CIRCLE_OUTER_RADIUS) / 2,
                                       RANDOM_INIT_CIRCLE_OUTER_RADIUS]:
                        fallback_xy = mug_center_xy + test_radius * np.array([np.cos(rel_angle_rad_center), np.sin(rel_angle_rad_center)])
                        
                        rel_pos_fallback = fallback_xy - np.array([ARM_BASE_X, ARM_BASE_Y])
                        rel_dist_fallback = np.linalg.norm(rel_pos_fallback)
                        rel_angle_rad_fallback = np.arctan2(rel_pos_fallback[1], rel_pos_fallback[0])
                        rel_angle_deg_fallback = np.rad2deg(rel_angle_rad_fallback)
                        if rel_angle_deg_fallback > 180:
                            rel_angle_deg_fallback -= 360
                        elif rel_angle_deg_fallback < -180:
                            rel_angle_deg_fallback += 360
                        
                        if (RANDOM_INIT_RADIUS_MIN <= rel_dist_fallback <= RANDOM_INIT_RADIUS_MAX and 
                            RANDOM_INIT_ANGLE_MIN <= rel_angle_deg_fallback <= RANDOM_INIT_ANGLE_MAX):
                            x_target = fallback_xy[0]
                            y_target = fallback_xy[1]
                            print(f"   ✅ Found fallback point (center in sector, radius={test_radius*100:.1f}cm)")
                            break
                    
                    if x_target is None:
                        fallback_xy = mug_center_xy + RANDOM_INIT_CIRCLE_OUTER_RADIUS * np.array([np.cos(rel_angle_rad_center), np.sin(rel_angle_rad_center)])
                        x_target = fallback_xy[0]
                        y_target = fallback_xy[1]
                        print(f"   ⚠️ Using outer radius as fallback (may not be in sector)")
                else:
                    direction_to_mug = rel_pos_center / rel_dist_center if rel_dist_center > 1e-6 else np.array([1.0, 0.0])
                    
                    for test_radius in np.linspace(RANDOM_INIT_RADIUS_MIN, RANDOM_INIT_RADIUS_MAX, 10):
                        fallback_xy = np.array([ARM_BASE_X, ARM_BASE_Y]) + test_radius * direction_to_mug
                        
                        dist_to_mug = np.linalg.norm(fallback_xy - mug_center_xy)
                        if RANDOM_INIT_CIRCLE_INNER_RADIUS <= dist_to_mug <= RANDOM_INIT_CIRCLE_OUTER_RADIUS:
                            rel_angle_rad_fallback = np.arctan2(fallback_xy[1] - ARM_BASE_Y, fallback_xy[0] - ARM_BASE_X)
                            rel_angle_deg_fallback = np.rad2deg(rel_angle_rad_fallback)
                            if rel_angle_deg_fallback > 180:
                                rel_angle_deg_fallback -= 360
                            elif rel_angle_deg_fallback < -180:
                                rel_angle_deg_fallback += 360
                            
                            if RANDOM_INIT_ANGLE_MIN <= rel_angle_deg_fallback <= RANDOM_INIT_ANGLE_MAX:
                                x_target = fallback_xy[0]
                                y_target = fallback_xy[1]
                                print(f"   ✅ Found fallback point (sector to annulus, radius={test_radius:.3f}m)")
                                break
                    
                    if x_target is None:
                        print(f"   ⚠️ Warning: Intersection area is empty. Will fallback to V1.")
                        return None
        
        # Z坐标使用新版范围
        z_target = random.uniform(RANDOM_INIT_Z_MIN_V2, RANDOM_INIT_Z_MAX_V2)
        
        rel_pos_display = np.array([x_target, y_target]) - np.array([ARM_BASE_X, ARM_BASE_Y])
        rel_dist_display = np.linalg.norm(rel_pos_display)
        rel_angle_display = np.rad2deg(np.arctan2(rel_pos_display[1], rel_pos_display[0]))
        
        print(f"🎲 Random Init (V2): pos=({x_target:.3f}, {y_target:.3f}, {z_target:.3f})")
        try:
            mug_dist = np.linalg.norm(np.array([x_target, y_target]) - mug_center_xy)
            print(f"   → Mug center: ({mug_center_xy[0]:.3f}, {mug_center_xy[1]:.3f}), distance from mug: {mug_dist*100:.1f}cm")
        except:
            pass
        print(f"   → Relative to base: dist={rel_dist_display:.3f}m, angle={rel_angle_display:.1f}°")
        
        return x_target, y_target, z_target
    
    def smooth_move_to_random(self):
        """
        🔥 平滑地将机械臂移动到随机初始化位置
        """
        if not self.moving_to_random or self.random_target_q is None:
            return False
        
        self.random_interp_steps += 1
        alpha = min(self.random_interp_steps / RANDOM_INIT_MOVE_STEPS, 1.0)
        
        alpha_smooth = alpha * alpha * (3 - 2 * alpha)
        
        target_q = self.random_start_q * (1 - alpha_smooth) + self.random_target_q * alpha_smooth
        
        gripper_cmd = self.current_arm_q[6:10] if hasattr(self, 'current_arm_q') else np.array([0.0]*4)
        self.current_arm_q = np.concatenate([target_q, gripper_cmd])
        
        self.env.forward(q=target_q, joint_names=self.joint_names, increase_tick=False)
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        self.p0 = p_current
        self.R0 = R_current
        
        if self.random_interp_steps >= RANDOM_INIT_MOVE_STEPS:
            self.current_arm_q = np.concatenate([self.random_target_q, gripper_cmd])
            self.env.forward(q=self.random_target_q, joint_names=self.joint_names, increase_tick=False)
            p_final, R_final = self.env.get_pR_body(body_name='tcp_link')
            self.p0 = p_final
            self.R0 = R_final
            
            self.moving_to_random = False
            print(f"\n✅ Reached random position!")
            print(f"   → Press [Y] to start expert policy with recording")
            return True
        
        return True
    
    def smooth_return_home(self):
        """
        平滑地将机械臂移动到初始位置
        """
        if self.arm_home_q is None:
            return False
        
        if self.returning_home:
            self.home_interp_steps += 1
            alpha = min(self.home_interp_steps / self.home_total_steps, 1.0)
            
            alpha_smooth = alpha * alpha * (3 - 2 * alpha)
            
            target_q = self.home_start_q * (1 - alpha_smooth) + self.arm_home_q * alpha_smooth
            
            gripper_cmd = self.current_arm_q[6:10] if hasattr(self, 'current_arm_q') else np.array([0.0]*4)
            self.current_arm_q = np.concatenate([target_q, gripper_cmd])
            
            self.env.forward(q=target_q, joint_names=self.joint_names, increase_tick=False)
            p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
            self.p0 = p_current
            self.R0 = R_current
            
            if self.home_interp_steps >= self.home_total_steps:
                self.returning_home = False
                self.home_interp_steps = 0
                self.home_start_q = None
                self.current_arm_q = np.concatenate([self.arm_home_q, gripper_cmd])
                self.env.forward(q=self.arm_home_q, joint_names=self.joint_names, increase_tick=False)
                return True
            
            return True
        else:
            self.home_start_q = self.env.get_qpos_joints(joint_names=self.joint_names).copy()
            self.returning_home = True
            self.home_interp_steps = 0
            return True

    def reset(self, seed = None, mode=None):
        '''
        Reset the environment
        Move the robot to a initial position, set the object positions based on the seed
        '''
        if seed != None: np.random.seed(seed=seed)
        
        # V2环境只有arm模式，mode参数保留以兼容接口
        if mode is not None:
            self.control_mode = 'arm'  # V2环境强制为arm模式
        
        # 🔥 机械臂初始化：总是先初始化到标准位置
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero,ik_err_stack,ik_info = solve_ik(
            env = self.env,
            joint_names_for_ik = self.joint_names,
            body_name_trgt     = 'tcp_link',
            q_init       = q_init,
            p_trgt       = np.array([0.3,0.0,1.0]),
            R_trgt       = rpy2r(np.deg2rad([90,-0.,90 ])),
        )
        self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)
        
        # set plate position (固定位置)
        plate_xyz = np.array([0.3, -0.25, 0.82])
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        
        # 🔥 红色杯子位置随机化（扇形区域，相对于机械臂基座）
        # 1. 极坐标生成
        r = random.uniform(MUG_MIN_DIST, MUG_MAX_DIST)  # 0.30 ~ 0.40m
        theta = random.uniform(np.deg2rad(MUG_MIN_ANGLE), np.deg2rad(MUG_MAX_ANGLE))  # 0° ~ 45°
        
        # 2. 转换为笛卡尔坐标（相对于机械臂基座 ARM_BASE_X, ARM_BASE_Y）
        mug_x = ARM_BASE_X + r * np.cos(theta)
        mug_y = ARM_BASE_Y + r * np.sin(theta)
        mug_z = TABLE_Z_HEIGHT
        
        # 3. 限制在桌面范围内（Clip，留点边缘余量）
        mug_x = np.clip(mug_x, TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05)
        mug_y = np.clip(mug_y, TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05)
        
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=np.array([mug_x, mug_y, mug_z]))
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        
        # 🔥 隐藏蓝色杯子（移到远处）
        self.env.set_p_base_body(body_name='body_obj_mug_6',p=np.array([20.0, 0.0, 1.0]))
        self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        
        self.env.forward(increase_tick=False)

        # 🔥 如果启用了随机初始化，生成随机目标位置并准备平滑移动
        if self.random_init_enabled == 1:
            x_target, y_target, z_target = self._sample_random_init_v1()
        elif self.random_init_enabled == 2:
            result = self._sample_random_init_v2()
            if result is None:
                print("   → Falling back to V1 random initialization.")
                x_target, y_target, z_target = self._sample_random_init_v1()
            else:
                x_target, y_target, z_target = result
        
        # 如果启用了随机初始化，设置目标位置并准备平滑移动
        if self.random_init_enabled in [1, 2]:
            target_pos = np.array([x_target, y_target, z_target])
            self.random_target_q, _, _ = solve_ik(
                self.env, self.joint_names, 'tcp_link', q_zero, 
                target_pos, rpy2r(np.deg2rad([90, -0., 90]))
            )
            self.random_target_pos = target_pos
            
            self.moving_to_random = True
            self.random_start_q = q_zero.copy()
            self.random_interp_steps = 0
            
            print(f"   → Will smoothly move from standard position to random position")
            print(f"   → After reaching: Press [Y] to start expert policy with recording")
        else:
            self.moving_to_random = False
            self.random_target_q = None
            self.random_start_q = None
            self.random_interp_steps = 0
            self.random_target_pos = None

        # Set the initial pose of the robot
        self.last_q = copy.deepcopy(q_zero)
        # 🔥 根据开关设置初始夹爪状态
        initial_gripper_state = 0.0 if self.random_init_gripper_open else 1.0
        self.q = np.concatenate([q_zero, np.array([initial_gripper_state]*4)])
        self.current_arm_q = np.concatenate([q_zero, np.array([initial_gripper_state]*4)])
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        
        # 🔥 保存初始关节角度（用于平滑归位）
        self.arm_home_q = copy.deepcopy(q_zero)
        self.returning_home = False
        self.home_start_q = None
        self.home_interp_steps = 0
        
        # 🤖 重置专家策略状态
        self.expert_trajectory = []
        self.expert_trajectory_idx = 0
        self.expert_executing = False
        self.expert_pending = False
        self.expert_lift_target_z = None
        self.expert_lift_start_pos = None
        self.expert_lift_interp_steps = 0
        self.expert_lift_tremor_prev = np.zeros(2)
        self.expert_countdown = 0
        self.is_recording = False
        self.expert_waiting_save = False
        self.expert_post_countdown = 0
        self.expert_auto_save = False
        
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        # 🔥 只保存红色杯子和盘子的位置（蓝色杯子已隐藏）
        self.obj_init_pose = np.concatenate([mug_red_init_pose, plate_init_pose],dtype=np.float32)
        
        for _ in range(100):
            self.step_env()
        self.set_instruction()
        print("DONE INITIALIZATION")
        self.gripper_state = bool(initial_gripper_state)  # 🔥 同步更新 gripper_state
        
        # 重置时刷新一次图像缓存
        self.grab_image()

    def set_instruction(self, given = None):
        """
        Set the instruction for the task
        🔥 只支持红色杯子
        """
        if given is None:
            self.instruction = 'Place the red mug on the plate.'
            self.obj_target = 'body_obj_mug_5'
            self.target_color = 'red'
        else:
            self.instruction = given
            if 'red' in self.instruction.lower():
                self.obj_target = 'body_obj_mug_5'
                self.target_color = 'red'
            else:
                # 默认使用红色杯子
                self.obj_target = 'body_obj_mug_5'
                self.target_color = 'red'
                print(f"⚠️ Warning: Instruction does not contain 'red'. Using red mug as default.")

    def step(self, action, mode='arm'):
        '''
        Take a step in the environment
        args:
            action: np.array of shape (7,), action to take
            mode: str, 保留参数以兼容接口，V2环境只有arm模式
        returns:
            state: np.array, state of the environment after taking the action
        '''
        # V2环境只有arm模式
        self.control_mode = 'arm'
        
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'tcp_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')
        
        gripper_cmd = np.array([action[-1]]*4)
        gripper_cmd[[1,3]] *= 0.8
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])
        self.q = q
        self.current_arm_q = q
        
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        # 🔥 V2环境只有机械臂，没有轮子，所以只使用current_arm_q（10维：6关节+4夹爪）
        if hasattr(self, 'current_arm_q'):
            self.env.step(self.current_arm_q)
        else:
            self.env.step(self.q)

    def grab_image(self):
        '''
        grab images from the environment
        returns:
            images_dict: dict, {'agent': rgb_agent, 'wrist': rgb_ego}
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric')
        self.rgb_side = self.env.get_fixed_cam_rgb(cam_name='sideview')
        
        # 返回字典格式（与v4一致）
        images = {}
        images['agent'] = self.rgb_agent
        images['wrist'] = self.rgb_ego
        
        return images

    def render(self, teleop=False, idx = 0):
        '''
        Render the environment
        '''
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        
        # 确保图像已获取
        if not hasattr(self, 'rgb_ego'):
            self.grab_image()
        
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Wrist View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        self.env.plot_T(p = np.array([0.1,0.0,1.0]), label=f"Episode {idx}", plot_axis=False, plot_sphere=False)
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
        
        if getattr(self, 'instruction', None) is not None:
            self.env.viewer_text_overlay(text1='Task',text2=self.instruction)
        self.env.render()

    def get_joint_state(self):
        '''
        Get the joint state of the robot
        returns:
            q: np.array, joint angles of the robot + gripper state (0 for open, 1 for closed)
            [j1,j2,j3,j4,j5,j6,gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def get_delta_q(self):
        '''
        Get the delta joint angles of the robot
        returns:
            delta: np.array, delta joint angles of the robot + gripper state (0 for open, 1 for closed)
            [dj1,dj2,dj3,dj4,dj5,dj6,gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('rh_r1')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def check_success(self):
        '''
        Check if the mug is placed on the plate
        + Gripper should be open and move upward above 0.9
        '''
        p_mug = self.env.get_p_body(self.obj_target)
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if np.linalg.norm(p_mug[:2] - p_plate[:2]) < 0.1 and np.linalg.norm(p_mug[2] - p_plate[2]) < 0.6 and self.env.get_qpos_joint('rh_r1') < 0.1:
            p = self.env.get_p_body('tcp_link')[2]
            if p > 0.9:
                return True
        return False
    
    def get_obj_pose(self):
        '''
        returns: 
            p_mug_red: np.array, position of the red mug
            p_plate: np.array, position of the plate
        🔥 只返回红色杯子和盘子（蓝色杯子已隐藏）
        '''
        p_mug_red = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        # 🔥 为了保持兼容性，返回两个杯子位置（蓝色杯子位置设为0）
        p_mug_blue = np.zeros(3)  # 蓝色杯子已隐藏，返回零向量

        return p_mug_red, p_mug_blue, p_plate
    
    def set_obj_pose(self, p_mug_red, p_mug_blue, p_plate):
        '''
        Set the object poses
        args:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug (忽略，已隐藏)
            p_plate: np.array, position of the plate
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug_red)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        # 🔥 蓝色杯子保持隐藏
        self.env.set_p_base_body(body_name='body_obj_mug_6',p=np.array([20.0, 0.0, 1.0]))
        self.env.set_R_base_body(body_name='body_obj_mug_6',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()

    def get_ee_pose(self):
        '''
        get the end effector pose of the robot + gripper state
        '''
        p, R = self.env.get_pR_body(body_name='tcp_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
    
    # ====== 🤖 Expert Policy Helper Functions (专家策略辅助函数) ======
    
    def interpolate_move(self, start_pos, end_pos, steps, gripper_state, add_tremor=False):
        """线性插值移动（可选添加抖动）"""
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps
            pos = start_pos + t * (end_pos - start_pos)
            waypoints.append((pos.copy(), gripper_state))
        
        if add_tremor and EXPERT_TREMOR_ENABLED:
            waypoints = self.add_orthogonal_tremor(
                waypoints, start_pos, end_pos, 
                EXPERT_TREMOR_AMPLITUDE, 
                EXPERT_TREMOR_SMOOTHNESS
            )
        
        return waypoints
    
    def bezier_move(self, p0, p1, p2, steps, gripper_state, add_tremor=False):
        """二阶贝塞尔曲线插值移动（可选添加抖动）"""
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps
            pos = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            waypoints.append((pos.copy(), gripper_state))
        
        if add_tremor and EXPERT_TREMOR_ENABLED:
            waypoints = self.add_orthogonal_tremor(
                waypoints, p0, p2, 
                EXPERT_TREMOR_AMPLITUDE, 
                EXPERT_TREMOR_SMOOTHNESS
            )
        
        return waypoints
    
    def create_gripper_action(self, pos, gripper_state, steps):
        """创建原地开关夹爪并等待的航点序列"""
        return [(pos.copy(), gripper_state) for _ in range(steps)]
    
    def distance_based_steps(self, start, end, speed_per_step, min_steps=None, max_steps=None):
        """基于距离动态计算插值步数"""
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
        """为等待阶段生成带随机扰动的步数"""
        noise = np.random.randint(-noise_range, noise_range + 1)
        return max(3, base_steps + noise)
    
    def bezier_curve_length(self, p0, p1, p2, num_samples=20):
        """估算二阶贝塞尔曲线的弧长"""
        length = 0.0
        prev_point = p0
        for i in range(1, num_samples + 1):
            t = i / num_samples
            point = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            length += np.linalg.norm(point - prev_point)
            prev_point = point
        return length
    
    def sample_point_in_circle(self, center_xy, radius):
        """在圆内均匀采样随机点"""
        r = np.sqrt(np.random.uniform(0, 1)) * radius
        theta = np.random.uniform(0, 2 * np.pi)
        point_xy = center_xy + np.array([r * np.cos(theta), r * np.sin(theta)])
        return point_xy
    
    def add_orthogonal_tremor(self, trajectory, start_pos, end_pos, amplitude, smoothness=0.7):
        """在轨迹上添加正交于移动方向的抖动"""
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
        
        for i, (pos, gripper_state) in enumerate(trajectory):
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
    
    def _compute_expert_trajectory(self, current_pos):
        """计算专家策略轨迹"""
        if not hasattr(self, 'obj_target'):
            print("⚠️ No target object set. Call set_instruction() first.")
            return []
        obj_pos = self.env.get_p_body(self.obj_target)
        
        # 获取盘子位置
        try:
            plate_pos = self.env.get_p_body('body_obj_plate_11')
        except:
            print("⚠️ Cannot find plate body.")
            return []
        
        # ====== 0. 检查是否需要平滑上升 ======
        mid_z = EXPERT_FUNNEL_MID_Z
        mid_z_tolerance = 0.005
        lift_target_z = None
        lift_start_pos = None
        
        if current_pos[2] < mid_z - mid_z_tolerance:
            lift_target_z = mid_z + np.random.uniform(-mid_z_tolerance, mid_z_tolerance)
            lift_start_pos = current_pos.copy()
            lift_end_pos = np.array([current_pos[0], current_pos[1], lift_target_z])
            print(f"   ⬆️ Current Z ({current_pos[2]:.3f}m) below mid Z ({mid_z:.3f}m). Will add smooth lift to {lift_target_z:.3f}m in trajectory.")
        else:
            lift_end_pos = current_pos.copy()
        
        # ====== 1. 准备阶段：计算带噪声的目标位置 ======
        z_travel = EXPERT_Z_TRAVEL_BASE + np.random.uniform(-EXPERT_Z_TRAVEL_NOISE, EXPERT_Z_TRAVEL_NOISE)
        
        # 🔥 动态调整悬停点高度（根据初始化夹爪位置）
        # 计算默认悬停点高度
        default_hover_z = EXPERT_FUNNEL_HOVER_Z if EXPERT_FUNNEL_HOVER_Z is not None else z_travel
        
        # 计算悬停点和中间点的中点高度
        hover_mid_z = (default_hover_z + EXPERT_FUNNEL_MID_Z) / 2.0
        
        # 检查条件：当前高度低于悬停点但高于中点
        if current_pos[2] < default_hover_z and current_pos[2] > hover_mid_z:
            # 将悬停点高度调整为当前夹爪高度
            adjusted_hover_z = current_pos[2]
            print(f"   🔧 Adjusted hover Z: {default_hover_z:.3f}m -> {adjusted_hover_z:.3f}m (current_pos[2]={current_pos[2]:.3f}m, mid_point={hover_mid_z:.3f}m)")
        else:
            adjusted_hover_z = default_hover_z
        
        # 🔥 杯子抓取：使用Y轴偏移抓取杯子把手（与红色方块不同）
        y_grasp_offset = EXPERT_Y_GRASP_OFFSET  # 抓取杯子把手时的Y轴偏移
        # 🔥 抓取中心点（已加上Y轴偏移，作为悬停点和中间点的圆心）
        grasp_center_xy = np.array([obj_pos[0], obj_pos[1] + y_grasp_offset])
        
        grasp_pos = np.array([
            obj_pos[0] + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            obj_pos[1] + y_grasp_offset + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            EXPERT_Z_GRASP_BASE + np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE)
        ])
        
        # 🔥 悬停点：以grasp_center_xy（已包含Y轴偏移）为圆心，使用调整后的高度
        hover_xy = self.sample_point_in_circle(grasp_center_xy, radius=EXPERT_FUNNEL_HOVER_RADIUS)
        hover_pos = np.array([hover_xy[0], hover_xy[1], adjusted_hover_z])
        
        # 🔥 中间点：以grasp_center_xy（已包含Y轴偏移）为圆心
        mid_xy = self.sample_point_in_circle(grasp_center_xy, radius=EXPERT_FUNNEL_MID_RADIUS)
        mid_pos = np.array([mid_xy[0], mid_xy[1], EXPERT_FUNNEL_MID_Z])
        
        # 放置点（盘子中心，带噪声和Y轴偏移）
        y_place_offset = EXPERT_Y_PLACE_OFFSET  # 放置时的Y轴偏移（用于调整杯子在盘子上的位置）
        place_pos = np.array([
            plate_pos[0] + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            plate_pos[1] + y_place_offset + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            EXPERT_Z_PLACE_BASE + np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE)
        ])
        
        place_hover_pos = np.array([place_pos[0], place_pos[1], z_travel])
        
        # ====== 动态计算各阶段步数 ======
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], z_travel])
        retract_pos = np.array([place_pos[0], place_pos[1], EXPERT_RETRACT_HEIGHT])
        
        dist_to_hover = np.linalg.norm(lift_end_pos - hover_pos)
        dist_to_mid = np.linalg.norm(lift_end_pos - mid_pos)
        
        if dist_to_mid < dist_to_hover:
            use_hover = False
            approach_steps = self.distance_based_steps(lift_end_pos, mid_pos, EXPERT_SPEED_APPROACH)
            descend_steps_1 = 0
            descend_steps_2 = self.distance_based_steps(mid_pos, grasp_pos, EXPERT_SPEED_DESCEND)
        else:
            use_hover = True
            approach_steps = self.distance_based_steps(lift_end_pos, hover_pos, EXPERT_SPEED_APPROACH)
            descend_steps_1 = self.distance_based_steps(hover_pos, mid_pos, EXPERT_SPEED_DESCEND)
            descend_steps_2 = self.distance_based_steps(mid_pos, grasp_pos, EXPERT_SPEED_DESCEND)
        
        grasp_wait_steps = self.rand_wait_steps(EXPERT_GRASP_WAIT_BASE, EXPERT_GRASP_WAIT_NOISE)
        lift_steps = self.distance_based_steps(grasp_pos, lift_pos, EXPERT_SPEED_LIFT)
        lower_steps = self.distance_based_steps(place_hover_pos, place_pos, EXPERT_SPEED_LOWER)
        place_wait_steps = self.rand_wait_steps(EXPERT_PLACE_WAIT_BASE, EXPERT_PLACE_WAIT_NOISE)
        retract_steps = self.distance_based_steps(place_pos, retract_pos, EXPERT_SPEED_RETRACT)
        
        # ====== 生成完整轨迹 ======
        trajectory = []
        
        # 🔥 0. 预处理阶段：如果夹爪闭合，先等待1帧，再张开夹爪
        if self.gripper_state:
            trajectory.extend(self.create_gripper_action(current_pos, gripper_state=1.0, steps=1))
            open_wait_steps = self.rand_wait_steps(EXPERT_OPEN_WAIT_BASE, EXPERT_OPEN_WAIT_NOISE)
            trajectory.extend(self.create_gripper_action(current_pos, gripper_state=0.0, steps=open_wait_steps))
            print(f"   🔓 Opening gripper first (was closed): 1 frame wait + {open_wait_steps} steps to open")
        
        # 🔥 0.5. 平滑上升阶段（如果需要）
        if lift_target_z is not None:
            lift_smooth_steps = self.distance_based_steps(lift_start_pos, lift_end_pos, EXPERT_SPEED_LIFT)
            trajectory.extend(self.interpolate_move(lift_start_pos, lift_end_pos, lift_smooth_steps, gripper_state=0.0, add_tremor=True))
            print(f"   ⬆️ Smooth lift: {lift_start_pos[2]:.3f}m -> {lift_target_z:.3f}m ({lift_smooth_steps} steps)")
        
        # ====== 2. 接近阶段 ======
        if use_hover:
            trajectory.extend(self.interpolate_move(lift_end_pos, hover_pos, approach_steps, gripper_state=0.0, add_tremor=True))
            trajectory.extend(self.interpolate_move(hover_pos, mid_pos, descend_steps_1, gripper_state=0.0, add_tremor=True))
        else:
            trajectory.extend(self.interpolate_move(lift_end_pos, mid_pos, approach_steps, gripper_state=0.0, add_tremor=True))
        
        trajectory.extend(self.interpolate_move(mid_pos, grasp_pos, descend_steps_2, gripper_state=0.0, add_tremor=True))
        
        # ====== 3. 抓取阶段 ======
        trajectory.extend(self.create_gripper_action(grasp_pos, gripper_state=1.0, steps=grasp_wait_steps))
        
        # ====== 4. 运输阶段 ======
        trajectory.extend(self.interpolate_move(grasp_pos, lift_pos, lift_steps, gripper_state=1.0, add_tremor=True))
        
        # 4.2 贝塞尔曲线运输到盘子 above
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
        
        trajectory.extend(self.bezier_move(lift_pos, control_point, place_hover_pos, transport_steps, gripper_state=1.0, add_tremor=True))
        
        # ====== 5. 放置阶段 ======
        trajectory.extend(self.interpolate_move(place_hover_pos, place_pos, lower_steps, gripper_state=1.0, add_tremor=True))
        trajectory.extend(self.create_gripper_action(place_pos, gripper_state=0.0, steps=place_wait_steps))
        
        # ====== 6. 撤离阶段 ======
        trajectory.extend(self.interpolate_move(place_pos, retract_pos, retract_steps, gripper_state=0.0))
        
        path_type = "hover->mid->grasp" if use_hover else "mid->grasp (direct)"
        hover_z_display = f"{hover_pos[2]:.2f}" if EXPERT_FUNNEL_HOVER_Z is not None else f"z_travel({z_travel:.2f})"
        
        print(f"🤖 Expert trajectory generated: {len(trajectory)} steps")
        print(f"   Target object: {self.obj_target} (red mug) -> Plate (终点: 盘子)")
        print(f"   🔥 Y-axis grasp offset: {y_grasp_offset*1000:.1f}mm (用于抓取杯子把手，已应用到抓取中心、悬停点和中间点)")
        print(f"   🔥 Grasp center (圆心): ({grasp_center_xy[0]:.3f}, {grasp_center_xy[1]:.3f}) [Y偏移: {y_grasp_offset*1000:.1f}mm]")
        print(f"   🔥 Smart Path Selection: {path_type} (dist_to_hover={dist_to_hover:.3f}m, dist_to_mid={dist_to_mid:.3f}m)")
        print(f"   🔥 Funnel Approach: hover (grasp center+offset, z={hover_z_display}, r={EXPERT_FUNNEL_HOVER_RADIUS*100:.1f}cm) -> mid (grasp center+offset, z={EXPERT_FUNNEL_MID_Z:.2f}, r={EXPERT_FUNNEL_MID_RADIUS*100:.1f}cm) -> grasp")
        print(f"   Grasp pos (杯子把手): ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}) [Y偏移: {y_grasp_offset*1000:.1f}mm]")
        print(f"   Place pos (盘子): ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}) [Y偏移: {y_place_offset*1000:.1f}mm]")
        print(f"   🔥 Dynamic Steps: approach={approach_steps}, descend1={descend_steps_1}, descend2={descend_steps_2}, grasp_wait={grasp_wait_steps}")
        print(f"                    lift={lift_steps}, transport={transport_steps}, lower={lower_steps}")
        print(f"                    place_wait={place_wait_steps}, retract={retract_steps}")
        
        return trajectory
    
    def auto_execute_task(self, record=False):
        """自动执行专家策略"""
        self.is_recording = record
        
        current_pos = self.p0.copy()
        
        if not hasattr(self, 'obj_target'):
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
        
        print(f"🤖 Expert policy initialized. Trajectory computed: {len(trajectory)} steps.")
        if record:
            print(f"   🎥 Recording Mode: Enabled")
        else:
            print(f"   🧪 Test Mode: No recording")
    
    def get_expert_action(self):
        """获取专家策略的下一个动作"""
        if self.expert_pending:
            self.expert_countdown -= 1
            if self.expert_countdown <= 0:
                self.expert_pending = False
                self.expert_executing = True
                print("🚀 Motion Start!")
            else:
                return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        if self.expert_waiting_save:
            self.expert_post_countdown -= 1
            if self.expert_post_countdown <= 0:
                self.expert_waiting_save = False
                self.is_recording = False
                if self.expert_auto_save:
                    print(f"\n⏰ Post-execution wait finished. Auto-saving...")
                else:
                    print(f"\n⏰ Post-execution wait finished. Recording PAUSED (not saved).")
                    print(f"   👉 Press [U] to SAVE, or [I] to DISCARD.")
                return None
            else:
                return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        if len(self.expert_trajectory) == 0:
            print("⚠️ Warning: Trajectory is empty. This should not happen.")
            return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        if self.expert_trajectory_idx >= len(self.expert_trajectory):
            self.expert_executing = False
            self.expert_waiting_save = True
            self.expert_post_countdown = EXPERT_POST_WAIT
            print(f"\n✅ Expert trajectory finished! Entering {EXPERT_POST_WAIT/20:.1f}s post-wait period...")
            return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        target_pos, gripper_state = self.expert_trajectory[self.expert_trajectory_idx]
        self.expert_trajectory_idx += 1
        
        delta_pos = target_pos - self.p0
        delta_rot = np.zeros(3)
        
        action = np.concatenate([delta_pos, delta_rot, [gripper_state]], dtype=np.float32)
        
        self.gripper_state = bool(gripper_state)
        
        return action
    
    def teleop_robot(self, mode='arm'):
        '''
        Teleoperate the robot using keyboard
        🔥 V2环境只有arm模式，mode参数保留以兼容接口
        '''
        dpos = np.zeros(3)
        drot = np.eye(3)
        reset = False
        
        if self.env.is_key_pressed_once(key=glfw.KEY_Z): reset = True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE): self.gripper_state = not self.gripper_state
        
        # 🔥 O 键：平滑归位机械臂
        if self.env.is_key_pressed_once(key=glfw.KEY_O):
            if not self.returning_home:
                print("🏠 [O] Smooth Return Origin Pose : Moving arm to initial position...")
                self.smooth_return_home()
            else:
                print("⚠️ Arm is already returning home.")
        
        # 🤖 T 键：测试模式（仅执行，不录制）
        if self.env.is_key_pressed_once(key=glfw.KEY_T):
            if self.moving_to_random:
                print("⚠️ Currently moving to random position. Please wait for movement to complete.")
            elif not self.expert_executing and not self.expert_pending and not self.expert_waiting_save:
                print("🤖 [T] Test Mode: Auto Execute Expert Policy (No Recording)")
                self.auto_execute_task(record=False)
            else:
                print("⚠️ Expert policy already running or waiting for save. Press Z to reset.")
        
        # 🎥 Y 键：录制模式（执行并开启录制，自动保存）
        if self.env.is_key_pressed_once(key=glfw.KEY_Y):
            if self.moving_to_random:
                print("⚠️ Currently moving to random position. Please wait for movement to complete.")
            elif not self.expert_executing and not self.expert_pending and not self.expert_waiting_save:
                print("🎥 [Y] Record Mode: Auto Execute Expert Policy + Start Recording + Auto Save")
                self.expert_auto_save = True
                self.auto_execute_task(record=True)
            else:
                print("⚠️ Expert policy already running or waiting for save. Press Z to reset.")
        
        # 🔥 如果正在移动到随机位置，优先执行移动
        if self.moving_to_random:
            self.smooth_move_to_random()
            return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32), reset
        
        # 🤖 如果专家策略处于pending、executing或waiting_save状态，优先返回专家动作
        if self.expert_pending or self.expert_executing or self.expert_waiting_save:
            action = self.get_expert_action()
            if action is not None:
                if self.expert_pending:
                    print(f"   ⏳ Buffer: {self.expert_countdown}/{EXPERT_START_DELAY} steps...", end='\r')
                elif self.expert_waiting_save:
                    print(f"   ⏰ Post-wait: {self.expert_post_countdown}/{EXPERT_POST_WAIT} steps (recording)...", end='\r')
                elif len(self.expert_trajectory) == 0:
                    print(f"   ⚠️ Warning: Trajectory is empty...", end='\r')
                else:
                    progress = self.expert_trajectory_idx / len(self.expert_trajectory) * 100
                    print(f"   🤖 Expert: {self.expert_trajectory_idx}/{len(self.expert_trajectory)} ({progress:.1f}%)", end='\r')
                return action, reset
        
        # 如果正在归位中，优先执行归位动作
        if self.returning_home:
            self.smooth_return_home()
            return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32), reset
        
        speed = 0.007; rot_speed = 0.03
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W): dpos[0] = -speed
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S): dpos[0] = speed
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A): dpos[1] = -speed
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D): dpos[1] = speed
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R): dpos[2] = speed
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F): dpos[2] = -speed
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q): drot = rotation_matrix(angle=rot_speed, direction=[0,0,1])[:3,:3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E): drot = rotation_matrix(angle=-rot_speed, direction=[0,0,1])[:3,:3]
        
        drot_rpy = r2rpy(drot)
        action = np.concatenate([dpos, drot_rpy, [float(self.gripper_state)]], dtype=np.float32)
        return action, reset
