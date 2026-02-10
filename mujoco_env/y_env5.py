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

# ====== V4 版本任务说明 ======
# 任务：小车和机械臂协同，机械臂把红色杯子放到盘子上
# V4 场景特点（与V2一致）：
#   - 桌面高度 0.83m
#   - 机械臂基座在原点 (0, 0)
#   - 使用杯子和盘子
#   - 小车在远处固定位置 (x=0, y=3)

# ====== 🤖 Expert Policy Parameters (专家策略参数) ======
# 这些参数用于自动化录制数据的专家策略，可根据需要调整

# --- 高度参数 ---
# 🔥 V4环境调整：桌面0.83m，盘子0.82m (与V2一致)
EXPERT_Z_TRAVEL_BASE = 1.0           # 巡航基础高度 (Transport cruise height) = 桌面0.83 + 0.17
EXPERT_Z_TRAVEL_NOISE = 0.03         # 巡航高度随机扰动范围 ± (Cruise height noise)
EXPERT_Z_GRASP_BASE = 0.84           # 抓取高度基础值 (Grasp height) = 桌面高度
EXPERT_Z_PLACE_BASE = 0.87           # 放置高度基础值 (Place height) = 盘子高度
EXPERT_RETRACT_HEIGHT = 1.0          # 撤离安全高度 (Retract safe height) = 巡航高度附近

# --- 位置偏移与噪声 ---
EXPERT_XY_NOISE_SCALE = 0.01        # 端点随机噪声范围 ±3mm (Endpoint noise for robustness)
EXPERT_Y_GRASP_OFFSET = 0.067       # 抓取Y轴固定偏移（用于抓取杯子把手）(Y offset for cup handle)
EXPERT_Y_PLACE_OFFSET = 0.03        # 放置时的 Y 轴固定偏移
EXPERT_HOVER_NOISE = 0.01            # 悬停点误差 (Hover point noise)
EXPERT_Z_NOISE = 0.005               # Z轴高度微小随机噪声 (Z height noise)

# --- 🔥 漏斗移动逻辑参数 (Funnel Approach Parameters) ---
# 🔥 悬停点和中间点的圆半径参数（用于漏斗式接近策略，方便调参）
EXPERT_FUNNEL_HOVER_RADIUS = 0.03   # 悬停点圆半径（米）(Hover point circle radius, 3cm)
EXPERT_FUNNEL_MID_RADIUS = 0.01     # 中间点圆半径（米）(Mid point circle radius, 1cm)
# 🔥 悬停点和中间点的Z坐标参数（圆心高度）
EXPERT_FUNNEL_HOVER_Z = None        # 悬停点Z坐标（米），None=使用巡航高度z_travel
EXPERT_FUNNEL_MID_Z = 0.92          # 中间点Z坐标（米）(Mid point Z coordinate) = 桌面0.83 + 0.09

# --- 🔥 人类抖动模拟参数 (Human Tremor Simulation Parameters) ---
EXPERT_TREMOR_ENABLED = True           # 是否启用抖动 (Enable tremor)
EXPERT_TREMOR_AMPLITUDE = 0.002       # 抖动幅度（米）(Tremor amplitude, 2mm)
EXPERT_TREMOR_SMOOTHNESS = 0.7        # 抖动平滑度 (0-1，越大越平滑) (Tremor smoothness)

# --- 🔥 动态步数参数（基于距离计算，提高数据多样性）---
# 末端执行器速度参数（米/步），用于根据距离动态计算步数
# 🔥 V4.1: 速度减半，使单条数据集时长变为原来的2倍
EXPERT_SPEED_APPROACH = 0.008        # 接近阶段速度 (Approach speed, m/step) - 原0.012减半
EXPERT_SPEED_DESCEND = 0.005         # 下降阶段速度 (Descend speed, m/step) - 原0.010减半
EXPERT_SPEED_LIFT = 0.006            # 提升阶段速度 (Lift speed, m/step) - 原0.008减半
EXPERT_SPEED_TRANSPORT = 0.008       # 运输阶段速度 (Transport speed, m/step) - 原0.008减半
EXPERT_SPEED_LOWER = 0.006           # 下降到放置点速度 (Lower speed, m/step) - 原0.008减半
EXPERT_SPEED_RETRACT = 0.006         # 撤离阶段速度 (Retract speed, m/step) - 原0.010减半

# 最小/最大步数限制（防止极端情况）
EXPERT_MIN_STEPS = 8                 # 任何阶段的最小步数 (Min steps for any phase)
EXPERT_MAX_STEPS = 200                # 任何阶段的最大步数 (Max steps for any phase)

# 等待阶段步数（带随机扰动范围）
EXPERT_OPEN_WAIT_BASE = 4            # 🔥 初始张开夹爪等待基础步数 (Initial open gripper wait base steps)
EXPERT_OPEN_WAIT_NOISE = 1           # 🔥 初始张开夹爪等待随机扰动 ± (Initial open gripper wait noise)
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
EXPERT_START_DELAY = 0               # 🔥 启动缓冲期步数（已禁用，立即开始录制）(Pre-recording buffer steps)
EXPERT_POST_WAIT = 60                # 🔥 执行完成后等待步数（3秒，20Hz下）用于人工确认 (Post-execution wait steps)

# --- 🔥 夹爪随机初始化参数 (Random Initialization Parameters) ---
RANDOM_INIT_ENABLED = 0              # 🔥 随机初始化模式（由 collect_data_v4.py 传入）: 0=关闭, 1=旧版(扇形区域), 2=新版(环形交集)
RANDOM_INIT_CIRCLE_INNER_RADIUS = 0.02  # 🔥 新版随机初始化：以红色杯子为中心的环形区域内半径（米）
RANDOM_INIT_CIRCLE_OUTER_RADIUS = 0.05  # 🔥 新版随机初始化：以红色杯子为中心的环形区域外半径（米）
RANDOM_INIT_ANGLE_MIN = 0            # 角度范围最小值（度）(Min angle in degrees, 与V2一致)
RANDOM_INIT_ANGLE_MAX = 45           # 角度范围最大值（度）(Max angle in degrees, 与V2一致)
RANDOM_INIT_RADIUS_MIN = 0.3         # 径向距离最小值（米）(Min radial distance in meters)
RANDOM_INIT_RADIUS_MAX = 0.4         # 径向距离最大值（米）(Max radial distance in meters)
RANDOM_INIT_Z_MIN = 0.9              # Z坐标最小值（米）(Min Z coordinate in meters) - 旧版使用 = 桌面0.83 + 0.07
RANDOM_INIT_Z_MAX = 1.0              # Z坐标最大值（米）(Max Z coordinate in meters) - 旧版使用 = 巡航高度附近
RANDOM_INIT_Z_MIN_V2 = 0.88          # 🔥 新版随机初始化Z坐标最小值（米） = 中间点高度
RANDOM_INIT_Z_MAX_V2 = 1.0           # 🔥 新版随机初始化Z坐标最大值（米） = 巡航高度附近
RANDOM_INIT_GRIPPER_OPEN = True      # 🔥 初始化时夹爪是否张开 (True=张开, False=闭合)
RANDOM_INIT_MOVE_STEPS = 75          # 🔥 平滑移动到随机位置的步数（约7.5秒，20Hz）(Steps for smooth move to random position)

# ====== 🎲 Object Initialization Parameters (物体初始化参数) ======
# 这些参数用于控制红色杯子的随机初始化（与V2一致）

# --- 机械臂基座位置 ---
ARM_BASE_X = 0.0                     # 机械臂基座X坐标 (Arm base X position, 与V2一致)
ARM_BASE_Y = 0.0                     # 机械臂基座Y坐标 (Arm base Y position, 与V2一致)

# --- 桌面参数 ---
TABLE_Z_HEIGHT = 0.83                # 桌面高度 (Table height, 与V2一致)

# --- 桌面范围限制（根据XML文件中的实际桌子尺寸）---
# 桌子body: pos="0 0 0", geom: pos="0.5 0 0.4", size="1.0 0.7 0.4" (半尺寸)
# 实际范围: X=[-0.5, 1.5], Y=[-0.7, 0.7]
# 留安全边距，避免物体掉出桌面
TABLE_X_MIN = -0.4                   # 桌面X轴最小值 (Table X min boundary, 留0.1m边距)
TABLE_X_MAX = 1.4                    # 桌面X轴最大值 (Table X max boundary, 留0.1m边距)
TABLE_Y_MIN = -0.6                   # 桌面Y轴最小值 (Table Y min boundary, 留0.1m边距)
TABLE_Y_MAX = 0.6                    # 桌面Y轴最大值 (Table Y max boundary, 留0.1m边距)

# --- 🔥 扇形区域初始化参数（红色杯子，与V2一致）---
MUG_MIN_DIST = 0.30                  # 离机械臂基座最近距离（米）(Min distance from arm base)
MUG_MAX_DIST = 0.40                  # 离机械臂基座最远距离（米）(Max distance from arm base)
MUG_MIN_ANGLE = 0.0                  # 左偏角度（度）(Min angle in degrees, 0° = 正前方)
MUG_MAX_ANGLE = 45.0                 # 右偏角度（度）(Max angle in degrees, 45° = 右偏45度)

# --- 隐藏物体参数 ---
HIDDEN_OBJ_X = 20.0                  # 隐藏物体的X坐标（远离场景）
HIDDEN_OBJ_Y_INTERVAL = 0.5          # 隐藏物体沿Y轴的间隔（米）
HIDDEN_OBJ_Z = 1.0                   # 隐藏物体的Z坐标（高度）

NAV_INSTRUCTIONS = [
    # 基础指令 (Basic)
    "Go to the workbench.",
    "Navigate to the workbench.",
    "Drive to the workbench.",
    "Move to the workbench."
]

# 🔥 V4与V2一致：只使用红色杯子放到盘子上
# 任务指令固定为 "Place the red mug on the plate."
ARM_INSTRUCTION = "Place the red mug on the plate."

class SimpleEnv4:
    def __init__(self, xml_path, action_type='eef_pose', state_type='joint_angle', seed=None, 
                 random_init_enabled=False, random_init_gripper_open=True):
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        
        # 🔥 随机初始化开关（由外部传入）
        self.random_init_enabled = random_init_enabled
        self.random_init_gripper_open = random_init_gripper_open
        
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
        
        # 🤖 专家策略状态变量
        self.expert_trajectory = []       # 专家轨迹列表 [(pos, gripper_state), ...]
        self.expert_trajectory_idx = 0    # 当前执行到的轨迹索引
        self.expert_executing = False     # 是否正在执行专家策略
        self.expert_pending = False       # 是否处于缓冲期（等待启动）
        self.expert_countdown = 0         # 缓冲期倒计时
        self.is_recording = False         # 录制标志位（用于外部检测是否需要保存图像）
        self.expert_waiting_save = False  # 🔥 是否处于等待保存状态（执行完毕后的等待期）
        self.expert_lifting_to_mid = False  # 🔥 是否正在升高到中间点高度
        self.expert_mid_z_target = None   # 🔥 目标中间点Z坐标（用于升高检查）
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
        self.env.reset()
        self.env.init_viewer(
            distance=2.0, elevation=-30, transparent=False, black_sky=True,
            use_rgb_overlay=False, loc_rgb_overlay='top right',
        )

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
        # 🔥 获取红色杯子的位置（与V2一致）
        try:
            red_mug_pos = self.env.get_p_body('body_obj_mug_5')
            mug_center_xy = red_mug_pos[:2]  # [x, y]
        except Exception as e:
            print(f"⚠️ Cannot get red mug position: {e}. Falling back to V1.")
            return None
        
        # 🔥 计算交集区域：环形（以红色杯子为中心）∩ 扇形区域（从基座出发）
        
        # ====== 🔥 改进1：交集预检测 ======
        # 计算红色杯子到基座的距离和方向
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
        
        # 方法：交替在环形和扇形区域内采样，检查是否同时满足两个条件（交集）
        max_attempts = 1000  # 🔥 改进3：增加采样次数（从500增加到1000）
        x_target = None
        y_target = None
        
        # 🔥 改进3：使用更智能的采样策略（如果预检测显示交集可能很小，增加采样密度）
        if not intersection_possible:
            print(f"   ⚠️ Warning: Intersection may be empty. Will try boundary search.")
        
        for attempt in range(max_attempts):
            # 策略：交替使用两种采样方法，提高找到交集的概率
            if attempt % 2 == 0:
                # 方法1：在环形区域内采样，然后检查是否在扇形区域内
                # 在环形区域内均匀采样（内半径到外半径之间）
                r_inner_sq = RANDOM_INIT_CIRCLE_INNER_RADIUS ** 2
                r_outer_sq = RANDOM_INIT_CIRCLE_OUTER_RADIUS ** 2
                r_sq = np.random.uniform(r_inner_sq, r_outer_sq)
                r_annulus = np.sqrt(r_sq)
                theta_circle = np.random.uniform(0, 2 * np.pi)
                candidate_xy = mug_center_xy + np.array([r_annulus * np.cos(theta_circle), r_annulus * np.sin(theta_circle)])
            else:
                # 方法2：在扇形区域内采样，然后检查是否在环形区域内
                angle_deg = random.uniform(RANDOM_INIT_ANGLE_MIN, RANDOM_INIT_ANGLE_MAX)
                angle_rad = np.deg2rad(angle_deg)
                radius = random.uniform(RANDOM_INIT_RADIUS_MIN, RANDOM_INIT_RADIUS_MAX)
                candidate_xy = np.array([ARM_BASE_X, ARM_BASE_Y]) + radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            
            # 🔥 交集判断：必须同时满足两个条件
            # 条件1：在环形区域内
            dist_to_mug = np.linalg.norm(candidate_xy - mug_center_xy)
            in_annulus = (RANDOM_INIT_CIRCLE_INNER_RADIUS <= dist_to_mug <= RANDOM_INIT_CIRCLE_OUTER_RADIUS)
            
            # 条件2：在扇形区域内
            rel_pos = candidate_xy - np.array([ARM_BASE_X, ARM_BASE_Y])
            rel_dist = np.linalg.norm(rel_pos)
            rel_angle_rad = np.arctan2(rel_pos[1], rel_pos[0])
            rel_angle_deg = np.rad2deg(rel_angle_rad)
            
            # 将角度归一化到 [-180, 180] 范围
            if rel_angle_deg > 180:
                rel_angle_deg -= 360
            elif rel_angle_deg < -180:
                rel_angle_deg += 360
            
            in_sector = (RANDOM_INIT_RADIUS_MIN <= rel_dist <= RANDOM_INIT_RADIUS_MAX and 
                        RANDOM_INIT_ANGLE_MIN <= rel_angle_deg <= RANDOM_INIT_ANGLE_MAX)
            
            # 🔥 交集：同时满足两个条件（环形区域 ∩ 扇形区域）
            if in_annulus and in_sector:
                x_target = candidate_xy[0]
                y_target = candidate_xy[1]
                break
        
        # ====== 🔥 改进2和4：如果采样失败，使用边界点搜索和智能回退 ======
        if x_target is None or y_target is None:
            print(f"   🔍 Random sampling failed ({max_attempts} attempts). Trying boundary search...")
            
            # 🔥 改进2：边界点搜索 - 系统性地搜索环形区域边界与扇形区域的交点
            boundary_search_success = False
            
            for theta in np.linspace(0, 2 * np.pi, 36):  # 每10度一个点
                for r_annulus in [RANDOM_INIT_CIRCLE_INNER_RADIUS, RANDOM_INIT_CIRCLE_OUTER_RADIUS]:
                    candidate_xy = mug_center_xy + r_annulus * np.array([np.cos(theta), np.sin(theta)])
                    
                    # 检查是否在扇形区域内
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
                # 回退策略1：从红色方块中心向扇形中心方向，在环形区域内找一个点
                center_in_sector = (RANDOM_INIT_RADIUS_MIN <= rel_dist_center <= RANDOM_INIT_RADIUS_MAX and 
                                   RANDOM_INIT_ANGLE_MIN <= rel_angle_deg_center <= RANDOM_INIT_ANGLE_MAX)
                
                if center_in_sector:
                    # 红色杯子中心在扇形内，从中心向外在环形区域内找一个点
                    for test_radius in [RANDOM_INIT_CIRCLE_INNER_RADIUS, 
                                       (RANDOM_INIT_CIRCLE_INNER_RADIUS + RANDOM_INIT_CIRCLE_OUTER_RADIUS) / 2,
                                       RANDOM_INIT_CIRCLE_OUTER_RADIUS]:
                        fallback_xy = mug_center_xy + test_radius * np.array([np.cos(rel_angle_rad_center), np.sin(rel_angle_rad_center)])
                        
                        # 验证这个点是否在扇形区域内
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
                        # 如果都不行，使用外半径（至少保证在环形区域内）
                        fallback_xy = mug_center_xy + RANDOM_INIT_CIRCLE_OUTER_RADIUS * np.array([np.cos(rel_angle_rad_center), np.sin(rel_angle_rad_center)])
                        x_target = fallback_xy[0]
                        y_target = fallback_xy[1]
                        print(f"   ⚠️ Using outer radius as fallback (may not be in sector)")
                else:
                    # 回退策略2：在扇形区域内找一个尽可能靠近红色杯子中心的点
                    direction_to_mug = rel_pos_center / rel_dist_center if rel_dist_center > 1e-6 else np.array([1.0, 0.0])
                    
                    # 在扇形区域内，沿着指向红色杯子的方向搜索
                    for test_radius in np.linspace(RANDOM_INIT_RADIUS_MIN, RANDOM_INIT_RADIUS_MAX, 10):
                        fallback_xy = np.array([ARM_BASE_X, ARM_BASE_Y]) + test_radius * direction_to_mug
                        
                        # 检查这个点是否在环形区域内
                        dist_to_mug = np.linalg.norm(fallback_xy - mug_center_xy)
                        if RANDOM_INIT_CIRCLE_INNER_RADIUS <= dist_to_mug <= RANDOM_INIT_CIRCLE_OUTER_RADIUS:
                            # 验证角度是否在扇形范围内
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
                        # 最后回退：如果交集确实为空，返回 None 让调用者使用 V1 逻辑
                        print(f"   ⚠️ Warning: Intersection area is empty. Will fallback to V1.")
                        return None
        
        # Z坐标使用新版范围
        z_target = random.uniform(RANDOM_INIT_Z_MIN_V2, RANDOM_INIT_Z_MAX_V2)
        
        # 打印信息
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

    def reset(self, seed=None, mode=None):
        if seed is not None: np.random.seed(seed)
        
        # 如果传入了 mode 参数，更新 control_mode（用于 set_instruction 判断）
        if mode is not None:
            self.control_mode = mode
        
        # 🔥 机械臂初始化：总是先初始化到标准位置（与V2一致）
        q_init = np.deg2rad([0,0,0,0,0,0])
        # 固定初始化：机械臂归位 (与V2一致，机械臂基座在原点 (0, 0))
        # 目标: 基座前方 0.3m -> [0.3, 0.0, 1.0] (桌面0.83 + 安全高度0.17)
        q_zero, _, _ = solve_ik(self.env, self.joint_names, 'tcp_link', q_init, np.array([0.3, 0.0, 1.0]), rpy2r(np.deg2rad([90, -0., 90])))
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)
        
        # 🔥 如果启用了随机初始化，生成随机目标位置并准备平滑移动
        if self.random_init_enabled == 1:
            # ====== 旧版随机初始化：扇形区域 ======
            x_target, y_target, z_target = self._sample_random_init_v1()
            
        elif self.random_init_enabled == 2:
            # ====== 新版随机初始化：环形交集（仅简单模式）======
            result = self._sample_random_init_v2()
            if result is None:
                # V2 失败，回退到 V1
                print("   → Falling back to V1 random initialization.")
                x_target, y_target, z_target = self._sample_random_init_v1()
            else:
                x_target, y_target, z_target = result
        
        # 如果启用了随机初始化，设置目标位置并准备平滑移动
        if self.random_init_enabled in [1, 2]:
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
            # 🔥 小车初始位置：固定在远处 (x=0, y=3)，不干扰机械臂工作
            x_init = 0.0
            y_init = 3.0
            z_init = 0.0
            yaw_init = np.deg2rad(-90)  # -90度，转换为弧度
            
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
        self.expert_trajectory = []
        self.expert_trajectory_idx = 0
        self.expert_executing = False
        self.expert_pending = False
        self.expert_lifting_to_mid = False
        self.expert_mid_z_target = None
        self.expert_lift_target_z = None
        self.expert_lift_start_pos = None
        self.expert_lift_interp_steps = 0
        self.expert_lift_tremor_prev = np.zeros(2)  # 🔥 重置抖动状态
        self.expert_countdown = 0
        self.is_recording = False
        self.expert_waiting_save = False
        self.expert_post_countdown = 0
        self.expert_auto_save = False
        self.expert_trajectory_start_pos = None  # 🔥 轨迹起始位置（用于位置对齐）
        
        # 🔥 注意：moving_to_random 和 random_target_q 在 reset 中根据开关设置，这里不重置

        # 物体初始化
        self._init_objects_demo()
        
        # 获取物体位置（与V2一致）
        mug_red_init_pose, mug_blue_init_pose, plate_init_pose = self.get_obj_pose()
        # 🔥 只保存红色杯子和盘子的位置（蓝色杯子已隐藏，与V2一致）
        self.obj_init_pose = np.concatenate([mug_red_init_pose, plate_init_pose], dtype=np.float32)

        for _ in range(100):
            self.step_env()
            
        self.set_instruction()  # 现在会根据 self.control_mode 自动设置正确的任务文本
        # 🔥 gripper_state 已在上面根据 random_init_gripper_open 设置，这里不再重置
        
        # 重置时刷新一次图像缓存
        self.grab_image()

    def _init_objects_demo(self):
        # ====== 与V2一致：使用红色杯子和盘子 ======
        
        # 设置盘子位置 (固定位置，与V2一致)
        plate_xyz = np.array([0.3, -0.25, 0.82])
        self.env.set_p_base_body(body_name='body_obj_plate_11', p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11', R=np.eye(3,3))
        
        # 🔥 红色杯子位置随机化（扇形区域，相对于机械臂基座，与V2一致）
        # 1. 极坐标生成
        r = random.uniform(MUG_MIN_DIST, MUG_MAX_DIST)  # 0.30 ~ 0.40m
        theta = random.uniform(np.deg2rad(MUG_MIN_ANGLE), np.deg2rad(MUG_MAX_ANGLE))  # 0° ~ 45°
        
        # 2. 转换为笛卡尔坐标（相对于机械臂基座 ARM_BASE_X, ARM_BASE_Y）
        mug_x = ARM_BASE_X + r * np.cos(theta)
        mug_y = ARM_BASE_Y + r * np.sin(theta)
        mug_z = TABLE_Z_HEIGHT
        
        # 3. 限制在桌面范围内（Clip，留点边缘余量）
        # 🔥 修复：使用更小的边缘余量（0.01m），避免过度限制随机分布
        mug_x = np.clip(mug_x, TABLE_X_MIN + 0.01, TABLE_X_MAX - 0.01)
        mug_y = np.clip(mug_y, TABLE_Y_MIN + 0.01, TABLE_Y_MAX - 0.01)
        
        self.env.set_p_base_body(body_name='body_obj_mug_5', p=np.array([mug_x, mug_y, mug_z]))
        self.env.set_R_base_body(body_name='body_obj_mug_5', R=np.eye(3,3))
        
        # 🔥 隐藏蓝色杯子和其他物体（移到远处，与V2一致）
        self.env.set_p_base_body(body_name='body_obj_mug_6', p=np.array([HIDDEN_OBJ_X, 0.0, HIDDEN_OBJ_Z]))
        self.env.set_R_base_body(body_name='body_obj_mug_6', R=np.eye(3,3))
        
        # 隐藏其他杯子
        hidden_mugs = ['body_obj_mug_7', 'body_obj_mug_8']
        for idx, mug_name in enumerate(hidden_mugs):
            hidden_y = HIDDEN_OBJ_Y_INTERVAL * (idx + 1)
            self.env.set_p_base_body(
                body_name=mug_name,
                p=np.array([HIDDEN_OBJ_X, hidden_y, HIDDEN_OBJ_Z])
            )
            self.env.set_R_base_body(body_name=mug_name, R=np.eye(3,3))
        
        # 隐藏红色方块（如果存在）
        try:
            self.env.set_p_base_body(body_name='body_obj_red_block', p=np.array([HIDDEN_OBJ_X, 2.0, HIDDEN_OBJ_Z]))
            self.env.set_R_base_body(body_name='body_obj_red_block', R=np.eye(3,3))
        except:
            pass  # 可能不存在红色方块
        
        # 记录桌上的物体信息（供 set_instruction 使用）
        self.mugs_on_table = ['body_obj_mug_5']
        self.mug_colors_on_table = {'body_obj_mug_5': 'red'}

    def set_instruction(self, given=None, task_type=None):
        """
        设置任务指令（与V2一致：红色杯子放到盘子上）
        
        Parameters:
            given: 手动指定的指令文本
            task_type: 任务类型，可选值:
                - 'nav': 导航任务 (小车移动)
                - 'arm': 机械臂任务 (红色杯子放到盘子上，与V2一致)
                - None: 自动根据 control_mode 决定
        """
        # 保存任务类型
        if task_type is not None:
            self.task_type = task_type
        elif not hasattr(self, 'task_type'):
            # 默认：arm 模式用 arm 任务，base 模式用 nav 任务
            self.task_type = 'arm' if self.control_mode == 'arm' else 'nav'
        
        if given is None:
            # 根据控制模式和任务类型设置不同的任务文本
            if self.control_mode == 'base':
                # Base 模式：导航任务
                self.task_type = 'nav'
                available_instructions = [inst for inst in NAV_INSTRUCTIONS 
                                         if inst != self.current_nav_instruction]
                if len(available_instructions) == 0:
                    available_instructions = NAV_INSTRUCTIONS
                
                self.instruction = random.choice(available_instructions)
                self.current_nav_instruction = self.instruction
                
            else:
                # Arm 模式：与V2一致，红色杯子放到盘子上
                self.task_type = 'arm'
                self.instruction = ARM_INSTRUCTION  # "Place the red mug on the plate."
                self.obj_target = 'body_obj_mug_5'
                self.target_color = 'red'
        else:
            self.instruction = given
            # 解析 obj_target 和 target_color (与V2一致，只支持红色杯子)
            if self.control_mode == 'arm' or self.task_type == 'arm':
                if 'red' in self.instruction.lower():
                    self.obj_target = 'body_obj_mug_5'
                    self.target_color = 'red'
                else:
                    # 默认使用红色杯子
                    self.obj_target = 'body_obj_mug_5'
                    self.target_color = 'red'
                    print(f"⚠️ Warning: Instruction does not contain 'red'. Using red mug as default.")
            elif self.control_mode == 'base':
                self.current_nav_instruction = given

    def step(self, action, mode='arm'):
        # 记录当前的控制模式，供 render 使用
        self.control_mode = mode
        
        if mode == 'arm':
            self.current_wheel_vel = np.zeros(2)
            if self.action_type == 'eef_pose':
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
            elif self.action_type == 'joint_angle':
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

    # ====== 🤖 Expert Policy Helper Functions (专家策略辅助函数) ======
    
    def interpolate_move(self, start_pos, end_pos, steps, gripper_state, add_tremor=False):
        """
        线性插值移动（可选添加抖动）
        
        Parameters:
            start_pos: 起始位置 (3,)
            end_pos: 终止位置 (3,)
            steps: 插值步数
            gripper_state: 夹爪状态 (0.0=打开, 1.0=关闭)
            add_tremor: 是否添加抖动
            
        Returns:
            waypoints: List of (pos, gripper_state) tuples
        """
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps  # t 从 1/steps 到 1.0
            pos = start_pos + t * (end_pos - start_pos)
            waypoints.append((pos.copy(), gripper_state))
        
        # 如果启用抖动，添加正交抖动
        if add_tremor and EXPERT_TREMOR_ENABLED:
            waypoints = self.add_orthogonal_tremor(
                waypoints, start_pos, end_pos, 
                EXPERT_TREMOR_AMPLITUDE, 
                EXPERT_TREMOR_SMOOTHNESS
            )
        
        return waypoints
    
    def bezier_move(self, p0, p1, p2, steps, gripper_state, add_tremor=False):
        """
        二阶贝塞尔曲线插值移动（可选添加抖动）
        B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
        
        Parameters:
            p0: 起点 (3,)
            p1: 控制点 (3,) - 决定曲线弧度
            p2: 终点 (3,)
            steps: 插值步数
            gripper_state: 夹爪状态
            add_tremor: 是否添加抖动
            
        Returns:
            waypoints: List of (pos, gripper_state) tuples
        """
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps  # t 从 1/steps 到 1.0
            # 二阶贝塞尔曲线公式
            pos = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            waypoints.append((pos.copy(), gripper_state))
        
        # 如果启用抖动，添加正交抖动（使用起点和终点构建正交平面）
        if add_tremor and EXPERT_TREMOR_ENABLED:
            waypoints = self.add_orthogonal_tremor(
                waypoints, p0, p2, 
                EXPERT_TREMOR_AMPLITUDE, 
                EXPERT_TREMOR_SMOOTHNESS
            )
        
        return waypoints
    
    def create_gripper_action(self, pos, gripper_state, steps):
        """
        创建原地开关夹爪并等待的航点序列
        
        Parameters:
            pos: 当前位置 (3,)
            gripper_state: 夹爪状态 (0.0=打开, 1.0=关闭)
            steps: 等待步数
            
        Returns:
            waypoints: List of (pos, gripper_state) tuples
        """
        return [(pos.copy(), gripper_state) for _ in range(steps)]
    
    def distance_based_steps(self, start, end, speed_per_step, min_steps=None, max_steps=None):
        """
        🔥 基于距离动态计算插值步数（提高数据多样性）
        
        Parameters:
            start: 起始位置 (3,)
            end: 终止位置 (3,)
            speed_per_step: 每步移动的距离（米/步）
            min_steps: 最小步数限制（默认使用全局 EXPERT_MIN_STEPS）
            max_steps: 最大步数限制（默认使用全局 EXPERT_MAX_STEPS）
            
        Returns:
            steps: 计算出的步数（整数）
        """
        if min_steps is None:
            min_steps = EXPERT_MIN_STEPS
        if max_steps is None:
            max_steps = EXPERT_MAX_STEPS
            
        # 计算欧氏距离
        dist = np.linalg.norm(end - start)
        
        # 添加速度随机扰动（±EXPERT_SPEED_NOISE_RATIO）
        noise = np.random.uniform(-EXPERT_SPEED_NOISE_RATIO, EXPERT_SPEED_NOISE_RATIO)
        actual_speed = speed_per_step * (1.0 + noise)
        
        # 计算步数并限制范围
        steps = int(dist / actual_speed)
        steps = np.clip(steps, min_steps, max_steps)
        
        return steps
    
    def rand_wait_steps(self, base_steps, noise_range):
        """
        🔥 为等待阶段生成带随机扰动的步数
        
        Parameters:
            base_steps: 基础步数
            noise_range: 随机扰动范围 ±
            
        Returns:
            steps: 带扰动的步数（至少为 3）
        """
        noise = np.random.randint(-noise_range, noise_range + 1)
        return max(3, base_steps + noise)
    
    def bezier_curve_length(self, p0, p1, p2, num_samples=20):
        """
        🔥 估算二阶贝塞尔曲线的弧长（用于动态计算运输步数）
        
        Parameters:
            p0: 起点 (3,)
            p1: 控制点 (3,)
            p2: 终点 (3,)
            num_samples: 采样点数量
            
        Returns:
            length: 估算的曲线长度（米）
        """
        length = 0.0
        prev_point = p0
        for i in range(1, num_samples + 1):
            t = i / num_samples
            # 二阶贝塞尔曲线公式
            point = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            length += np.linalg.norm(point - prev_point)
            prev_point = point
        return length
    
    def sample_point_in_circle(self, center_xy, radius):
        """
        🔥 在圆内均匀采样随机点（用于漏斗移动逻辑）
        
        Parameters:
            center_xy: 圆心坐标 (2,) [x, y]
            radius: 圆的半径（米）
            
        Returns:
            point_xy: 圆内的随机点坐标 (2,) [x, y]
        """
        # 使用极坐标方法：在半径范围内均匀采样
        # 为了确保在圆内均匀分布，使用 sqrt(r) 来采样半径
        r = np.sqrt(np.random.uniform(0, 1)) * radius
        theta = np.random.uniform(0, 2 * np.pi)
        
        point_xy = center_xy + np.array([r * np.cos(theta), r * np.sin(theta)])
        return point_xy
    
    def add_orthogonal_tremor(self, trajectory, start_pos, end_pos, amplitude, smoothness=0.7):
        """
        🔥 在轨迹上添加正交于移动方向的抖动（模拟人类手部抖动）
        
        Parameters:
            trajectory: 原始轨迹列表 [(pos, gripper_state), ...]
            start_pos: 起始位置 (3,)
            end_pos: 终止位置 (3,)
            amplitude: 抖动幅度（米）
            smoothness: 平滑度 (0-1)，越大抖动越平滑
            
        Returns:
            trajectory_with_tremor: 添加抖动后的轨迹
        """
        if len(trajectory) == 0:
            return trajectory
        
        # 计算移动方向向量
        move_dir = end_pos - start_pos
        move_len = np.linalg.norm(move_dir)
        
        # 如果移动距离太小，不添加抖动
        if move_len < 1e-6:
            return trajectory
        
        move_dir = move_dir / move_len  # 归一化
        
        # 构建正交平面：找到两个与移动方向正交的向量
        # 方法1：使用一个固定方向（如[0,0,1]）与移动方向叉乘
        if abs(move_dir[2]) < 0.9:  # 如果移动方向不接近垂直
            ortho1 = np.cross(move_dir, np.array([0, 0, 1]))
            ortho1 = ortho1 / np.linalg.norm(ortho1)
        else:  # 如果移动方向接近垂直，使用另一个方向
            ortho1 = np.cross(move_dir, np.array([1, 0, 0]))
            ortho1 = ortho1 / np.linalg.norm(ortho1)
        
        # 第二个正交向量
        ortho2 = np.cross(move_dir, ortho1)
        ortho2 = ortho2 / np.linalg.norm(ortho2)
        
        # 生成抖动轨迹
        trajectory_with_tremor = []
        prev_tremor = np.zeros(2)  # 用于平滑（在正交平面的2D坐标）
        
        for i, (pos, gripper_state) in enumerate(trajectory):
            # 生成随机抖动（在正交平面上）
            random_tremor = np.random.uniform(-1, 1, size=2)
            
            # 平滑处理（随机游走 + 平滑）
            tremor_2d = smoothness * prev_tremor + (1 - smoothness) * random_tremor
            prev_tremor = tremor_2d.copy()
            
            # 归一化并缩放到目标幅度
            tremor_magnitude = np.linalg.norm(tremor_2d)
            if tremor_magnitude > 1e-6:
                tremor_2d = tremor_2d / tremor_magnitude * amplitude * np.random.uniform(0.5, 1.5)
            else:
                tremor_2d = np.zeros(2)
            
            # 将2D抖动转换到3D空间
            tremor_3d = tremor_2d[0] * ortho1 + tremor_2d[1] * ortho2
            
            # 叠加到原始位置
            pos_with_tremor = pos + tremor_3d
            
            trajectory_with_tremor.append((pos_with_tremor.copy(), gripper_state))
        
        return trajectory_with_tremor
    
    def _compute_expert_trajectory(self, current_pos):
        """
        🔥 内部函数：计算专家策略轨迹（一次性计算完整轨迹，包括平滑上升）
        
        Parameters:
            current_pos: 当前夹爪位置 (3,)，可能是任意位置（包括高度不够的情况）
        
        Returns:
            trajectory: 完整轨迹列表 [(pos, gripper_state), ...]，包括平滑上升段（如果需要）
        """
        # 获取目标物体位置
        if not hasattr(self, 'obj_target'):
            print("⚠️ No target object set. Call set_instruction() first.")
            return []
        obj_pos = self.env.get_p_body(self.obj_target)
        
        # 获取盘子位置（与V2一致）
        try:
            plate_pos = self.env.get_p_body('body_obj_plate_11')
        except:
            print("⚠️ Cannot find plate body.")
            return []
        
        # ====== 🔥 0. 检查是否需要平滑上升，如果需要则计算上升终点 ======
        mid_z = EXPERT_FUNNEL_MID_Z
        mid_z_tolerance = 0.005  # 5mm容差
        lift_target_z = None
        lift_start_pos = None
        
        # 检查高度是否足够
        if current_pos[2] < mid_z - mid_z_tolerance:
            # 需要平滑上升：计算上升终点（保持XY不变，只提升Z）
            lift_target_z = mid_z + np.random.uniform(-mid_z_tolerance, mid_z_tolerance)
            lift_start_pos = current_pos.copy()
            # 上升终点位置（XY保持不变，Z提升到目标高度）
            lift_end_pos = np.array([current_pos[0], current_pos[1], lift_target_z])
            print(f"   ⬆️ Current Z ({current_pos[2]:.3f}m) below mid Z ({mid_z:.3f}m). Will add smooth lift to {lift_target_z:.3f}m in trajectory.")
        else:
            # 高度足够，不需要平滑上升
            lift_end_pos = current_pos.copy()
        
        # ====== 1. 准备阶段：计算带噪声的目标位置 ======
        
        # 随机生成巡航高度
        z_travel = EXPERT_Z_TRAVEL_BASE + np.random.uniform(-EXPERT_Z_TRAVEL_NOISE, EXPERT_Z_TRAVEL_NOISE)
        
        # 🔥 动态调整悬停点高度（根据初始化夹爪位置）
        # 计算默认悬停点高度
        default_hover_z = EXPERT_FUNNEL_HOVER_Z if EXPERT_FUNNEL_HOVER_Z is not None else z_travel
        
        # 🔥 确定实际起始高度（如果添加了平滑上升，使用上升后的高度；否则使用当前高度）
        actual_start_z = lift_end_pos[2] if lift_target_z is not None else current_pos[2]
        
        # 🔥 如果实际起始高度低于悬停点，则将悬停点调整为实际起始高度
        if actual_start_z < default_hover_z:
            adjusted_hover_z = actual_start_z
            if lift_target_z is not None:
                print(f"   🔧 Adjusted hover Z: {default_hover_z:.3f}m -> {adjusted_hover_z:.3f}m (after smooth lift from {current_pos[2]:.3f}m to {lift_end_pos[2]:.3f}m)")
            else:
                print(f"   🔧 Adjusted hover Z: {default_hover_z:.3f}m -> {adjusted_hover_z:.3f}m (current_pos[2]={current_pos[2]:.3f}m)")
        else:
            adjusted_hover_z = default_hover_z
        
        # 🔥 漏斗移动逻辑：两个圆的圆心都使用抓取中心点（X物体, Y物体+offset），即抓取点的目标位置（不含噪声）
        # 🔥 杯子抓取：使用Y轴偏移抓取杯子把手（与V2一致）
        y_grasp_offset = EXPERT_Y_GRASP_OFFSET  # 抓取杯子把手时的Y轴偏移
        
        grasp_center_xy = np.array([obj_pos[0], obj_pos[1] + y_grasp_offset])
        
        # 抓取点（在物体真实坐标上叠加噪声和偏移）
        grasp_pos = np.array([
            obj_pos[0] + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            obj_pos[1] + y_grasp_offset + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            EXPERT_Z_GRASP_BASE + np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE)
        ])
        
        # 🔥 漏斗移动逻辑：悬停点在物体上方，以抓取中心点 (grasp_center_xy) 为中心、半径可调的圆内随机
        hover_xy = self.sample_point_in_circle(grasp_center_xy, radius=EXPERT_FUNNEL_HOVER_RADIUS)
        # 🔥 使用调整后的悬停点高度（动态调整逻辑已在上方完成）
        hover_pos = np.array([
            hover_xy[0],
            hover_xy[1],
            adjusted_hover_z
        ])
        
        # 🔥 漏斗移动逻辑：中间点，XY坐标在抓取中心点 (grasp_center_xy) 为中心、半径可调的圆内随机
        mid_xy = self.sample_point_in_circle(grasp_center_xy, radius=EXPERT_FUNNEL_MID_RADIUS)
        mid_pos = np.array([
            mid_xy[0],
            mid_xy[1],
            EXPERT_FUNNEL_MID_Z  # 🔥 使用可调参数设置中间点Z坐标
        ])
        
        # 放置点（盘子上方，带噪声，与V2一致）
        # 🔥 杯子放置：使用Y轴偏移调整杯子在盘子上的位置
        y_place_offset = EXPERT_Y_PLACE_OFFSET  # 放置时的Y轴偏移
        
        place_pos = np.array([
            plate_pos[0] + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            plate_pos[1] + y_place_offset + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            EXPERT_Z_PLACE_BASE + np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE)
        ])
        
        # 放置悬停点（盘子正上方）
        place_hover_pos = np.array([
            place_pos[0],
            place_pos[1],
            z_travel
        ])
        
        # ====== 🔥 动态计算各阶段步数（基于距离）======
        # 提前计算关键位置（供步数计算使用）
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], z_travel])
        retract_pos = np.array([place_pos[0], place_pos[1], EXPERT_RETRACT_HEIGHT])
        
        # 🔥 移除路径选择逻辑：始终使用漏斗路径（悬停点 -> 中间点 -> 抓取点）
        # 计算各阶段步数
        approach_steps = self.distance_based_steps(lift_end_pos, hover_pos, EXPERT_SPEED_APPROACH)
        descend_steps_1 = self.distance_based_steps(hover_pos, mid_pos, EXPERT_SPEED_DESCEND)
        descend_steps_2 = self.distance_based_steps(mid_pos, grasp_pos, EXPERT_SPEED_DESCEND)
        
        grasp_wait_steps = self.rand_wait_steps(EXPERT_GRASP_WAIT_BASE, EXPERT_GRASP_WAIT_NOISE)
        lift_steps = self.distance_based_steps(grasp_pos, lift_pos, EXPERT_SPEED_LIFT)
        lower_steps = self.distance_based_steps(place_hover_pos, place_pos, EXPERT_SPEED_LOWER)
        place_wait_steps = self.rand_wait_steps(EXPERT_PLACE_WAIT_BASE, EXPERT_PLACE_WAIT_NOISE)
        retract_steps = self.distance_based_steps(place_pos, retract_pos, EXPERT_SPEED_RETRACT)
        # transport_steps 需要在计算控制点后计算（见下方）
        
        # ====== 生成完整轨迹 ======
        trajectory = []
        
        # 🔥 0. 预处理阶段：如果夹爪闭合，先等待1帧，再张开夹爪
        if self.gripper_state:  # 如果当前夹爪是闭合的
            # 先等待1帧（保持闭合状态）
            trajectory.extend(self.create_gripper_action(current_pos, gripper_state=1.0, steps=1))
            # 然后张开夹爪并等待（使用独立的张开等待参数）
            open_wait_steps = self.rand_wait_steps(EXPERT_OPEN_WAIT_BASE, EXPERT_OPEN_WAIT_NOISE)
            trajectory.extend(self.create_gripper_action(current_pos, gripper_state=0.0, steps=open_wait_steps))
            print(f"   🔓 Opening gripper first (was closed): 1 frame wait + {open_wait_steps} steps to open")
        
        # 🔥 0.5. 平滑上升阶段（如果需要）：从当前位置平滑上升到目标高度（保持XY不变）
        if lift_target_z is not None:
            # 计算平滑上升的步数（使用提升速度）
            lift_smooth_steps = self.distance_based_steps(lift_start_pos, lift_end_pos, EXPERT_SPEED_LIFT)
            # 生成平滑上升轨迹（添加抖动）
            trajectory.extend(self.interpolate_move(lift_start_pos, lift_end_pos, lift_smooth_steps, gripper_state=0.0, add_tremor=True))
            print(f"   ⬆️ Smooth lift: {lift_start_pos[2]:.3f}m -> {lift_target_z:.3f}m ({lift_smooth_steps} steps)")
        
        # ====== 2. 接近阶段 (Approach) ======
        # 🔥 漏斗路径：上升终点 -> 悬停点 -> 中间点 -> 抓取点
        # 注意：从 lift_end_pos（上升终点）开始，而不是 current_pos
        # 2.1 移动到物体上方悬停点（夹爪打开）
        # 🔥 漏斗移动逻辑：悬停点在物体上方，以抓取中心点 (grasp_center_xy) 为中心、半径可调的圆内随机
        # 🔥 添加抖动：模拟人类手部轻微抖动
        trajectory.extend(self.interpolate_move(lift_end_pos, hover_pos, approach_steps, gripper_state=0.0, add_tremor=True))
        
        # 🔥 漏斗移动逻辑：下降分为两段，实现XY坐标逐渐收敛
        # 2.2 第一段：从悬停点下降到中间点（Z={EXPERT_FUNNEL_MID_Z}m），XY坐标从半径{EXPERT_FUNNEL_HOVER_RADIUS*100:.1f}cm收敛到半径{EXPERT_FUNNEL_MID_RADIUS*100:.1f}cm
        # 🔥 添加抖动：精细操作时的轻微抖动
        trajectory.extend(self.interpolate_move(hover_pos, mid_pos, descend_steps_1, gripper_state=0.0, add_tremor=True))
        
        # 2.3 第二段：从中间点下降到抓取点（平滑下降，夹爪打开）
        # 🔥 添加抖动：精细操作时的轻微抖动
        trajectory.extend(self.interpolate_move(mid_pos, grasp_pos, descend_steps_2, gripper_state=0.0, add_tremor=True))
        
        # ====== 3. 抓取阶段 (Grasp) ======
        # 3.1 闭合夹爪并等待物理引擎结算
        trajectory.extend(self.create_gripper_action(grasp_pos, gripper_state=1.0, steps=grasp_wait_steps))
        
        # ====== 4. 运输阶段 (Transport) ======
        # 4.1 垂直提升至巡航高度
        # 🔥 添加抖动：提升时的轻微抖动
        trajectory.extend(self.interpolate_move(grasp_pos, lift_pos, lift_steps, gripper_state=1.0, add_tremor=True))
        
        # 4.2 贝塞尔曲线运输到托盘上方
        # 🔥 优化：基于机械臂基座避障的贝塞尔曲线生成
        # 核心逻辑：
        #   1. 在起点和终点之间画一条直线
        #   2. 以机械臂基座为圆心，EXPERT_BEZIER_AVOID_RADIUS 为半径画圆
        #   3. 如果直线不穿过圆 → 往远离圆心方向画一个小弧线
        #   4. 如果直线穿过圆 → 往远离圆心方向画一个更大的弧线，至少与圆相切
        
        # 机械臂基座位置（圆心）
        arm_center = np.array([ARM_BASE_X, ARM_BASE_Y])
        
        # 直线起点终点（XY平面）
        line_start = lift_pos[:2]
        line_end = place_hover_pos[:2]
        
        # 计算直线向量
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq < 1e-10:
            # 起点终点几乎重合，使用简单的控制点
            mid_point_xy = line_start
            perp_dir = np.array([1.0, 0.0])
            offset_dist = EXPERT_BEZIER_SMALL_CURVE
        else:
            # 计算直线上离圆心最近的点
            # 参数化直线：P(t) = line_start + t * line_vec, t ∈ [0, 1]
            # 最近点参数：t = dot(center - start, vec) / |vec|^2
            t_closest = np.dot(arm_center - line_start, line_vec) / line_len_sq
            t_closest = np.clip(t_closest, 0.0, 1.0)  # 限制在线段范围内
            closest_point = line_start + t_closest * line_vec
            
            # 圆心到直线的最近距离
            dist_to_line = np.linalg.norm(arm_center - closest_point)
            
            # 计算垂直于直线的方向向量
            line_dir = line_vec / np.sqrt(line_len_sq)
            perp_dir = np.array([-line_dir[1], line_dir[0]])  # 垂直于直线的方向
            
            # 确保 perp_dir 指向远离圆心的方向
            mid_point_xy = (line_start + line_end) / 2
            vec_to_center = arm_center - mid_point_xy
            if np.dot(perp_dir, vec_to_center) > 0:
                perp_dir = -perp_dir  # 反转方向，使其远离圆心
            
            # 根据直线和圆的关系决定弧线偏移量
            if dist_to_line > EXPERT_BEZIER_AVOID_RADIUS:
                # ✅ 直线不穿过圆：画一个小弧线
                offset_dist = EXPERT_BEZIER_SMALL_CURVE
            else:
                # ⚠️ 直线穿过圆或离圆心太近：需要绕开
                # 控制点需要偏移足够远，使曲线与圆相切
                # 对于二阶贝塞尔，曲线最大偏离是控制点偏移的一半
                # 所以控制点需要偏移 2 * (radius - dist + margin) 才能让曲线离圆 margin 远
                required_curve_offset = EXPERT_BEZIER_AVOID_RADIUS - dist_to_line + EXPERT_BEZIER_TANGENT_MARGIN
                offset_dist = 2.0 * required_curve_offset
        
        # 添加一些随机性
        offset_dist += np.random.uniform(0.0, EXPERT_BEZIER_XY_OFFSET)
        
        # 计算控制点
        control_point_xy = mid_point_xy + perp_dir * offset_dist
        control_point_z = z_travel + np.random.uniform(EXPERT_BEZIER_Z_OFFSET_MIN, EXPERT_BEZIER_Z_OFFSET_MAX)
        
        control_point = np.array([control_point_xy[0], control_point_xy[1], control_point_z])
        
        # 🔥 基于贝塞尔曲线弧长动态计算运输步数
        bezier_length = self.bezier_curve_length(lift_pos, control_point, place_hover_pos)
        # 添加速度随机扰动
        noise = np.random.uniform(-EXPERT_SPEED_NOISE_RATIO, EXPERT_SPEED_NOISE_RATIO)
        actual_transport_speed = EXPERT_SPEED_TRANSPORT * (1.0 + noise)
        transport_steps = int(bezier_length / actual_transport_speed)
        transport_steps = np.clip(transport_steps, EXPERT_MIN_STEPS, EXPERT_MAX_STEPS)
        
        # 🔥 添加抖动：运输时的轻微抖动
        trajectory.extend(self.bezier_move(lift_pos, control_point, place_hover_pos, transport_steps, gripper_state=1.0, add_tremor=True))
        
        # ====== 5. 放置阶段 (Place) ======
        # 5.1 垂直下降到放置点
        # 🔥 添加抖动：精细放置时的轻微抖动
        trajectory.extend(self.interpolate_move(place_hover_pos, place_pos, lower_steps, gripper_state=1.0, add_tremor=True))
        
        # 5.2 松开夹爪并等待物体落稳
        trajectory.extend(self.create_gripper_action(place_pos, gripper_state=0.0, steps=place_wait_steps))
        
        # ====== 6. 撤离阶段 (Retract) ======
        # 🔥 关键：严格锁定 X/Y 轴，仅提升 Z 轴，避免碰倒刚放好的物体
        trajectory.extend(self.interpolate_move(place_pos, retract_pos, retract_steps, gripper_state=0.0))
        
        # 计算悬停点Z坐标显示值
        hover_z_display = f"{hover_pos[2]:.2f}" if EXPERT_FUNNEL_HOVER_Z is not None else f"z_travel({z_travel:.2f})"
        
        print(f"🤖 Expert trajectory generated: {len(trajectory)} steps")
        print(f"   Target object: {self.obj_target} (red mug) -> Plate")
        print(f"   🔥 Y-axis grasp offset: {y_grasp_offset*1000:.1f}mm (用于抓取杯子把手)")
        print(f"   🔥 Funnel Approach: hover (grasp center, z={hover_z_display}, r={EXPERT_FUNNEL_HOVER_RADIUS*100:.1f}cm) -> mid (grasp center, z={EXPERT_FUNNEL_MID_Z:.2f}, r={EXPERT_FUNNEL_MID_RADIUS*100:.1f}cm) -> grasp")
        print(f"   Grasp center: ({grasp_center_xy[0]:.3f}, {grasp_center_xy[1]:.3f}) [Y偏移: {y_grasp_offset*1000:.1f}mm]")
        print(f"   Hover pos: ({hover_pos[0]:.3f}, {hover_pos[1]:.3f}, {hover_pos[2]:.3f})")
        print(f"   Mid pos: ({mid_pos[0]:.3f}, {mid_pos[1]:.3f}, {mid_pos[2]:.3f})")
        print(f"   Grasp pos: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        print(f"   Place pos: ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f})")
        print(f"   🔥 Dynamic Steps: approach={approach_steps}, descend1={descend_steps_1}, descend2={descend_steps_2}, grasp_wait={grasp_wait_steps}")
        print(f"                    lift={lift_steps}, transport={transport_steps}, lower={lower_steps}")
        print(f"                    place_wait={place_wait_steps}, retract={retract_steps}")
        print(f"   📏 Bezier arc length: {bezier_length:.3f}m")
        
        return trajectory
    
    def auto_execute_task(self, record=False):
        """
        🤖 自动执行专家策略，一次性计算完整轨迹（包括平滑上升，如果需要）
        
        Parameters:
            record (bool): 是否开启录制模式。True=录制模式，False=测试模式
        
        Note: 🔥 轨迹一次性计算完成，包括平滑上升段（如果需要）
        """
        # 🔥 注意：队列和缓冲区的清空应该在调用此函数之前完成（由collect_data_v4.py负责）
        # 🔥 立即设置录制标志，确保所有轨迹（包括预处理阶段的张开夹爪和平滑上升）都能被录制
        self.is_recording = record
        
        # 获取当前末端执行器位置（使用 self.p0 确保与 delta 计算一致）
        current_pos = self.p0.copy()
        
        # 检查目标物体是否存在
        if not hasattr(self, 'obj_target'):
            print("⚠️ No target object set. Call set_instruction() first.")
            return
        
        # 🔥 一次性计算完整轨迹（包括平滑上升段，如果需要）
        trajectory = self._compute_expert_trajectory(current_pos)
        
        if len(trajectory) == 0:
            print("⚠️ Failed to compute trajectory. Stopping expert policy.")
            return
        
        # 保存轨迹到实例变量
        self.expert_trajectory = trajectory
        self.expert_trajectory_idx = 0
        
        # 🔥 设置Pending状态（缓冲期已禁用）
        # 注意：is_recording 已在函数开始时设置
        self.expert_pending = False  # 🔥 直接开始执行，不等待缓冲期
        self.expert_countdown = 0
        self.expert_executing = True  # 🔥 立即开始执行轨迹
        
        print(f"🤖 Expert policy initialized. Trajectory computed: {len(trajectory)} steps (including smooth lift if needed).")
        if record:
            print(f"   🎥 Recording Mode: Enabled")
        else:
            print(f"   🧪 Test Mode: No recording")
    
    def get_expert_action(self):
        """
        获取专家策略的下一个动作
        
        Returns:
            action: (7,) array [dx, dy, dz, drx, dry, drz, gripper] or None if trajectory finished
        """
        # 🔥 开头处理 Pending（缓冲期已禁用，直接跳过）
        if self.expert_pending:
            self.expert_countdown -= 1
            if self.expert_countdown <= 0:
                # 缓冲期结束，开始执行
                self.expert_pending = False
                self.expert_executing = True
                print("🚀 Motion Start!")
                # 继续执行下面的逻辑，返回第一个动作
            else:
                # 缓冲期中，返回全零动作（保持静止）
                return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        # 🔥 处理执行完毕后的等待期
        if self.expert_waiting_save:
            self.expert_post_countdown -= 1
            if self.expert_post_countdown <= 0:
                # 等待期结束
                self.expert_waiting_save = False
                self.is_recording = False
                # 🔥 如果设置了自动保存标志，则触发自动保存（由collect_data_v4.py检测并执行）
                if self.expert_auto_save:
                    print(f"\n⏰ Post-execution wait finished. Auto-saving...")
                    # 保持expert_auto_save=True，让collect_data_v4.py检测并保存
                else:
                    print(f"\n⏰ Post-execution wait finished. Recording PAUSED (not saved).")
                    print(f"   👉 Press [U] to SAVE, or [I] to DISCARD.")
                return None
            else:
                # 等待期中，继续录制但返回静止动作
                return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        # 🔥 如果轨迹还未计算，返回静止动作（这种情况不应该发生，因为 auto_execute_task 已经计算了轨迹）
        if len(self.expert_trajectory) == 0:
            print("⚠️ Warning: Trajectory is empty. This should not happen.")
            return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        # 🔥 结尾处理：轨迹执行完毕，进入等待期
        if self.expert_trajectory_idx >= len(self.expert_trajectory):
            self.expert_executing = False
            # 🔥 不立即停止录制，而是进入等待期继续录制
            self.expert_waiting_save = True
            self.expert_post_countdown = EXPERT_POST_WAIT
            print(f"\n✅ Expert trajectory finished! Entering {EXPERT_POST_WAIT/20:.1f}s post-wait period...")
            # 返回静止动作，继续录制
            return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        # 🔥 如果轨迹还未计算，返回静止动作（等待平滑上升完成）
        if len(self.expert_trajectory) == 0:
            return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32)
        
        # 执行轨迹中的下一个动作
        target_pos, gripper_state = self.expert_trajectory[self.expert_trajectory_idx]
        self.expert_trajectory_idx += 1
        
        # 计算 delta（相对于当前 self.p0）
        delta_pos = target_pos - self.p0
        delta_rot = np.zeros(3)  # 保持姿态不变
        
        # 组合成 action: [dx, dy, dz, drx, dry, drz, gripper]
        action = np.concatenate([delta_pos, delta_rot, [gripper_state]], dtype=np.float32)
        
        # 更新夹爪状态（用于渲染显示）
        self.gripper_state = bool(gripper_state)
        
        return action

    def teleop_robot(self, mode='arm'):
        dpos = np.zeros(3)
        drot = np.eye(3)
        wheel_v = np.zeros(2)
        reset = False
        
        if self.env.is_key_pressed_once(key=glfw.KEY_Z): reset = True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE): self.gripper_state = not self.gripper_state
        # 🔥 O 键：平滑归位机械臂（仅 Arm 模式）
        if self.env.is_key_pressed_once(key=glfw.KEY_O) and mode == 'arm':
            if not self.returning_home:
                print("🏠 [O] Smooth Return Origin Pose : Moving arm to initial position...")
                self.smooth_return_home()
            else:
                print("⚠️ Arm is already returning home.")
        
        # 🤖 T 键：测试模式（仅执行，不录制）
        if self.env.is_key_pressed_once(key=glfw.KEY_T) and mode == 'arm':
            if self.moving_to_random:
                print("⚠️ Currently moving to random position. Please wait for movement to complete.")
            elif not self.expert_executing and not self.expert_pending and not self.expert_waiting_save:
                print("🤖 [T] Test Mode: Auto Execute Expert Policy (No Recording)")
                print("   → Recording flag disabled, test mode only...")
                self.auto_execute_task(record=False)
            else:
                print("⚠️ Expert policy already running or waiting for save. Press Z to reset.")
        
        # 🎥 Y 键：录制模式（执行并开启录制，自动保存）
        if self.env.is_key_pressed_once(key=glfw.KEY_Y) and mode == 'arm':
            if self.moving_to_random:
                print("⚠️ Currently moving to random position. Please wait for movement to complete.")
            elif not self.expert_executing and not self.expert_pending and not self.expert_waiting_save:
                print("🎥 [Y] Record Mode: Auto Execute Expert Policy + Start Recording + Auto Save")
                print("   → Recording flag enabled, auto-stop after 3s post-wait...")
                print("   → After completion: Auto-saving (no manual save needed)")
                self.expert_auto_save = True  # 🔥 设置自动保存标志
                self.auto_execute_task(record=True)
            else:
                print("⚠️ Expert policy already running or waiting for save. Press Z to reset.")

        if mode == 'arm':
            # 🔥 如果正在移动到随机位置，优先执行移动
            if self.moving_to_random:
                self.smooth_move_to_random()
                # 返回一个零动作，让移动逻辑控制机械臂
                return np.concatenate([np.zeros(6), [float(self.gripper_state)]], dtype=np.float32), reset
            
            # 🤖 如果专家策略处于pending、executing或waiting_save状态，优先返回专家动作
            if self.expert_pending or self.expert_executing or self.expert_waiting_save:
                action = self.get_expert_action()
                if action is not None:
                    # 显示进度（不同阶段显示不同信息）
                    if self.expert_pending:
                        print(f"   ⏳ Buffer: {self.expert_countdown}/{EXPERT_START_DELAY} steps...", end='\r')
                    elif self.expert_waiting_save:
                        # 🔥 显示等待期倒计时
                        print(f"   ⏰ Post-wait: {self.expert_post_countdown}/{EXPERT_POST_WAIT} steps (recording)...", end='\r')
                    elif len(self.expert_trajectory) == 0:
                        # 🔥 轨迹还未计算（这种情况不应该发生，因为 auto_execute_task 已经计算了轨迹）
                        print(f"   ⚠️ Warning: Trajectory is empty...", end='\r')
                    else:
                        # 轨迹已计算，显示执行进度
                        progress = self.expert_trajectory_idx / len(self.expert_trajectory) * 100
                        print(f"   🤖 Expert: {self.expert_trajectory_idx}/{len(self.expert_trajectory)} ({progress:.1f}%)", end='\r')
                    return action, reset
                # 如果轨迹执行完毕，继续正常的 teleop 逻辑
            
            # 如果正在归位中，优先执行归位动作
            if self.returning_home:
                self.smooth_return_home()
                # 返回一个零动作，让归位逻辑控制机械臂
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

        elif mode == 'base':
            v_move = 15.0; v_turn = 6.0
            if self.env.is_key_pressed_repeat(key=glfw.KEY_W): wheel_v = [v_move, v_move]
            if self.env.is_key_pressed_repeat(key=glfw.KEY_S): wheel_v = [-v_move, -v_move]
            if self.env.is_key_pressed_repeat(key=glfw.KEY_A): wheel_v = [-v_turn, v_turn]
            if self.env.is_key_pressed_repeat(key=glfw.KEY_D): wheel_v = [v_turn, -v_turn]
            
            return np.array(wheel_v, dtype=np.float32), reset

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
        只返回本体的真实速度 (Proprioception)。
        千万不要返回 x, y 坐标，否则模型会过拟合环境位置！
        
        返回 shape: (2,) -> [v_left_real, v_right_real]
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

        # 2. 归一化 (可选，但推荐)
        # 如果你的速度范围大概是 -20 到 +20，除以 20 可以让数值落在 -1 到 1 之间
        # 这里的 20 是基于 XML 里 actuator ctrlrange="-30 30" 估算的，你可以根据实际手感调整
        # scale_factor = 30.0 
        # return np.array([v_left_real / scale_factor, v_right_real / scale_factor], dtype=np.float32)
        
        # 暂时先返回原始值，LeRobot 内部通常会有 Normalize 层
        return np.array([v_left_real, v_right_real], dtype=np.float32)

    def check_success(self):
        """
        V4 版本成功判定（与V2一致）：红色杯子放到盘子上
        """
        if not hasattr(self, 'obj_target'):
            return False
            
        p_mug = self.env.get_p_body(self.obj_target)
        
        # 判断红色杯子是否在盘子上（与V2一致）
        try:
            p_plate = self.env.get_p_body('body_obj_plate_11')
            # 盘子范围：xy 距离 < 0.1m，z 高度在盘子面附近 (±0.6m)
            xy_dist = np.linalg.norm(p_mug[:2] - p_plate[:2])
            z_diff = abs(p_mug[2] - p_plate[2])
            if xy_dist < 0.1 and z_diff < 0.6:
                # 夹爪已松开
                gripper = self.env.get_qpos_joint('rh_r1')
                if gripper < 0.1:
                    # 机械臂已撤离（z > 0.9m，与V2一致）
                    tcp_z = self.env.get_p_body('tcp_link')[2]
                    if tcp_z > 0.9:
                        return True
        except Exception as e:
            print(f"Warning: check_success error: {e}")
        return False

    def get_obj_pose(self):
        """
        获取物体位置（与V2一致：返回红色杯子和盘子的位置）
        Returns:
            p_mug_red: np.array, position of the red mug
            p_mug_blue: np.array, position of the blue mug (零向量，已隐藏)
            p_plate: np.array, position of the plate
        """
        p_mug_red = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        # 🔥 为了保持兼容性，返回三个位置（蓝色杯子已隐藏，返回零向量）
        p_mug_blue = np.zeros(3)
        return p_mug_red, p_mug_blue, p_plate
    
    def grab_image(self):
        # 初始化返回字典
        images = {}

        if self.control_mode == 'arm':
            # === 机械臂模式：只获取 ARM 相关的 2 个相机 ===
            self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='agentview')
            self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric') 
            
            # 存入字典（用于保存数据）
            images['agent'] = self.rgb_agent
            images['wrist'] = self.rgb_ego
            
            # 设置渲染标题
            self.agent_title = 'Agent View'
            self.ego_title = 'Wrist View'
            
        else:
            # === Base (小车) 模式：只获取 BASE 相关的 3 个相机 ===
            try:
                self.rgb_front = self.env.get_fixed_cam_rgb(cam_name='tb3_view')
                self.rgb_left = self.env.get_fixed_cam_rgb(cam_name='tb3_left')
                self.rgb_right = self.env.get_fixed_cam_rgb(cam_name='tb3_right')
                
                # 存入字典（用于保存数据）
                images['front'] = self.rgb_front
                images['left'] = self.rgb_left
                images['right'] = self.rgb_right
                
            except Exception as e:
                print(f"Error grabbing TB3 cameras: {e}")
                # Fallback: 使用 agentview 作为备用
                rgb_fallback = self.env.get_fixed_cam_rgb(cam_name='agentview')
                self.rgb_front = rgb_fallback
                self.rgb_left = rgb_fallback
                self.rgb_right = rgb_fallback
                images = {'front': rgb_fallback, 'left': rgb_fallback, 'right': rgb_fallback}

        return images
        
    def render(self, teleop=False, idx=0):
        self.env.plot_time()
        
        # 绘制绿色辅助线 (目标末端点)
        p_current, R_current = self.env.get_pR_body(body_name='tcp_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        
        # 确保图像已获取
        if self.control_mode == 'arm':
            if not hasattr(self, 'rgb_ego'):
                self.grab_image()
        else:
            if not hasattr(self, 'rgb_front'): self.grab_image()

        # =================== 🔥 画面显示逻辑 (核心修改) 🔥 ===================
        
        if self.control_mode == 'arm':
            # === Arm 模式：保持原样 ===
            # 1. 右上角: 全局视角
            rgb_agent_view = add_title_to_img(self.rgb_agent, text=self.agent_title, shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_agent_view, loc='top right')
            
            # 2. 右下角: 手腕视角
            rgb_egocentric_view = add_title_to_img(self.rgb_ego, text=self.ego_title, shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_egocentric_view, loc='bottom right')

        else:
            # === Base 模式：三摄分离显示 ===
            
            # 1. 右上角 (Top Right): 正前方 (主视角)
            img_front = add_title_to_img(self.rgb_front, text="Front View", shape=(640, 480))
            self.env.viewer_rgb_overlay(img_front, loc='top right')

            # 2. 左上角 (Top Left): 左侧视角
            img_left = add_title_to_img(self.rgb_left, text="Left View", shape=(640, 480))
            self.env.viewer_rgb_overlay(img_left, loc='top left')

            # 3. 右下角 (Bottom Right): 右侧视角
            img_right = add_title_to_img(self.rgb_right, text="Right View", shape=(640, 480))
            self.env.viewer_rgb_overlay(img_right, loc='bottom right')

            # 注意：左下角 (bottom left) 留空，或者被下面的 Task 文字覆盖
            
        # ===================================================================

        # GUI 相机自动跟随逻辑 (保持不变)
        if self.env.viewer is not None:
            if self.control_mode == 'base':
                self.env.viewer.cam.type = 2 
                try:
                    cam_id = self.env.model.camera('tb3_chase').id
                    self.env.viewer.cam.fixedcamid = cam_id
                except Exception as e:
                    self.env.viewer.cam.type = 1
                    try: self.env.viewer.cam.trackbodyid = self.env.model.body('tb3_base').id
                    except: pass
            elif self.control_mode == 'arm':
                self.env.viewer.cam.type = 0
                self.env.viewer.cam.trackbodyid = -1
        
        # 显示任务文字 (覆盖在左下角，Base模式和Arm模式都显示)
        if getattr(self, 'instruction', None):
            self.env.viewer_text_overlay(text1='Task', text2=self.instruction)
        
        # V4 新增：显示小车实时坐标 (左下角)
        try:
            tb3_pos = self.env.get_p_body('tb3_base')
            tb3_rot = self.env.get_R_body('tb3_base')
            theta = np.arctan2(tb3_rot[1, 0], tb3_rot[0, 0])  # Yaw 角
            coord_text = f"X:{tb3_pos[0]:+.3f}  Y:{tb3_pos[1]:+.3f}  Z:{tb3_pos[2]:.3f}  Yaw:{np.rad2deg(theta):+.1f}"
            self.env.viewer_text_overlay(text1='TB3 Pose', text2=coord_text)
        except Exception as e:
            pass  # 静默处理，避免影响渲染
        
        # V4 新增：显示机械臂夹爪（TCP）实时坐标 (左下角)
        try:
            tcp_pos = self.env.get_p_body('tcp_link')
            tcp_coord_text = f"TCP X:{tcp_pos[0]:+.3f}  Y:{tcp_pos[1]:+.3f}  Z:{tcp_pos[2]:+.3f}"
            self.env.viewer_text_overlay(text1='TCP Pose', text2=tcp_coord_text)
        except Exception as e:
            pass  # 静默处理，避免影响渲染
        
        # V4 新增：显示桌上物体的XYZ坐标 (左下角)
        try:
            # 只显示桌上存在的物体（与V2一致，只有红色杯子）
            if hasattr(self, 'mug_colors_on_table') and len(self.mug_colors_on_table) > 0:
                mug_texts = []
                color_names = {'red': 'Red', 'blue': 'Blue', 'yellow': 'Yellow', 'green': 'Green'}
                for mug_name, color in self.mug_colors_on_table.items():
                    p_mug = self.env.get_p_body(mug_name)
                    display_name = color_names.get(color, color.capitalize())
                    mug_texts.append(f"{display_name}:({p_mug[0]:+.3f},{p_mug[1]:+.3f},{p_mug[2]:.3f})")
                
                # 每两个换一行
                lines = []
                for i in range(0, len(mug_texts), 2):
                    line = "  ".join(mug_texts[i:i+2])
                    lines.append(line)
                mugs_text = "\n".join(lines)
                
                # 显示桌上物体数量
                num_mugs = len(self.mug_colors_on_table)
                self.env.viewer_text_overlay(text1=f'Mugs on Table ({num_mugs})', text2=mugs_text)
        except Exception as e:
            pass  # 静默处理，避免影响渲染
                
        self.env.render()