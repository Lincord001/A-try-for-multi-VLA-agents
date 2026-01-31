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
# 任务：小车和机械臂协同，机械臂把杯子放到小车托盘上
# V4 场景特点：
#   - 桌面高度降低到 0.3m（和托盘差不多高）
#   - 桌子x范围：[-0.5, 0.5]
#   - 机械臂基座相应降低
#   - 小车带有托盘（高度约 0.28m）
#   - 杯子位置更靠近桌边

# ====== 🤖 Expert Policy Parameters (专家策略参数) ======
# 这些参数用于自动化录制数据的专家策略，可根据需要调整

# --- 高度参数 ---
EXPERT_Z_TRAVEL_BASE = 0.52          # 巡航基础高度 (Transport cruise height)
EXPERT_Z_TRAVEL_NOISE = 0.03         # 巡航高度随机扰动范围 ± (Cruise height noise)
EXPERT_Z_GRASP_BASE = 0.35           # 抓取高度基础值 (Grasp height)
EXPERT_Z_PLACE_BASE = 0.355          # 放置高度基础值 (Place height)
EXPERT_RETRACT_HEIGHT = 0.55         # 撤离安全高度 (Retract safe height)

# --- 位置偏移与噪声 ---
EXPERT_XY_NOISE_SCALE = 0.003        # 端点随机噪声范围 ±3mm (Endpoint noise for robustness)
EXPERT_Y_GRASP_OFFSET = 0.067        # 杯子抓取Y轴固定偏移 (Y offset for cup handle)
# 🔥 [新增] 放置时的 Y 轴固定偏移 (正数向左/上移，负数向右/下移，取决于你的视角)
EXPERT_Y_PLACE_OFFSET = 0.03         # 建议先试 0.02 (2cm)，觉得不够就改成 0.04
EXPERT_HOVER_NOISE = 0.01            # 悬停点误差 (Hover point noise)
EXPERT_Z_NOISE = 0.005               # Z轴高度微小随机噪声 (Z height noise)

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
EXPERT_START_DELAY = 15              # 启动缓冲期步数（约0.75秒，20Hz下）(Pre-recording buffer steps)
EXPERT_POST_WAIT = 60                # 🔥 执行完成后等待步数（3秒，20Hz下）用于人工确认 (Post-execution wait steps)

# ====== 🎲 Object Initialization Parameters (物体初始化参数) ======
# 这些参数用于控制杯子的随机初始化，可根据需要调整

# --- 机械臂基座位置 ---
ARM_BASE_X = -0.4                    # 机械臂基座X坐标 (Arm base X position)
ARM_BASE_Y = 0.0                     # 机械臂基座Y坐标 (Arm base Y position)

# --- 桌面参数 ---
TABLE_Z_HEIGHT = 0.33                # 桌面高度 0.3 + 杯子偏移 0.03 (Table height + cup offset)

# --- 抓取范围参数（相对于机械臂基座）---
# 🔥 V4.1: 杯子位置固定，每个杯子只在固定位置附近小范围随机
CUP_POSITION_NOISE = 0.02            # 杯子位置随机范围 ±2cm (Cup position noise)

# 🔥 V4.1: 四个杯子的固定位置（红蓝黄绿，从左到右）
# 基座在 (-0.4, 0)，杯子距离基座约 0.35m
# 🔥 V4.1.2: 拉开杯子间距（均匀分布，每个间隔 0.20m）
CUP_FIXED_POSITIONS = {
    'body_obj_mug_5': (-0.12, -0.30, 0.345),   # 红色 - 最左
    'body_obj_mug_6': (-0.04, -0.15, 0.345),   # 蓝色 - 左中
    'body_obj_mug_7': (-0.10,  0.10, 0.345),   # 黄色 - 右中
    'body_obj_mug_8': (-0.24,  0.20, 0.345),   # 绿色 - 最右
}

# --- 杯子数量概率分布 ---
# 🔥 V4.1: 固定为4个杯子
CUP_COUNT_WEIGHTS = [0.0, 0.0, 0.0, 1.0]  # 永远选择4个杯子

# --- 杯子间距参数 ---
CUP_MIN_DISTANCE = 0.10              # 杯子之间的最小距离（米）- 缩小以适应固定位置
CUP_PLACEMENT_MAX_ATTEMPTS = 100     # 每个杯子的最大尝试次数，避免无限循环 (Max placement attempts)

# --- 桌面范围限制 ---
TABLE_X_MIN = -0.5                   # 桌面X轴最小值 (Table X min boundary)
TABLE_X_MAX = 0.5                    # 桌面X轴最大值 (Table X max boundary)
TABLE_Y_MIN = -0.5                   # 桌面Y轴最小值 (Table Y min boundary)
TABLE_Y_MAX = 0.5                    # 桌面Y轴最大值 (Table Y max boundary)

# --- 隐藏杯子参数 ---
HIDDEN_CUP_X = 20.0                  # 隐藏杯子的X坐标（远离场景）(Hidden cup X position)
HIDDEN_CUP_Y_INTERVAL = 0.5          # 隐藏杯子沿Y轴的间隔（米）(Hidden cup Y spacing)
HIDDEN_CUP_Z = 1.0                   # 隐藏杯子的Z坐标（高度）(Hidden cup Z position)

NAV_INSTRUCTIONS = [
    # 基础指令 (Basic)
    "Go to the workbench.",
    "Navigate to the workbench.",
    "Drive to the workbench.",
    "Move to the workbench."
]

# 🔥 V4.1: 简化指令集 - 只用一种模板，颜色自动填充
# 这样只有格式上的统一，但仍有 4 种最终指令（红/蓝/黄/绿）
COLLAB_INSTRUCTIONS = [
    {"text": "Place the {color} mug on the tray.", "color": None},
]

class SimpleEnv4:
    def __init__(self, xml_path, action_type='eef_pose', state_type='joint_angle', seed=None):
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        
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
        
        # 🤖 专家策略状态变量
        self.expert_trajectory = []       # 专家轨迹列表 [(pos, gripper_state), ...]
        self.expert_trajectory_idx = 0    # 当前执行到的轨迹索引
        self.expert_executing = False     # 是否正在执行专家策略
        self.expert_pending = False       # 是否处于缓冲期（等待启动）
        self.expert_countdown = 0         # 缓冲期倒计时
        self.is_recording = False         # 录制标志位（用于外部检测是否需要保存图像）
        self.expert_waiting_save = False  # 🔥 是否处于等待保存状态（执行完毕后的等待期）
        self.expert_post_countdown = 0    # 🔥 执行完毕后的等待倒计时

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

    def reset(self, seed=None, mode=None):
        if seed is not None: np.random.seed(seed)
        
        # 如果传入了 mode 参数，更新 control_mode（用于 set_instruction 判断）
        if mode is not None:
            self.control_mode = mode
        
        # 机械臂归位 (V4: 机械臂基座在 x=-0.4，配合新桌面高度 0.3m)
        q_init = np.deg2rad([0,0,0,0,0,0])
        # 机械臂基座在 x=-0.4，目标位置相应调整
        # 目标: 基座前方 0.3m -> [-0.4+0.3, 0.0, 0.5] = [-0.1, 0.0, 0.5] (桌面0.3 + 安全高度0.2)
        q_zero, _, _ = solve_ik(self.env, self.joint_names, 'tcp_link', q_init, np.array([-0.1, 0.0, 0.5]), rpy2r(np.deg2rad([90, -0., 90])))
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)

        try:
            # V4.1: 小车初始位置根据控制模式设置
            # 🔥 V4.1: 缩小小车随机范围至 ±5cm
            if self.control_mode == 'base':
                # 导航模式：小车放在随机位置，需要移动到工作台
                x_init = random.uniform(-0.640, -0.643)
                y_init = random.uniform(-0.4, 0.4)
                self.env.set_pR_base_body(
                    body_name='tb3_base',
                    p=np.array([x_init, y_init, 0.0]),
                    R=np.eye(3)
                )
            else:
                # 协同模式：小车放在工作台旁边，等待接收物品
                # 🔥 V4.1: y 方向随机范围从 ±40cm 缩小到 ±5cm
                x_init = random.uniform(-0.640, -0.643)
                y_init = random.uniform(-0.05, 0.05)  # 缩小范围：原 ±0.4 改为 ±0.05
                self.env.set_pR_base_body(
                    body_name='tb3_base',
                    p=np.array([x_init, y_init, 0.0]),
                    R=np.eye(3)  # YAW=0，朝向 +x 方向
                )
        except Exception as e:
            print(f"Warning: Could not reset TB3 base pose: {e}")
        
        # 状态重置
        self.last_q = copy.deepcopy(q_zero)
        self.current_arm_q = np.concatenate([q_zero, np.array([0.0]*4)]) 
        self.current_wheel_vel = np.zeros(2)
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')
        
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
        self.expert_countdown = 0
        self.is_recording = False
        self.expert_waiting_save = False
        self.expert_post_countdown = 0

        # 物体初始化
        self._init_objects_demo()
        
        obj_poses = self.get_obj_pose()
        self.obj_init_pose = np.concatenate(obj_poses, dtype=np.float32)

        for _ in range(100):
            self.step_env()
            
        self.set_instruction()  # 现在会根据 self.control_mode 自动设置正确的任务文本
        self.gripper_state = False
        
        # 重置时刷新一次图像缓存
        self.grab_image()

    def _init_objects_demo(self):
        # ====== V4.1 版本物品初始化 (固定顺序 + 小范围随机) ======
        # 🔥 V4.1: 四个杯子固定顺序（红蓝黄绿，从左到右）
        # 每个杯子只在固定位置附近 ±2cm 范围内随机
        
        # 杯子名称和颜色映射
        mug_info = {
            'body_obj_mug_5': 'red',
            'body_obj_mug_6': 'blue',
            'body_obj_mug_7': 'yellow',
            'body_obj_mug_8': 'green'
        }
        mug_names = list(mug_info.keys())
        
        # 🔥 V4.1: 永远4个杯子都在桌上
        mugs_on_table = mug_names.copy()
        
        # 记录桌上的杯子信息（供 set_instruction 使用）
        self.mugs_on_table = mugs_on_table
        self.mug_colors_on_table = {name: mug_info[name] for name in mugs_on_table}
        
        # 🔥 V4.1: 按固定顺序放置杯子，每个杯子在固定位置附近小范围随机
        for mug_name in mug_names:
            # 获取该杯子的固定位置
            base_x, base_y, base_z = CUP_FIXED_POSITIONS[mug_name]
            
            # 在固定位置附近添加小范围随机扰动 (±2cm)
            x = base_x + random.uniform(-CUP_POSITION_NOISE, CUP_POSITION_NOISE)
            y = base_y + random.uniform(-CUP_POSITION_NOISE, CUP_POSITION_NOISE)
            z = base_z
            
            # 确保杯子在桌面范围内
            x = np.clip(x, TABLE_X_MIN, TABLE_X_MAX)
            y = np.clip(y, TABLE_Y_MIN, TABLE_Y_MAX)
            
            # 放置杯子
            self.env.set_p_base_body(body_name=mug_name, p=np.array([x, y, z]))
            self.env.set_R_base_body(body_name=mug_name, R=np.eye(3,3))

    def set_instruction(self, given=None, task_type=None):
        """
        设置任务指令
        
        Parameters:
            given: 手动指定的指令文本
            task_type: 任务类型，可选值:
                - 'nav': 导航任务 (小车移动)
                - 'arm': 原版机械臂任务 (杯子放盘子)
                - 'collab': V4 协同任务 (杯子放小车托盘)
                - None: 自动根据 control_mode 决定
        """
        # 保存任务类型
        if task_type is not None:
            self.task_type = task_type
        elif not hasattr(self, 'task_type'):
            # 默认：arm 模式用 collab 任务，base 模式用 nav 任务
            self.task_type = 'collab' if self.control_mode == 'arm' else 'nav'
        
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
                # Arm 模式：根据 task_type 决定任务
                # 🔥 V4: 只从桌上存在的杯子中选择目标
                # 完整的杯子名称到颜色映射
                full_mug_colors = {
                    'body_obj_mug_5': 'red',
                    'body_obj_mug_6': 'blue', 
                    'body_obj_mug_7': 'yellow',
                    'body_obj_mug_8': 'green'
                }
                
                # 获取当前桌上的杯子（如果没有初始化，默认全部在桌上）
                if hasattr(self, 'mug_colors_on_table') and len(self.mug_colors_on_table) > 0:
                    available_mugs = self.mug_colors_on_table  # {body_name: color}
                else:
                    available_mugs = full_mug_colors
                
                if self.task_type == 'collab':
                    # V4 协同任务：杯子放到小车托盘
                    # 随机选择一个指令模板
                    instruction_template = random.choice(COLLAB_INSTRUCTIONS)
                    
                    # 🔥 只从桌上存在的杯子中随机选择
                    target_body_name = random.choice(list(available_mugs.keys()))
                    color = available_mugs[target_body_name]
                    
                    # 设置目标物体
                    self.obj_target = target_body_name
                    
                    # 格式化指令文本
                    self.instruction = instruction_template['text'].format(color=color)
                    # 保存颜色属性（用于数据记录等）
                    self.target_color = color
                else:
                    # 原版任务：杯子放到盘子
                    self.task_type = 'arm'
                    # 🔥 只从桌上存在的杯子中随机选择
                    target_body_name = random.choice(list(available_mugs.keys()))
                    color = available_mugs[target_body_name]
                    self.obj_target = target_body_name
                    # 🔥 V4.1: 简化指令，不使用同义词替换
                    self.instruction = f'Place the {color} mug on the plate.'
                    self.target_color = color
        else:
            self.instruction = given
            # 解析 obj_target 和 target_color (支持四种颜色)
            if self.control_mode == 'arm' or self.task_type in ['arm', 'collab']:
                # 🔥 解析指定的颜色，并检查是否在桌上
                parsed_target = None
                parsed_color = None
                
                if 'red' in self.instruction.lower() or 'mug_5' in self.instruction:
                    parsed_target = 'body_obj_mug_5'
                    parsed_color = 'red'
                elif 'blue' in self.instruction.lower() or 'mug_6' in self.instruction:
                    parsed_target = 'body_obj_mug_6'
                    parsed_color = 'blue'
                elif 'yellow' in self.instruction.lower() or 'mug_7' in self.instruction:
                    parsed_target = 'body_obj_mug_7'
                    parsed_color = 'yellow'
                elif 'green' in self.instruction.lower() or 'mug_8' in self.instruction:
                    parsed_target = 'body_obj_mug_8'
                    parsed_color = 'green'
                
                # 检查解析的目标是否在桌上
                if hasattr(self, 'mug_colors_on_table') and len(self.mug_colors_on_table) > 0:
                    if parsed_target and parsed_target in self.mug_colors_on_table:
                        self.obj_target = parsed_target
                        self.target_color = parsed_color
                    else:
                        # 🔥 指定的杯子不在桌上，警告并随机选择桌上的一个
                        print(f"⚠️ Warning: Specified mug '{parsed_color}' is not on table. Selecting from available mugs.")
                        fallback_target = random.choice(list(self.mug_colors_on_table.keys()))
                        self.obj_target = fallback_target
                        self.target_color = self.mug_colors_on_table[fallback_target]
                else:
                    # 没有桌上杯子信息，使用解析结果或默认值
                    if parsed_target:
                        self.obj_target = parsed_target
                        self.target_color = parsed_color
                    else:
                        self.obj_target = 'body_obj_mug_5'
                        self.target_color = 'red'
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
            
            # 返回机械臂状态 (6,)
            return self.get_joint_state()[:6]
            
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

    # ====== 🤖 Expert Policy Helper Functions (专家策略辅助函数) ======
    
    def interpolate_move(self, start_pos, end_pos, steps, gripper_state):
        """
        线性插值移动
        
        Parameters:
            start_pos: 起始位置 (3,)
            end_pos: 终止位置 (3,)
            steps: 插值步数
            gripper_state: 夹爪状态 (0.0=打开, 1.0=关闭)
            
        Returns:
            waypoints: List of (pos, gripper_state) tuples
        """
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps  # t 从 1/steps 到 1.0
            pos = start_pos + t * (end_pos - start_pos)
            waypoints.append((pos.copy(), gripper_state))
        return waypoints
    
    def bezier_move(self, p0, p1, p2, steps, gripper_state):
        """
        二阶贝塞尔曲线插值移动
        B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
        
        Parameters:
            p0: 起点 (3,)
            p1: 控制点 (3,) - 决定曲线弧度
            p2: 终点 (3,)
            steps: 插值步数
            gripper_state: 夹爪状态
            
        Returns:
            waypoints: List of (pos, gripper_state) tuples
        """
        waypoints = []
        for i in range(steps):
            t = (i + 1) / steps  # t 从 1/steps 到 1.0
            # 二阶贝塞尔曲线公式
            pos = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            waypoints.append((pos.copy(), gripper_state))
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
    
    def auto_execute_task(self, record=False):
        """
        🤖 自动执行专家策略，生成高质量演示轨迹
        
        Parameters:
            record (bool): 是否开启录制模式。True=录制模式，False=测试模式
        
        执行阶段 (State Machine):
        1. 准备阶段: 计算带噪声的目标抓取点
        2. 接近 (Approach): 移动到悬停点，垂直下降到抓取点
        3. 抓取 (Grasp): 闭合夹爪，等待物理引擎结算
        4. 运输 (Transport): 提升后用贝塞尔曲线移动到托盘上方
        5. 放置 (Place): 垂直下降，松开夹爪
        6. 撤离 (Retract): 仅提升Z轴，避免碰倒杯子
        """
        # 获取当前末端执行器位置（使用 self.p0 确保与 delta 计算一致）
        current_pos = self.p0.copy()
        
        # 获取目标物体位置
        if not hasattr(self, 'obj_target'):
            print("⚠️ No target object set. Call set_instruction() first.")
            return
        obj_pos = self.env.get_p_body(self.obj_target)
        
        # 获取托盘位置
        try:
            tray_pos = self.env.get_p_body('tb3_tray')
        except:
            print("⚠️ Cannot find tb3_tray body.")
            return
        
        # ====== 1. 准备阶段：计算带噪声的目标位置 ======
        
        # 随机生成巡航高度
        z_travel = EXPERT_Z_TRAVEL_BASE + np.random.uniform(-EXPERT_Z_TRAVEL_NOISE, EXPERT_Z_TRAVEL_NOISE)
        
        # 抓取点（在物体真实坐标上叠加噪声和偏移）
        grasp_pos = np.array([
            obj_pos[0] + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            obj_pos[1] + EXPERT_Y_GRASP_OFFSET + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            EXPERT_Z_GRASP_BASE + np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE)
        ])
        
        # 悬停点（物体上方，带微小误差）
        hover_pos = np.array([
            grasp_pos[0] + np.random.uniform(-EXPERT_HOVER_NOISE, EXPERT_HOVER_NOISE),
            grasp_pos[1] + np.random.uniform(-EXPERT_HOVER_NOISE, EXPERT_HOVER_NOISE),
            z_travel
        ])
        
        # 放置点（托盘上方，带噪声）
        # 🔥 [修改后]：在 tray_pos[1] 后面加上 + EXPERT_Y_PLACE_OFFSET
        place_pos = np.array([
            tray_pos[0] + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            # 这里加上偏移量 👇
            tray_pos[1] + EXPERT_Y_PLACE_OFFSET + np.random.uniform(-EXPERT_XY_NOISE_SCALE, EXPERT_XY_NOISE_SCALE),
            EXPERT_Z_PLACE_BASE + np.random.uniform(-EXPERT_Z_NOISE, EXPERT_Z_NOISE)
        ])
        
        # 放置悬停点（托盘正上方）
        place_hover_pos = np.array([
            place_pos[0],
            place_pos[1],
            z_travel
        ])
        
        # ====== 🔥 动态计算各阶段步数（基于距离）======
        # 提前计算关键位置（供步数计算使用）
        lift_pos = np.array([grasp_pos[0], grasp_pos[1], z_travel])
        retract_pos = np.array([place_pos[0], place_pos[1], EXPERT_RETRACT_HEIGHT])
        
        # 动态步数计算
        approach_steps = self.distance_based_steps(current_pos, hover_pos, EXPERT_SPEED_APPROACH)
        descend_steps = self.distance_based_steps(hover_pos, grasp_pos, EXPERT_SPEED_DESCEND)
        grasp_wait_steps = self.rand_wait_steps(EXPERT_GRASP_WAIT_BASE, EXPERT_GRASP_WAIT_NOISE)
        lift_steps = self.distance_based_steps(grasp_pos, lift_pos, EXPERT_SPEED_LIFT)
        lower_steps = self.distance_based_steps(place_hover_pos, place_pos, EXPERT_SPEED_LOWER)
        place_wait_steps = self.rand_wait_steps(EXPERT_PLACE_WAIT_BASE, EXPERT_PLACE_WAIT_NOISE)
        retract_steps = self.distance_based_steps(place_pos, retract_pos, EXPERT_SPEED_RETRACT)
        # transport_steps 需要在计算控制点后计算（见下方）
        
        # ====== 生成完整轨迹 ======
        trajectory = []
        
        # ====== 2. 接近阶段 (Approach) ======
        # 2.1 移动到物体上方悬停点（夹爪打开）
        trajectory.extend(self.interpolate_move(current_pos, hover_pos, approach_steps, gripper_state=0.0))
        
        # 2.2 垂直下降到抓取点（平滑下降，夹爪打开）
        trajectory.extend(self.interpolate_move(hover_pos, grasp_pos, descend_steps, gripper_state=0.0))
        
        # ====== 3. 抓取阶段 (Grasp) ======
        # 3.1 闭合夹爪并等待物理引擎结算
        trajectory.extend(self.create_gripper_action(grasp_pos, gripper_state=1.0, steps=grasp_wait_steps))
        
        # ====== 4. 运输阶段 (Transport) ======
        # 4.1 垂直提升至巡航高度
        trajectory.extend(self.interpolate_move(grasp_pos, lift_pos, lift_steps, gripper_state=1.0))
        
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
        
        trajectory.extend(self.bezier_move(lift_pos, control_point, place_hover_pos, transport_steps, gripper_state=1.0))
        
        # ====== 5. 放置阶段 (Place) ======
        # 5.1 垂直下降到放置点
        trajectory.extend(self.interpolate_move(place_hover_pos, place_pos, lower_steps, gripper_state=1.0))
        
        # 5.2 松开夹爪并等待物体落稳
        trajectory.extend(self.create_gripper_action(place_pos, gripper_state=0.0, steps=place_wait_steps))
        
        # ====== 6. 撤离阶段 (Retract) ======
        # 🔥 关键：严格锁定 X/Y 轴，仅提升 Z 轴，避免碰倒刚放好的杯子
        trajectory.extend(self.interpolate_move(place_pos, retract_pos, retract_steps, gripper_state=0.0))
        
        # 保存轨迹到实例变量
        self.expert_trajectory = trajectory
        self.expert_trajectory_idx = 0
        
        # 🔥 设置录制状态和Pending状态（缓冲期）
        self.is_recording = record
        self.expert_pending = True
        self.expert_countdown = EXPERT_START_DELAY
        self.expert_executing = False  # 先不执行，等待缓冲期结束
        
        print(f"🤖 Expert trajectory generated: {len(trajectory)} steps")
        print(f"   Target object: {self.obj_target} ({getattr(self, 'target_color', 'unknown')} mug)")
        print(f"   Grasp pos: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        print(f"   Place pos: ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f})")
        print(f"   🔥 Dynamic Steps: approach={approach_steps}, descend={descend_steps}, grasp_wait={grasp_wait_steps}")
        print(f"                    lift={lift_steps}, transport={transport_steps}, lower={lower_steps}")
        print(f"                    place_wait={place_wait_steps}, retract={retract_steps}")
        print(f"   📏 Bezier arc length: {bezier_length:.3f}m")
        if record:
            print(f"   🎥 Recording Mode: Buffer {EXPERT_START_DELAY} steps before motion start")
        else:
            print(f"   🧪 Test Mode: No recording")
    
    def get_expert_action(self):
        """
        获取专家策略的下一个动作
        
        Returns:
            action: (7,) array [dx, dy, dz, drx, dry, drz, gripper] or None if trajectory finished
        """
        # 🔥 开头处理 Pending（缓冲期）
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
                # 等待期结束，自动停止录制（但不保存）
                self.expert_waiting_save = False
                self.is_recording = False
                print(f"\n⏰ Post-execution wait finished. Recording PAUSED (not saved).")
                print(f"   👉 Press [U] to SAVE, or [I] to DISCARD.")
                return None
            else:
                # 等待期中，继续录制但返回静止动作
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
            if not self.expert_executing and not self.expert_pending and not self.expert_waiting_save:
                print("🤖 [T] Test Mode: Auto Execute Expert Policy (No Recording)")
                print("   → Recording flag disabled, test mode only...")
                self.auto_execute_task(record=False)
            else:
                print("⚠️ Expert policy already running or waiting for save. Press Z to reset.")
        
        # 🎥 Y 键：录制模式（执行并开启录制）
        if self.env.is_key_pressed_once(key=glfw.KEY_Y) and mode == 'arm':
            if not self.expert_executing and not self.expert_pending and not self.expert_waiting_save:
                print("🎥 [Y] Record Mode: Auto Execute Expert Policy + Start Recording")
                print("   → Recording flag enabled, auto-stop after 3s post-wait...")
                print("   → After completion: [U] to SAVE, [I] to DISCARD")
                self.auto_execute_task(record=True)
            else:
                print("⚠️ Expert policy already running or waiting for save. Press Z to reset.")

        if mode == 'arm':
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
                    else:
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
        V4 版本成功判定：杯子放到小车托盘上
        (盘子已移除，只支持协同任务)
        """
        if not hasattr(self, 'obj_target'):
            return False
            
        p_obj = self.env.get_p_body(self.obj_target)
        
        # 协同任务：判断杯子是否在小车托盘上
        try:
            p_tray = self.env.get_p_body('tb3_tray')
            # 托盘范围：xy 距离 < 0.15m，z 高度在托盘面附近 (±0.1m)
            xy_dist = np.linalg.norm(p_obj[:2] - p_tray[:2])
            z_diff = abs(p_obj[2] - p_tray[2])
            if xy_dist < 0.15 and z_diff < 0.1:
                # 机械臂已撤离（z > 0.4m，因为桌面降低了）
                tcp_z = self.env.get_p_body('tcp_link')[2]
                if tcp_z > 0.4:
                    return True
        except Exception as e:
            print(f"Warning: check_success error: {e}")
        return False

    def get_obj_pose(self):
        # V4: 返回四个杯子的位置 (盘子已移除)
        p_mug5 = self.env.get_p_body('body_obj_mug_5')
        p_mug6 = self.env.get_p_body('body_obj_mug_6')
        p_mug7 = self.env.get_p_body('body_obj_mug_7')
        p_mug8 = self.env.get_p_body('body_obj_mug_8')
        return (p_mug5, p_mug6, p_mug7, p_mug8)
    
    def grab_image(self):
        # 初始化返回字典
        images = {}

        if self.control_mode == 'arm':
            # === 机械臂模式：只获取 ARM 相关的 3 个相机 ===
            self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='agentview')
            self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric') 
            self.rgb_back = self.env.get_fixed_cam_rgb(cam_name='backview')
            
            # 存入字典（用于保存数据）
            images['agent'] = self.rgb_agent
            images['wrist'] = self.rgb_ego
            images['back'] = self.rgb_back
            
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
            if not hasattr(self, 'rgb_ego'): self.grab_image()
            if teleop and not hasattr(self, 'rgb_back'): self.grab_image()
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
            
            # 3. 左上角: 后视图 (仅 Teleop 时)
            if teleop and hasattr(self, 'rgb_back'):
                rgb_back_view = add_title_to_img(self.rgb_back, text='Back View', shape=(640,480))
                self.env.viewer_rgb_overlay(rgb_back_view, loc='top left')

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
        
        # V4 新增：显示桌上杯子的XYZ坐标 (左下角)
        try:
            # 只显示桌上存在的杯子
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
                
                # 显示桌上杯子数量
                num_mugs = len(self.mug_colors_on_table)
                self.env.viewer_text_overlay(text1=f'Mugs on Table ({num_mugs})', text2=mugs_text)
        except Exception as e:
            pass  # 静默处理，避免影响渲染
                
        self.env.render()