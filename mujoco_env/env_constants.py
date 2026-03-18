import numpy as np

# ====== V6 版本任务说明 ======
# 任务：小车和机械臂协同，机械臂把红色杯子放到盘子上
# V6 场景特点（基于V5_2）：
#   - 桌面高度 0.83m
#   - 机械臂基座在原点 (0, 0)
#   - 使用杯子和盘子
#   - 小车在远处固定位置 (x=0, y=3)

# ====== 🤖 Expert Policy Parameters (专家策略参数) ======
# 这些参数用于自动化录制数据的专家策略，可根据需要调整

# ====== V6 Height Offset (统一高度偏移) ======
# 作用范围：y_env6.py 中所有任务相关的 Z 目标（抓取/放置/初始化/成功判定等）
# 调整方式：正值整体抬高，负值整体降低。
# 注意：该常量不会自动修改 XML 中桌子/托盘几何高度；XML 高度需单独调整。
V6_Z_OFFSET = 0.15

# ====== V7 Docking Notch Parameters (桌子停靠凹坑参数) ======
# 仅对 object_table_v7.xml 生效：
# - 停靠区 [V7_DOCK_X_MIN, V7_DOCK_X_MAX] 在桌子 Y 负方向保持不延申
# - 其余区域向 Y 负方向延申 V7_TABLE_Y_NEG_EXTENSION
V7_DOCK_X_MIN = -0.08
V7_DOCK_X_MAX = 0.50
V7_TABLE_Y_NEG_EXTENSION = 0.10

# object_table_v7 的基准几何参数（与 XML 保持一致）
V7_TABLE_CENTER_X = 0.5
V7_TABLE_HALF_X = 1.0
V7_TABLE_HALF_Y = 0.7
V7_TABLE_HALF_Z = 0.14

# 仅保留桌面 y >= 该阈值的部分（其余裁剪掉）
V6_TABLE_Y_MIN = -0.11

# 成功判定时 TCP 的撤离高度容差（米）：
# 与专家撤离高度联动，留 5cm 容差，避免阈值与轨迹配置漂移
SUCCESS_TCP_Z_MARGIN = 0.05

# --- TB3 初始 X 轴随机化参数（均匀分布）---
TB3_X_CENTER = 0.21
TB3_X_OFFSET_STD = 0.04
TB3_X_OFFSET_MIN = -0.115
TB3_X_OFFSET_MAX = 0.115
TB3_X_OFFSET_MAX_ATTEMPTS = 50
TB3_X_MIN = 0.095
TB3_X_MAX = 0.325

# --- Base 模式：H 键自动停车流程参数 ---
BASE_AUTO_STAGE1_TARGET_XY = np.array([0.2, -0.8], dtype=np.float32)
BASE_AUTO_STAGE2_TARGET_XY = np.array([0.2, -0.25], dtype=np.float32)
BASE_AUTO_STAGE2_TARGET_YAW = np.deg2rad(90.0)  # 车头朝向 +Y
BASE_AUTO_POS_TOL = 0.03
BASE_AUTO_YAW_TOL = np.deg2rad(3.0)
BASE_AUTO_STAGE1_X_TOL = 0.03
BASE_AUTO_STAGE1_Y_TOL = 0.05
BASE_AUTO_FINAL_POS_TOL = 0.015
BASE_AUTO_FINAL_YAW_TOL = np.deg2rad(2.0)
BASE_AUTO_FINAL_X_TOL = 0.02
BASE_AUTO_FINAL_Y_EPS = 0.003
BASE_AUTO_MAX_FWD_V = 15.0
BASE_AUTO_MAX_TURN_V = 6.0
BASE_AUTO_KP_LIN = 18.0
BASE_AUTO_KP_ANG = 8.0
BASE_AUTO_WAIT_SEC = 3.0
BASE_AUTO_WAIT_FRAMES = 60
BASE_AUTO_PUSH_SEC = 2.5
BASE_AUTO_PUSH_FWD_V = 4.0
BASE_AUTO_STAGE2_Y_NEAR_MARGIN = 0.02
BASE_AUTO_STAGE_TIMEOUT_SEC = 20.0
BASE_AUTO_ROTATE_IN_PLACE_TH = np.deg2rad(20.0)
BASE_AUTO_MIN_TURN_CMD = 1.0
BASE_AUTO_MIN_FWD_CMD = 2.0

# --- Base 模式：直行航向闭环辅助（手动 W/S + H 键前推复用）---
BASE_STRAIGHT_ASSIST_ENABLED = True
BASE_STRAIGHT_KP_YAW = 8.0
BASE_STRAIGHT_MAX_TURN_V = 3.0
BASE_STRAIGHT_YAW_DEADBAND = np.deg2rad(0.5)

# --- 高度参数 ---
# 🔥 V6沿用原有抓取逻辑参数：桌面与盘子任务参数保持兼容
EXPERT_Z_TRAVEL_BASE = 0.43 + V6_Z_OFFSET  # 巡航基础高度（支持统一偏移）
EXPERT_Z_TRAVEL_NOISE = 0.03         # 巡航高度随机扰动范围 ± (Cruise height noise)
EXPERT_Z_GRASP_BASE = 0.32 + V6_Z_OFFSET   # 抓取高度基础值（支持统一偏移）
EXPERT_Z_PLACE_BASE = 0.33 + V6_Z_OFFSET   # 放置高度基础值（支持统一偏移）
EXPERT_RETRACT_HEIGHT = 0.48 + V6_Z_OFFSET # 撤离安全高度（支持统一偏移）

# --- 位置偏移与噪声 ---
EXPERT_XY_NOISE_SCALE = 0.01        # 端点随机噪声范围 ±3mm (Endpoint noise for robustness)
EXPERT_Y_GRASP_OFFSET = 0.067       # 抓取Y轴固定偏移（用于抓取杯子把手）(Y offset for cup handle)
EXPERT_Y_PLACE_OFFSET = 0.03        # 放置时的 Y 轴固定偏移
EXPERT_HOVER_NOISE = 0.01            # 悬停点误差 (Hover point noise)
EXPERT_Z_NOISE = 0.005               # Z轴高度微小随机噪声 (Z height noise)

# --- 🔥 漏斗移动逻辑参数 (Funnel Approach Parameters) ---
# 🔥 悬停点和中间点的圆半径参数（用于漏斗式接近策略，方便调参）
EXPERT_FUNNEL_HOVER_RADIUS = 0.02   # 悬停点圆半径（米）(Hover point circle radius, 3cm)
EXPERT_FUNNEL_MID_RADIUS = 0.01     # 中间点圆半径（米）(Mid point circle radius, 1cm)
# 🔥 悬停点和中间点的Z坐标参数（圆心高度）
EXPERT_FUNNEL_HOVER_Z = None        # 悬停点Z坐标（米），None=使用巡航高度z_travel
EXPERT_FUNNEL_MID_Z = 0.40 + V6_Z_OFFSET  # 中间点Z坐标（支持统一偏移）

# --- 🔥 人类抖动模拟参数 (Human Tremor Simulation Parameters) ---
EXPERT_TREMOR_ENABLED = True           # 是否启用抖动 (Enable tremor)
EXPERT_TREMOR_AMPLITUDE = 0.002       # 抖动幅度（米）(Tremor amplitude, 2mm)
EXPERT_TREMOR_SMOOTHNESS = 0.7        # 抖动平滑度 (0-1，越大越平滑) (Tremor smoothness)

# --- 🔥 动态步数参数（基于距离计算，提高数据多样性）---
# 末端执行器速度参数（米/步），用于根据距离动态计算步数
# 🔥 V4.1: 速度减半，使单条数据集时长变为原来的2倍
EXPERT_SPEED_APPROACH = 0.006        # 接近阶段速度 (Approach speed, m/step) - 原0.012减半
EXPERT_SPEED_APPROACH_SMALLER_ANGLE = 0.004  # 🔥 角度较小的杯子接近阶段速度 (Approach speed for smaller angle mug, m/step)
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
# 🔥 V1模式专用角度参数
RANDOM_INIT_ANGLE_MIN_V1 = 0         # V1模式角度范围最小值（度）(V1 Min angle in degrees)
RANDOM_INIT_ANGLE_MAX_V1 = 15        # V1模式角度范围最大值（度）(V1 Max angle in degrees, 0~15度)
RANDOM_INIT_RADIUS_MIN = 0.3         # 径向距离最小值（米）(Min radial distance in meters)
RANDOM_INIT_RADIUS_MAX = 0.4         # 径向距离最大值（米）(Max radial distance in meters)
# 🔥 V1模式专用径向距离参数
RANDOM_INIT_RADIUS_MIN_V1 = 0.275   # V1模式径向距离最小值（米）(V1 Min radial distance in meters)
RANDOM_INIT_RADIUS_MAX_V1 = 0.325   # V1模式径向距离最大值（米）(V1 Max radial distance in meters)
RANDOM_INIT_Z_MIN = 0.38 + V6_Z_OFFSET  # Z坐标最小值（支持统一偏移）
RANDOM_INIT_Z_MAX = 0.48 + V6_Z_OFFSET  # Z坐标最大值（支持统一偏移）
# 🔥 V1模式专用Z轴参数
RANDOM_INIT_Z_MIN_V1 = 0.43 + V6_Z_OFFSET  # V1模式Z坐标最小值（支持统一偏移）
RANDOM_INIT_Z_MAX_V1 = 0.48 + V6_Z_OFFSET  # V1模式Z坐标最大值（支持统一偏移）
RANDOM_INIT_Z_MIN_V2 = 0.36 + V6_Z_OFFSET  # V2模式Z坐标最小值（支持统一偏移）
RANDOM_INIT_Z_MAX_V2 = 0.48 + V6_Z_OFFSET  # V2模式Z坐标最大值（支持统一偏移）
RANDOM_INIT_GRIPPER_OPEN = True      # 🔥 初始化时夹爪是否张开 (True=张开, False=闭合)
RANDOM_INIT_MOVE_STEPS = 75          # 🔥 平滑移动到随机位置的步数（约7.5秒，20Hz）(Steps for smooth move to random position)

# ====== 🎲 Object Initialization Parameters (物体初始化参数) ======
# 这些参数用于控制红色杯子的随机初始化（与V2一致）

# --- 机械臂基座位置 ---
ARM_BASE_X = 0.0                     # 机械臂基座X坐标 (Arm base X position, 与V2一致)
ARM_BASE_Y = 0.0                     # 机械臂基座Y坐标 (Arm base Y position, 与V2一致)

# --- 桌面参数 ---
TABLE_Z_HEIGHT = 0.325 + V6_Z_OFFSET  # 桌面高度（杯子初始化Z高度，支持统一偏移）

# --- 桌面范围限制（与V2一致）---
TABLE_X_MIN = 0                   # 桌面X轴最小值 (Table X min boundary)
TABLE_X_MAX = 1.5                  # 桌面X轴最大值 (Table X max boundary)
TABLE_Y_MIN = -0.1                  # 桌面Y轴最小值 (Table Y min boundary)
TABLE_Y_MAX = 1                   # 桌面Y轴最大值 (Table Y max boundary)

# --- 🔥 红色和蓝色杯子初始化参数（两个弧线段）---
# 弧线段1：距离机械臂基座0.3米，角度从0度到向右45度
MUG_ARC1_RADIUS = 0.25                # 弧线段1的半径（米）(Arc 1 radius in meters)
MUG_ARC1_ANGLE_MIN = 0.0            # 弧线段1的最小角度（度）(Arc 1 min angle in degrees, 0° = 正前方)
MUG_ARC1_ANGLE_MAX = 45.0           # 弧线段1的最大角度（度）(Arc 1 max angle in degrees, 45° = 右偏45度)

# 弧线段2：距离机械臂基座0.4米，角度从0度到向右30度
MUG_ARC2_RADIUS = 0.35                # 弧线段2的半径（米）(Arc 2 radius in meters)
MUG_ARC2_ANGLE_MIN = 0.0            # 弧线段2的最小角度（度）(Arc 2 min angle in degrees, 0° = 正前方)
MUG_ARC2_ANGLE_MAX = 45.0           # 弧线段2的最大角度（度）(Arc 2 max angle in degrees, 30° = 右偏30度)

# --- 🔥 红色和蓝色杯子间距参数 ---
MUG_MIN_SPACING = 0.15              # 红色和蓝色杯子之间的最小间隔（米）(Min spacing between red and blue mugs, 5cm)

# --- 以下参数保留用于夹爪随机初始化逻辑（V1版本仍使用扇形区域）---
MUG_MIN_DIST = 0.30                  # 离机械臂基座最近距离（米）(Min distance from arm base) - 用于V1随机初始化
MUG_MAX_DIST = 0.40                  # 离机械臂基座最远距离（米）(Max distance from arm base) - 用于V1随机初始化
MUG_MIN_ANGLE = 0.0                  # 左偏角度（度）(Min angle in degrees, 0° = 正前方) - 用于V1随机初始化
MUG_MAX_ANGLE = 45.0                 # 右偏角度（度）(Max angle in degrees, 45° = 右偏45度) - 用于V1随机初始化

# --- 隐藏物体参数 ---
HIDDEN_OBJ_X = 20.0                  # 隐藏物体的X坐标（远离场景）
HIDDEN_OBJ_Y_INTERVAL = 0.5          # 隐藏物体沿Y轴的间隔（米）
HIDDEN_OBJ_Z = 1.0                   # 隐藏物体的Z坐标（高度）

NAV_INSTRUCTIONS = [
    # 基础指令 (Basic)
    "Go to the workbench.",
    "Drive to the workbench."
]

# 🔥 任务指令：支持红色和蓝色杯子
ARM_INSTRUCTIONS = [
    "Place the red mug on the plate.",
    "Place the blue mug on the plate."
]
