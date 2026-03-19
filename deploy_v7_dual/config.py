import numpy as np

# ==========================================
# 🔥 环境配置
# ==========================================
# 任务超时判定（秒）：超过该时间未成功则判定失败并自动重置
TASK_TIMEOUT_SEC = 60
# 任务循环次数：执行多少个任务后自动退出（0表示无限循环）
TASK_LOOP_COUNT = 100  # 0 = 无限循环，>0 = 执行指定次数后退出
# 任务步数/结果日志文件
STEP_LOG_PATH = "./task_steps_log.csv"
# 成功统计输出目录（与 cup_position 文件夹一致）
TASK_STATS_OUTPUT_DIR = "./cup_position"

# ==========================================
# 🔧 模型配置
# ==========================================

# Arm 模型配置 (V5.3)
ARM_CONFIG = {
    "model_path": "./ckpt/pi0_arm/pretrained_model_arm_v7_1",
    "dataset_repo_id": "omy_arm_data_v7_1",
    "dataset_root": "./demo_data_arm_v7_1",
    "chunk_size": 64,  # 🔥 V5.3: 从 5 改为 64
    "n_action_steps": 64,  # 🔥 V5.3: 从 5 改为 64
    "image_size": 224,  # arm 模型使用 224x224
    "state_dim": 7,  # 🔥 修复：应该是7维 [q1, q2, q3, q4, q5, q6, gripper]
    "action_dim": 7,
    "camera_keys": ["agent", "wrist"],
}

# Base 模型配置 (V3)
BASE_CONFIG = {
    "model_path": "./ckpt/pi0_base/pretrained_model_base_v7_6",
    "dataset_repo_id": "omy_base_data_v7_6",
    "dataset_root": "./demo_data_base_v7_6",
    "chunk_size": 32,  # V6: 与 arm 对齐
    "n_action_steps": 32,  # V6: 与 arm 对齐
    "image_size": 224,  # base 模型使用 256x256
    "state_dim": 4,  # [v_left, v_right, sin(yaw), cos(yaw)]
    "action_dim": 2,
    "camera_keys": ["front", "left", "right"],
}

# ==========================================
# 🔥 控制频率配置（Hz）
# ==========================================
CONTROL_FREQUENCY = 20
CONTROL_DT = 1.0 / CONTROL_FREQUENCY

# ==========================================
# 🔧 推理参数配置
# ==========================================
# ARM 推理模式总开关：
#   - "sync":  同步推理（每帧在主线程推理）
#   - "async": 异步推理（后台线程推理，主线程不阻塞）
ARM_INFERENCE_MODE = "async"
if ARM_INFERENCE_MODE.lower() not in ("sync", "async"):
    raise ValueError("ARM_INFERENCE_MODE must be 'sync' or 'async'")
ARM_SYNC_INFERENCE = ARM_INFERENCE_MODE.lower() == "sync"

# 异步推理参数（仅当 ARM_SYNC_INFERENCE = False 时有效）：
ACTION_HORIZON = 8
CHUNK_THRESHOLD = 0
# Base 异步推理参数（与 Arm 逻辑保持一致）
BASE_ACTION_HORIZON = 8
BASE_CHUNK_THRESHOLD = 0

# Base 自动推理动作后处理（可选）
BASE_POSTPROC_ENABLED = True
BASE_POSTPROC_HEADING_HOLD_ENABLED = True
BASE_POSTPROC_KP_YAW = 64.0
BASE_POSTPROC_MAX_TURN_V = 2.0
BASE_POSTPROC_YAW_DEADBAND = np.deg2rad(0.5)
BASE_POSTPROC_STRAIGHT_DELTA_TH = 1.5
BASE_POSTPROC_MIN_ABS_SPEED = 1.0
BASE_POSTPROC_MAX_WHEEL_ABS = 30.0
# Base 轮速降速
BASE_FORWARD_SPEED_SCALE_ENABLED = True
BASE_FORWARD_SPEED_SCALE = 0.5

# 同步推理参数（仅当 ARM_SYNC_INFERENCE = True 时有效）
ARM_EXEC_HORIZON = 8

# ==========================================
# 🔧 动作平滑配置（防颤抖）
# ==========================================
SMOOTHING_ENABLED = True
SMOOTHING_ALPHA_JOINTS = 0.15
SMOOTHING_ALPHA_GRIPPER = 0.25

# 夹爪迟滞控制
GRIPPER_HYSTERESIS_ENABLED = True
GRIPPER_OPEN_THRESH = 0.7
GRIPPER_CLOSE_THRESH = 0.25

# ==========================================
# 🔧 模型加载选择
# ==========================================
LOAD_ARM_MODEL = True
LOAD_BASE_MODEL = True

# ==========================================
# 🔧 Arm 模式难度选择
# ==========================================
ARM_PILOT_RUN_MODE = True  # 🔥 已废弃，保留用于兼容性

# ==========================================
# 🔧 随机初始化配置
# ==========================================
RANDOM_INIT_ENABLED = 0  # 0: 关闭, 1: 旧版, 2: 新版
RANDOM_INIT_GRIPPER_OPEN = True

# ==========================================
# 🚗 TB3 托盘 X 坐标随机化配置
# ==========================================
TB3_X_GAUSSIAN_ENABLED = True
TB3_X_CENTER = 0.20
TB3_X_OFFSET_STD = 0.04
TB3_X_OFFSET_MIN = -0.10
TB3_X_OFFSET_MAX = 0.10
