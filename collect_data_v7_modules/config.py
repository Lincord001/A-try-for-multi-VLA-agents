"""
config.py
---------
所有用户可调配置常量、多配置录制列表、派生常量以及状态机枚举值。
这是最常需要修改的文件。
"""

import os

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    🎛️ 常用配置（频繁调整区）🎛️                               ║
# ║          以下参数是您最可能需要修改的，已按使用频率排序                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- 🔥 GPU 选择（双 4090 环境可切换）---
# 设置为 "0" 或 "1"，控制当前脚本使用哪张显卡
GPU_DEVICE_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE_ID

# --- 🔥 随机初始化开关 (Random Initialization Switch) ---
# ⚠️ 注意：这些参数现在作为默认值，实际值由 MULTI_CONFIG_RECORDING 中的配置覆盖
RANDOM_INIT_ENABLED = 0            # 0: 关闭, 1: 旧版(扇形区域), 2: 新版(圆形交集，仅简单模式)
RANDOM_INIT_GRIPPER_OPEN = True    # True: 初始化时夹爪张开, False: 初始化时夹爪闭合

# --- 🔥 杯子选择模式开关 (Mug Selection Mode Switch) ---
SELECT_SMALLER_ANGLE_MUG = False   # True: 总是选择偏转角度更小的杯子, False: 随机选择

# --- 🚗 TB3 托盘 X 坐标随机化开关（可被 MULTI_CONFIG_RECORDING 覆盖）---
TB3_X_RANDOM_ENABLED = True       # True: 在区间内均匀随机；False: 固定为区间中点
TB3_X_MIN = 0.095                 # X 随机下限（米）
TB3_X_MAX = 0.330                 # X 随机上限（米）

# --- 🧺 托盘任务路由开关（可被 MULTI_CONFIG_RECORDING 覆盖）---
# True: ARM 自动录制时强制路由到 arm_01（tray -> table）
# False: ARM 自动录制时强制路由到 arm_default（on the plate）
TRAY_INIT_ON_TB3_ENABLED = False

# --- 🤖 多配置自动录制 [P键] ---
# 🔥 定义多个配置，按顺序自动录制
# 每个配置包含：
#   - name: 配置名称（用于日志显示）
#   - target_episodes: 该配置的目标录制条数
#   - random_init_enabled: 随机初始化开关 (0/1/2)
#   - random_init_gripper_open: 夹爪初始状态 (True/False)
#   - select_smaller_angle_mug: 杯子选择模式 (True/False, 可选，默认使用全局 SELECT_SMALLER_ANGLE_MUG)
#   - tb3_x_random_enabled: 是否启用 TB3 X 区间随机化 (可选，默认全局 TB3_X_RANDOM_ENABLED)
#   - tb3_x_min/tb3_x_max: TB3 X 随机化区间（可选，默认全局）
#   - tray_init_on_tb3_enabled: 是否启用托盘任务路由（可选，默认全局 TRAY_INIT_ON_TB3_ENABLED）
#
MULTI_CONFIG_RECORDING = [
    {
        'name': 'stage_4',
        'target_episodes': 200,
        'random_init_enabled': 0,
        'random_init_gripper_open': True,
        'select_smaller_angle_mug': False,  # 可选，不设置则使用全局默认值
        'tb3_x_random_enabled': True,  # 可选，不设置则使用全局默认值
        'tray_init_on_tb3_enabled': True,  # 可选，不设置则使用全局默认值
    },
    {
        'name': 'stage_5',
        'target_episodes': 200,
        'random_init_enabled': 1,
        'random_init_gripper_open': True,
        'select_smaller_angle_mug': False,  # 可选，不设置则使用全局默认值
        'tb3_x_random_enabled': True,  # 可选，不设置则使用全局默认值
        'tray_init_on_tb3_enabled': True,  # 可选，不设置则使用全局默认值
    },
]

# MULTI_CONFIG_RECORDING = [
#     {
#         'name': 'stage_1',
#         'target_episodes': 200,
#         'random_init_enabled': 0,
#         'random_init_gripper_open': True,
#         'select_smaller_angle_mug': False,
#         'tb3_x_random_enabled': True,
#         'tray_init_on_tb3_enabled': False,
#     },
#     {
#         'name': 'stage_2',
#         'target_episodes': 200,
#         'random_init_enabled': 1,
#         'random_init_gripper_open': True,
#         'select_smaller_angle_mug': False,
#         'tb3_x_random_enabled': True,
#         'tray_init_on_tb3_enabled': False,
#     },
#     {
#         'name': 'stage_3',
#         'target_episodes': 200,
#         'random_init_enabled': 1,
#         'random_init_gripper_open': True,
#         'select_smaller_angle_mug': True,
#         'tb3_x_random_enabled': True,
#         'tray_init_on_tb3_enabled': False,
#     },
# ]

# 🔥 如果 MULTI_CONFIG_RECORDING 为空列表，则使用单配置模式（向后兼容）
# 单配置模式使用以下参数：
AUTO_RECORD_TARGET_EPISODES = 150    # 🎯 单配置模式的目标录制条数
AUTO_SHUTDOWN_ON_COMPLETE = True     # 🔌 完成后是否自动关闭仿真环境

# --- 🎮 初始模式 ---
INITIAL_MODE = 'arm'                 # 启动时的模式: 'arm' 或 'base'

# --- 📁 数据集名称与路径 ---
ARM_DATASET_NAME = 'omy_arm_data_test'       # Arm 模式数据集名称
ARM_DATASET_ROOT = './demo_data_arm_test'    # Arm 模式数据集保存路径
BASE_DATASET_NAME = 'omy_base_data_RAG'     # Base 模式数据集名称
BASE_DATASET_ROOT = './demo_data_base_RAG'  # Base 模式数据集保存路径

# --- 🖼️ 图像与录制 ---
IMG_SIZE = 224                       # 图像分辨率 (224=ViT标准, 256=兼容旧数据)
FPS = 20                             # 录制帧率 (Hz)
MAX_EPISODE_SEC = 200                # 单条数据最大时长（秒），超时自动丢弃

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    ⚙️ 高级配置（一般不需要修改）⚙️                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# --- 全自动录制细节参数 ---
AUTO_RESET_WAIT_FRAMES = 40          # 重置后等待帧数（约2秒，让物理引擎稳定）
AUTO_CUP_CHECK_TOLERANCE = 0.05      # 物体 z 坐标容差（±5cm，超出判定为倒了）
AUTO_POST_SAVE_WAIT_FRAMES = 20      # 保存后等待帧数（约1秒）
AUTO_MAX_RESET_RETRIES = 5           # 物体倒了时最大重试次数
ARM_POST_EXEC_WAIT_FRAMES = 3 * FPS  # Y键专家执行完成后继续录制3秒

# --- 场景配置 ---
SEED = 0
XML_PATH = './asset/example_scene_y7.xml'

# --- 派生配置（自动计算，勿手动修改）---
MAX_FRAMES = MAX_EPISODE_SEC * FPS

DATASET_CONFIG = {
    'arm': {
        'repo_name': ARM_DATASET_NAME,
        'root': ARM_DATASET_ROOT,
    },
    'base': {
        'repo_name': BASE_DATASET_NAME,
        'root': BASE_DATASET_ROOT,
    }
}

MODE_CONFIG = {
    'arm': {
        'action_shape': (7,),
        'state_shape': (7,),  # [q1, q2, q3, q4, q5, q6, gripper]
    },
    'base': {
        'action_shape': (2,),
        'state_shape': (4,),  # [v_left, v_right, sin(yaw), cos(yaw)]
    }
}

# --- 自动录制状态机常量（内部使用，勿修改）---
AUTO_STATE_IDLE = 0
AUTO_STATE_RESETTING = 1
AUTO_STATE_CHECK_CUPS = 2
AUTO_STATE_START_EXPERT = 3
AUTO_STATE_EXECUTING = 4
AUTO_STATE_WAIT_QUEUE = 5
AUTO_STATE_SAVING = 6
AUTO_STATE_POST_SAVE = 7
AUTO_STATE_SWITCHING_CONFIG = 8  # 🔥 配置切换状态
