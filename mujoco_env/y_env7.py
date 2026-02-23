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
import time
import glfw

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

# --- TB3 初始 X 轴随机化参数（方案A：全程高斯扰动）---
TB3_X_CENTER = 0.30
TB3_X_OFFSET_STD = 0.04
TB3_X_OFFSET_MIN = -0.10
TB3_X_OFFSET_MAX = 0.10
TB3_X_OFFSET_MAX_ATTEMPTS = 50

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


def _parse_vec_attr(attr_text):
    return [float(x) for x in attr_text.strip().split()]


def _format_vec_attr(values):
    return " ".join(f"{v:.6f}".rstrip("0").rstrip(".") for v in values)


def _offset_attr_z(elem, attr_name, z_index=2, offset=0.0):
    if elem is None:
        return
    attr_val = elem.get(attr_name)
    if not attr_val:
        return
    vec = _parse_vec_attr(attr_val)
    if len(vec) <= z_index:
        return
    vec[z_index] += offset
    elem.set(attr_name, _format_vec_attr(vec))


def _stretch_box_geom_upward(geom_elem, offset):
    """
    让 box 几何体“向上增高”而不是整体平移：
    - 底面保持不变
    - 顶面抬高 offset
    """
    if geom_elem is None:
        return
    if geom_elem.get("type", "box") != "box":
        return

    size_attr = geom_elem.get("size")
    pos_attr = geom_elem.get("pos")
    if not size_attr or not pos_attr:
        return

    size_vec = _parse_vec_attr(size_attr)
    pos_vec = _parse_vec_attr(pos_attr)
    if len(size_vec) < 3 or len(pos_vec) < 3:
        return

    # MuJoCo box size 的 z 分量是半高，向上增高 offset 需要半高增加 offset/2，
    # 同时中心上移 offset/2 才能保持底面不动。
    new_half_height = size_vec[2] + 0.5 * offset
    size_vec[2] = max(1e-5, new_half_height)
    pos_vec[2] += 0.5 * offset

    geom_elem.set("size", _format_vec_attr(size_vec))
    geom_elem.set("pos", _format_vec_attr(pos_vec))


def _clip_box_geom_y_min(geom_elem, y_min):
    """
    将 box 几何体在 y 方向裁剪为 [y_min, +inf)：
    - 保留 y >= y_min 的部分
    - y < y_min 的部分删除
    """
    if geom_elem is None:
        return
    if geom_elem.get("type", "box") != "box":
        return

    size_attr = geom_elem.get("size")
    pos_attr = geom_elem.get("pos")
    if not size_attr or not pos_attr:
        return

    size_vec = _parse_vec_attr(size_attr)
    pos_vec = _parse_vec_attr(pos_attr)
    if len(size_vec) < 2 or len(pos_vec) < 2:
        return

    half_y = size_vec[1]
    center_y = pos_vec[1]
    cur_min_y = center_y - half_y
    cur_max_y = center_y + half_y

    # 完全在保留范围内，无需处理
    if cur_min_y >= y_min:
        return

    # 完全在裁剪线以下，收缩到极小厚度避免非法尺寸
    if cur_max_y <= y_min:
        size_vec[1] = 1e-5
        pos_vec[1] = y_min + 1e-5
    else:
        # 与裁剪线相交：更新中心和半尺寸，保留 [y_min, cur_max_y]
        new_min_y = y_min
        new_max_y = cur_max_y
        size_vec[1] = max(1e-5, 0.5 * (new_max_y - new_min_y))
        pos_vec[1] = 0.5 * (new_max_y + new_min_y)

    geom_elem.set("size", _format_vec_attr(size_vec))
    geom_elem.set("pos", _format_vec_attr(pos_vec))


def _extend_cylinder_geom_upward(geom_elem, offset):
    """
    让 cylinder（用于支架）“向上加长”而不是整体平移：
    - 下端保持不变
    - 上端抬高 offset
    """
    if geom_elem is None:
        return
    if geom_elem.get("type") != "cylinder":
        return

    size_attr = geom_elem.get("size")
    pos_attr = geom_elem.get("pos")
    if not size_attr or not pos_attr:
        return

    size_vec = _parse_vec_attr(size_attr)
    pos_vec = _parse_vec_attr(pos_attr)
    if len(size_vec) < 2 or len(pos_vec) < 3:
        return

    # MuJoCo cylinder size[1] 是半高。
    new_half_height = size_vec[1] + 0.5 * offset
    size_vec[1] = max(1e-5, new_half_height)
    pos_vec[2] += 0.5 * offset

    geom_elem.set("size", _format_vec_attr(size_vec))
    geom_elem.set("pos", _format_vec_attr(pos_vec))


def _configure_v7_table_notch(table_root):
    """
    根据常量重写 object_table_v7 的两块 Y- 延申区域，
    形成“中间停靠区不延申、两侧延申”的凹坑结构。
    """
    table_body = table_root.find(".//body[@name='front_object_table']")
    if table_body is None:
        return

    # 直接读取主桌板几何，按其真实 x 范围构造两侧延申块。
    # 注意：V7_DOCK_X_* 使用与场景一致的 x 坐标系，不再额外减中心偏移。
    front_table = table_body.find(".//geom[@name='front_object_table']")
    if front_table is None:
        return
    front_size = _parse_vec_attr(front_table.get("size", "1.0 0.7 0.14"))
    front_pos = _parse_vec_attr(front_table.get("pos", "0.5 0 0.14"))
    if len(front_size) < 3 or len(front_pos) < 3:
        return

    x_local_min = front_pos[0] - front_size[0]
    x_local_max = front_pos[0] + front_size[0]
    dock_local_min = V7_DOCK_X_MIN
    dock_local_max = V7_DOCK_X_MAX

    # 防御性裁剪，避免参数越界导致非法尺寸
    dock_local_min = np.clip(dock_local_min, x_local_min, x_local_max)
    dock_local_max = np.clip(dock_local_max, x_local_min, x_local_max)
    if dock_local_max < dock_local_min:
        dock_local_min, dock_local_max = dock_local_max, dock_local_min

    ext_half_y = max(1e-5, 0.5 * V7_TABLE_Y_NEG_EXTENSION)
    # 关键：主桌面会被裁剪到 y>=V6_TABLE_Y_MIN，凹坑延申应从该“可见前沿”往 -Y 推出。
    ext_pos_y = V6_TABLE_Y_MIN - ext_half_y

    left_elem = table_body.find(".//geom[@name='front_object_table_ext_left']")
    right_elem = table_body.find(".//geom[@name='front_object_table_ext_right']")

    if left_elem is not None:
        left_len = max(0.0, dock_local_min - x_local_min)
        left_half_x = max(1e-5, 0.5 * left_len)
        left_pos_x = 0.5 * (x_local_min + dock_local_min)
        left_elem.set("size", _format_vec_attr([left_half_x, ext_half_y, V7_TABLE_HALF_Z]))
        left_elem.set("pos", _format_vec_attr([left_pos_x, ext_pos_y, V7_TABLE_HALF_Z]))

    if right_elem is not None:
        right_len = max(0.0, x_local_max - dock_local_max)
        right_half_x = max(1e-5, 0.5 * right_len)
        right_pos_x = 0.5 * (dock_local_max + x_local_max)
        right_elem.set("size", _format_vec_attr([right_half_x, ext_half_y, V7_TABLE_HALF_Z]))
        right_elem.set("pos", _format_vec_attr([right_pos_x, ext_pos_y, V7_TABLE_HALF_Z]))


def _sample_tb3_x_offset(offset_std=TB3_X_OFFSET_STD, offset_min=TB3_X_OFFSET_MIN, offset_max=TB3_X_OFFSET_MAX):
    """
    从截断高斯分布采样 TB3 的 X 轴偏移：
    - 均值 0，标准差 TB3_X_OFFSET_STD
    - 截断范围 [TB3_X_OFFSET_MIN, TB3_X_OFFSET_MAX]
    """
    for _ in range(TB3_X_OFFSET_MAX_ATTEMPTS):
        offset = np.random.normal(0.0, offset_std)
        if offset_min <= offset <= offset_max:
            return float(offset)

    # 极少数情况下回退到 clip，避免意外无限循环
    return float(np.clip(offset, offset_min, offset_max))


def _build_offset_xml_bundle(base_scene_xml_path, z_offset):
    """
    当 V6_Z_OFFSET != 0 时，生成一套带高度偏移的 v6 XML。
    返回可直接传给 MuJoCoParserClass 的 scene xml 绝对路径。
    """
    if abs(z_offset) < 1e-9:
        return base_scene_xml_path

    scene_abs = os.path.abspath(base_scene_xml_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # 关键：把生成文件放在原 asset 层级附近，保持原有相对 include 路径可用。
    asset_root = os.path.join(repo_root, "asset")

    table_src = os.path.join(repo_root, "asset", "tabletop_v6", "object", "object_table_v7.xml")
    omy_src = os.path.join(repo_root, "asset", "robotis_omy_v6", "omy_v6.xml")
    tb3_src = os.path.join(repo_root, "asset", "robotis_tb3_v6", "turtlebot3_waffle_pi_v6.xml")

    table_dst = os.path.join(
        repo_root, "asset", "tabletop_v6", "object", "object_table_v7_offset_generated.xml"
    )
    omy_dst = os.path.join(
        repo_root, "asset", "robotis_omy_v6", "omy_v6_offset_generated.xml"
    )
    tb3_dst = os.path.join(
        repo_root, "asset", "robotis_tb3_v6", "turtlebot3_waffle_pi_v6_offset_generated.xml"
    )
    scene_dst = os.path.join(asset_root, "example_scene_y7_offset_generated.xml")

    # 1) 桌子 + 台上相机
    table_tree = ET.parse(table_src)
    table_root = table_tree.getroot()
    _configure_v7_table_notch(table_root)
    # V7: 主桌板需要裁剪；两块延申板不裁剪，否则会被阈值直接裁没。
    front_table = table_root.find(".//geom[@name='front_object_table']")
    _clip_box_geom_y_min(front_table, V6_TABLE_Y_MIN)
    _stretch_box_geom_upward(front_table, z_offset)

    for ext_name in ["front_object_table_ext_left", "front_object_table_ext_right"]:
        ext_geom = table_root.find(f".//geom[@name='{ext_name}']")
        _stretch_box_geom_upward(ext_geom, z_offset)
    for cam_body_name in ["camera", "camera2", "camera3"]:
        cam_body = table_root.find(f".//body[@name='{cam_body_name}']")
        # 相机跟随桌面高度整体上移。
        _offset_attr_z(cam_body, "pos", z_index=2, offset=z_offset)
    table_tree.write(table_dst, encoding="utf-8", xml_declaration=False)

    # 2) 机械臂基座高度
    omy_tree = ET.parse(omy_src)
    omy_root = omy_tree.getroot()
    link1_body = omy_root.find(".//body[@name='link1']")
    _offset_attr_z(link1_body, "pos", z_index=2, offset=z_offset)
    omy_tree.write(omy_dst, encoding="utf-8", xml_declaration=False)

    # 3) 小车托盘高度（托盘盘面上移 + 立柱向上加长，底端不离开车体）
    tb3_tree = ET.parse(tb3_src)
    tb3_root = tb3_tree.getroot()
    tray_body = tb3_root.find(".//body[@name='tb3_tray']")
    _offset_attr_z(tray_body, "pos", z_index=2, offset=z_offset)
    for pillar_name in ["tray_pillar_fl", "tray_pillar_fr", "tray_pillar_bl", "tray_pillar_br"]:
        pillar_geom = tb3_root.find(f".//geom[@name='{pillar_name}']")
        _extend_cylinder_geom_upward(pillar_geom, z_offset)
    tb3_tree.write(tb3_dst, encoding="utf-8", xml_declaration=False)

    # 4) 场景入口：改 include 指向上面三份偏移后的 XML
    scene_tree = ET.parse(scene_abs)
    scene_root = scene_tree.getroot()
    for include in scene_root.findall(".//include"):
        inc = include.get("file", "")
        if "tabletop_v6/object/object_table_v7.xml" in inc:
            include.set("file", "./tabletop_v6/object/object_table_v7_offset_generated.xml")
        elif "robotis_omy_v6/omy_v6.xml" in inc:
            include.set("file", "./robotis_omy_v6/omy_v6_offset_generated.xml")
        elif "robotis_tb3_v6/turtlebot3_waffle_pi_v6.xml" in inc:
            include.set("file", "./robotis_tb3_v6/turtlebot3_waffle_pi_v6_offset_generated.xml")
    scene_tree.write(scene_dst, encoding="utf-8", xml_declaration=False)

    print(
        f"[V7_Z_OFFSET] Using runtime XML bundle with z_offset={z_offset:+.3f} at: {scene_dst}"
    )
    return scene_dst

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
    "Navigate to the workbench.",
    "Drive to the workbench.",
    "Move to the workbench."
]

# 🔥 任务指令：支持红色和蓝色杯子
ARM_INSTRUCTIONS = [
    "Place the red mug on the plate.",
    "Place the blue mug on the plate."
]

class SimpleEnv6:
    def __init__(self, xml_path, action_type='eef_pose', state_type='joint_angle', seed=None, 
                 random_init_enabled=False, random_init_gripper_open=True, select_smaller_angle_mug=False,
                 tb3_x_gaussian_enabled=True, tb3_x_center=TB3_X_CENTER,
                 tb3_x_offset_std=TB3_X_OFFSET_STD, tb3_x_offset_min=TB3_X_OFFSET_MIN,
                 tb3_x_offset_max=TB3_X_OFFSET_MAX):
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

        # 🔥 TB3 初始 X 轴随机化开关与参数（由外部传入）
        self.tb3_x_gaussian_enabled = tb3_x_gaussian_enabled
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

        # 🚗 Base 模式自动停车（L 键）状态
        self.base_auto_active = False
        self.base_auto_stage = 'idle'
        self.base_auto_wait_counter = 0
        self.base_auto_wait_deadline = None  # 使用 wall time 控制等待时长
        self.base_auto_stage_steps = 0
        self.base_auto_recording_active = False  # 由外层采集脚本每帧同步
        self.base_auto_record_stop_requested = False  # 结束后置 True，由外层决定是否停录
        self.base_auto_push_target_yaw = None  # push_forward 阶段锁定航向
        self.base_straight_assist_active = False
        self.base_straight_target_yaw = None
        self.base_straight_last_cmd_sign = 0  # +1: 前进, -1: 后退, 0: 空闲
        self.base_action_intent = np.zeros(2, dtype=np.float32)  # 记录用于数据集的“未纠偏意图动作”

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

    @staticmethod
    def _wrap_to_pi(angle):
        """把角度归一化到 [-pi, pi]。"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_tb3_pose_xy_yaw(self):
        """返回底盘当前 (x, y, yaw)。"""
        p_tb3 = self.env.get_p_body('tb3_base')
        R_tb3 = self.env.get_R_body('tb3_base')
        yaw = np.arctan2(R_tb3[1, 0], R_tb3[0, 0])
        return p_tb3[:2], yaw

    def _base_nav_to_target(
        self,
        target_xy,
        desired_yaw=None,
        pos_tol=BASE_AUTO_POS_TOL,
        yaw_tol=BASE_AUTO_YAW_TOL,
        allow_stop=True,
    ):
        """
        Base 自动流程的底层控制器：
        - desired_yaw=None: 朝向目标点导航到目标点
        - desired_yaw!=None: 先对齐到指定朝向，再到目标点
        """
        xy, yaw = self._get_tb3_pose_xy_yaw()
        delta = target_xy - xy
        dist = np.linalg.norm(delta)

        if desired_yaw is None:
            yaw_ref = np.arctan2(delta[1], delta[0]) if dist > 1e-9 else yaw
        else:
            yaw_ref = desired_yaw

        yaw_err = self._wrap_to_pi(yaw_ref - yaw)

        # 角度误差很大时先原地转向，并保证命令足够大以克服静摩擦。
        if abs(yaw_err) > BASE_AUTO_ROTATE_IN_PLACE_TH:
            turn_mag = max(BASE_AUTO_MIN_TURN_CMD, abs(BASE_AUTO_KP_ANG * yaw_err))
            turn = float(np.clip(np.sign(yaw_err) * turn_mag, -BASE_AUTO_MAX_TURN_V, BASE_AUTO_MAX_TURN_V))
            return np.array([-turn, turn], dtype=np.float32), False

        # Some higher-level state machines use stricter rectangular checks
        # than this radial tolerance. Keep moving until the caller decides to stop.
        if allow_stop and dist <= pos_tol and abs(yaw_err) <= yaw_tol:
            return np.zeros(2, dtype=np.float32), True

        # 前进段给一个最小线速度，避免“朝向基本正确但不走”的停滞。
        fwd = float(np.clip(BASE_AUTO_KP_LIN * dist, BASE_AUTO_MIN_FWD_CMD, BASE_AUTO_MAX_FWD_V))

        turn_raw = BASE_AUTO_KP_ANG * yaw_err
        if abs(yaw_err) > yaw_tol:
            turn_mag = max(BASE_AUTO_MIN_TURN_CMD, abs(turn_raw))
            turn = float(np.clip(np.sign(yaw_err) * turn_mag, -BASE_AUTO_MAX_TURN_V, BASE_AUTO_MAX_TURN_V))
        else:
            turn = float(np.clip(turn_raw, -BASE_AUTO_MAX_TURN_V, BASE_AUTO_MAX_TURN_V))
        return np.array([fwd - turn, fwd + turn], dtype=np.float32), False

    def _base_drive_with_heading_hold(self, fwd_v, target_yaw):
        """
        以给定线速度行驶，并用 yaw 闭环做差速纠偏，抑制直线段偏航。
        """
        _, yaw = self._get_tb3_pose_xy_yaw()
        yaw_err = self._wrap_to_pi(target_yaw - yaw)
        if abs(yaw_err) <= BASE_STRAIGHT_YAW_DEADBAND:
            turn = 0.0
        else:
            turn = float(np.clip(BASE_STRAIGHT_KP_YAW * yaw_err, -BASE_STRAIGHT_MAX_TURN_V, BASE_STRAIGHT_MAX_TURN_V))
        return np.array([fwd_v - turn, fwd_v + turn], dtype=np.float32)

    def _set_base_action_intent(self, cmd):
        """记录 base 模式用于写数据集的意图动作（未经过二次纠偏）。"""
        self.base_action_intent = np.array(cmd, dtype=np.float32).copy()

    def get_base_action_intent(self):
        """获取当前 base 模式的意图动作（用于采集脚本存盘）。"""
        return self.base_action_intent.copy()

    def _start_base_auto_parking(self):
        self.base_auto_active = True
        self.base_auto_stage = 'goto_stage1'
        self.base_auto_wait_counter = 0
        self.base_auto_wait_deadline = None
        self.base_auto_stage_steps = 0
        self.base_auto_record_stop_requested = False
        self.base_auto_push_target_yaw = None
        print(
            "\n🚗 [H] Base auto parking started: "
            "goto (0.2,-0.8) -> goto (0.2,-0.25) -> push forward -> wait 3s -> request stop recording"
        )

    def _stop_base_auto_parking(self, reason):
        """停止 Base 自动停车并释放控制权。"""
        self.base_auto_active = False
        self.base_auto_stage = 'idle'
        self.base_auto_wait_counter = 0
        self.base_auto_wait_deadline = None
        self.base_auto_stage_steps = 0
        self.base_auto_record_stop_requested = False
        self.base_auto_push_target_yaw = None
        print(f"🛑 Base auto parking stopped: {reason}")

    def _run_base_auto_parking(self):
        """执行 H 键自动停车流程，返回当前帧 wheel 命令。"""
        if not self.base_auto_active:
            self._set_base_action_intent([0.0, 0.0])
            return np.zeros(2, dtype=np.float32)

        timeout_frames = int(BASE_AUTO_STAGE_TIMEOUT_SEC * 20)

        if self.base_auto_stage == 'goto_stage1':
            self.base_auto_stage_steps += 1
            if self.base_auto_stage_steps > timeout_frames:
                self._stop_base_auto_parking("stage1 timeout")
                self._set_base_action_intent([0.0, 0.0])
                return np.zeros(2, dtype=np.float32)
            cmd, _ = self._base_nav_to_target(
                BASE_AUTO_STAGE1_TARGET_XY,
                desired_yaw=None,
                pos_tol=max(BASE_AUTO_STAGE1_X_TOL, BASE_AUTO_STAGE1_Y_TOL),  # 控制器容差
                yaw_tol=np.pi,  # 第一阶段不要求姿态
                allow_stop=False,  # 由 stage1 的 x/y 阈值决定是否停车，避免提前停住
            )
            xy, _ = self._get_tb3_pose_xy_yaw()
            x_ok = abs(float(xy[0]) - float(BASE_AUTO_STAGE1_TARGET_XY[0])) <= BASE_AUTO_STAGE1_X_TOL
            y_ok = abs(float(xy[1]) - float(BASE_AUTO_STAGE1_TARGET_XY[1])) <= BASE_AUTO_STAGE1_Y_TOL
            if x_ok and y_ok:
                self.base_auto_stage = 'goto_stage2'
                self.base_auto_stage_steps = 0
                print("   ✅ Stage 1 passed near: (0.2, -0.8), switching to stage 2 target tracking...")
            self._set_base_action_intent(cmd)
            return cmd

        if self.base_auto_stage == 'goto_stage2':
            self.base_auto_stage_steps += 1
            if self.base_auto_stage_steps > timeout_frames:
                self._stop_base_auto_parking("stage2 timeout")
                self._set_base_action_intent([0.0, 0.0])
                return np.zeros(2, dtype=np.float32)
            cmd, _ = self._base_nav_to_target(
                BASE_AUTO_STAGE2_TARGET_XY,
                desired_yaw=None,  # 朝向终点，不强制 +Y
                pos_tol=BASE_AUTO_FINAL_POS_TOL,   # 终点更精确
                yaw_tol=BASE_AUTO_FINAL_YAW_TOL,   # 终点朝向也更精确
                allow_stop=False,  # 由 stage2 的 x/y/yaw 条件决定是否停车，避免提前停住
            )
            xy, _ = self._get_tb3_pose_xy_yaw()
            x_ok = abs(float(xy[0]) - float(BASE_AUTO_STAGE2_TARGET_XY[0])) <= BASE_AUTO_FINAL_X_TOL
            # 与桌边接触时常会停在 y≈-0.263，先进入短暂前推阶段再结束。
            y_near = float(xy[1]) >= float(BASE_AUTO_STAGE2_TARGET_XY[1]) - BASE_AUTO_STAGE2_Y_NEAR_MARGIN
            if x_ok and y_near:
                self.base_auto_stage = 'push_forward'
                self.base_auto_wait_deadline = time.monotonic() + BASE_AUTO_PUSH_SEC
                self.base_auto_stage_steps = 0
                _, yaw_now = self._get_tb3_pose_xy_yaw()
                self.base_auto_push_target_yaw = yaw_now
                print(
                    f"   ✅ Stage 2 near target. Pushing forward for {BASE_AUTO_PUSH_SEC:.1f}s "
                    f"to settle against table edge..."
                )
                self._set_base_action_intent([BASE_AUTO_PUSH_FWD_V, BASE_AUTO_PUSH_FWD_V])
                if BASE_STRAIGHT_ASSIST_ENABLED:
                    return self._base_drive_with_heading_hold(BASE_AUTO_PUSH_FWD_V, self.base_auto_push_target_yaw)
                return np.array([BASE_AUTO_PUSH_FWD_V, BASE_AUTO_PUSH_FWD_V], dtype=np.float32)
            self._set_base_action_intent(cmd)
            return cmd

        if self.base_auto_stage == 'push_forward':
            if self.base_auto_wait_deadline is None:
                self.base_auto_wait_deadline = time.monotonic() + BASE_AUTO_PUSH_SEC
            if time.monotonic() >= self.base_auto_wait_deadline:
                self.base_auto_wait_deadline = None
                self.base_auto_push_target_yaw = None
                # 仅在录制时等待 60 帧（约 3 秒@20Hz）；未录制则直接完成自动停车。
                if self.base_auto_recording_active:
                    self.base_auto_stage = 'wait_done'
                    self.base_auto_wait_counter = BASE_AUTO_WAIT_FRAMES
                    self.base_auto_stage_steps = 0
                    print("   ✅ Push-forward done. Waiting 60 frames before finish (recording active)...")
                    self._set_base_action_intent([0.0, 0.0])
                    return np.zeros(2, dtype=np.float32)
                self.base_auto_active = False
                self.base_auto_stage = 'idle'
                self.base_auto_stage_steps = 0
                self.base_auto_record_stop_requested = True
                print("   ✅ Push-forward done. Recording inactive, skip 3s wait.")
                print("   🛑 Base auto parking finished. Stop-recording requested.")
                self._set_base_action_intent([0.0, 0.0])
                return np.zeros(2, dtype=np.float32)
            self._set_base_action_intent([BASE_AUTO_PUSH_FWD_V, BASE_AUTO_PUSH_FWD_V])
            if BASE_STRAIGHT_ASSIST_ENABLED:
                if self.base_auto_push_target_yaw is None:
                    _, yaw_now = self._get_tb3_pose_xy_yaw()
                    self.base_auto_push_target_yaw = yaw_now
                return self._base_drive_with_heading_hold(BASE_AUTO_PUSH_FWD_V, self.base_auto_push_target_yaw)
            return np.array([BASE_AUTO_PUSH_FWD_V, BASE_AUTO_PUSH_FWD_V], dtype=np.float32)

        if self.base_auto_stage == 'wait_done':
            self.base_auto_wait_counter -= 1
            if self.base_auto_wait_counter <= 0:
                self.base_auto_active = False
                self.base_auto_stage = 'idle'
                self.base_auto_stage_steps = 0
                self.base_auto_record_stop_requested = True
                print("   🛑 Base auto parking finished. Stop-recording requested.")
            self._set_base_action_intent([0.0, 0.0])
            return np.zeros(2, dtype=np.float32)

        # 未知状态兜底
        self.base_auto_active = False
        self.base_auto_stage = 'idle'
        self.base_auto_wait_deadline = None
        self._set_base_action_intent([0.0, 0.0])
        return np.zeros(2, dtype=np.float32)

    def _sample_random_init_v1(self):
        """
        🔥 旧版随机初始化：在扇形区域内采样
        
        Returns:
            tuple: (x_target, y_target, z_target) 目标位置坐标
        """
        # 1. 生成随机角度和径向距离（使用V1专用参数：角度0~15度，径向距离0.275~0.325米）
        angle_deg = random.uniform(RANDOM_INIT_ANGLE_MIN_V1, RANDOM_INIT_ANGLE_MAX_V1)
        angle_rad = np.deg2rad(angle_deg)
        radius = random.uniform(RANDOM_INIT_RADIUS_MIN_V1, RANDOM_INIT_RADIUS_MAX_V1)
        
        # 2. 转换为笛卡尔坐标（相对于机械臂基座 ARM_BASE_X, ARM_BASE_Y）
        x_target = ARM_BASE_X + radius * np.cos(angle_rad)
        y_target = ARM_BASE_Y + radius * np.sin(angle_rad)
        # 🔥 使用V1专用Z轴参数：0.95~1.0米（正常初始化高度）
        z_target = random.uniform(RANDOM_INIT_Z_MIN_V1, RANDOM_INIT_Z_MAX_V1)
        
        print(f"🎲 Random Init (V1): pos=({x_target:.3f}, {y_target:.3f}, {z_target:.3f}), angle={angle_deg:.1f}° (range: {RANDOM_INIT_ANGLE_MIN_V1}~{RANDOM_INIT_ANGLE_MAX_V1}°), radius={radius:.3f}m (range: {RANDOM_INIT_RADIUS_MIN_V1}~{RANDOM_INIT_RADIUS_MAX_V1}m), z_range: {RANDOM_INIT_Z_MIN_V1}~{RANDOM_INIT_Z_MAX_V1}m")
        
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

    def reset(self, seed=None, mode=None, force_fixed_arm_init=False):
        if seed is not None: np.random.seed(seed)
        
        # 如果传入了 mode 参数，更新 control_mode（用于 set_instruction 判断）
        if mode is not None:
            self.control_mode = mode
        
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
            # - 开启时：高斯扰动（截断）x = center + N(0, std), clipped to [min, max]
            # - 关闭时：固定 x = center
            if self.tb3_x_gaussian_enabled:
                x_offset = _sample_tb3_x_offset(
                    offset_std=self.tb3_x_offset_std,
                    offset_min=self.tb3_x_offset_min,
                    offset_max=self.tb3_x_offset_max,
                )
                x_init = self.tb3_x_center + x_offset
            else:
                x_offset = 0.0
                x_init = self.tb3_x_center
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
        self.base_auto_active = False
        self.base_auto_stage = 'idle'
        self.base_auto_wait_counter = 0
        self.base_auto_wait_deadline = None
        self.base_auto_stage_steps = 0
        self.base_auto_recording_active = False
        self.base_auto_record_stop_requested = False
        self.base_auto_push_target_yaw = None
        self.base_straight_assist_active = False
        self.base_straight_target_yaw = None
        self.base_straight_last_cmd_sign = 0
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
            
        self.set_instruction()  # 现在会根据 self.control_mode 自动设置正确的任务文本
        # 🔥 gripper_state 已在上面根据 random_init_gripper_open 设置，这里不再重置
        
        # 重置时刷新一次图像缓存
        self.grab_image()

    def _init_objects_demo(self):
        # ====== 使用红色杯子、蓝色杯子和盘子 ======
        
        # 盘子按需求移出场景（等效删除）
        plate_xyz = np.array([HIDDEN_OBJ_X, 0.0, HIDDEN_OBJ_Z])
        self.env.set_p_base_body(body_name='body_obj_plate_11', p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11', R=np.eye(3,3))
        
        # 🔥 红色和蓝色杯子位置随机化（两个弧线段）
        # 弧线段1：距离0.3m，角度0°-45°
        # 弧线段2：距离0.4m，角度0°-30°
        # 红色和蓝色杯子各随机初始化在一个弧线段上，确保间隔>=MUG_MIN_SPACING
        
        max_attempts = 100  # 最大尝试次数，确保找到满足间距要求的位置
        
        # 🔥 固定分配：红色杯子在弧线段1（半径0.3m），蓝色杯子在弧线段2（半径0.4m）
        for attempt in range(max_attempts):
            # 红色杯子固定在弧线段1：距离0.3m，角度0°-45°
            angle_red = np.deg2rad(random.uniform(MUG_ARC1_ANGLE_MIN, MUG_ARC1_ANGLE_MAX))
            mug_red_pos = np.array([
                ARM_BASE_X + MUG_ARC1_RADIUS * np.cos(angle_red),
                ARM_BASE_Y + MUG_ARC1_RADIUS * np.sin(angle_red),
                TABLE_Z_HEIGHT
            ])
            
            # 蓝色杯子固定在弧线段2：距离0.4m，角度0°-30°
            angle_blue = np.deg2rad(random.uniform(MUG_ARC2_ANGLE_MIN, MUG_ARC2_ANGLE_MAX))
            mug_blue_pos = np.array([
                ARM_BASE_X + MUG_ARC2_RADIUS * np.cos(angle_blue),
                ARM_BASE_Y + MUG_ARC2_RADIUS * np.sin(angle_blue),
                TABLE_Z_HEIGHT
            ])
            
            # 检查间距是否满足要求
            spacing = np.linalg.norm(mug_red_pos[:2] - mug_blue_pos[:2])
            if spacing >= MUG_MIN_SPACING:
                break  # 满足要求，退出循环
        
        # 如果所有尝试都不满足要求，使用最后一次尝试的结果（并打印警告）
        if spacing < MUG_MIN_SPACING:
            print(f"⚠️ Warning: Could not find positions with spacing >= {MUG_MIN_SPACING}m after {max_attempts} attempts. Using spacing={spacing:.3f}m")
        
        # 限制在桌面范围内（Clip，留点边缘余量）
        mug_red_pos[0] = np.clip(mug_red_pos[0], TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05)
        mug_red_pos[1] = np.clip(mug_red_pos[1], TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05)
        mug_blue_pos[0] = np.clip(mug_blue_pos[0], TABLE_X_MIN + 0.05, TABLE_X_MAX - 0.05)
        mug_blue_pos[1] = np.clip(mug_blue_pos[1], TABLE_Y_MIN + 0.05, TABLE_Y_MAX - 0.05)
        
        # 设置红色杯子位置
        self.env.set_p_base_body(body_name='body_obj_mug_5', p=mug_red_pos)
        self.env.set_R_base_body(body_name='body_obj_mug_5', R=np.eye(3,3))
        
        # 设置蓝色杯子位置
        self.env.set_p_base_body(body_name='body_obj_mug_6', p=mug_blue_pos)
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
        self.mugs_on_table = ['body_obj_mug_5', 'body_obj_mug_6']
        self.mug_colors_on_table = {'body_obj_mug_5': 'red', 'body_obj_mug_6': 'blue'}

    def set_instruction(self, given=None, task_type=None):
        """
        设置任务指令（支持红色和蓝色杯子）
        
        Parameters:
            given: 手动指定的指令文本
            task_type: 任务类型，可选值:
                - 'nav': 导航任务 (小车移动)
                - 'arm': 机械臂任务 (红色或蓝色杯子放到盘子上)
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
                # Arm 模式：随机选择红色或蓝色杯子
                self.task_type = 'arm'
                
                # 🔥 如果启用了选择偏转角度更小的杯子模式
                if self.select_smaller_angle_mug:
                    try:
                        # 获取两个杯子的位置
                        red_mug_pos = self.env.get_p_body('body_obj_mug_5')
                        blue_mug_pos = self.env.get_p_body('body_obj_mug_6')
                        
                        # 🔥 只计算偏转角度，不计算距离
                        # 计算相对于机械臂基座的位置向量（仅用于计算角度）
                        red_rel_pos = red_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
                        blue_rel_pos = blue_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
                        
                        # 🔥 计算偏转角度（弧度），使用 atan2(y, x)
                        # atan2 返回从 x 轴正方向（正前方）到点 (x,y) 的角度
                        # 范围 [-π, π]，0 表示正前方，正值表示右侧，负值表示左侧
                        red_angle = np.arctan2(red_rel_pos[1], red_rel_pos[0])
                        blue_angle = np.arctan2(blue_rel_pos[1], blue_rel_pos[0])
                        
                        # 🔥 只比较角度绝对值，选择更接近正前方（角度更小）的杯子
                        # 注意：这里只比较角度，不比较距离
                        red_angle_abs = abs(red_angle)
                        blue_angle_abs = abs(blue_angle)
                        
                        if red_angle_abs <= blue_angle_abs:
                            # 红色杯子角度更小（更接近正前方）
                            self.obj_target = 'body_obj_mug_5'
                            self.target_color = 'red'
                            self.instruction = "Place the red mug on the plate."
                        else:
                            # 蓝色杯子角度更小（更接近正前方）
                            self.obj_target = 'body_obj_mug_6'
                            self.target_color = 'blue'
                            self.instruction = "Place the blue mug on the plate."
                    except Exception as e:
                        # 如果获取位置失败，回退到随机选择
                        print(f"⚠️ Warning: Failed to get mug positions for angle selection: {e}. Falling back to random selection.")
                        self.instruction = random.choice(ARM_INSTRUCTIONS)
                        if 'red' in self.instruction.lower():
                            self.obj_target = 'body_obj_mug_5'
                            self.target_color = 'red'
                        elif 'blue' in self.instruction.lower():
                            self.obj_target = 'body_obj_mug_6'
                            self.target_color = 'blue'
                        else:
                            self.obj_target = 'body_obj_mug_5'
                            self.target_color = 'red'
                else:
                    # 默认：随机选择
                    self.instruction = random.choice(ARM_INSTRUCTIONS)
                    # 根据指令内容确定目标物体
                    if 'red' in self.instruction.lower():
                        self.obj_target = 'body_obj_mug_5'
                        self.target_color = 'red'
                    elif 'blue' in self.instruction.lower():
                        self.obj_target = 'body_obj_mug_6'
                        self.target_color = 'blue'
                    else:
                        # 默认使用红色杯子
                        self.obj_target = 'body_obj_mug_5'
                        self.target_color = 'red'
                        print(f"⚠️ Warning: Instruction does not contain 'red' or 'blue'. Using red mug as default.")
        else:
            self.instruction = given
            # 解析 obj_target 和 target_color (支持红色和蓝色杯子)
            if self.control_mode == 'arm' or self.task_type == 'arm':
                # 🔥 如果启用了选择偏转角度更小的杯子模式，忽略指令中的颜色，直接选择角度更小的
                if self.select_smaller_angle_mug:
                    try:
                        # 获取两个杯子的位置
                        red_mug_pos = self.env.get_p_body('body_obj_mug_5')
                        blue_mug_pos = self.env.get_p_body('body_obj_mug_6')
                        
                        # 🔥 只计算偏转角度，不计算距离
                        # 计算相对于机械臂基座的位置向量（仅用于计算角度）
                        red_rel_pos = red_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
                        blue_rel_pos = blue_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
                        
                        # 🔥 计算偏转角度（弧度），使用 atan2(y, x)
                        # atan2 返回从 x 轴正方向（正前方）到点 (x,y) 的角度
                        # 范围 [-π, π]，0 表示正前方，正值表示右侧，负值表示左侧
                        red_angle = np.arctan2(red_rel_pos[1], red_rel_pos[0])
                        blue_angle = np.arctan2(blue_rel_pos[1], blue_rel_pos[0])
                        
                        # 🔥 只比较角度绝对值，选择更接近正前方（角度更小）的杯子
                        # 注意：这里只比较角度，不比较距离
                        red_angle_abs = abs(red_angle)
                        blue_angle_abs = abs(blue_angle)
                        
                        if red_angle_abs <= blue_angle_abs:
                            # 红色杯子角度更小（更接近正前方）
                            self.obj_target = 'body_obj_mug_5'
                            self.target_color = 'red'
                            # 更新指令文本以匹配选择
                            self.instruction = "Place the red mug on the plate."
                        else:
                            # 蓝色杯子角度更小（更接近正前方）
                            self.obj_target = 'body_obj_mug_6'
                            self.target_color = 'blue'
                            # 更新指令文本以匹配选择
                            self.instruction = "Place the blue mug on the plate."
                    except Exception as e:
                        # 如果获取位置失败，回退到按指令解析
                        print(f"⚠️ Warning: Failed to get mug positions for angle selection: {e}. Falling back to instruction parsing.")
                        if 'red' in self.instruction.lower():
                            self.obj_target = 'body_obj_mug_5'
                            self.target_color = 'red'
                        elif 'blue' in self.instruction.lower():
                            self.obj_target = 'body_obj_mug_6'
                            self.target_color = 'blue'
                        else:
                            self.obj_target = 'body_obj_mug_5'
                            self.target_color = 'red'
                            print(f"⚠️ Warning: Instruction does not contain 'red' or 'blue'. Using red mug as default.")
                else:
                    # 默认：按指令内容解析
                    if 'red' in self.instruction.lower():
                        self.obj_target = 'body_obj_mug_5'
                        self.target_color = 'red'
                    elif 'blue' in self.instruction.lower():
                        self.obj_target = 'body_obj_mug_6'
                        self.target_color = 'blue'
                    else:
                        # 默认使用红色杯子
                        self.obj_target = 'body_obj_mug_5'
                        self.target_color = 'red'
                        print(f"⚠️ Warning: Instruction does not contain 'red' or 'blue'. Using red mug as default.")
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
        
        # 获取小车当前位置：放置目标的 XY 基准改为小车坐标（非盘子）
        try:
            tb3_pos = self.env.get_p_body('tb3_base')
        except:
            print("⚠️ Cannot find tb3_base body.")
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
        
        # 🔥 强制分阶段运动：接近阶段始终先 XY 对齐，再进行后续下降
        # 做法：悬停点 Z 固定为 lift_end_pos 的 Z，保证 lift_end_pos -> hover_pos 为恒定 Z 的平面移动
        # NOTE: z_travel 仍用于后续运输与放置阶段
        adjusted_hover_z = lift_end_pos[2]
        
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
        # 🔥 悬停点 Z 与 lift_end_pos 一致，保证第一段接近只在 XY 平面运动
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
        
        # 放置点：
        # - XY：使用当前小车坐标作为目标中心
        # - Y：继续叠加放置偏移
        # - Z：保持现有高度逻辑不变
        y_place_offset = 0.25 * EXPERT_Y_PLACE_OFFSET  # 放到小车上时，Y偏置减半
        
        place_pos = np.array([
            tb3_pos[0],
            tb3_pos[1] + y_place_offset,
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
        # 🔥 检查选中的杯子是否是角度较小的那一个，如果是则使用更快的接近速度
        approach_speed = EXPERT_SPEED_APPROACH  # 默认接近速度
        try:
            # 获取两个杯子的位置
            red_mug_pos = self.env.get_p_body('body_obj_mug_5')
            blue_mug_pos = self.env.get_p_body('body_obj_mug_6')
            
            # 计算相对于机械臂基座的位置向量（仅用于计算角度）
            red_rel_pos = red_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
            blue_rel_pos = blue_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
            
            # 计算偏转角度（弧度），使用 atan2(y, x)
            red_angle = np.arctan2(red_rel_pos[1], red_rel_pos[0])
            blue_angle = np.arctan2(blue_rel_pos[1], blue_rel_pos[0])
            
            # 只比较角度绝对值，选择更接近正前方（角度更小）的杯子
            red_angle_abs = abs(red_angle)
            blue_angle_abs = abs(blue_angle)
            
            # 判断哪个杯子角度更小
            smaller_angle_mug = 'red' if red_angle_abs <= blue_angle_abs else 'blue'
            
            # 检查选中的杯子是否是角度较小的那一个
            if hasattr(self, 'target_color') and self.target_color == smaller_angle_mug:
                # 选中的是角度较小的杯子，使用更快的接近速度
                approach_speed = EXPERT_SPEED_APPROACH_SMALLER_ANGLE
                print(f"   ⚡ Selected mug ({self.target_color}) is the smaller angle mug. Using faster approach speed: {approach_speed:.3f} m/step")
        except Exception as e:
            # 如果获取位置失败，使用默认速度
            print(f"   ⚠️ Warning: Failed to check mug angles for speed adjustment: {e}. Using default approach speed.")
        
        # 计算各阶段步数
        approach_steps = self.distance_based_steps(lift_end_pos, hover_pos, approach_speed)
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
        color_name = self.target_color if hasattr(self, 'target_color') else 'unknown'
        print(f"   Target object: {self.obj_target} ({color_name} mug) -> Plate")
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
            if self.env.is_key_pressed_once(key=glfw.KEY_H):
                if not self.base_auto_active:
                    self._start_base_auto_parking()
                else:
                    self._stop_base_auto_parking("manual cancel by [H]")

            if self.base_auto_active:
                return self._run_base_auto_parking(), reset

            v_move = 15.0; v_turn = 6.0
            w_pressed = self.env.is_key_pressed_repeat(key=glfw.KEY_W)
            s_pressed = self.env.is_key_pressed_repeat(key=glfw.KEY_S)
            a_pressed = self.env.is_key_pressed_repeat(key=glfw.KEY_A)
            d_pressed = self.env.is_key_pressed_repeat(key=glfw.KEY_D)

            # 与历史手感保持一致：A/D 具有更高优先级，可直接覆盖 W/S 直行。
            if a_pressed and not d_pressed:
                self.base_straight_assist_active = False
                self.base_straight_target_yaw = None
                self.base_straight_last_cmd_sign = 0
                wheel_v = np.array([-v_turn, v_turn], dtype=np.float32)
                self._set_base_action_intent(wheel_v)
                return wheel_v, reset
            if d_pressed and not a_pressed:
                self.base_straight_assist_active = False
                self.base_straight_target_yaw = None
                self.base_straight_last_cmd_sign = 0
                wheel_v = np.array([v_turn, -v_turn], dtype=np.float32)
                self._set_base_action_intent(wheel_v)
                return wheel_v, reset

            # 直行段：锁定按键瞬间航向并持续闭环纠偏。
            if w_pressed and not s_pressed:
                self._set_base_action_intent([v_move, v_move])
                if (
                    not BASE_STRAIGHT_ASSIST_ENABLED
                    or not self.base_straight_assist_active
                    or self.base_straight_last_cmd_sign != 1
                ):
                    _, yaw_now = self._get_tb3_pose_xy_yaw()
                    self.base_straight_target_yaw = yaw_now
                    self.base_straight_assist_active = True
                    self.base_straight_last_cmd_sign = 1
                if BASE_STRAIGHT_ASSIST_ENABLED:
                    return self._base_drive_with_heading_hold(v_move, self.base_straight_target_yaw), reset
                return np.array([v_move, v_move], dtype=np.float32), reset

            if s_pressed and not w_pressed:
                self._set_base_action_intent([-6.0, -6.0])
                if (
                    not BASE_STRAIGHT_ASSIST_ENABLED
                    or not self.base_straight_assist_active
                    or self.base_straight_last_cmd_sign != -1
                ):
                    _, yaw_now = self._get_tb3_pose_xy_yaw()
                    self.base_straight_target_yaw = yaw_now
                    self.base_straight_assist_active = True
                    self.base_straight_last_cmd_sign = -1
                if BASE_STRAIGHT_ASSIST_ENABLED:
                    return self._base_drive_with_heading_hold(-6.0, self.base_straight_target_yaw), reset
                return np.array([-6.0, -6.0], dtype=np.float32), reset

            # 退出直行段或手动转向时，释放航向锁定。
            self.base_straight_assist_active = False
            self.base_straight_target_yaw = None
            self.base_straight_last_cmd_sign = 0

            wheel_v = np.zeros(2, dtype=np.float32)
            self._set_base_action_intent(wheel_v)

            return wheel_v, reset

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
            p_tb3 = self.env.get_p_body('tb3_base')
            # 与放置阶段保持一致：目标点 = (tb3_x, tb3_y + y_place_offset)
            y_place_offset = 0.25 * EXPERT_Y_PLACE_OFFSET
            target_xy = np.array([p_tb3[0], p_tb3[1] + y_place_offset], dtype=np.float32)
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
        # Clear previous-frame RGB overlays first, otherwise mode switch can leave
        # stale windows (e.g. base top-left camera remains in arm mode).
        #
        # NOTE: overlay buffers are stored on viewer (self.env.viewer), not env.
        viewer = getattr(self.env, 'viewer', None)
        if viewer is not None:
            # Compatibility: some viewer versions expose reset_rgb_overlay(),
            # some only expose raw buffer attributes.
            if hasattr(viewer, 'reset_rgb_overlay'):
                viewer.reset_rgb_overlay()
            for attr in (
                'rgb_overlay_top_right',
                'rgb_overlay_top_left',
                'rgb_overlay_bottom_right',
                'rgb_overlay_bottom_left',
                'rgb_overlay',  # legacy single-overlay buffer
            ):
                if hasattr(viewer, attr):
                    setattr(viewer, attr, None)
        
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
            # 显示桌上存在的物体（红色杯子和蓝色杯子）
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


# Backward-compatible alias for code that still imports SimpleEnv4 from this file.
SimpleEnv4 = SimpleEnv6
SimpleEnv7 = SimpleEnv6