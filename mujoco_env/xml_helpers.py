import os
import xml.etree.ElementTree as ET
import numpy as np

from .env_constants import (
    V6_Z_OFFSET,
    V7_DOCK_X_MIN,
    V7_DOCK_X_MAX,
    V7_TABLE_Y_NEG_EXTENSION,
    V7_TABLE_HALF_Z,
    V6_TABLE_Y_MIN,
    TB3_X_MIN,
    TB3_X_MAX
)

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


def _sample_tb3_x_uniform(x_min=TB3_X_MIN, x_max=TB3_X_MAX):
    """在 [x_min, x_max] 区间上均匀采样 TB3 的初始 x。"""
    x_low = float(min(x_min, x_max))
    x_high = float(max(x_min, x_max))
    return float(np.random.uniform(x_low, x_high))


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
