import random

import numpy as np

from .env_constants import (
    ARM_BASE_X,
    ARM_BASE_Y,
    HIDDEN_OBJ_X,
    HIDDEN_OBJ_Y_INTERVAL,
    HIDDEN_OBJ_Z,
    MUG_ARC1_ANGLE_MAX,
    MUG_ARC1_ANGLE_MIN,
    MUG_ARC1_RADIUS,
    MUG_ARC2_ANGLE_MAX,
    MUG_ARC2_ANGLE_MIN,
    MUG_ARC2_RADIUS,
    MUG_MIN_SPACING,
    RANDOM_INIT_ANGLE_MAX,
    RANDOM_INIT_ANGLE_MAX_V1,
    RANDOM_INIT_ANGLE_MIN,
    RANDOM_INIT_ANGLE_MIN_V1,
    RANDOM_INIT_CIRCLE_INNER_RADIUS,
    RANDOM_INIT_CIRCLE_OUTER_RADIUS,
    RANDOM_INIT_RADIUS_MAX,
    RANDOM_INIT_RADIUS_MAX_V1,
    RANDOM_INIT_RADIUS_MIN,
    RANDOM_INIT_RADIUS_MIN_V1,
    RANDOM_INIT_Z_MAX_V1,
    RANDOM_INIT_Z_MAX_V2,
    RANDOM_INIT_Z_MIN_V1,
    RANDOM_INIT_Z_MIN_V2,
    TABLE_X_MAX,
    TABLE_X_MIN,
    TABLE_Y_MAX,
    TABLE_Y_MIN,
    TABLE_Z_HEIGHT,
)
from .transforms import rpy2r


class SceneInitMixin:
    """Mixin providing random-init sampling and object-placement logic."""

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

        mug_red_R = np.eye(3,3)
        mug_blue_R = np.eye(3,3)
        # 记录"桌面参考坐标"：即该颜色杯子在非托盘初始化时的桌面位置
        # 后续若该杯子被放到托盘上，专家策略可复用该桌面坐标作为放置终点。
        self.table_reference_positions = {
            'red': mug_red_pos.copy(),
            'blue': mug_blue_pos.copy(),
        }

        # 🔥 新初始化逻辑（仅 Arm 模式）：根据当前任务，将对应颜色的杯子初始化到托盘上
        desired_tray_color = getattr(self, '_active_tray_init_color', None)
        if self.control_mode == 'arm' and desired_tray_color in ('red', 'blue'):
            try:
                tb3_pos = self.env.get_p_body('tb3_base')
            except Exception:
                tb3_pos = None
                print("⚠️ [Init] Cannot find tb3_base body, skip place-endpoint mug initialization.")

            if tb3_pos is not None:
                # 以你配置后的托盘杯子中心点为圆心，在半径 3cm 圆内随机采样初始化点
                mug_center_pos = np.array([tb3_pos[0], tb3_pos[1] - 0.06, 0.49], dtype=np.float32)
                sampled_xy = self.sample_point_in_circle(mug_center_pos[:2], radius=0.03)
                mug_at_tb3_pos = np.array([sampled_xy[0], sampled_xy[1], mug_center_pos[2]], dtype=np.float32)
                tray_mug_yaw = random.uniform(-np.pi, np.pi)
                tray_mug_R = rpy2r(np.array([0.0, 0.0, tray_mug_yaw]))
                if desired_tray_color == 'red':
                    self.tray_initialized_color = 'red'
                    self.tray_initialized_body = 'body_obj_mug_5'
                    mug_red_pos = mug_at_tb3_pos.copy()
                    mug_red_R = tray_mug_R
                    print(
                        f"🎯 [Init] Red mug initialized on tray circle (r<=3cm): "
                        f"({mug_at_tb3_pos[0]:.3f}, {mug_at_tb3_pos[1]:.3f}, {mug_at_tb3_pos[2]:.3f}), "
                        f"yaw={np.rad2deg(tray_mug_yaw):+.1f}°"
                    )
                else:
                    self.tray_initialized_color = 'blue'
                    self.tray_initialized_body = 'body_obj_mug_6'
                    mug_blue_pos = mug_at_tb3_pos.copy()
                    mug_blue_R = tray_mug_R
                    print(
                        f"🎯 [Init] Blue mug initialized on tray circle (r<=3cm): "
                        f"({mug_at_tb3_pos[0]:.3f}, {mug_at_tb3_pos[1]:.3f}, {mug_at_tb3_pos[2]:.3f}), "
                        f"yaw={np.rad2deg(tray_mug_yaw):+.1f}°"
                    )
        
        # 设置红色杯子位置
        self.env.set_p_base_body(body_name='body_obj_mug_5', p=mug_red_pos)
        self.env.set_R_base_body(body_name='body_obj_mug_5', R=mug_red_R)
        
        # 设置蓝色杯子位置
        self.env.set_p_base_body(body_name='body_obj_mug_6', p=mug_blue_pos)
        self.env.set_R_base_body(body_name='body_obj_mug_6', R=mug_blue_R)
        
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
