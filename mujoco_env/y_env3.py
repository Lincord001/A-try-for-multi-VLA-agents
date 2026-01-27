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

# 任务：从厨房 (Kitchen) -> 工作台 (Workbench/Arm)
# 核心语义：移动、工作台、机械臂、左侧房间

NAV_INSTRUCTIONS = [
    # 基础指令 (Basic)
    "Go to the workbench.",
    "Navigate to the workbench.",
    "Drive to the workbench.",
    "Move to the workbench."
    
]

class SimpleEnv3:
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
        
        # 机械臂归位
        q_init = np.deg2rad([0,0,0,0,0,0])
        q_zero, _, _ = solve_ik(self.env, self.joint_names, 'tcp_link', q_init, np.array([0.3,0.0,1.0]), rpy2r(np.deg2rad([90,-0.,90 ])))
        self.env.forward(q=q_zero, joint_names=self.joint_names, increase_tick=False)

        try:
            self.env.set_pR_base_body(
                body_name='tb3_base',
                p=np.array([0.0, 3.0, 0.0]), # 比如放在 x=-1 的位置
                R=np.eye(3)
            )
        except:
            print("Warning: Could not reset TB3 base pose.")
        
        # 状态重置
        self.last_q = copy.deepcopy(q_zero)
        self.current_arm_q = np.concatenate([q_zero, np.array([0.0]*4)]) 
        self.current_wheel_vel = np.zeros(2)
        self.p0, self.R0 = self.env.get_pR_body(body_name='tcp_link')

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
        # 初始化盘子
        plate_xyz = np.array([0, -0.45, 0.82])
        self.env.set_p_base_body(body_name='body_obj_plate_11', p=plate_xyz)
        self.env.set_R_base_body(body_name='body_obj_plate_11', R=np.eye(3,3))
        
        # 初始化两个杯子（使用与V2版本相同的位置范围）
        # mug_5 (红色杯子) - V2版本位置: x_range=[+0.32,+0.33], y_range=[-0.00,+0.02], z=0.83
        mug5_xyzs = sample_xyzs(
            1,
            x_range=[+0.32, +0.33],
            y_range=[-0.00, +0.02],
            z_range=[0.83, 0.83],
            min_dist=0.16,
            xy_margin=0.0
        )
        self.env.set_p_base_body(body_name='body_obj_mug_5', p=mug5_xyzs[0,:])
        self.env.set_R_base_body(body_name='body_obj_mug_5', R=np.eye(3,3))
        
        # mug_6 (蓝色杯子) - V2版本位置: x_range=[+0.29,+0.3], y_range=[0.19,+0.21], z=0.83
        mug6_xyzs = sample_xyzs(
            1,
            x_range=[+0.29, +0.3],
            y_range=[0.19, +0.21],
            z_range=[0.83, 0.83],
            min_dist=0.16,
            xy_margin=0.0
        )
        self.env.set_p_base_body(body_name='body_obj_mug_6', p=mug6_xyzs[0,:])
        self.env.set_R_base_body(body_name='body_obj_mug_6', R=np.eye(3,3))

    def set_instruction(self, given=None):
        if given is None:
            # 根据控制模式设置不同的任务文本
            if self.control_mode == 'base':
                # Base 模式：从 NAV_INSTRUCTIONS 中随机选择一条不同于当前指令的新指令
                available_instructions = [inst for inst in NAV_INSTRUCTIONS 
                                         if inst != self.current_nav_instruction]
                # 如果所有指令都相同（或只有一条指令），则允许重复
                if len(available_instructions) == 0:
                    available_instructions = NAV_INSTRUCTIONS
                
                self.instruction = random.choice(available_instructions)
                self.current_nav_instruction = self.instruction
                # Base 模式下不需要 obj_target
            else:
                # Arm 模式：随机选择红色或蓝色杯子
                target_name = random.choice(['mug_5', 'mug_6'])
                color = 'red' if target_name == 'mug_5' else 'blue'
                self.instruction = f'Place the {color} mug on the plate.'
                self.obj_target = f'body_obj_{target_name}'
        else:
            self.instruction = given
            # 只有在 arm 模式下才需要解析 obj_target
            if self.control_mode == 'arm':
                if 'red' in self.instruction.lower() or 'mug_5' in self.instruction:
                    self.obj_target = 'body_obj_mug_5'
                elif 'blue' in self.instruction.lower() or 'mug_6' in self.instruction:
                    self.obj_target = 'body_obj_mug_6'
            elif self.control_mode == 'base':
                # 如果手动设置了 base 模式的指令，也更新 current_nav_instruction
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

    def teleop_robot(self, mode='arm'):
        dpos = np.zeros(3)
        drot = np.eye(3)
        wheel_v = np.zeros(2)
        reset = False
        
        if self.env.is_key_pressed_once(key=glfw.KEY_Z): reset = True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE): self.gripper_state = not self.gripper_state

        if mode == 'arm':
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
        p_obj = self.env.get_p_body(self.obj_target)
        p_plate = self.env.get_p_body('body_obj_plate_11')
        if np.linalg.norm(p_obj[:2]-p_plate[:2]) < 0.1 and np.linalg.norm(p_obj[2]-p_plate[2]) < 0.1:
            if self.env.get_p_body('tcp_link')[2] > 0.9: return True
        return False

    def get_obj_pose(self):
        # 返回两个杯子和盘子的位置
        p_mug5 = self.env.get_p_body('body_obj_mug_5')
        p_mug6 = self.env.get_p_body('body_obj_mug_6')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return (p_mug5, p_mug6, p_plate)
    
    def grab_image(self):
        # 初始化返回字典
        images = {}

        # 1. 获取 Agent View (作为 Arm 模式的默认或 Fallback)
        rgb_agent_raw = self.env.get_fixed_cam_rgb(cam_name='agentview')

        if self.control_mode == 'arm':
            # === 机械臂模式 ===
            self.rgb_agent = rgb_agent_raw 
            self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric') 
            
            # 存入字典（用于保存数据）
            images['agent'] = self.rgb_agent
            images['wrist'] = self.rgb_ego
            
            # 侧视图 (仅用于渲染辅助)
            self.rgb_side = self.env.get_fixed_cam_rgb(cam_name='sideview')
            
            # 设置渲染标题
            self.agent_title = 'Agent View'
            self.ego_title = 'Wrist View'
            
        else:
            # === Base (小车) 模式 ===
            try:
                # 1. 获取三个车载摄像头原始数据
                self.rgb_front = self.env.get_fixed_cam_rgb(cam_name='tb3_view')
                self.rgb_left = self.env.get_fixed_cam_rgb(cam_name='tb3_left')
                self.rgb_right = self.env.get_fixed_cam_rgb(cam_name='tb3_right')
                
                # 存入字典（用于保存数据）
                images['front'] = self.rgb_front
                images['left'] = self.rgb_left
                images['right'] = self.rgb_right
                
            except Exception as e:
                print(f"Error grabbing TB3 cameras: {e}")
                # Fallback 防止报错
                self.rgb_front = rgb_agent_raw
                self.rgb_left = rgb_agent_raw
                self.rgb_right = rgb_agent_raw
                images = {'front': rgb_agent_raw, 'left': rgb_agent_raw, 'right': rgb_agent_raw}

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
            
            # 3. 左上角: 侧视图 (仅 Teleop 时)
            if teleop and hasattr(self, 'rgb_side'):
                rgb_side_view = add_title_to_img(self.rgb_side, text='Side View', shape=(640,480))
                self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')

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
                
        self.env.render()