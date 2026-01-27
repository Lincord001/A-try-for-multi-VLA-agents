import sys
import random
import numpy as np
import copy
import glfw

# 确保这些模块在您的 mujoco_env 文件夹下存在
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy

class AlohaEnv:
    def __init__(self, 
                 xml_path='./asset/aloha/scene.xml',
                 action_type='eef_pose', 
                 state_type='joint_angle',
                 seed=None):
        """
        初始化 Aloha 2 仿真环境。
        """
        # ---------------------------------
        # 1. 初始化 MuJoCo 模型
        # ---------------------------------
        self.xml_path = xml_path
        self.env = MuJoCoParserClass(name='Aloha_Bimanual', rel_xml_path=self.xml_path)
        self.action_type = action_type
        self.state_type = state_type
        
        # ---------------------------------
        # 2. 关节与硬件配置
        # ---------------------------------
        # === 右臂 (Active Agent) ===
        self.right_joint_names = [
            'right/waist', 'right/shoulder', 'right/elbow', 
            'right/forearm_roll', 'right/wrist_angle', 'right/wrist_rotate'
        ]
        self.right_gripper_name = 'right/left_finger'
        self.right_eef_name = 'right/gripper_link'

        # === 左臂 (Passive / Stow) ===
        self.left_joint_names = [
            'left/waist', 'left/shoulder', 'left/elbow', 
            'left/forearm_roll', 'left/wrist_angle', 'left/wrist_rotate'
        ]
        self.left_gripper_name = 'left/left_finger'
        
        # 定义左臂的待机姿态 (弧度)
        self.left_stow_q = np.array([0, -0.96, 1.16, 0, -0.3, 0])

        # === 摄像头 ===
        self.cam_names = {
            'agent': 'overhead_cam',
            'ego': 'wrist_cam_right',
            'left': 'wrist_cam_left'
        }
        
        self.obj_names = {
            'red_mug': 'body_obj_mug_5',
            'blue_mug': 'body_obj_mug_6',
            'plate': 'body_obj_plate_11'
        }

        self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        """初始化 Viewer"""
        self.env.reset()
        self.env.init_viewer(
            distance=1.5,
            elevation=-30, 
            azimuth=90,
            transparent=False,
            black_sky=False,
            use_rgb_overlay=False,
        )

    def _get_qpos_addr(self, joint_name):
        """
        [辅助函数] 获取关节在 qpos 数组中的索引地址
        使用 mujoco_parser 提供的 get_idxs_fwd 方法
        """
        return self.env.get_idxs_fwd([joint_name])[0]

    def _get_actuator_id(self, joint_name):
        """
        [辅助函数] 根据关节名称查找对应的 Actuator ID
        """
        try:
            # ctrl_qpos_names 存储了每个 actuator 控制的关节名称
            return self.env.ctrl_qpos_names.index(joint_name)
        except ValueError:
            return -1

    def reset(self, seed=None):
        """重置环境，将右臂初始化为与左臂相同的自然姿态"""
        if seed is not None: 
            np.random.seed(seed)
            random.seed(seed)
        
        # ---------------------------------------------------------
        # 1. 直接定义理想的关节角度 (使用左臂的待机数据)
        # ---------------------------------------------------------
        # [腰, 肩, 肘, 前臂旋转, 手腕角度, 手腕旋转]
        # 这个姿态就是你看到的那个“蛮好的”姿态
        q_right_init = np.array([0, -0.96, 1.16, 0, -0.3, 0])

        # ---------------------------------------------------------
        # 2. 将关节角度直接写入物理引擎 (qpos)
        # ---------------------------------------------------------
        # 设置右臂 (Active)
        right_idxs = self.env.get_idxs_fwd(self.right_joint_names)
        self.env.data.qpos[right_idxs] = q_right_init
        
        # 设置右夹爪 (Open = 0.037)
        rg_addr = self._get_qpos_addr(self.right_gripper_name)
        self.env.data.qpos[rg_addr] = 0.037

        # 设置左臂 (Stow - 保持不动)
        left_idxs = self.env.get_idxs_fwd(self.left_joint_names)
        self.env.data.qpos[left_idxs] = self.left_stow_q
            
        # 设置左夹爪 (Closed = 0.002)
        lg_addr = self._get_qpos_addr(self.left_gripper_name)
        self.env.data.qpos[lg_addr] = 0.002
        # ---------------------------------------------------------
        # 【新增】初始化物体位置 (适配 Aloha 桌面的坐标系)
        # ---------------------------------------------------------
        # Aloha 桌面中心大约在 (0, -0.3, 0) 附近
        # 盘子位置 (放在两个机械臂中间稍微靠前的位置)
        # Z 轴高度设为 0.0 (桌面)，原项目是 0.83，这里必须改，否则会浮空
        plate_pos = np.array([0.0, -0.4, 0.0]) 
        self.env.set_p_base_body(body_name=self.obj_names['plate'], p=plate_pos)
        self.env.set_R_base_body(body_name=self.obj_names['plate'], R=np.eye(3))

        # 红色马克杯位置 (随机偏移或固定)
        mug1_pos = np.array([-0.2, -0.4, 0.0])
        self.env.set_p_base_body(body_name=self.obj_names['red_mug'], p=mug1_pos)
        self.env.set_R_base_body(body_name=self.obj_names['red_mug'], R=np.eye(3))

        # 蓝色马克杯位置
        mug2_pos = np.array([0.2, -0.4, 0.0])
        self.env.set_p_base_body(body_name=self.obj_names['blue_mug'], p=mug2_pos)
        self.env.set_R_base_body(body_name=self.obj_names['blue_mug'], R=np.eye(3))

        # ---------------------------------------------------------
        # 3. 关键：运行正向运动学 (Forward Kinematics)
        # ---------------------------------------------------------
        # 让 MuJoCo 根据刚才设置的关节角度，算出末端执行器(手)具体在哪儿
        self.env.forward(increase_tick=False)

        # ---------------------------------------------------------
        # 4. 反向初始化 IK 目标
        # ---------------------------------------------------------
        # 我们不需要猜坐标了，直接问环境：“我现在手在哪？”
        # 然后把这个位置设为 IK 控制器的起始目标点 (p0, R0)
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.right_eef_name)

        # ---------------------------------------------------------
        # 5. 初始化内部变量
        # ---------------------------------------------------------
        self.last_q = q_right_init.copy()
        self.compute_q = q_right_init.copy() 
        self.gripper_state = True 
        
        # 预热控制器，确保稳定
        self.q = self._get_full_ctrl_vector(q_right_init, 0.037) 
        for _ in range(50):
            self.step_env()
        
        self.instruction = "Task: Place the Red Mug on the Plate" 
        print("DONE INITIALIZATION (Aloha Env - Natural Pose)")
        return self.get_joint_state()
    
    def check_success(self):
        """
        检查红色杯子是否在盘子上
        """
        p_mug = self.env.get_p_body(self.obj_names['red_mug'])
        p_plate = self.env.get_p_body(self.obj_names['plate'])
        
        # 距离判断：XY平面距离小于 10cm，且杯子高度 Z 类似（或者比盘子略高）
        dist = np.linalg.norm(p_mug[:2] - p_plate[:2])
        if dist < 0.1 and p_mug[2] > p_plate[2]: 
            return True
        return False
    
    def _get_full_ctrl_vector(self, right_arm_q, right_gripper_val):
        """
        组装完整控制向量
        """
        ctrl = np.zeros(self.env.model.nu)
        
        # 1. 右臂
        for i, name in enumerate(self.right_joint_names):
            act_id = self._get_actuator_id(name)
            if act_id != -1:
                ctrl[act_id] = right_arm_q[i]
        
        # 2. 右夹爪
        rg_act_id = self._get_actuator_id(self.right_gripper_name)
        if rg_act_id != -1:
            ctrl[rg_act_id] = right_gripper_val

        # 3. 左臂
        for i, name in enumerate(self.left_joint_names):
            act_id = self._get_actuator_id(name)
            if act_id != -1:
                ctrl[act_id] = self.left_stow_q[i]
        
        # 4. 左夹爪
        lg_act_id = self._get_actuator_id(self.left_gripper_name)
        if lg_act_id != -1:
            ctrl[lg_act_id] = 0.002
            
        return ctrl

    def step(self, action):
        """
        action: [dx, dy, dz, dr, dp, dy, gripper]
        """
        # 1. IK 解算 (右臂)
        if self.action_type == 'eef_pose':
            q_current = self.env.get_qpos_joints(joint_names=self.right_joint_names)
            
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            
            q_solved, _, _ = solve_ik(
                env=self.env,
                joint_names_for_ik=self.right_joint_names,
                body_name_trgt=self.right_eef_name,
                q_init=q_current,
                p_trgt=self.p0,
                R_trgt=self.R0,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False
            )
            self.compute_q = q_solved 
        
        elif self.action_type == 'joint_angle':
            self.compute_q = action[:-1]
        else:
            raise ValueError('Unsupported action type')

        # 2. 夹爪
        gripper_ctrl = 0.037 if action[-1] > 0.5 else 0.002 
        
        # 3. 组装信号
        self.q = self._get_full_ctrl_vector(self.compute_q, gripper_ctrl)
        
        # 4. 步进
        self.env.step(self.q) 
        
        return self.get_joint_state()

    def step_env(self):
        if hasattr(self, 'q'):
            self.env.step(self.q)
        else:
            self.env.step(np.zeros(self.env.model.nu))
    
    def get_joint_state(self):
        qpos = self.env.get_qpos_joints(joint_names=self.right_joint_names)
        gripper_qpos = self.env.get_qpos_joint(self.right_gripper_name)[0]
        gripper_state = 1.0 if gripper_qpos > 0.015 else 0.0
        return np.concatenate([qpos, [gripper_state]], dtype=np.float32)

    def teleop_robot(self):
        dpos = np.zeros(3)
        drot = np.eye(3)
        reset = False

        if self.env.is_key_pressed_repeat(key=glfw.KEY_S): dpos[0] += 0.005
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W): dpos[0] -= 0.005
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A): dpos[1] -= 0.005
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D): dpos[1] += 0.005
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R): dpos[2] += 0.005
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F): dpos[2] -= 0.005

        angle_speed = 0.05
        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):  drot = drot @ rotation_matrix(angle=angle_speed, direction=[0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT): drot = drot @ rotation_matrix(angle=-angle_speed, direction=[0, 1, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):    drot = drot @ rotation_matrix(angle=-angle_speed, direction=[1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):  drot = drot @ rotation_matrix(angle=angle_speed, direction=[1, 0, 0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):     drot = drot @ rotation_matrix(angle=angle_speed, direction=[0, 0, 1])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):     drot = drot @ rotation_matrix(angle=-angle_speed, direction=[0, 0, 1])[:3, :3]

        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            reset = True
            return np.zeros(7, dtype=np.float32), reset
        
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state

        drot_rpy = r2rpy(drot)
        
        action = np.concatenate([
            dpos, 
            drot_rpy, 
            np.array([1.0 if self.gripper_state else 0.0], dtype=np.float32)
        ])
        return action, reset

    def grab_image(self):
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name=self.cam_names['agent'])
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name=self.cam_names['ego'])
        self.rgb_left = self.env.get_fixed_cam_rgb(cam_name=self.cam_names['left'])

        return self.rgb_agent, self.rgb_ego, self.rgb_left

    def render(self, teleop=False, idx=0):
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name=self.right_eef_name)
        #self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95, 0.05, 0.05, 0.5])
        
        if hasattr(self, 'rgb_agent'):
            rgb_agent_view = add_title_to_img(self.rgb_agent, text='Agent View', shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_agent_view, loc='top right')
        
        if hasattr(self, 'rgb_ego'):
            rgb_ego_view = add_title_to_img(self.rgb_ego, text='Wrist View', shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_ego_view, loc='bottom right')

        if hasattr(self, 'rgb_left'):
            rgb_left_view = add_title_to_img(self.rgb_left, text='Left Wrist', shape=(640, 480))
            self.env.viewer_rgb_overlay(rgb_left_view, loc='bottom left')
        
        if teleop:
            self.env.viewer_text_overlay(text1='Key Pressed', text2=str(self.env.get_key_pressed_list()))
        
        if hasattr(self, 'instruction'):
            self.env.viewer_text_overlay(text1='Instruction', text2=self.instruction)
            
        self.env.render()

    def check_success(self):
        return False
        
    def set_instruction(self, instruction):
        self.instruction = instruction
