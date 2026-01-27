import sys
import numpy as np
import os
import zmq
import pickle
import time
import mujoco
import mujoco.viewer 
from PIL import Image

# 相对路径配置
XML_PATH = './asset/aloha/scene.xml'

class AlohaZMQEnv:
    def __init__(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")
        
        print(f"🚀 Loading model from {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        
        # =========================================================
        # [关键修复] 动态修改模型的离屏渲染缓冲区大小
        # 将其设得比 1280x720 大，防止 ValueError
        # =========================================================
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        
        self.data = mujoco.MjData(self.model)
        
        # 初始化渲染器 (现在 1280 宽度是安全的了)
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        
        # 获取相机 ID
        self.cam_ids = {
            "cam_high": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam"),
            "cam_left_wrist": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam_left"),
            "cam_right_wrist": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam_right"),
        }

        self.obj_names = {
            'red_mug': 'body_obj_mug_5',
            'blue_mug': 'body_obj_mug_6',
            'plate': 'body_obj_plate_11'
        }
        
        assert self.model.nu == 14, f"Error: Expected 14 actuators, found {self.model.nu}"

        self.viewer = None
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def set_object_pose(self, body_name, pos, quat=[1, 0, 0, 0]):
        """
        [新增辅助函数] 原生 MuJoCo 设置自由关节物体位置的方法
        """
        try:
            # 1. 查找 Body 的 ID
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid == -1:
                print(f"⚠️ Warning: Body '{body_name}' not found in XML.")
                return

            # 2. 查找该 Body 关联的 Joint ID (通常 free joint 是附着在 body 上的第一个关节)
            jid = self.model.body_jntadr[bid]
            if jid == -1:
                print(f"⚠️ Warning: Body '{body_name}' has no joint (cannot move).")
                return
            
            # 3. 获取该关节在 qpos 数组中的起始地址
            qpos_adr = self.model.jnt_qposadr[jid]
            
            # 4. 修改 qpos (前3位是位置，后4位是四元数)
            # 注意：Aloha Z轴桌面高度约为 0.0，这里直接设置绝对位置
            self.data.qpos[qpos_adr:qpos_adr+3] = pos
            self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
            
        except Exception as e:
            print(f"❌ Error setting pose for {body_name}: {e}")
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        
        # ---------------------------------------------------------
        # 1. 移植 keyframe_ctrl.xml 的 "neutral_pose"
        # ---------------------------------------------------------
        # 官方定义的 qpos (16维): 6关节 + 2指头 + 6关节 + 2指头
        # 数值来源: 0 -0.96 1.16 0 -0.3 0 0.0084 0.0084 (x2)
        qpos_neutral = np.array([
            0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084,  # 左臂
            0, -0.96, 1.16, 0, -0.3, 0, 0.0084, 0.0084   # 右臂
        ])
        
        # 官方定义的 ctrl (14维): 6关节 + 1夹爪 + 6关节 + 1夹爪
        # 数值来源: 0 -0.96 1.16 0 -0.3 0 0.0084 (x2)
        ctrl_neutral = np.array([
            0, -0.96, 1.16, 0, -0.3, 0, 0.0084,  # 左臂
            0, -0.96, 1.16, 0, -0.3, 0, 0.0084   # 右臂
        ])

        # 设置关节位置 (注意这里是 [:16])
        if self.model.nq >= 16:
            self.data.qpos[:16] = qpos_neutral
            
        # 设置控制信号 (注意这里是 [:])
        if self.model.nu == 14:
             self.data.ctrl[:] = ctrl_neutral

        print("✅ Reset to 'neutral_pose' from keyframe (Hardcoded).")

        # ---------------------------------------------------------
        # 2. 设置物体位置 (调整 Z 轴避免悬浮)
        # ---------------------------------------------------------
        z_height = 0.05  # 根据观察，Aloha 桌面可能就是 0.0
        # 如果还是悬浮，可以尝试微调为 -0.02，或者让下面的物理预热自动处理下落
        
        self.set_object_pose(self.obj_names['plate'], [0.0, -0.2, z_height])
        self.set_object_pose(self.obj_names['red_mug'], [-0.2, 0.2, z_height])
        self.set_object_pose(self.obj_names['blue_mug'], [0.2, 0.2, z_height])

        # ---------------------------------------------------------
        # 3. 物理预热 (让物体自然落地，让机械臂稳定)
        # ---------------------------------------------------------
        for _ in range(50):
            self.data.ctrl[:] = ctrl_neutral # 持续发送 neutral 姿态信号
            mujoco.mj_step(self.model, self.data)

        if self.viewer: self.viewer.sync()
    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        if self.viewer: self.viewer.sync()

    def get_observation(self):
        qpos = self.data.qpos[:14].copy() 
        
        images = {}
        for name, cam_id in self.cam_ids.items():
            self.renderer.update_scene(self.data, camera=cam_id)
            img = self.renderer.render() 
            images[name] = np.array(img) 

        return {
            "qpos": qpos,
            "images": images
        }

    def check_success(self):
        """
        【新增】检测任务是否成功
        逻辑：红色杯子 (red_mug) 与 盘子 (plate) 的 XY 平面距离小于 10cm，且杯子在盘子上方。
        """
        try:
            # 获取 Body ID
            plate_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.obj_names['plate'])
            mug_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.obj_names['red_mug'])
            
            if plate_body_id == -1 or mug_body_id == -1:
                return False

            # 获取全局坐标 (xpos)
            p_plate = self.data.xpos[plate_body_id]
            p_mug = self.data.xpos[mug_body_id]

            # 计算 XY 平面距离
            dist = np.linalg.norm(p_mug[:2] - p_plate[:2])
            
            # 判定条件：
            # 1. 水平距离小于 0.1m (10cm)
            # 2. 杯子 Z 轴高度大于盘子 Z 轴 (防止穿模或在盘子底下)
            # 3. 并且杯子没有掉到地上 (假设桌面 Z=0, 地面 Z=-0.75，杯子应该在 Z > -0.1)
            if dist < 0.1 and p_mug[2] > p_plate[2] and p_mug[2] > -0.1:
                return True
            
        except Exception as e:
            # 避免报错中断服务
            print(f"Check success error: {e}")
            return False
        
        return False
def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("📡 ZMQ Server (HD Mode 720P) listening on port 5555...")

    env = AlohaZMQEnv(XML_PATH)
    env.reset()
    print("🤖 Environment Ready.")

    # 成功计数器
    success_counter = 0 

    while env.viewer.is_running():
        try:
            message = socket.recv(flags=zmq.NOBLOCK)
            data = pickle.loads(message)
            
            cmd = data.get('cmd')
            obs = None
            
            if cmd == 'reset':
                env.reset()
                obs = env.get_observation()
                success_counter = 0
                # 这里的 obs 不包含 success 字段，保持和下面一致或者加上都可以，不影响
                print("🔄 Environment Reset triggered by Client.")
                
            elif cmd == 'step':
                action = data.get('action')
                if action is not None:
                    # 确保动作执行
                    env.step(action)
                
                # --- 逻辑修正：只检测，不重置 ---
                is_success = False
                if env.check_success():
                    success_counter += 1
                    if success_counter >= 10:
                        is_success = True
                        # print("✅ Success detected (waiting for client to reset)") 
                else:
                    success_counter = 0
                # -----------------------------

                obs = env.get_observation()
                # 将成功状态放入 obs，客户端想用就用，不想用就不管
                obs['success'] = is_success 
            
            socket.send(pickle.dumps(obs))
            
        except zmq.Again:
            if env.viewer: 
                env.viewer.sync()
                time.sleep(0.005) 

    print("Closing...")
    socket.close()
    context.term()

if __name__ == "__main__":
    run_server()