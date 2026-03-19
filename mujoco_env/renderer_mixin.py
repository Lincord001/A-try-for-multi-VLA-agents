import numpy as np

from .utils import add_title_to_img


class RendererMixin:
    """Mixin providing camera image capture and MuJoCo viewer rendering."""

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
