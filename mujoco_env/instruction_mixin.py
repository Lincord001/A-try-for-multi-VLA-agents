import random

import numpy as np

from .env_constants import ARM_BASE_X, ARM_BASE_Y, ARM_INSTRUCTIONS, NAV_INSTRUCTIONS


class InstructionMixin:
    """Mixin providing task-instruction management and mug-selection helpers."""

    def _is_target_initialized_on_tray(self):
        """当前目标杯子是否在本轮 reset 时被初始化到了托盘上。"""
        target_color = getattr(self, 'target_color', None)
        tray_color = getattr(self, 'tray_initialized_color', None)
        return target_color in ('red', 'blue') and tray_color == target_color

    def _get_tb3_tray_center_pos(self):
        """返回托盘抓取中心点（与初始化托盘杯子的中心定义保持一致）。"""
        try:
            tb3_pos = self.env.get_p_body('tb3_base')
        except Exception:
            return None
        return np.array([tb3_pos[0], tb3_pos[1] - 0.06, 0.49], dtype=np.float32)

    def _select_mug_by_smaller_angle(self):
        """选择偏转角度更小的杯子，返回 (obj_target, target_color, instruction)"""
        red_mug_pos = self.env.get_p_body('body_obj_mug_5')
        blue_mug_pos = self.env.get_p_body('body_obj_mug_6')
        
        red_rel_pos = red_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
        blue_rel_pos = blue_mug_pos[:2] - np.array([ARM_BASE_X, ARM_BASE_Y])
        
        red_angle = np.arctan2(red_rel_pos[1], red_rel_pos[0])
        blue_angle = np.arctan2(blue_rel_pos[1], blue_rel_pos[0])
        
        if abs(red_angle) <= abs(blue_angle):
            return 'body_obj_mug_5', 'red', "Place the red mug on the plate."
        else:
            return 'body_obj_mug_6', 'blue', "Place the blue mug on the plate."

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
                        self.obj_target, self.target_color, self.instruction = self._select_mug_by_smaller_angle()
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
                is_tray_to_table_instruction = (
                    'tray' in self.instruction.lower() and 'table' in self.instruction.lower()
                )
                force_tray_init = bool(getattr(self, 'force_tray_init_enabled', False))
                # 约束：开启托盘初始化模式时，禁用"小角度优先选杯子"逻辑
                # 避免托盘初始化与角度选杯产生冲突。
                enable_smaller_angle_selection = self.select_smaller_angle_mug and (not force_tray_init)
                # 🔥 如果启用了选择偏转角度更小的杯子模式，忽略指令中的颜色，直接选择角度更小的
                if enable_smaller_angle_selection and not is_tray_to_table_instruction:
                    try:
                        self.obj_target, self.target_color, self.instruction = self._select_mug_by_smaller_angle()
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
                if not self._suppress_pending_tray_init_update:
                    self.pending_tray_init_color = self.target_color if (is_tray_to_table_instruction or force_tray_init) else None
            elif self.control_mode == 'base':
                self.current_nav_instruction = given
