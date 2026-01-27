import sys
import numpy as np
import os
import glfw # 确保导入了 glfw

# 关键: 相对路径导入，因为它在 mujoco_env 文件夹内
try:
    from mujoco_env.aloha_env import AlohaEnv
except ImportError as e:
    print(f"❌ 导入错误：{e}")
    print("请确保 'mujoco_env' 文件夹及其所有子模块在 Python 路径中。")
    sys.exit(1)

# ================= 配置区域 =================
# 场景文件路径 (相对于 view_aloha.py 的位置)
XML_PATH = './asset/aloha/scene.xml'
# ===========================================

def main():
    # 1. 初始化 ALOHA 环境
    print(f"🚀 Initializing ALOHA Playground from: {XML_PATH}")
    
    if not os.path.exists(XML_PATH):
        print(f"❌ 错误：找不到场景文件: {XML_PATH}")
        print("请检查 asset/aloha/scene.xml 路径是否正确。")
        return

    try:
        # 实例化环境
        env = AlohaEnv(xml_path=XML_PATH, action_type='eef_pose', state_type='joint_angle')
    except Exception as e:
        print(f"❌ 初始化环境时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 打印操作指南（略）
    print("\n" + "="*50)
    print("  🤖 ALOHA 2 Visualization & Control Playground 🤖")
    print("  Controls: W/S/A/D, R/F, Arrows, Q/E, Space, Z (Reset)")
    print("="*50 + "\n")

    action = np.zeros(7)

    try:
        # 主循环
        while env.env.is_viewer_alive():
            env.step_env()
            
            if env.env.loop_every(HZ=20):
                
                action, reset = env.teleop_robot()
                
                if reset:
                    print("🔄 Resetting robot...")
                    env.reset()
                    action = np.zeros(7) # 重置 action，避免继续运动

                env.grab_image()
                env.step(action)
                env.render(teleop=True, idx=0)

    except KeyboardInterrupt:
        print("\n👋 Playground closed by user.")
    finally:
        print("Closing viewer...")
        if 'env' in locals() and hasattr(env, 'env'):
            env.env.close_viewer()

if __name__ == "__main__":
    main()
