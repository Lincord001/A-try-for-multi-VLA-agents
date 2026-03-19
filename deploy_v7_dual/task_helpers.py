import numpy as np


def get_target_cup_init(env):
    """获取目标杯子的初始坐标和颜色。"""
    meta = env.get_task_metadata()
    target_color = meta["target_color"]
    obj_init_pose = meta["obj_init_pose"]

    if target_color is None:
        instruction = meta.get("instruction", "") or ""
        if "red" in instruction.lower():
            target_color = "red"
        elif "blue" in instruction.lower():
            target_color = "blue"

    if obj_init_pose is None:
        return target_color, np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    if target_color == "blue":
        cup_init = np.array(obj_init_pose[3:6], dtype=np.float32)
    else:
        target_color = "red"
        cup_init = np.array(obj_init_pose[0:3], dtype=np.float32)
    return target_color, cup_init


def get_tb3_init_pose(env):
    """获取当前任务对应的 TB3 初始坐标。"""
    try:
        p_tb3 = env.env.get_p_body("tb3_base")
        return np.array(p_tb3[:3], dtype=np.float32)
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)


def build_reset_options(
    random_init_enabled,
    random_init_gripper_open,
    tb3_x_gaussian_enabled,
    tb3_x_center,
    tb3_x_offset_std,
    tb3_x_offset_min,
    tb3_x_offset_max,
):
    """构造环境 reset 所需 options。"""
    return {
        "random_init_enabled": random_init_enabled,
        "random_init_gripper_open": random_init_gripper_open,
        "tb3_x_gaussian_enabled": tb3_x_gaussian_enabled,
        "tb3_x_center": tb3_x_center,
        "tb3_x_offset_std": tb3_x_offset_std,
        "tb3_x_offset_min": tb3_x_offset_min,
        "tb3_x_offset_max": tb3_x_offset_max,
    }


def perform_auto_reset(
    env,
    control_mode,
    instruction_group_indices,
    last_instruction_by_mode,
    arm_runner,
    arm_policy,
    arm_smoother,
    auto_mode_arm,
    arm_sync_inference,
    apply_instruction_fn,
    reset_options,
):
    """执行自动重置的通用逻辑。"""
    if (not arm_sync_inference) and arm_runner:
        arm_runner.stop()

    if arm_policy:
        arm_policy.reset()

    if arm_runner:
        arm_runner.reset_state()
    arm_smoother.reset()

    apply_instruction_fn(
        env,
        control_mode,
        instruction_group_indices,
        last_instruction_by_mode,
        log_prefix="🔄 [AUTO-RESET]",
        reinitialize_arm=(control_mode == "arm"),
    )

    if control_mode != "arm":
        env.reset(mode=control_mode, options=reset_options)

    if (not arm_sync_inference) and arm_runner and auto_mode_arm:
        arm_runner.start()
