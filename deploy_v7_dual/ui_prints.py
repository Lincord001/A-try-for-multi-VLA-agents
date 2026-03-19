def print_startup_banner():
    print("\n" + "=" * 70)
    print("🎮 V6 DUAL-MODE DEPLOYMENT (ARM + BASE)")
    print("=" * 70)


def print_model_loading_config(
    load_arm_model,
    load_base_model,
    arm_inference_mode,
    arm_sync_inference,
    chunk_threshold,
    arm_exec_horizon,
    action_horizon,
    base_action_horizon,
    base_chunk_threshold,
    base_postproc_enabled,
    base_postproc_heading_hold_enabled,
    base_postproc_kp_yaw,
    base_postproc_max_turn_v,
    base_forward_speed_scale_enabled,
    base_forward_speed_scale,
    arm_pilot_run_mode,
    random_init_enabled,
    random_init_gripper_open,
    tb3_x_gaussian_enabled,
    tb3_x_center,
    tb3_x_offset_std,
    tb3_x_offset_min,
    tb3_x_offset_max,
    task_timeout_sec,
    task_loop_count,
    task_stats_output_dir,
):
    print(f"\n📋 Model Loading Configuration:")
    print(f"   LOAD_ARM_MODEL: {load_arm_model}")
    print(f"   LOAD_BASE_MODEL: {load_base_model}")
    print(
        f"   ARM_INFERENCE_MODE: {arm_inference_mode} "
        f"({'同步推理' if arm_sync_inference else '异步推理'})"
    )
    print(f"   CHUNK_THRESHOLD: {chunk_threshold}")
    print(f"   ARM_EXEC_HORIZON: {arm_exec_horizon} (SYNC only)")
    print(f"   ACTION_HORIZON: {action_horizon} (ASYNC only)")
    print(f"   BASE_ACTION_HORIZON: {base_action_horizon} (ASYNC only)")
    print(f"   BASE_CHUNK_THRESHOLD: {base_chunk_threshold}")
    print(
        f"   BASE_POSTPROC: {'Enabled' if base_postproc_enabled else 'Disabled'} "
        f"(heading_hold={'On' if base_postproc_heading_hold_enabled else 'Off'}, "
        f"kp={base_postproc_kp_yaw:.2f}, max_turn={base_postproc_max_turn_v:.2f})"
    )
    if base_forward_speed_scale_enabled:
        base_scale_status = f"ON ({base_forward_speed_scale:.2f}x)"
    else:
        base_scale_status = "OFF"
    print(f"   BASE_FORWARD_SPEED_SCALE: {base_scale_status}")
    print(
        f"   ARM_PILOT_RUN_MODE: {arm_pilot_run_mode} "
        f"({'简单模式' if arm_pilot_run_mode else '困难模式'}) [已废弃]"
    )
    random_init_mode_str = (
        "关闭"
        if random_init_enabled == 0
        else "V1 (扇形区域)"
        if random_init_enabled == 1
        else "V2 (圆形交集)"
    )
    print(f"   RANDOM_INIT_ENABLED: {random_init_enabled} ({random_init_mode_str})")
    print(f"   RANDOM_INIT_GRIPPER_OPEN: {random_init_gripper_open}")
    print(
        f"   TB3_X_GAUSSIAN_ENABLED: {tb3_x_gaussian_enabled} "
        f"(center={tb3_x_center:.3f}, std={tb3_x_offset_std:.3f}, "
        f"range=[{tb3_x_offset_min:+.3f}, {tb3_x_offset_max:+.3f}])"
    )
    print(f"\n📋 Task Configuration:")
    print(f"   TASK_TIMEOUT_SEC: {task_timeout_sec}s")
    loop_desc = "无限循环" if task_loop_count == 0 else f"执行 {task_loop_count} 次后退出"
    print(f"   TASK_LOOP_COUNT: {task_loop_count} ({loop_desc})")
    print(f"   TASK_STATS_OUTPUT_DIR: {task_stats_output_dir}")


def print_action_smoother_config(
    smoothing_enabled,
    smoothing_alpha_joints,
    smoothing_alpha_gripper,
    gripper_hysteresis_enabled,
    gripper_open_thresh,
    gripper_close_thresh,
):
    print(f"🔧 [ARM] Action Smoother initialized:")
    print(f"   - Smoothing Enabled: {smoothing_enabled}")
    print(f"   - Joint Alpha: {smoothing_alpha_joints}")
    print(f"   - Gripper Alpha: {smoothing_alpha_gripper}")
    print(
        f"   - Gripper Hysteresis: {gripper_hysteresis_enabled} "
        f"(open>{gripper_open_thresh}, close<{gripper_close_thresh})"
    )


def print_controls_guide(control_mode):
    print("\n" + "=" * 70)
    print("🎮 V6 DUAL-MODE READY")
    print("=" * 70)
    print("Controls:")
    print("  [C] Switch between ARM/BASE mode (no env reset)")
    print("  [N] Start PI0 Auto Control (current mode)")
    print("  [M] Switch to Manual Control")
    print("  [Z] Reset Environment")
    print("  [K] Cycle BASE speed scaling: OFF -> 0.75x -> 0.50x")
    print("  [←]/[→] Switch Instruction Group (current mode only)")
    print("  [L] 🔥 Toggle Auto-Control + Auto-Detection (ARM mode only)")
    print("      → Press once to ENABLE: model auto-execute + auto-check success/fail + auto-reset")
    print("      → Press again to DISABLE")
    print("  [G] Teleport Base (and cup if locked) to (4.25, 3.5, 0) yaw=-s57.1")
    print("  [Q] Quit")
    print("=" * 70 + "\n")
    print(f"🎯 Current Mode: {control_mode.upper()}")
