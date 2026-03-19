def clear_runtime_state(
    arm_policy,
    arm_runner,
    base_runner,
    arm_smoother,
    base_postproc,
):
    """统一清理策略/runner 缓存和后处理器状态。"""
    if arm_policy:
        arm_policy.reset()
    if arm_runner:
        arm_runner.reset_state()
    if base_runner:
        base_runner.reset_state()
    arm_smoother.reset()
    base_postproc.reset()
    return None, 0


def deactivate_arm_auto(
    auto_mode_arm,
    auto_check_enabled,
    arm_runner,
    arm_smoother,
    arm_sync_inference,
    disable_auto_check=False,
    reset_runner_state=False,
):
    """停止 ARM 自动控制，可选关闭自动检测与清理 runner 状态。"""
    if disable_auto_check:
        auto_check_enabled = False
    if not auto_mode_arm:
        return auto_mode_arm, auto_check_enabled, None, 0

    auto_mode_arm = False
    arm_smoother.reset()
    if not arm_sync_inference and arm_runner:
        arm_runner.stop()
        if reset_runner_state:
            arm_runner.reset_state()
    return auto_mode_arm, auto_check_enabled, None, 0


def deactivate_base_auto(
    auto_mode_base,
    base_runner,
    base_postproc,
    reset_runner_state=False,
):
    """停止 BASE 自动控制，可选清理 runner 状态。"""
    if not auto_mode_base:
        return auto_mode_base

    auto_mode_base = False
    if base_runner:
        base_runner.stop()
        if reset_runner_state:
            base_runner.reset_state()
    base_postproc.reset()
    return auto_mode_base


def activate_arm_auto(
    auto_check_enabled,
    arm_policy,
    arm_runner,
    arm_smoother,
    arm_sync_inference,
    enable_auto_check=False,
):
    """启动 ARM 自动控制，返回更新后的状态与 chunk 指针。"""
    if enable_auto_check:
        auto_check_enabled = True
    auto_mode_arm = True
    arm_smoother.reset()
    arm_policy.reset()
    if not arm_sync_inference and arm_runner:
        arm_runner.start()
    return auto_mode_arm, auto_check_enabled, None, 0
