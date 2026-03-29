"""
Shared instruction-group utilities for deploy and data-collection scripts.

Both `7_2_pi0_deploy_*.py` and `collect_data_*.py` used to carry their own
(nearly identical) copy of these helpers. This module is the single source of
truth so that future edits only need to happen in one place.
"""

import random

# ==========================================
# Instruction Groups (shared config)
# ==========================================
INSTRUCTION_GROUPS = {
    'arm': [
        {
            'name': 'arm_default',
            'instructions': [
                "Place the red mug on the plate.",
                "Place the blue mug on the plate.",
            ],
        },
        {
            'name': 'arm_01',
            'instructions': [
                "Take the red mug from the tray to the table.",
                "Take the blue mug from the tray to the table.",
            ],
        },
    ],
    'base': [
        {
            'name': 'base_0',
            'instructions': [
                "Go to the workbench.",
                "Drive to the workbench.",
            ],
        },
        {
            'name': 'base_1',
            'instructions': [
                "Move to the kitchen refrigerator.",
            ],
        },
    ],
}

AUTO_ARM_GROUP_DEFAULT_NAME = 'arm_default'
AUTO_ARM_GROUP_TRAY_NAME = 'arm_01'


# ==========================================
# Core helpers (used by both scripts)
# ==========================================

def validate_instruction_groups(extra_required_arm_groups=None):
    """Validate that every mode has at least one non-empty group.

    Parameters
    ----------
    extra_required_arm_groups : list[str] | None
        Additional arm group names that must exist (e.g. for auto-routing).
    """
    for mode in ('arm', 'base'):
        groups = INSTRUCTION_GROUPS.get(mode, [])
        if len(groups) == 0:
            raise ValueError(f"INSTRUCTION_GROUPS['{mode}'] cannot be empty.")
        for group in groups:
            name = group.get('name', '<unnamed>')
            instructions = group.get('instructions', [])
            if len(instructions) == 0:
                raise ValueError(
                    f"Instruction group '{name}' in mode '{mode}' has no instructions."
                )

    if extra_required_arm_groups:
        arm_group_names = {g.get('name') for g in INSTRUCTION_GROUPS.get('arm', [])}
        for required_name in extra_required_arm_groups:
            if required_name not in arm_group_names:
                raise ValueError(
                    f"Missing required arm instruction group '{required_name}'. "
                    f"Please define it in INSTRUCTION_GROUPS['arm']."
                )


def get_group_info(mode, group_indices):
    """Return (idx, total, group_name, instructions) for the active group."""
    groups = INSTRUCTION_GROUPS[mode]
    total = len(groups)
    idx = group_indices[mode] % total
    group = groups[idx]
    group_name = group.get('name', f'{mode}_group_{idx + 1}')
    instructions = list(group.get('instructions', []))
    return idx, total, group_name, instructions


def pick_instruction_from_active_group(mode, group_indices, last_instruction_by_mode):
    """Pick one instruction from the active group, avoiding the previous one."""
    _, _, _, instructions = get_group_info(mode, group_indices)
    last_inst = last_instruction_by_mode.get(mode)
    candidates = [inst for inst in instructions if inst != last_inst]
    if len(candidates) == 0:
        candidates = instructions
    picked = random.choice(candidates)
    last_instruction_by_mode[mode] = picked
    return picked


def apply_instruction_from_group(
    PnPEnv,
    mode,
    group_indices,
    last_instruction_by_mode,
    log_prefix="",
    reinitialize_arm=False,
):
    """Pick and apply an instruction from the active group to the environment."""
    idx, total, group_name, _ = get_group_info(mode, group_indices)
    picked_instruction = pick_instruction_from_active_group(
        mode, group_indices, last_instruction_by_mode,
    )
    task_type = 'arm' if mode == 'arm' else 'nav'
    PnPEnv.set_instruction(given=picked_instruction, task_type=task_type)
    if mode == 'arm' and reinitialize_arm:
        PnPEnv.reset(mode='arm', preserve_instruction=True)
    extra = f"{log_prefix} " if log_prefix else ""
    print(
        f"{extra}🧭 [{mode.upper()}] Instruction Group: {group_name} "
        f"({idx + 1}/{total}) | Task: {PnPEnv.instruction}"
    )


# ==========================================
# Auto-routing helpers (collect_data only)
# ==========================================

def find_group_index_by_name(mode, group_name):
    """Find group index by name; return None if not found."""
    groups = INSTRUCTION_GROUPS.get(mode, [])
    for idx, group in enumerate(groups):
        if group.get('name') == group_name:
            return idx
    return None


def route_arm_group_for_auto(group_indices, tray_init_on_tb3_enabled):
    """Route ARM instruction group based on tray toggle.

    Returns the target group name.
    """
    target_group_name = (
        AUTO_ARM_GROUP_TRAY_NAME if tray_init_on_tb3_enabled
        else AUTO_ARM_GROUP_DEFAULT_NAME
    )
    target_idx = find_group_index_by_name('arm', target_group_name)
    if target_idx is None:
        raise ValueError(f"Cannot find arm instruction group '{target_group_name}'")
    group_indices['arm'] = target_idx
    return target_group_name
