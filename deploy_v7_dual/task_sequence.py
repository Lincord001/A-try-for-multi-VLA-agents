"""
task_sequence.py
----------------
JSON 串行任务队列的加载、启动与推进逻辑。

设计原则：
- 任务队列属于部署运行时行为，放在 deploy_v7_dual，而不是 orchestration。
- 只做“串行编排 + 状态推进”，底层执行仍复用现有 arm/base 控制流。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _submit_vla_retrieval(vla_instruction_rag, request_id, query, cluster_caption, target_caption, target_image_path):
    result = vla_instruction_rag.retrieve_instruction(
        query=query,
        cluster_caption=cluster_caption,
        target_caption=target_caption,
        target_image_path=target_image_path,
    )
    result["request_id"] = int(request_id)
    return result


def _submit_arm_instruction_retrieval(arm_instruction_rag, task_index, query):
    result = arm_instruction_rag.retrieve_instruction(query)
    result["task_index"] = int(task_index)
    return result


def load_task_sequence(json_path: str) -> list[dict[str, Any]]:
    """Load and validate the serial task list JSON."""
    path = Path(json_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        tasks = payload.get("tasks")
    else:
        tasks = payload

    if not isinstance(tasks, list) or len(tasks) == 0:
        raise ValueError("任务列表 JSON 必须是非空数组，或包含非空 tasks 数组。")

    normalized: list[dict[str, Any]] = []
    for idx, raw_task in enumerate(tasks, start=1):
        if not isinstance(raw_task, dict):
            raise ValueError(f"任务 #{idx} 必须是 object。")

        task = dict(raw_task)
        executor = str(task.get("executor", "")).strip().lower()
        if executor not in {"arm", "base"}:
            raise ValueError(f"任务 #{idx} 的 executor 只能是 'arm' 或 'base'。")

        task["executor"] = executor
        task["name"] = str(task.get("name") or f"{executor}_task_{idx}")

        if executor == "arm":
            instruction = str(task.get("instruction", "")).strip()
            query = str(task.get("query", "")).strip()
            if not instruction and not query:
                raise ValueError(f"ARM 任务 #{idx} 至少需要提供 instruction 或 query。")
            task["instruction"] = instruction
            task["query"] = query
            task["reinitialize_arm"] = bool(task.get("reinitialize_arm", True))
        else:
            query = str(task.get("query", "")).strip()
            target_node = str(task.get("target_node", "")).strip()
            if not query and not target_node:
                raise ValueError(
                    f"BASE 任务 #{idx} 至少需要提供 query 或 target_node 其中之一。"
                )
            task["query"] = query
            task["target_node"] = target_node
            task["cluster_caption"] = str(task.get("cluster_caption", "")).strip()
            task["target_caption"] = str(task.get("target_caption", "")).strip()

        normalized.append(task)

    return normalized


def start_task_sequence_from_file(state, json_path: str) -> None:
    """Load a task list and mark it active."""
    tasks = load_task_sequence(json_path)
    state.start_task_sequence(tasks=tasks, source_path=str(Path(json_path).expanduser().resolve()))
    print(
        f"\n▶️ [TASK QUEUE] Loaded {len(tasks)} tasks from: "
        f"{state.task_sequence_source_path}"
    )


def advance_task_sequence(
    state,
    env,
    arm_policy,
    arm_runner,
    arm_smoother,
    base_runner,
    base_postproc,
    rag_executor,
    rag_navigator=None,
    vla_instruction_rag=None,
    vla_instruction_executor=None,
    arm_instruction_rag=None,
    arm_instruction_executor=None,
    trace_manager=None,
    arm_orchestrator=None,
    rag_output_json: str = "",
) -> bool:
    """Drive the task queue forward: consume results, then start the next task when idle."""
    if not state.task_sequence_active:
        return False

    _prefetch_arm_task_queries(
        state=state,
        arm_instruction_rag=arm_instruction_rag,
        arm_instruction_executor=arm_instruction_executor,
    )

    if _step_arm_return_home_if_needed(state=state, env=env):
        return True

    if state.task_sequence_pending_result is not None:
        result = dict(state.task_sequence_pending_result)
        task = state.current_task_sequence_task() or {}
        task_name = str(task.get("name", f"task_{state.task_sequence_index + 1}"))
        state.task_sequence_results.append(
            {
                "task_index": int(state.task_sequence_index),
                "task_name": task_name,
                "executor": task.get("executor"),
                **result,
            }
        )
        print(
            "\n🏁 [TASK QUEUE] "
            f"{task_name} finished: status={result.get('status')} "
            f"| reason={result.get('reason')}"
        )
        state.task_sequence_index += 1
        state.task_sequence_started = False
        state.task_sequence_pending_result = None

        if state.task_sequence_index >= len(state.task_sequence_tasks):
            total = len(state.task_sequence_results)
            succeeded = sum(
                1 for item in state.task_sequence_results
                if str(item.get("status")) in {"success", "completed"}
            )
            print(
                "\n✅ [TASK QUEUE] All tasks finished. "
                f"success_like={succeeded}/{total}"
            )
            state.stop_task_sequence(reason="completed_all")
            return False

    if state.task_sequence_started:
        return False

    task = state.current_task_sequence_task()
    if task is None:
        state.stop_task_sequence(reason="no_current_task")
        return False

    try:
        _start_task(
            state=state,
            env=env,
            task=task,
            arm_policy=arm_policy,
            arm_runner=arm_runner,
            arm_smoother=arm_smoother,
            base_runner=base_runner,
            base_postproc=base_postproc,
            rag_executor=rag_executor,
            rag_navigator=rag_navigator,
            vla_instruction_rag=vla_instruction_rag,
            vla_instruction_executor=vla_instruction_executor,
            arm_instruction_rag=arm_instruction_rag,
            arm_instruction_executor=arm_instruction_executor,
            trace_manager=trace_manager,
            arm_orchestrator=arm_orchestrator,
            rag_output_json=rag_output_json,
        )
    except Exception as exc:
        task_name = str(task.get("name", f"task_{state.task_sequence_index + 1}"))
        print(f"\n❌ [TASK QUEUE] Failed to start {task_name}: {exc}")
        state.stop_task_sequence(reason=f"start_failure:{task_name}")
    return False


def _stop_all_execution_modes(
    state,
    arm_runner,
    arm_smoother,
    base_runner,
    base_postproc,
    trace_manager=None,
    arm_orchestrator=None,
) -> None:
    if trace_manager is not None:
        trace_manager.stop_arm_episode(reason="task_queue_mode_switch")
        trace_manager.stop_base_episode(reason="task_queue_mode_switch")
    if arm_orchestrator is not None:
        arm_orchestrator.on_auto_stop("task_queue_mode_switch")
    state.deactivate_arm_auto(
        arm_runner,
        arm_smoother,
        disable_auto_check=True,
        reset_runner_state=True,
    )
    state.deactivate_base_auto(
        base_runner,
        base_postproc,
        reset_runner_state=True,
    )
    state.stop_navigation()
    state.rag_vla_handoff_waiting = False


def _start_task(
    state,
    env,
    task,
    arm_policy,
    arm_runner,
    arm_smoother,
    base_runner,
    base_postproc,
    rag_executor,
    rag_navigator=None,
    vla_instruction_rag=None,
    vla_instruction_executor=None,
    arm_instruction_rag=None,
    arm_instruction_executor=None,
    trace_manager=None,
    arm_orchestrator=None,
    rag_output_json: str = "",
) -> None:
    executor = str(task["executor"])
    task_name = str(task["name"])
    _stop_all_execution_modes(
        state,
        arm_runner,
        arm_smoother,
        base_runner,
        base_postproc,
        trace_manager=trace_manager,
        arm_orchestrator=arm_orchestrator,
    )

    if executor == "arm":
        _start_arm_task(
            state,
            env,
            task,
            arm_policy,
            arm_runner,
            arm_smoother,
            arm_instruction_rag=arm_instruction_rag,
            arm_instruction_executor=arm_instruction_executor,
            trace_manager=trace_manager,
            arm_orchestrator=arm_orchestrator,
        )
    else:
        _start_base_task(
            state,
            env,
            task,
            base_runner,
            base_postproc,
            rag_executor,
            rag_navigator=rag_navigator,
            vla_instruction_rag=vla_instruction_rag,
            vla_instruction_executor=vla_instruction_executor,
            trace_manager=trace_manager,
            rag_output_json=rag_output_json,
        )

    state.task_sequence_started = True
    print(
        f"\n🚀 [TASK QUEUE] Started {task_name} "
        f"({state.task_sequence_index + 1}/{len(state.task_sequence_tasks)})"
    )


def _start_arm_task(
    state,
    env,
    task,
    arm_policy,
    arm_runner,
    arm_smoother,
    arm_instruction_rag=None,
    arm_instruction_executor=None,
    trace_manager=None,
    arm_orchestrator=None,
) -> None:
    if arm_policy is None:
        raise RuntimeError("ARM policy not loaded, cannot run ARM task queue item.")

    instruction = str(task.get("instruction", "")).strip()
    query = str(task.get("query", "")).strip()
    if not instruction:
        if not query:
            raise RuntimeError("ARM task queue item requires instruction or query.")
        rag_result = state.task_sequence_arm_query_results.get(int(state.task_sequence_index))
        if rag_result is None:
            if arm_instruction_rag is None:
                raise RuntimeError("ARM instruction retriever unavailable, cannot resolve arm query.")
            rag_result = arm_instruction_rag.retrieve_instruction(query)
        if not rag_result.get("matched", True) or not rag_result.get("instruction"):
            raise RuntimeError(f"ARM query has no matching standardized instruction: {query}")
        instruction = str(rag_result["instruction"])
        print(
            "\n🦾 [ARM-RAG] Task queue normalized instruction: "
            f"{instruction} | reason={rag_result.get('llm_reason', '')}"
        )
    reinitialize_arm = bool(task.get("reinitialize_arm", True))

    state.control_mode = "arm"
    env.control_mode = "arm"
    env.set_instruction(given=instruction, task_type="arm")
    state.last_instruction_by_mode["arm"] = instruction
    if reinitialize_arm:
        env.reinitialize_arm_only()
        print("\n🔄 [ARM] Reinitialized arm only; preserved base and scene objects.")

    state.activate_arm_auto(
        arm_policy,
        arm_runner,
        arm_smoother,
        enable_auto_check=True,
        reset_timer=True,
        env=env,
    )
    if trace_manager is not None:
        trace_manager.start_arm_episode(
            env,
            step=state.step,
            reason="task_queue_start",
        )
    if arm_orchestrator is not None:
        arm_orchestrator.on_auto_start(env)


def _prefetch_arm_task_queries(
    state,
    arm_instruction_rag=None,
    arm_instruction_executor=None,
):
    if not state.task_sequence_active or arm_instruction_rag is None:
        return

    if not state.task_sequence_prefetch_started:
        for idx, task in enumerate(state.task_sequence_tasks):
            if str(task.get("executor")) != "arm":
                continue
            if str(task.get("instruction", "")).strip():
                continue
            query = str(task.get("query", "")).strip()
            if not query:
                continue
            if arm_instruction_executor is None:
                continue
            state.task_sequence_arm_query_futures[idx] = arm_instruction_executor.submit(
                _submit_arm_instruction_retrieval,
                arm_instruction_rag,
                idx,
                query,
            )
        state.task_sequence_prefetch_started = True
        if state.task_sequence_arm_query_futures:
            print(
                "\n🦾 [ARM-RAG] Started async prefetch for "
                f"{len(state.task_sequence_arm_query_futures)} queued arm task(s)."
            )

    completed = []
    for idx, future in state.task_sequence_arm_query_futures.items():
        if not future.done():
            continue
        try:
            result = future.result()
            state.task_sequence_arm_query_results[idx] = dict(result)
            print(
                "\n🦾 [ARM-RAG] Prefetch ready for queued task "
                f"{idx + 1}: {result.get('instruction') or 'NO_MATCH'}"
            )
            if result.get("llm_reason"):
                print(f"   LLM reason: {result['llm_reason']}")
        except Exception as exc:
            state.task_sequence_arm_query_results[idx] = {
                "matched": False,
                "instruction": None,
                "llm_reason": str(exc),
                "task_index": int(idx),
            }
            print(f"\n⚠️ [ARM-RAG] Prefetch failed for queued task {idx + 1}: {exc}")
        completed.append(idx)

    for idx in completed:
        state.task_sequence_arm_query_futures.pop(idx, None)


def _step_arm_return_home_if_needed(state, env) -> bool:
    if not state.task_sequence_arm_return_home_active:
        return False

    state.control_mode = "arm"
    env.control_mode = "arm"
    env.smooth_return_home()
    env.render(teleop=False, idx=state.step)
    state.step += 1

    if getattr(env, "returning_home", False):
        if state.step % 20 == 0:
            print("[TASK QUEUE][ARM] Waiting for smooth return-home to finish.")
        return True

    payload = dict(state.task_sequence_arm_post_home_result or {})
    print("\n🏠 [TASK QUEUE][ARM] Smooth return-home finished.")
    if payload:
        print(
            "[TASK QUEUE][ARM] Post-home result ready: "
            f"status={payload.get('status')} | reason={payload.get('reason')}"
        )
    state.task_sequence_arm_return_home_active = False
    state.task_sequence_arm_post_home_result = None
    if payload:
        state.task_sequence_pending_result = payload
    return True


def _start_base_task(
    state,
    env,
    task,
    base_runner,
    base_postproc,
    rag_executor,
    rag_navigator=None,
    vla_instruction_rag=None,
    vla_instruction_executor=None,
    trace_manager=None,
    rag_output_json: str = "",
) -> None:
    state.control_mode = "base"
    env.control_mode = "base"
    state.clear_retrieval_result()

    query_text = str(task.get("query", "")).strip()
    explicit_target_node = str(task.get("target_node", "")).strip()
    cluster_caption = str(task.get("cluster_caption", "")).strip()
    target_caption = str(task.get("target_caption", "")).strip()

    if explicit_target_node:
        retrieval_result = {
            "query": query_text or explicit_target_node,
            "cluster_id": "task_queue_explicit_target",
            "cluster_caption": cluster_caption,
            "target_node": explicit_target_node,
            "target_caption": target_caption,
            "score": 1.0,
        }
    else:
        if rag_navigator is None:
            raise RuntimeError("RAG navigator unavailable, cannot resolve base task query.")
        retrieval_result = rag_navigator.retrieve_top_leaf(query_text)

    state.update_retrieval_result(query_text or retrieval_result["query"], retrieval_result)
    target_node = str(retrieval_result["target_node"])
    query_for_env = str(retrieval_result.get("query") or query_text or target_node)
    env.set_instruction(given=query_for_env, task_type="nav")
    state.last_instruction_by_mode["base"] = query_for_env
    state.step = 0
    state.reset_task_timer(env)

    if vla_instruction_rag is not None and vla_instruction_executor is not None:
        state.rag_vla_request_seq += 1
        request_id = int(state.rag_vla_request_seq)
        state.rag_vla_pending_request_id = request_id
        state.rag_vla_retrieval_pending = True
        state.rag_vla_pending_future = vla_instruction_executor.submit(
            _submit_vla_retrieval,
            vla_instruction_rag,
            request_id,
            query_for_env,
            str(retrieval_result.get("cluster_caption", "")),
            str(retrieval_result.get("target_caption", "")),
            retrieval_result.get("target_image_path"),
        )
        print(
            "\n🧠 [TASK QUEUE][BASE] Async VLA retrieval started "
            f"for target={target_node}"
        )
    elif vla_instruction_rag is not None:
        # 保持与原逻辑一致：没有异步执行器时，不在这里提前做同步检索，
        # 留给 RAG 结束后的 handoff 阶段按原流程处理。
        print(
            "\nℹ️ [TASK QUEUE][BASE] VLA retriever available but no async executor. "
            "Handoff decision will stay on the original post-RAG path."
        )

    p_tb3, R_tb3 = env.env.get_pR_body("tb3_base")
    yaw = float(np.arctan2(float(R_tb3[1, 0]), float(R_tb3[0, 0])))
    start_pose = np.array([float(p_tb3[0]), float(p_tb3[1]), yaw], dtype=np.float64)
    plan_result = rag_executor.plan_dense_waypoints(start_pose=start_pose, target_node=target_node)
    if rag_output_json:
        rag_executor.save_result(plan_result, rag_output_json)
    state.start_navigation(plan_result["dense_waypoints"], target_node)
    base_postproc.reset()
    if trace_manager is not None:
        trace_manager.start_base_episode(
            env,
            step=state.step,
            reason="task_queue_base_start",
        )
