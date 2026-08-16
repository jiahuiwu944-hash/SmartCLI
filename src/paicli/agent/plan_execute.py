from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from paicli.agent.query import query
from paicli.agent.verifier import CompletionVerifier
from paicli.config import PaiCliConfig
from paicli.llm.base import LlmClient
from paicli.memory import MemoryManager, capture_approved_memories
from paicli.plan import ExecutionPlan, Planner, Task, TaskStatus
from paicli.prompt import PromptAssembler
from paicli.runtime.budget import BudgetManager
from paicli.runtime.run_state import RunState, RunStateStore, is_resume_request
from paicli.skill import SkillContextBuffer
from paicli.snapshot import SnapshotService
from paicli.tools.hooks import ToolHookManager
from paicli.tools.registry import ToolRegistry
from paicli.types import Message


@dataclass(slots=True)
class TaskRunResult:
    task: Task
    text: str
    tokens: int
    turns: int
    messages: list[Message]
    error: Exception | None = None


class PlanExecuteAgent:
    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: PaiCliConfig,
        cwd: str,
        approval_callback=None,
        continuation_callback=None,
        tool_hook_manager: ToolHookManager | None = None,
        planner: Planner | None = None,
        max_task_turns: int = 8,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.continuation_callback = continuation_callback
        self.tool_hook_manager = tool_hook_manager
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max_task_turns
        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()
        self.verifier = CompletionVerifier(llm_client)

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        total_tokens = 0
        total_turns = 0
        final_text = ""
        termination_reason = "completed"
        try:
            store = RunStateStore(self.cwd)
            resumed = store.latest_paused("plan") if is_resume_request(message) else None
            if resumed:
                plan = _plan_from_dict(resumed.payload["plan"])
                state = resumed
                state.status = "RUNNING"
                yield {"type": "text_delta", "text": f"Resuming plan {state.run_id}.\n\n"}
            else:
                yield {"type": "text_delta", "text": f"Planning task: {message}\n\n"}
                plan = await self.planner.create_plan(message)
                state = store.create("plan", message)
                state.payload["plan"] = _plan_to_dict(plan)
                store.save(state)
            yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}
            budget = BudgetManager(
                turn_limit=self.config.agent.max_turns,
                token_limit=self.config.agent.max_total_tokens,
                callback=self.continuation_callback,
            )
            budget.turns = state.turns
            budget.tokens = state.tokens
            async for event in self._execute_plan(plan, state, store, budget):
                if event.get("type") == "usage":
                    usage = event.get("usage") or {}
                    total_tokens += int(usage.get("input_tokens") or 0)
                    total_tokens += int(usage.get("output_tokens") or 0)
                elif event.get("type") == "plan_task_done":
                    total_turns += int(event.get("turns") or 0)
                    continue
                elif event.get("type") == "text_delta":
                    final_text += str(event.get("text") or "")
                elif event.get("type") == "run_stopped":
                    termination_reason = str(event.get("reason") or "budget")
                yield event
            self.history = [
                Message(role="user", content=message),
                Message(role="assistant", content=final_text),
            ]
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "error": exc}
            return
        finally:
            with suppress(Exception):
                snapshot.create("post-turn")
        yield {
            "type": "done",
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "termination_reason": termination_reason,
            "completed": termination_reason == "completed" and state.status == "COMPLETED",
            "messages": self.history,
        }

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        state: RunState,
        store: RunStateStore,
        budget: BudgetManager,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "text_delta", "text": "Executing plan...\n\n"}
        plan.mark_started()
        while True:
            if budget.reached():
                approved, request = await budget.request_extension(
                    additional_turns=self.config.agent.budget_extension_turns,
                    additional_tokens=self.config.agent.budget_extension_tokens,
                    mode="plan",
                )
                yield {"type": "budget_extension_requested", **request}
                if not approved:
                    state.status = "PAUSED"
                    state.turns, state.tokens = budget.turns, budget.tokens
                    state.payload["plan"] = _plan_to_dict(plan)
                    store.save(state)
                    yield {
                        "type": "run_stopped",
                        "reason": request["reason"],
                        "message": (f"Plan {state.run_id} paused; run /plan 继续 to resume it."),
                    }
                    return
                yield {"type": "budget_extended", **request}
            executable = _executable_tasks_in_order(plan)
            if not executable:
                break
            if len(executable) == 1:
                result = await self._execute_task(plan, executable[0])
                async for event in self._apply_task_result(result):
                    yield event
                budget.consume(turns=result.turns, tokens=result.tokens)
                _save_plan_state(state, store, plan, budget)
                continue
            yield {
                "type": "text_delta",
                "text": (
                    f"Running parallel batch: {', '.join(task.id for task in executable)}\n\n"
                ),
            }
            results = await asyncio.gather(
                *(self._execute_task(plan, task) for task in executable),
                return_exceptions=False,
            )
            for result in results:
                async for event in self._apply_task_result(result):
                    yield event
                budget.consume(turns=result.turns, tokens=result.tokens)
            _save_plan_state(state, store, plan, budget)

        if plan.has_failed():
            plan.mark_failed()
            yield {"type": "text_delta", "text": "Plan partially completed with failed tasks.\n\n"}
        elif plan.is_all_completed():
            final_result = _build_plan_result(plan)
            evidence = _plan_evidence(plan)
            verdict = await self.verifier.verify_final(
                original_request=plan.goal,
                proposed_answer=final_result,
                messages=evidence,
            )
            yield {
                "type": "stop_hook_review",
                "approved": verdict.approved,
                "feedback": verdict.feedback,
                "mode": "plan",
            }
            if verdict.approved:
                if (
                    self.config.features.memory
                    and self.config.memory.long_term_enabled
                    and self.config.memory.auto_memory_enabled
                    and verdict.memory_candidates
                ):
                    try:
                        report = capture_approved_memories(
                            MemoryManager(
                                self.config.memory.long_term_db_path,
                                scope=self.cwd,
                            ),
                            verdict.memory_candidates,
                            original_request=plan.goal,
                            messages=evidence,
                            min_confidence=self.config.memory.auto_memory_min_confidence,
                            max_candidates=self.config.memory.auto_memory_max_candidates,
                        )
                        yield {
                            "type": "memory_capture",
                            "stored_ids": report.stored_ids,
                            "actions": [item.action for item in report.mutations],
                            "rejected": report.rejected,
                            "mode": "plan",
                        }
                    except Exception as exc:  # noqa: BLE001
                        yield {
                            "type": "memory_capture",
                            "stored_ids": [],
                            "actions": [],
                            "rejected": [f"memory persistence failed: {exc}"],
                            "mode": "plan",
                        }
                plan.mark_completed()
                state.status = "COMPLETED"
                yield {"type": "text_delta", "text": final_result}
            else:
                plan.mark_failed()
                state.status = "FAILED"
                yield {
                    "type": "text_delta",
                    "text": f"Plan verification failed: {verdict.feedback}\n\n",
                }
        else:
            plan.mark_failed()
            yield {
                "type": "text_delta",
                "text": "Plan stalled because dependencies were not satisfied.\n\n",
            }
        state.payload["plan"] = _plan_to_dict(plan)
        state.turns, state.tokens = budget.turns, budget.tokens
        if state.status == "RUNNING":
            state.status = "FAILED" if plan.has_failed() else plan.status.value
        store.save(state)

    async def _apply_task_result(self, result: TaskRunResult) -> AsyncIterator[dict[str, Any]]:
        if result.error:
            result.task.mark_failed(str(result.error))
            yield {"type": "text_delta", "text": f"Failed [{result.task.id}]: {result.error}\n\n"}
            return
        result.task.mark_completed(result.text)
        yield {
            "type": "text_delta",
            "text": f"Completed [{result.task.id}]: {_preview(result.text)}\n\n",
        }
        yield {
            "type": "usage",
            "usage": {"input_tokens": result.tokens, "output_tokens": 0},
        }
        yield {"type": "plan_task_done", "turns": result.turns, "tokens": result.tokens}

    async def _execute_task(self, plan: ExecutionPlan, task: Task) -> TaskRunResult:
        task.mark_started()
        text = ""
        tool_results: list[str] = []
        tokens = 0
        turns = 0
        messages: list[Message] = []
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._task_system_prompt(task),
                user_message=_task_context(plan, task),
                history=[],
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                tool_hook_manager=self.tool_hook_manager,
                # Planned tasks may execute concurrently; each task owns its skill lifecycle.
                skill_context_buffer=SkillContextBuffer(),
                max_turns=self.max_task_turns,
                stop_hook_enabled=False,
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "tool_result":
                    content = str(event.get("result") or "")
                    if content:
                        tool_results.append(content)
                elif event.get("type") == "usage":
                    usage = event.get("usage") or {}
                    tokens += int(usage.get("input_tokens") or 0)
                    tokens += int(usage.get("output_tokens") or 0)
                elif event.get("type") == "done":
                    turns += int(event.get("total_turns") or 0)
                    messages = list(event.get("messages") or [])
                elif event.get("type") == "run_stopped":
                    raise RuntimeError(str(event.get("message") or "Agent run stopped"))
                elif event.get("type") == "error":
                    raise event["error"]
            result_text = text.strip() or "\n".join(tool_results).strip()
            verdict = self.verifier.verify_task(
                description=task.description,
                result=result_text,
                messages=messages,
            )
            if not verdict.approved:
                raise RuntimeError(verdict.feedback)
            return TaskRunResult(task, result_text, tokens, turns, messages)
        except Exception as exc:  # noqa: BLE001
            return TaskRunResult(task, "", tokens, turns, messages, exc)

    def _task_system_prompt(self, task: Task) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build()
        return (
            base
            + "\n\nYou are executing one task inside a Plan-and-Execute DAG.\n"
            + f"Task id: {task.id}\nTask type: {task.type.value}\n"
            + "Complete this task concretely. Use tools when needed."
        )


def _executable_tasks_in_order(plan: ExecutionPlan) -> list[Task]:
    executable_ids = {task.id for task in plan.executable_tasks()}
    return [plan.tasks[task_id] for task_id in plan.execution_order() if task_id in executable_ids]


def _task_context(plan: ExecutionPlan, task: Task) -> str:
    lines = [
        f"Goal: {plan.goal}",
        f"Current task [{task.id}]: {task.description}",
        "",
        "Completed dependency results:",
    ]
    for dep_id in task.dependencies:
        dep = plan.get_task(dep_id)
        if dep and dep.status == TaskStatus.COMPLETED:
            lines.append(f"- [{dep.id}] {dep.description}: {_preview(dep.result, 800)}")
    return "\n".join(lines)


def _build_plan_result(plan: ExecutionPlan) -> str:
    lines = ["Plan execution completed.", "", "Task summary:"]
    for task in plan.all_tasks():
        lines.append(f"- [{task.id}] {task.status.value}: {task.description}")
        if task.result:
            lines.append(f"  Result: {_preview(task.result)}")
    return "\n".join(lines) + "\n"


def _preview(text: str, max_len: int = 160) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _plan_to_dict(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "goal": plan.goal,
        "summary": plan.summary,
        "status": plan.status.value,
        "tasks": [
            {
                "id": task.id,
                "description": task.description,
                "type": task.type.value,
                "dependencies": task.dependencies,
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
            }
            for task in plan.all_tasks()
        ],
    }


def _plan_from_dict(data: dict[str, Any]) -> ExecutionPlan:
    from paicli.plan import PlanStatus, TaskType

    plan = ExecutionPlan(
        id=str(data["id"]),
        goal=str(data["goal"]),
        summary=str(data.get("summary") or ""),
    )
    for item in data.get("tasks") or []:
        status = TaskStatus(str(item.get("status") or "PENDING"))
        if status == TaskStatus.RUNNING:
            status = TaskStatus.PENDING
        task = Task(
            id=str(item["id"]),
            description=str(item["description"]),
            type=TaskType(str(item.get("type") or "ANALYSIS")),
            dependencies=list(item.get("dependencies") or []),
            status=status,
            result=str(item.get("result") or ""),
            error=str(item.get("error") or ""),
        )
        plan.add_task(task)
    plan.status = PlanStatus.RUNNING
    return plan


def _save_plan_state(
    state: RunState,
    store: RunStateStore,
    plan: ExecutionPlan,
    budget: BudgetManager,
) -> None:
    state.payload["plan"] = _plan_to_dict(plan)
    state.turns, state.tokens = budget.turns, budget.tokens
    store.save(state)


def _plan_evidence(plan: ExecutionPlan) -> list[Message]:
    return [
        Message(
            role="tool",
            tool_call_id=f"plan_{task.id}",
            content=f"Task {task.id} status={task.status.value}: {task.result}",
        )
        for task in plan.all_tasks()
    ]
