from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from paicli.agent.query import query
from paicli.agent.verifier import CompletionVerifier, VerificationResult
from paicli.config import PaiCliConfig
from paicli.llm.base import LlmClient
from paicli.memory import MemoryManager, capture_approved_memories
from paicli.plan import ExecutionPlan, Planner, Task, TaskStatus, TaskType
from paicli.prompt import PromptAssembler
from paicli.runtime.budget import BudgetManager
from paicli.runtime.run_state import RunState, RunStateStore, resume_target
from paicli.skill import SkillContextBuffer
from paicli.snapshot import SnapshotService
from paicli.tools.hooks import ToolHookManager
from paicli.tools.registry import ToolRegistry
from paicli.types import Message


@dataclass(frozen=True, slots=True)
class OrchestrationPolicy:
    mode: str
    worker_count: int
    review_each_task: bool
    retry_limit: int


@dataclass(slots=True)
class TaskRunResult:
    task: Task
    text: str
    tokens: int
    turns: int
    messages: list[Message]
    error: Exception | None = None


_TASK_REVIEW_PROMPT = """You are a strict task reviewer in SmartCLI's orchestration kernel.
Judge only from the task result, acceptance criteria, and concrete tool evidence. A failed,
blocked, denied, or missing tool result is not success. Return JSON only:
{"approved": true|false, "issues": ["specific issue"]}
The approved field must be a JSON boolean.
"""

_SYNTHESIS_PROMPT = """You are SmartCLI's final synthesizer. Produce a concise answer to the
original request using only completed task results and their evidence. Do not invent actions,
files, commands, or verification. Explicitly identify any limitation that remains. Never mention
internal reviewers, Stop Hook verdicts, correction prompts, orchestration, or task IDs.
"""


class PlanExecuteAgent:
    """Shared DAG orchestration kernel used by both Plan and Team modes."""

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
        mode: str = "plan",
        worker_count: int = 4,
        review_each_task: bool = False,
        max_retries_per_task: int = 1,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.continuation_callback = continuation_callback
        self.tool_hook_manager = tool_hook_manager
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max(1, max_task_turns)
        self.policy = OrchestrationPolicy(
            mode=mode,
            worker_count=max(1, worker_count),
            review_each_task=review_each_task,
            retry_limit=max(0, max_retries_per_task),
        )
        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()
        self.verifier = CompletionVerifier(llm_client)

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        state: RunState | None = None
        budget = BudgetManager(
            turn_limit=self.config.agent.max_turns,
            token_limit=self.config.agent.max_total_tokens,
            callback=self.continuation_callback,
        )
        final_text = ""
        termination_reason = "completed"
        try:
            store = RunStateStore(self.cwd)
            target = resume_target(message)
            if target is not False:
                state = store.latest_resumable(self.policy.mode, run_id=target)
                if state is None:
                    suffix = f" {target}" if target else ""
                    raise ValueError(f"No resumable {self.policy.mode} run found{suffix}.")
                plan = _plan_from_dict(state.payload["plan"])
                state.status = "RUNNING"
                budget.turns, budget.tokens = state.turns, state.tokens
                store.save(state)
                yield {
                    "type": "text_delta",
                    "text": f"Resuming {self.policy.mode} run {state.run_id}.\n\n",
                }
            else:
                yield {"type": "text_delta", "text": f"Planning task: {message}\n\n"}
                plan = await self.planner.create_plan(message)
                budget.consume(turns=self.planner.last_turns, tokens=self.planner.last_tokens)
                if self.planner.last_warning:
                    yield {
                        "type": "planner_fallback",
                        "message": self.planner.last_warning,
                        "mode": self.policy.mode,
                    }
                state = store.create(self.policy.mode, message)
                state.turns, state.tokens = budget.turns, budget.tokens
                state.payload["plan"] = _plan_to_dict(plan)
                store.save(state)

            yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}
            deadline = monotonic() + max(1.0, self.config.agent.max_runtime_seconds)
            async for event in self._execute_plan(plan, state, store, budget, deadline):
                if event.get("type") == "text_delta":
                    final_text += str(event.get("text") or "")
                elif event.get("type") == "run_stopped":
                    termination_reason = str(event.get("reason") or "budget")
                yield event
            self.history = [
                Message(role="user", content=state.goal),
                Message(role="assistant", content=final_text),
            ]
        except Exception as exc:  # noqa: BLE001
            termination_reason = "error"
            yield {"type": "error", "error": exc}
            return
        finally:
            with suppress(Exception):
                snapshot.create("post-turn")

        assert state is not None
        yield {
            "type": "done",
            "total_turns": budget.turns,
            "total_tokens": budget.tokens,
            "termination_reason": termination_reason,
            "completed": termination_reason == "completed" and state.status == "COMPLETED",
            "messages": self.history,
            "run_id": state.run_id,
        }

    async def _execute_plan(
        self,
        plan: ExecutionPlan,
        state: RunState,
        store: RunStateStore,
        budget: BudgetManager,
        deadline: float,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "text_delta", "text": "Executing plan...\n\n"}
        plan.mark_started()
        _save_plan_state(state, store, plan, budget)

        while True:
            for task in plan.propagate_blocked():
                yield {"type": "task_blocked", "task_id": task.id, "reason": task.error}
            _save_plan_state(state, store, plan, budget)

            minimum_task_turns = 2 if self.policy.review_each_task else 1
            stop_reason = _execution_stop_reason(
                budget, deadline, minimum_turns=minimum_task_turns
            )
            if stop_reason:
                if stop_reason == "runtime_budget":
                    _pause_state(state, store, plan, budget)
                    yield _paused_event(state, stop_reason, self.policy.mode)
                    return
                approved, request = await budget.request_extension(
                    additional_turns=self.config.agent.budget_extension_turns,
                    additional_tokens=self.config.agent.budget_extension_tokens,
                    mode=self.policy.mode,
                    minimum_turns=minimum_task_turns,
                )
                yield {"type": "budget_extension_requested", **request}
                if not approved:
                    _pause_state(state, store, plan, budget)
                    yield _paused_event(state, request["reason"], self.policy.mode)
                    return
                yield {"type": "budget_extended", **request}

            executable = _executable_tasks_in_order(plan)
            if not executable:
                break
            budget_workers = max(1, budget.remaining_turns // minimum_task_turns)
            batch = _select_safe_batch(
                executable, min(self.policy.worker_count, budget_workers)
            )
            if len(batch) > 1:
                yield {
                    "type": "text_delta",
                    "text": f"Running safe parallel batch: {', '.join(t.id for t in batch)}\n\n",
                }
            async for event in self._run_batch(
                plan, batch, state, store, budget, deadline
            ):
                if event.get("type") == "_batch_stopped":
                    _pause_state(state, store, plan, budget)
                    yield _paused_event(state, "runtime_budget", self.policy.mode)
                    return
                yield event

        final_turns = 2 if self.config.agent.stop_hook_enabled else 1
        reasons = budget.reached(minimum_turns=final_turns)
        if monotonic() >= deadline and plan.is_all_completed():
            _pause_state(state, store, plan, budget)
            yield _paused_event(state, "runtime_budget", self.policy.mode)
            return
        if reasons and plan.is_all_completed():
            approved, request = await budget.request_extension(
                additional_turns=self.config.agent.budget_extension_turns,
                additional_tokens=self.config.agent.budget_extension_tokens,
                mode=self.policy.mode,
                minimum_turns=final_turns,
            )
            yield {"type": "budget_extension_requested", **request}
            if not approved:
                _pause_state(state, store, plan, budget)
                yield _paused_event(state, request["reason"], self.policy.mode)
                return
            yield {"type": "budget_extended", **request}

        if plan.has_failed():
            plan.mark_failed()
            state.status = "FAILED"
            yield {"type": "text_delta", "text": _build_plan_result(plan, self.policy.mode)}
        elif plan.is_all_completed():
            final_result, synth_tokens, synth_turns = await self._synthesize(plan)
            budget.consume(turns=synth_turns, tokens=synth_tokens)
            verdict = await self._verify_final(plan, final_result)
            budget.consume(
                turns=1 if self.config.agent.stop_hook_enabled else 0,
                tokens=verdict.input_tokens + verdict.output_tokens,
            )
            yield {
                "type": "stop_hook_review",
                "approved": verdict.approved,
                "feedback": verdict.feedback,
                "mode": self.policy.mode,
            }
            if not verdict.approved:
                correction_turns = 2 if self.config.agent.stop_hook_enabled else 1
                correction_reasons = budget.reached(minimum_turns=correction_turns)
                if correction_reasons:
                    approved, request = await budget.request_extension(
                        additional_turns=self.config.agent.budget_extension_turns,
                        additional_tokens=self.config.agent.budget_extension_tokens,
                        mode=self.policy.mode,
                        minimum_turns=correction_turns,
                    )
                    yield {"type": "budget_extension_requested", **request}
                    if not approved:
                        _pause_state(state, store, plan, budget)
                        yield _paused_event(state, request["reason"], self.policy.mode)
                        return
                    yield {"type": "budget_extended", **request}
                corrected, tokens, turns = await self._synthesize(
                    plan, correction=verdict.feedback
                )
                budget.consume(turns=turns, tokens=tokens)
                corrected_verdict = await self._verify_final(plan, corrected)
                budget.consume(
                    turns=1 if self.config.agent.stop_hook_enabled else 0,
                    tokens=corrected_verdict.input_tokens + corrected_verdict.output_tokens,
                )
                yield {
                    "type": "stop_hook_review",
                    "approved": corrected_verdict.approved,
                    "feedback": corrected_verdict.feedback,
                    "mode": self.policy.mode,
                    "correction": True,
                }
                final_result, verdict = corrected, corrected_verdict

            if verdict.approved:
                plan.mark_completed()
                state.status = "COMPLETED"
                async for event in self._capture_memory(plan, verdict):
                    yield event
                yield {"type": "text_delta", "text": final_result}
            else:
                plan.mark_failed()
                state.status = "FAILED"
                yield {
                    "type": "text_delta",
                    "text": f"Final verification failed: {verdict.feedback}\n\n{final_result}",
                }
        else:
            plan.mark_failed()
            state.status = "FAILED"
            yield {
                "type": "text_delta",
                "text": "Plan stalled because dependencies were not satisfied.\n",
            }
        _save_plan_state(state, store, plan, budget)

    async def _run_batch(
        self,
        plan: ExecutionPlan,
        batch: list[Task],
        state: RunState,
        store: RunStateStore,
        budget: BudgetManager,
        deadline: float,
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        batch_size = len(batch)
        turn_allocation = max(
            1, min(self.max_task_turns, budget.remaining_turns // batch_size)
        )
        token_allocation = budget.remaining_tokens // batch_size if budget.token_limit else 0

        def emit(event: dict[str, Any]) -> None:
            queue.put_nowait(event)

        async def run_one(task: Task) -> None:
            result = await self._execute_task(
                plan,
                task,
                turn_allocation=turn_allocation,
                token_allocation=token_allocation,
                emit=emit,
            )
            queue.put_nowait({"type": "_task_result", "result": result})

        for task in batch:
            task.mark_started()
            emit({"type": "task_started", "task_id": task.id, "attempt": task.attempt_count})
        _save_plan_state(state, store, plan, budget)
        jobs = [asyncio.create_task(run_one(task)) for task in batch]
        remaining = len(jobs)
        try:
            while remaining:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    yield {"type": "_batch_stopped"}
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                except TimeoutError:
                    yield {"type": "_batch_stopped"}
                    return
                if event.get("type") == "_task_result":
                    remaining -= 1
                    result = event["result"]
                    async for applied in self._apply_task_result(result):
                        yield applied
                    budget.consume(turns=result.turns, tokens=result.tokens)
                    _save_plan_state(state, store, plan, budget)
                else:
                    if event.get("type") == "task_attempt_failed":
                        _save_plan_state(state, store, plan, budget)
                    yield event
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            for task in batch:
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.PENDING

    async def _execute_task(
        self,
        plan: ExecutionPlan,
        task: Task,
        *,
        turn_allocation: int,
        token_allocation: int,
        emit: Callable[[dict[str, Any]], None],
    ) -> TaskRunResult:
        total_tokens = 0
        total_turns = 0
        all_messages: list[Message] = []
        retry_feedback = ""
        configured_attempts = min(task.retry_limit, self.policy.retry_limit) + 1
        requested_attempts = max(1, configured_attempts - task.attempt_count + 1)
        review_turns = 1 if self.policy.review_each_task else 0
        minimum_attempt_turns = 1 + review_turns
        attempts = max(
            1,
            min(requested_attempts, turn_allocation // minimum_attempt_turns),
        )
        retry_limit = attempts - 1
        per_attempt_turns = max(1, turn_allocation // attempts - review_turns)
        per_attempt_tokens = token_allocation // attempts if token_allocation else 0
        last_error: Exception | None = None

        for attempt in range(attempts):
            if attempt:
                task.mark_started()
                emit(
                    {
                        "type": "task_retry_started",
                        "task_id": task.id,
                        "attempt": task.attempt_count,
                        "feedback": retry_feedback,
                    }
                )
            text = ""
            tool_results: list[str] = []
            messages: list[Message] = []
            attempt_tokens = 0
            attempt_turns = 0
            try:
                worker_config = deepcopy(self.config)
                if per_attempt_tokens:
                    worker_config.agent.max_total_tokens = per_attempt_tokens
                registry = (
                    _read_only_registry(self.tool_registry)
                    if task.parallel_safe
                    else self.tool_registry
                )
                context = _task_context(plan, task, retry_feedback=retry_feedback)
                async for event in query(
                    llm_client=self.llm_client,
                    tool_registry=registry,
                    system_prompt=self._task_system_prompt(task, registry),
                    user_message=context,
                    history=[],
                    cwd=self.cwd,
                    config=worker_config,
                    approval_callback=self.approval_callback,
                    tool_hook_manager=self.tool_hook_manager,
                    skill_context_buffer=SkillContextBuffer(),
                    max_turns=per_attempt_turns,
                    stop_hook_enabled=False,
                ):
                    event_type = event.get("type")
                    if event_type == "text_delta":
                        text += str(event.get("text") or "")
                        emit(
                            {
                                "type": "task_text_delta",
                                "task_id": task.id,
                                "text": str(event.get("text") or ""),
                            }
                        )
                    elif event_type == "tool_result":
                        content = str(event.get("result") or "")
                        if content:
                            tool_results.append(content)
                        emit({**event, "task_id": task.id})
                    elif event_type in {
                        "tool_call",
                        "skill_activated",
                        "thinking_delta",
                        "llm_retry",
                        "context_compressed",
                    }:
                        emit({**event, "task_id": task.id})
                    elif event_type == "usage":
                        usage = event.get("usage") or {}
                        attempt_tokens += int(usage.get("input_tokens") or 0)
                        attempt_tokens += int(usage.get("output_tokens") or 0)
                        emit({**event, "task_id": task.id})
                    elif event_type == "done":
                        attempt_turns += int(event.get("total_turns") or 0)
                        messages = list(event.get("messages") or [])
                    elif event_type == "run_stopped":
                        raise RuntimeError(str(event.get("message") or "Agent run stopped"))
                    elif event_type == "error":
                        raise event["error"]

                result_text = text.strip() or "\n".join(tool_results).strip()
                verdict = self.verifier.verify_task(
                    description=task.description,
                    result=result_text,
                    messages=messages,
                )
                if verdict.approved and self.policy.review_each_task:
                    verdict = await self._review_task(task, result_text, messages)
                    attempt_tokens += verdict.input_tokens + verdict.output_tokens
                    attempt_turns += 1
                if not verdict.approved:
                    raise RuntimeError(verdict.feedback)

                total_tokens += attempt_tokens
                total_turns += attempt_turns
                all_messages.extend(messages)
                task.evidence = _message_evidence(all_messages)
                return TaskRunResult(task, result_text, total_tokens, total_turns, all_messages)
            except Exception as exc:  # noqa: BLE001
                total_tokens += attempt_tokens
                total_turns += attempt_turns
                all_messages.extend(messages)
                last_error = exc
                retry_feedback = str(exc)
                task.error = retry_feedback
                task.evidence = _message_evidence(all_messages)
                emit(
                    {
                        "type": "task_attempt_failed",
                        "task_id": task.id,
                        "attempt": task.attempt_count,
                        "error": retry_feedback,
                        "will_retry": attempt < retry_limit,
                    }
                )

        return TaskRunResult(
            task,
            "",
            total_tokens,
            total_turns,
            all_messages,
            last_error or RuntimeError("Task failed"),
        )

    async def _apply_task_result(
        self, result: TaskRunResult
    ) -> AsyncIterator[dict[str, Any]]:
        if result.error:
            result.task.mark_failed(str(result.error))
            yield {
                "type": "text_delta",
                "text": f"Failed [{result.task.id}]: {result.error}\n\n",
            }
        else:
            result.task.mark_completed(result.text)
            result.task.error = ""
            yield {
                "type": "text_delta",
                "text": f"Completed [{result.task.id}]: {_preview(result.text)}\n\n",
            }
        yield {
            "type": "plan_task_done",
            "task_id": result.task.id,
            "turns": result.turns,
            "tokens": result.tokens,
            "attempts": result.task.attempt_count,
            "status": result.task.status.value,
        }

    async def _review_task(
        self, task: Task, result: str, messages: list[Message]
    ) -> VerificationResult:
        payload = (
            f"Original task:\n{task.description}\n\n"
            f"Acceptance criteria:\n{json.dumps(task.acceptance_criteria, ensure_ascii=False)}\n\n"
            f"Execution result:\n{result}\n\n"
            f"Tool evidence:\n{json.dumps(_message_evidence(messages), ensure_ascii=False)}"
        )
        text = ""
        input_tokens = 0
        output_tokens = 0
        async for event in self.llm_client.chat(
            [Message(role="user", content=payload)], [], system_prompt=_TASK_REVIEW_PROMPT
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "usage":
                usage = event.get("usage") or {}
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
            elif event.get("type") == "error":
                return VerificationResult(False, str(event.get("error")))
        approved, issues = _parse_review(text)
        return VerificationResult(
            approved, "" if approved else issues, input_tokens, output_tokens
        )

    async def _synthesize(
        self, plan: ExecutionPlan, *, correction: str = ""
    ) -> tuple[str, int, int]:
        report = _build_plan_result(plan, self.policy.mode)
        payload = f"Original request:\n{plan.goal}\n\nCompleted task report:\n{report}"
        if correction:
            payload += (
                "\n\nPrivate quality requirements for the revision:\n"
                f"{correction}\nApply these requirements silently; do not mention them."
            )
        text = ""
        tokens = 0
        try:
            async for event in self.llm_client.chat(
                [Message(role="user", content=payload)],
                [],
                system_prompt=_SYNTHESIS_PROMPT,
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "usage":
                    usage = event.get("usage") or {}
                    tokens += int(usage.get("input_tokens") or 0)
                    tokens += int(usage.get("output_tokens") or 0)
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception:  # noqa: BLE001
            text = ""
        synthesis = text.strip()
        if synthesis:
            report = _build_synthesized_result(plan, self.policy.mode, synthesis)
        return report, tokens, 1

    async def _verify_final(
        self, plan: ExecutionPlan, final_result: str
    ) -> VerificationResult:
        if not self.config.agent.stop_hook_enabled:
            return VerificationResult(True)
        return await self.verifier.verify_final(
            original_request=plan.goal,
            proposed_answer=final_result,
            messages=_plan_evidence(plan),
        )

    async def _capture_memory(
        self, plan: ExecutionPlan, verdict: VerificationResult
    ) -> AsyncIterator[dict[str, Any]]:
        if not (
            self.config.features.memory
            and self.config.memory.long_term_enabled
            and self.config.memory.auto_memory_enabled
            and verdict.memory_candidates
        ):
            return
        try:
            report = capture_approved_memories(
                MemoryManager(self.config.memory.long_term_db_path, scope=self.cwd),
                verdict.memory_candidates,
                original_request=plan.goal,
                messages=_plan_evidence(plan),
                min_confidence=self.config.memory.auto_memory_min_confidence,
                max_candidates=self.config.memory.auto_memory_max_candidates,
            )
            yield {
                "type": "memory_capture",
                "stored_ids": report.stored_ids,
                "actions": [item.action for item in report.mutations],
                "rejected": report.rejected,
                "mode": self.policy.mode,
            }
        except Exception as exc:  # noqa: BLE001
            yield {
                "type": "memory_capture",
                "stored_ids": [],
                "actions": [],
                "rejected": [f"memory persistence failed: {exc}"],
                "mode": self.policy.mode,
            }

    def _task_system_prompt(self, task: Task, registry: ToolRegistry) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build()
        criteria = "; ".join(task.acceptance_criteria) or "return a concrete result"
        return (
            f"{base}\n\nYou are executing one task inside SmartCLI's shared DAG kernel.\n"
            f"Mode: {self.policy.mode}\nTask id: {task.id}\nTask type: {task.type.value}\n"
            f"Acceptance criteria: {criteria}\n"
            "Complete only this task. Use tools when needed and preserve concrete evidence."
        )


def _execution_stop_reason(
    budget: BudgetManager, deadline: float, *, minimum_turns: int = 1
) -> str:
    if monotonic() >= deadline:
        return "runtime_budget"
    return "+".join(budget.reached(minimum_turns=minimum_turns))


def _executable_tasks_in_order(plan: ExecutionPlan) -> list[Task]:
    executable_ids = {task.id for task in plan.executable_tasks()}
    return [
        plan.tasks[task_id]
        for task_id in plan.execution_order()
        if task_id in executable_ids
    ]


def _select_safe_batch(tasks: list[Task], worker_count: int) -> list[Task]:
    if not tasks:
        return []
    first = tasks[0]
    if not _parallel_candidate(first):
        return [first]
    batch = [first]
    for candidate in tasks[1:]:
        if len(batch) >= max(1, worker_count):
            break
        if _parallel_candidate(candidate) and all(
            not _resource_conflict(candidate, selected) for selected in batch
        ):
            batch.append(candidate)
    return batch


def _parallel_candidate(task: Task) -> bool:
    return task.parallel_safe and task.type not in {TaskType.FILE_WRITE, TaskType.COMMAND}


def _resource_conflict(left: Task, right: Task) -> bool:
    left_reads = {_normalize_path(item) for item in left.read_paths}
    right_reads = {_normalize_path(item) for item in right.read_paths}
    left_writes = {_normalize_path(item) for item in left.write_paths}
    right_writes = {_normalize_path(item) for item in right.write_paths}
    return bool(
        _path_sets_overlap(left_writes, right_writes | right_reads)
        or _path_sets_overlap(right_writes, left_reads)
    )


def _normalize_path(value: str) -> str:
    normalized = str(Path(value)).replace("\\", "/").rstrip("/").casefold()
    return normalized or "."


def _path_sets_overlap(left: set[str], right: set[str]) -> bool:
    return any(
        a == b or a.startswith(f"{b}/") or b.startswith(f"{a}/")
        for a in left
        for b in right
    )


def _read_only_registry(source: ToolRegistry) -> ToolRegistry:
    registry = ToolRegistry()
    for name in source.list_names():
        tool = source.get(name)
        if tool and tool.is_read_only and tool.is_concurrency_safe:
            registry.register(tool)
    return registry


def _task_context(
    plan: ExecutionPlan, task: Task, *, retry_feedback: str = ""
) -> str:
    lines = [
        f"Goal: {plan.goal}",
        f"Current task [{task.id}]: {task.description}",
        f"Acceptance criteria: {json.dumps(task.acceptance_criteria, ensure_ascii=False)}",
        "",
        "Completed dependency results:",
    ]
    for dep_id in task.dependencies:
        dep = plan.get_task(dep_id)
        if dep and dep.status == TaskStatus.COMPLETED:
            lines.append(f"- [{dep.id}] {dep.description}: {dep.result}")
    if retry_feedback:
        lines.extend(["", f"Previous attempt failed review: {retry_feedback}"])
    return "\n".join(lines)


def _build_plan_result(plan: ExecutionPlan, mode: str) -> str:
    heading = "Multi-Agent task completed." if mode == "team" else "Plan execution completed."
    lines = [heading, "", "Task report:"]
    for task in plan.all_tasks():
        lines.append(f"- [{task.id}] {task.status.value}: {task.description}")
        if task.result:
            lines.append(f"  Result: {task.result}")
        if task.error:
            lines.append(f"  Error: {task.error}")
        if task.evidence:
            lines.append(f"  Evidence: {' | '.join(task.evidence)}")
    return "\n".join(lines) + "\n"


def _build_synthesized_result(plan: ExecutionPlan, mode: str, synthesis: str) -> str:
    heading = "Multi-Agent task completed." if mode == "team" else "Plan execution completed."
    lines = [heading, "", synthesis.strip(), "", "Task outcomes:"]
    for task in plan.all_tasks():
        outcome = _preview(task.result, 180) if task.result else task.error or "no result"
        lines.append(f"- {task.description}: {outcome}")
    return "\n".join(lines) + "\n"


def _preview(text: str, max_len: int = 240) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _parse_review(text: str) -> tuple[bool, str]:
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return False, text.strip() or "Reviewer returned no verdict."
    if not isinstance(data, dict) or not isinstance(data.get("approved"), bool):
        return False, "Reviewer verdict must contain a JSON boolean approved field."
    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    feedback = "; ".join(str(item) for item in issues if str(item).strip())
    if not data["approved"] and not feedback:
        feedback = "Task reviewer rejected the result without details."
    return data["approved"], feedback


def _message_evidence(messages: list[Message], *, max_items: int = 20) -> list[str]:
    evidence: list[str] = []
    names: dict[str, str] = {}
    for message in messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                call_id = str(call.get("id") or "?")
                function = call.get("function") or {}
                names[call_id] = str(function.get("name") or "unknown")
        elif message.role == "tool":
            call_id = str(message.tool_call_id or "?")
            content = (
                message.content
                if isinstance(message.content, str)
                else json.dumps(message.content, ensure_ascii=False)
            )
            evidence.append(f"{call_id}:{names.get(call_id, 'tool')}={_preview(content, 500)}")
    return evidence[-max_items:]


def _plan_to_dict(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "schema_version": 2,
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
                "acceptance_criteria": task.acceptance_criteria,
                "read_paths": task.read_paths,
                "write_paths": task.write_paths,
                "parallel_safe": task.parallel_safe,
                "retry_limit": task.retry_limit,
                "attempt_count": task.attempt_count,
                "evidence": task.evidence,
            }
            for task in plan.all_tasks()
        ],
    }


def _plan_from_dict(data: dict[str, Any]) -> ExecutionPlan:
    from paicli.plan import PlanStatus

    plan = ExecutionPlan(
        id=str(data["id"]),
        goal=str(data["goal"]),
        summary=str(data.get("summary") or ""),
    )
    for item in data.get("tasks") or []:
        status = TaskStatus(str(item.get("status") or "PENDING"))
        if status == TaskStatus.RUNNING:
            status = TaskStatus.PENDING
        plan.add_task(
            Task(
                id=str(item["id"]),
                description=str(item["description"]),
                type=TaskType(str(item.get("type") or "ANALYSIS")),
                dependencies=list(item.get("dependencies") or []),
                status=status,
                result=str(item.get("result") or ""),
                error=str(item.get("error") or ""),
                acceptance_criteria=list(item.get("acceptance_criteria") or []),
                read_paths=list(item.get("read_paths") or []),
                write_paths=list(item.get("write_paths") or []),
                parallel_safe=item.get("parallel_safe") is True,
                retry_limit=max(0, int(item.get("retry_limit") or 0)),
                attempt_count=max(0, int(item.get("attempt_count") or 0)),
                evidence=list(item.get("evidence") or []),
            )
        )
    for task in plan.all_tasks():
        for dep_id in task.dependencies:
            dep = plan.get_task(dep_id)
            if dep:
                dep.add_dependent(task.id)
    errors = plan.validate()
    if errors:
        raise ValueError("Invalid saved plan: " + "; ".join(errors))
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


def _pause_state(
    state: RunState,
    store: RunStateStore,
    plan: ExecutionPlan,
    budget: BudgetManager,
) -> None:
    state.status = "PAUSED"
    _save_plan_state(state, store, plan, budget)


def _paused_event(state: RunState, reason: str, mode: str) -> dict[str, Any]:
    return {
        "type": "run_stopped",
        "reason": reason,
        "run_id": state.run_id,
        "message": (
            f"{mode.title()} run {state.run_id} paused; use /{mode} resume "
            f"{state.run_id} to continue it."
        ),
    }


def _plan_evidence(plan: ExecutionPlan) -> list[Message]:
    return [
        Message(
            role="tool",
            tool_call_id=f"{task.id}_evidence",
            content=(
                f"Task {task.id} status={task.status.value}\n"
                f"Result: {task.result}\nEvidence: {' | '.join(task.evidence)}"
            ),
        )
        for task in plan.all_tasks()
    ]
