from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from paicli.agent.query import query
from paicli.agent.verifier import CompletionVerifier
from paicli.config import PaiCliConfig
from paicli.context import ContextRuntime
from paicli.llm.base import LlmClient
from paicli.memory import MemoryManager, capture_approved_memories
from paicli.prompt import PromptAssembler
from paicli.runtime.budget import BudgetManager
from paicli.runtime.run_state import RunState, RunStateStore, is_resume_request
from paicli.skill import SkillContextBuffer
from paicli.snapshot import SnapshotService
from paicli.tools.hooks import ToolHookManager
from paicli.tools.registry import ToolRegistry
from paicli.types import Message


class AgentRole(StrEnum):
    PLANNER = "PLANNER"
    WORKER = "WORKER"
    REVIEWER = "REVIEWER"


class AgentMessageType(StrEnum):
    TASK = "TASK"
    RESULT = "RESULT"
    FEEDBACK = "FEEDBACK"
    APPROVAL = "APPROVAL"
    REJECTION = "REJECTION"
    ERROR = "ERROR"


class StepStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(slots=True)
class AgentMessage:
    from_agent: str
    from_role: AgentRole | None
    content: str
    type: AgentMessageType
    tokens: int = 0
    turns: int = 0
    tool_successes: int = 0
    tool_failures: int = 0

    @classmethod
    def task(cls, from_agent: str, content: str) -> AgentMessage:
        return cls(from_agent, None, content, AgentMessageType.TASK)

    @classmethod
    def result(cls, from_agent: str, role: AgentRole, content: str) -> AgentMessage:
        return cls(from_agent, role, content, AgentMessageType.RESULT)

    @classmethod
    def error(cls, from_agent: str, role: AgentRole, content: str) -> AgentMessage:
        return cls(from_agent, role, content, AgentMessageType.ERROR)


@dataclass(slots=True)
class ExecutionStep:
    id: str
    description: str
    type: str
    dependencies: list[str]
    result: str = ""
    status: StepStatus = StepStatus.PENDING
    retry_count: int = 0
    tokens: int = 0
    turns: int = 0

    def with_result(self, result: str) -> ExecutionStep:
        return replace(self, result=result, status=StepStatus.COMPLETED)

    def with_failed(self, result: str) -> ExecutionStep:
        return replace(self, result=result, status=StepStatus.FAILED)

    def started(self) -> ExecutionStep:
        return replace(self, status=StepStatus.RUNNING)


class SubAgent:
    def __init__(
        self,
        *,
        name: str,
        role: AgentRole,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: PaiCliConfig,
        cwd: str,
        approval_callback=None,
        tool_hook_manager: ToolHookManager | None = None,
        skill_context_buffer: SkillContextBuffer | None = None,
    ):
        self.name = name
        self.role = role
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.tool_hook_manager = tool_hook_manager
        self.skill_context_buffer = skill_context_buffer or SkillContextBuffer()
        self.history: list[Message] = []
        self.context_runtime = ContextRuntime(llm_client, config)

    async def execute(self, task: AgentMessage, context: str = "") -> AgentMessage:
        content = f"{context}\n\nCurrent task:\n{task.content}".strip() if context else task.content
        if self.role == AgentRole.WORKER:
            return await self._execute_worker(content)
        return await self._execute_without_tools(content)

    async def review(self, original_task: str, execution_result: str) -> AgentMessage:
        return await self.execute(
            AgentMessage.task(
                "orchestrator",
                f"Original task:\n{original_task}\n\nExecution result:\n{execution_result}",
            )
        )

    def clear_history(self) -> None:
        self.history = []
        self.context_runtime.reset()

    async def _execute_worker(self, content: str) -> AgentMessage:
        text = ""
        tool_results: list[str] = []
        tokens = 0
        turns = 0
        tool_successes = 0
        tool_failures = 0
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._system_prompt(),
                user_message=content,
                history=self.history,
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                tool_hook_manager=self.tool_hook_manager,
                skill_context_buffer=self.skill_context_buffer,
                context_runtime=self.context_runtime,
                max_turns=8,
                stop_hook_enabled=False,
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "tool_result":
                    tool_results.append(str(event.get("result") or ""))
                    if event.get("is_error"):
                        tool_failures += 1
                    else:
                        tool_successes += 1
                elif event.get("type") == "done":
                    self.history = list(event.get("messages") or [])
                    turns += int(event.get("total_turns") or 0)
                    tokens += int(event.get("total_tokens") or 0)
                elif event.get("type") == "run_stopped":
                    raise RuntimeError(str(event.get("message") or "Agent run stopped"))
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc))
        result = text.strip() or "\n".join(item for item in tool_results if item).strip()
        message = AgentMessage.result(self.name, self.role, result)
        message.tokens = tokens
        message.turns = turns
        message.tool_successes = tool_successes
        message.tool_failures = tool_failures
        return message

    async def _execute_without_tools(self, content: str) -> AgentMessage:
        text = ""
        tokens = 0
        messages = [*self.history, Message(role="user", content=content)]
        protected_message = messages[-1]
        system_prompt = self._system_prompt()
        try:
            prepared = await self.context_runtime.prepare(
                system_prompt=system_prompt,
                messages=messages,
                tools=[],
                protected_message=protected_message,
            )
            messages = prepared.messages
            tokens += prepared.internal_input_tokens + prepared.internal_output_tokens
            async for event in self.llm_client.chat(
                messages,
                [],
                system_prompt=system_prompt,
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "usage":
                    usage = event.get("usage") or {}
                    tokens += int(usage.get("input_tokens") or 0)
                    tokens += int(usage.get("output_tokens") or 0)
                    self.context_runtime.observe_usage(
                        int(usage.get("input_tokens") or 0),
                        prepared.estimated_input_tokens,
                    )
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc))
        self.history = [*messages, Message(role="assistant", content=text)]
        message = AgentMessage.result(self.name, self.role, text)
        message.tokens = tokens
        message.turns = 1
        return message

    def _system_prompt(self) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build()
        role_prompt = {
            AgentRole.PLANNER: (
                "You are the Planner in a multi-agent workflow. Return only JSON with a "
                "steps array. Each step needs id, description, type, and dependencies."
            ),
            AgentRole.WORKER: (
                "You are the Worker in a multi-agent workflow. Execute only the assigned "
                "step. Use tools when needed and return the concrete result."
            ),
            AgentRole.REVIEWER: (
                "You are the Reviewer in a multi-agent workflow. Return JSON only: "
                '{"approved": true|false, "summary": "...", "issues": []}.'
            ),
        }[self.role]
        return f"{base}\n\n{role_prompt}\nAgent name: {self.name}"


class AgentOrchestrator:
    max_retries_per_step = 2

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
        worker_count: int = 2,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.continuation_callback = continuation_callback
        self.tool_hook_manager = tool_hook_manager
        self.skill_context_buffer = SkillContextBuffer()
        self.planner = self._subagent("planner", AgentRole.PLANNER)
        self.workers = [
            self._subagent(f"worker-{index}", AgentRole.WORKER)
            for index in range(1, max(1, worker_count) + 1)
        ]
        self.reviewer = self._subagent("reviewer", AgentRole.REVIEWER)
        self.verifier = CompletionVerifier(llm_client)
        self.history: list[Message] = []

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        final_text = ""
        termination_reason = "completed"
        try:
            store = RunStateStore(self.cwd)
            resumed = store.latest_paused("team") if is_resume_request(message) else None
            if resumed:
                state = resumed
                state.status = "RUNNING"
                message = state.goal
                steps = _steps_from_dict(state.payload.get("steps") or [])
                yield {"type": "text_delta", "text": f"Resuming team {state.run_id}.\n\n"}
            else:
                yield {"type": "text_delta", "text": "Phase 1: planner\n\n"}
                plan_result = await self.planner.execute(
                    AgentMessage.task("orchestrator", f"Create an execution plan for:\n{message}")
                )
                self.planner.clear_history()
                if plan_result.type == AgentMessageType.ERROR:
                    raise RuntimeError(f"planner failed: {plan_result.content}")
                steps = self.parse_plan(plan_result.content)
                if not steps:
                    raise ValueError(f"planner output could not be parsed:\n{plan_result.content}")
                state = store.create("team", message)
                state.turns = plan_result.turns
                state.tokens = plan_result.tokens
                state.payload["steps"] = _steps_to_dict(steps)
                store.save(state)
            yield {"type": "text_delta", "text": self.summarize_steps(steps) + "\n"}
            yield {"type": "text_delta", "text": "Phase 2: workers and reviewer\n\n"}
            budget = BudgetManager(
                turn_limit=self.config.agent.max_turns,
                token_limit=self.config.agent.max_total_tokens,
                callback=self.continuation_callback,
            )
            budget.turns, budget.tokens = state.turns, state.tokens
            for event in await self._execute_steps(
                steps,
                lambda text: {"type": "text_delta", "text": text},
                state,
                store,
                budget,
            ):
                if event.get("type") == "run_stopped":
                    termination_reason = str(event.get("reason") or "budget")
                yield event
            if state.status == "PAUSED":
                final_text = f"Team {state.run_id} is paused with execution context saved."
            else:
                final_text = self.build_final_result(steps)
                all_completed = all(step.status == StepStatus.COMPLETED for step in steps)
                if all_completed:
                    evidence = _team_evidence(steps)
                    verdict = await self.verifier.verify_final(
                        original_request=message,
                        proposed_answer=final_text,
                        messages=evidence,
                    )
                    budget.consume(
                        tokens=verdict.input_tokens + verdict.output_tokens,
                    )
                    if verdict.input_tokens or verdict.output_tokens:
                        yield {
                            "type": "usage",
                            "usage": {
                                "input_tokens": verdict.input_tokens,
                                "output_tokens": verdict.output_tokens,
                            },
                            "source": "stop_hook",
                        }
                    yield {
                        "type": "stop_hook_review",
                        "approved": verdict.approved,
                        "feedback": verdict.feedback,
                        "mode": "team",
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
                                    original_request=message,
                                    messages=evidence,
                                    min_confidence=(
                                        self.config.memory.auto_memory_min_confidence
                                    ),
                                    max_candidates=(
                                        self.config.memory.auto_memory_max_candidates
                                    ),
                                )
                                yield {
                                    "type": "memory_capture",
                                    "stored_ids": report.stored_ids,
                                    "actions": [
                                        item.action for item in report.mutations
                                    ],
                                    "rejected": report.rejected,
                                    "mode": "team",
                                }
                            except Exception as exc:  # noqa: BLE001
                                yield {
                                    "type": "memory_capture",
                                    "stored_ids": [],
                                    "actions": [],
                                    "rejected": [f"memory persistence failed: {exc}"],
                                    "mode": "team",
                                }
                        state.status = "COMPLETED"
                        yield {"type": "text_delta", "text": final_text}
                    else:
                        state.status = "FAILED"
                        final_text = f"Team verification failed: {verdict.feedback}"
                        yield {"type": "text_delta", "text": final_text}
                else:
                    state.status = "FAILED"
                    yield {"type": "text_delta", "text": final_text}
                state.payload["steps"] = _steps_to_dict(steps)
                state.turns, state.tokens = budget.turns, budget.tokens
                store.save(state)
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
            "total_turns": state.turns,
            "total_tokens": state.tokens,
            "termination_reason": termination_reason,
            "completed": termination_reason == "completed" and state.status == "COMPLETED",
            "messages": self.history,
        }

    async def _execute_steps(
        self,
        steps: list[ExecutionStep],
        event_factory,
        state: RunState,
        store: RunStateStore,
        budget: BudgetManager,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        retry_count: dict[str, int] = {}
        worker_queue: asyncio.Queue[SubAgent] = asyncio.Queue()
        for worker in self.workers:
            worker_queue.put_nowait(worker)

        while True:
            if budget.reached():
                approved, request = await budget.request_extension(
                    additional_turns=self.config.agent.budget_extension_turns,
                    additional_tokens=self.config.agent.budget_extension_tokens,
                    mode="team",
                )
                events.append({"type": "budget_extension_requested", **request})
                if not approved:
                    state.status = "PAUSED"
                    state.payload["steps"] = _steps_to_dict(steps)
                    state.turns, state.tokens = budget.turns, budget.tokens
                    store.save(state)
                    events.append(
                        {
                            "type": "run_stopped",
                            "reason": request["reason"],
                            "message": (
                                f"Team {state.run_id} paused; run /team 继续 to resume it."
                            ),
                        }
                    )
                    break
                events.append({"type": "budget_extended", **request})
            executable = self.get_executable_steps(steps)
            if not executable:
                break
            if len(executable) > 1:
                events.append(
                    event_factory(
                        f"Parallel batch: {', '.join(step.id for step in executable)}\n\n"
                    )
                )
            usage = await asyncio.gather(
                *(
                    self._run_step_with_worker_queue(
                        step,
                        steps,
                        retry_count,
                        worker_queue,
                    )
                    for step in executable
                )
            )
            for turns, tokens in usage:
                budget.consume(turns=turns, tokens=tokens)
            state.payload["steps"] = _steps_to_dict(steps)
            state.turns, state.tokens = budget.turns, budget.tokens
            store.save(state)
        return events

    async def _run_step_with_worker_queue(
        self,
        step: ExecutionStep,
        steps: list[ExecutionStep],
        retry_count: dict[str, int],
        worker_queue: asyncio.Queue[SubAgent],
    ) -> tuple[int, int]:
        worker = await worker_queue.get()
        try:
            reviewer = self._subagent(f"reviewer-{step.id}", AgentRole.REVIEWER)
            return await self._run_step(step, steps, retry_count, worker, reviewer)
        finally:
            worker.clear_history()
            worker_queue.put_nowait(worker)

    async def _run_step(
        self,
        step: ExecutionStep,
        steps: list[ExecutionStep],
        retry_count: dict[str, int],
        worker: SubAgent,
        reviewer: SubAgent,
    ) -> tuple[int, int]:
        self._update_step(steps, step.id, step.started())
        context = self.build_step_context(steps, step)
        task_msg = AgentMessage.task("orchestrator", step.description)
        result = await worker.execute(task_msg, context)
        total_turns = result.turns
        total_tokens = result.tokens
        if result.type == AgentMessageType.ERROR or not result.content.strip():
            self._update_step(steps, step.id, step.with_failed(result.content or "empty result"))
            return total_turns, total_tokens
        evidence = [Message(role="tool", content="TOOL_OK") for _ in range(result.tool_successes)]
        evidence.extend(
            Message(role="tool", content='{"is_error": true}') for _ in range(result.tool_failures)
        )
        deterministic = self.verifier.verify_task(
            description=step.description,
            result=result.content,
            messages=evidence,
        )
        if not deterministic.approved:
            self._update_step(
                steps,
                step.id,
                step.with_failed(deterministic.feedback),
            )
            return total_turns, total_tokens

        accepted_result = result.content
        review = await reviewer.review(
            step.description,
            _review_evidence(accepted_result, result),
        )
        total_turns += review.turns
        total_tokens += review.tokens
        reviewer.clear_history()
        approved = self.parse_review_approval(review.content)
        issues = self.parse_review_issues(review.content)
        retries = retry_count.get(step.id, 0)
        while not approved and retries < self.max_retries_per_step:
            retries += 1
            retry_count[step.id] = retries
            retry_context = context + f"\n\nReviewer rejected the previous result:\n{issues}"
            retry_result = await worker.execute(task_msg, retry_context)
            total_turns += retry_result.turns
            total_tokens += retry_result.tokens
            if retry_result.type == AgentMessageType.ERROR or not retry_result.content.strip():
                issues = retry_result.content or "empty retry result"
                continue
            accepted_result = retry_result.content
            retry_review = await reviewer.review(
                step.description,
                _review_evidence(accepted_result, retry_result),
            )
            total_turns += retry_review.turns
            total_tokens += retry_review.tokens
            reviewer.clear_history()
            approved = self.parse_review_approval(retry_review.content)
            issues = self.parse_review_issues(retry_review.content)

        current = next(item for item in steps if item.id == step.id)
        current.retry_count = retries
        current.turns = total_turns
        current.tokens = total_tokens
        if approved:
            self._update_step(steps, step.id, current.with_result(accepted_result))
        else:
            self._update_step(
                steps,
                step.id,
                current.with_failed(issues or "reviewer rejected the result"),
            )
        return total_turns, total_tokens

    def parse_plan(self, plan_json: str) -> list[ExecutionStep]:
        try:
            data = _parse_json_object(plan_json)
        except (json.JSONDecodeError, ValueError):
            return []
        nodes = data.get("steps") or data.get("tasks") or []
        if not isinstance(nodes, list) or not nodes:
            return []
        id_mapping: dict[str, str] = {}
        steps: list[ExecutionStep] = []
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            original_id = str(node.get("id") or f"step_{index}")
            new_id = f"step_{index}"
            id_mapping[original_id] = new_id
            steps.append(
                ExecutionStep(
                    id=new_id,
                    description=str(node.get("description") or original_id),
                    type=str(node.get("type") or "COMMAND"),
                    dependencies=[],
                )
            )
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict) or index > len(steps):
                continue
            raw_deps = node.get("dependencies") or []
            if not isinstance(raw_deps, list):
                continue
            steps[index - 1].dependencies = [
                id_mapping.get(str(dep), str(dep)) for dep in raw_deps if str(dep)
            ]
        return steps

    def get_executable_steps(self, steps: list[ExecutionStep]) -> list[ExecutionStep]:
        status = {step.id: step.status for step in steps}
        return [
            step
            for step in steps
            if step.status == StepStatus.PENDING
            and all(status.get(dep) == StepStatus.COMPLETED for dep in step.dependencies)
        ]

    def parse_review_approval(self, review_content: str | None) -> bool:
        if not review_content:
            return False
        try:
            data = _parse_json_object(review_content)
            if "approved" not in data:
                return False
            return bool(data.get("approved"))
        except (json.JSONDecodeError, ValueError):
            lower = review_content.lower()
            negative = ["未通过", "不通过", "不合格", "有问题", '"approved": false']
            positive = ["通过", "合格", '"approved": true']
            if any(item in lower for item in negative):
                return False
            return any(item in lower for item in positive)

    def parse_review_issues(self, review_content: str | None) -> str:
        if not review_content:
            return ""
        try:
            data = _parse_json_object(review_content)
        except (json.JSONDecodeError, ValueError):
            return "review rejected the result"
        for key in ("issues", "suggestions"):
            value = data.get(key)
            if isinstance(value, list) and value:
                return "\n".join(f"- {item}" for item in value)
        return str(data.get("summary") or "review rejected the result")

    def build_step_context(self, steps: list[ExecutionStep], current_step: ExecutionStep) -> str:
        lines = ["Overall task context:"]
        for step in steps:
            if step.id in current_step.dependencies and step.status == StepStatus.COMPLETED:
                lines.append(f"[{step.id}] {step.description}")
                if step.result:
                    lines.append(f"Result: {_preview(step.result, 500)}")
        return "\n".join(lines)

    def summarize_steps(self, steps: list[ExecutionStep]) -> str:
        lines = ["Execution plan:"]
        for step in steps:
            deps = ", ".join(step.dependencies) if step.dependencies else "none"
            lines.append(f"- [{step.id}] {step.description} ({step.type}, deps: {deps})")
        return "\n".join(lines)

    def build_final_result(self, steps: list[ExecutionStep]) -> str:
        all_completed = all(step.status == StepStatus.COMPLETED for step in steps)
        failed = any(step.status == StepStatus.FAILED for step in steps)
        if all_completed:
            header = "Multi-Agent task completed."
        elif failed:
            header = "Multi-Agent task did not fully complete; failed steps remain."
        else:
            header = "Multi-Agent task partially completed; pending steps remain."
        lines = [header, "", "Execution summary:"]
        for step in steps:
            icon = {
                StepStatus.COMPLETED: "COMPLETED",
                StepStatus.FAILED: "FAILED",
                StepStatus.PENDING: "PENDING",
                StepStatus.RUNNING: "RUNNING",
            }[step.status]
            lines.append(f"- [{step.id}] {icon}: {step.description}")
            if step.result:
                lines.append(f"  Result: {_preview(step.result)}")
        return "\n".join(lines) + "\n"

    def _subagent(self, name: str, role: AgentRole) -> SubAgent:
        return SubAgent(
            name=name,
            role=role,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            tool_hook_manager=self.tool_hook_manager,
            # Do not share task-scoped skill instructions across concurrent sub-agents.
            skill_context_buffer=SkillContextBuffer(),
        )

    def _update_step(
        self,
        steps: list[ExecutionStep],
        step_id: str,
        updated: ExecutionStep,
    ) -> None:
        for index, step in enumerate(steps):
            if step.id == step_id:
                steps[index] = updated
                return


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty JSON")
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _preview(text: str, max_len: int = 160) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _steps_to_dict(steps: list[ExecutionStep]) -> list[dict[str, Any]]:
    return [
        {
            "id": step.id,
            "description": step.description,
            "type": step.type,
            "dependencies": step.dependencies,
            "result": step.result,
            "status": step.status.value,
            "retry_count": step.retry_count,
            "tokens": step.tokens,
            "turns": step.turns,
        }
        for step in steps
    ]


def _steps_from_dict(items: list[dict[str, Any]]) -> list[ExecutionStep]:
    steps: list[ExecutionStep] = []
    for item in items:
        status = StepStatus(str(item.get("status") or "PENDING"))
        if status == StepStatus.RUNNING:
            status = StepStatus.PENDING
        steps.append(
            ExecutionStep(
                id=str(item["id"]),
                description=str(item["description"]),
                type=str(item.get("type") or "COMMAND"),
                dependencies=list(item.get("dependencies") or []),
                result=str(item.get("result") or ""),
                status=status,
                retry_count=int(item.get("retry_count") or 0),
                tokens=int(item.get("tokens") or 0),
                turns=int(item.get("turns") or 0),
            )
        )
    return steps


def _review_evidence(result: str, message: AgentMessage) -> str:
    return (
        f"{result}\n\nTool evidence: {message.tool_successes} successful result(s), "
        f"{message.tool_failures} failed result(s). A failed tool must not be treated as "
        "successful unless later evidence clearly proves recovery."
    )


def _team_evidence(steps: list[ExecutionStep]) -> list[Message]:
    return [
        Message(
            role="tool",
            tool_call_id=f"team_{step.id}",
            content=f"Step {step.id} status={step.status.value}: {step.result}",
        )
        for step in steps
    ]
