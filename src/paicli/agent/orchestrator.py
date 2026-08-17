from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from paicli.agent.plan_execute import PlanExecuteAgent
from paicli.agent.query import query
from paicli.config import PaiCliConfig
from paicli.context import ContextRuntime
from paicli.llm.base import LlmClient
from paicli.plan import Planner, Task, TaskStatus
from paicli.prompt import PromptAssembler
from paicli.skill import SkillContextBuffer
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


# Compatibility aliases now point to the canonical plan model. There is no second
# multi-agent status machine or execution-step dataclass.
ExecutionStep = Task
StepStatus = TaskStatus


class SubAgent:
    """Compatibility facade for callers that use an individual worker directly."""

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
                event_type = event.get("type")
                if event_type == "text_delta":
                    text += str(event.get("text") or "")
                elif event_type == "tool_result":
                    tool_results.append(str(event.get("result") or ""))
                    if event.get("is_error"):
                        tool_failures += 1
                    else:
                        tool_successes += 1
                elif event_type == "usage":
                    usage = event.get("usage") or {}
                    tokens += int(usage.get("input_tokens") or 0)
                    tokens += int(usage.get("output_tokens") or 0)
                elif event_type == "done":
                    self.history = list(event.get("messages") or [])
                    turns = int(event.get("total_turns") or 0)
                elif event_type == "error":
                    raise event["error"]
                elif event_type == "run_stopped":
                    raise RuntimeError(str(event.get("message") or "Agent run stopped"))
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc))
        content = text.strip() or "\n".join(item for item in tool_results if item).strip()
        result = AgentMessage.result(self.name, self.role, content)
        result.tokens = tokens
        result.turns = turns
        result.tool_successes = tool_successes
        result.tool_failures = tool_failures
        return result

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
                messages, [], system_prompt=system_prompt
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
        result = AgentMessage.result(self.name, self.role, text)
        result.tokens = tokens
        result.turns = 1
        return result

    def _system_prompt(self) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build()
        role_prompt = {
            AgentRole.PLANNER: "Create a valid SmartCLI execution plan and return JSON only.",
            AgentRole.WORKER: "Execute only the assigned task and preserve concrete evidence.",
            AgentRole.REVIEWER: (
                'Return JSON only: {"approved": true|false, "issues": []}. '
                "The approved value must be a JSON boolean."
            ),
        }[self.role]
        return f"{base}\n\n{role_prompt}\nAgent name: {self.name}"


class AgentOrchestrator(PlanExecuteAgent):
    """Team policy over the shared PlanExecuteAgent orchestration kernel."""

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
        super().__init__(
            llm_client=llm_client,
            tool_registry=tool_registry,
            config=config,
            cwd=cwd,
            approval_callback=approval_callback,
            continuation_callback=continuation_callback,
            tool_hook_manager=tool_hook_manager,
            mode="team",
            worker_count=worker_count,
            review_each_task=True,
            max_retries_per_task=self.max_retries_per_step,
        )

    def parse_plan(self, plan_json: str) -> list[Task]:
        plan = Planner(self.llm_client).parse_plan("team task", plan_json)
        # Preserve the historical helper's step_* IDs without maintaining a second model.
        mapping = {task.id: task.id.replace("task_", "step_", 1) for task in plan.all_tasks()}
        for task in plan.all_tasks():
            task.id = mapping[task.id]
            task.dependencies = [mapping[item] for item in task.dependencies]
            task.dependents = [mapping[item] for item in task.dependents]
        return plan.all_tasks()

    def parse_review_approval(self, review_text: str) -> bool:
        data = _review_json(review_text)
        return bool(data and isinstance(data.get("approved"), bool) and data["approved"])

    def parse_review_issues(self, review_text: str) -> str:
        data = _review_json(review_text)
        if data is None:
            return review_text.strip()
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [issues]
        return "; ".join(str(item) for item in issues if str(item).strip())

    def get_executable_steps(self, steps: list[Task]) -> list[Task]:
        status = {step.id: step.status for step in steps}
        return [
            step
            for step in steps
            if step.status == TaskStatus.PENDING
            and all(status.get(dep) == TaskStatus.COMPLETED for dep in step.dependencies)
        ]

    def build_step_context(self, steps: list[Task], current_step: Task) -> str:
        rows = ["Completed dependency results:"]
        for step in steps:
            if step.id in current_step.dependencies and step.status == TaskStatus.COMPLETED:
                rows.append(f"- [{step.id}] {step.description}: {step.result}")
        return "\n".join(rows)

    def summarize_steps(self, steps: list[Task]) -> str:
        return "\n".join(
            f"- [{step.id}] {step.status.value}: {step.description}" for step in steps
        )

    def build_final_result(self, steps: list[Task]) -> str:
        lines = ["Multi-Agent task completed.", "", "Task report:"]
        for step in steps:
            lines.append(f"- [{step.id}] {step.status.value}: {step.description}")
            if step.result:
                lines.append(f"  Result: {step.result}")
            if step.error:
                lines.append(f"  Error: {step.error}")
        return "\n".join(lines) + "\n"


def _review_json(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
