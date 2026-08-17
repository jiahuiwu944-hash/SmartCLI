from __future__ import annotations

import json
import re
import time
from typing import Any

from paicli.llm.base import LlmClient
from paicli.plan.models import ExecutionPlan, Task, TaskType
from paicli.types import Message

PLANNER_PROMPT = """You are SmartCLI's planner.
Create a compact executable DAG for the user's task.
Return only JSON with this shape:
{
  "summary": "short summary",
  "tasks": [
    {
      "id": "stable_source_id",
      "description": "concrete executable step",
      "type": "FILE_READ|FILE_WRITE|COMMAND|ANALYSIS|VERIFICATION",
      "dependencies": ["stable_source_id"],
      "acceptance_criteria": ["observable completion condition"],
      "read_paths": ["known or expected workspace path"],
      "write_paths": ["known or expected workspace path"],
      "parallel_safe": false,
      "retry_limit": 1
    }
  ]
}
Use at most 24 tasks. Dependencies must reference unique task IDs from this plan.
Set parallel_safe=true only for read-only tasks whose declared paths do not conflict.
Every mutating task must declare its expected write_paths and concrete acceptance criteria.
When a workspace inventory is supplied, treat its paths as authoritative. Do not invent a
modules/ layout or create a separate path-discovery step for a path already in the inventory.
Keep each worker task end-to-end: do not split locating, reading, and analyzing one module into
separate tasks. A two-module comparison normally needs two independent tasks plus one synthesis.
"""


class Planner:
    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client
        self.last_tokens = 0
        self.last_turns = 0
        self.last_warning = ""

    async def create_plan(
        self,
        goal: str,
        *,
        workspace_context: str = "",
        max_tasks: int = 8,
    ) -> ExecutionPlan:
        self.last_tokens = 0
        self.last_turns = 0
        self.last_warning = ""
        if _is_simple_goal(goal):
            return _minimal_plan(goal)
        failure = ""
        for attempt in range(2):
            correction = (
                "\n\nThe previous response was invalid: "
                f"{failure}. Return a corrected JSON plan now."
                if attempt
                else ""
            )
            inventory = (
                f"\n\nAuthoritative workspace inventory:\n{workspace_context}"
                if workspace_context
                else ""
            )
            text, tokens = await _collect_text(
                self.llm_client,
                [
                    Message(
                        role="user",
                        content=(
                            f"Please create an execution plan for:\n{goal}\n\n"
                            f"Use at most {max(1, max_tasks)} tasks.{inventory}{correction}"
                        ),
                    )
                ],
                system_prompt=PLANNER_PROMPT,
            )
            self.last_tokens += tokens
            self.last_turns += 1
            try:
                return self.parse_plan(goal, text, task_limit=max_tasks)
            except (ValueError, json.JSONDecodeError) as exc:
                failure = str(exc)
        self.last_warning = (
            f"Planner returned no usable JSON after 2 attempts ({failure}); "
            "using a safe fallback plan."
        )
        return _fallback_plan(goal)

    async def replan(self, failed_plan: ExecutionPlan, failure_reason: str) -> ExecutionPlan:
        completed = "\n".join(
            f"- {task.id}: {task.description}"
            for task in failed_plan.all_tasks()
            if task.result and not task.error
        )
        return await self.create_plan(
            f"{failed_plan.goal}\nFailure reason: {failure_reason}\nCompleted tasks:\n{completed}"
        )

    def parse_plan(
        self, goal: str, plan_json: str, *, task_limit: int = 24
    ) -> ExecutionPlan:
        data = _parse_json_object(plan_json)
        task_nodes = data.get("tasks") or data.get("steps") or []
        if not isinstance(task_nodes, list) or not task_nodes:
            raise ValueError("planner output did not contain a non-empty tasks/steps array")
        limit = max(1, min(24, task_limit))
        if len(task_nodes) > limit:
            raise ValueError(f"planner output exceeds the {limit}-task limit")

        plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=goal)
        plan.summary = str(data.get("summary") or "")
        id_mapping: dict[str, str] = {}
        source_ids: set[str] = set()

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                raise ValueError(f"planner task {index} must be an object")
            original_id = str(node.get("id") or f"task_{index}")
            if original_id in source_ids:
                raise ValueError(f'duplicate planner task id: "{original_id}"')
            source_ids.add(original_id)
            new_id = f"task_{index}"
            id_mapping[original_id] = new_id
            task_type = _parse_task_type(str(node.get("type") or "ANALYSIS"))
            plan.add_task(
                Task(
                    id=new_id,
                    description=str(node.get("description") or original_id),
                    type=task_type,
                    acceptance_criteria=_string_list(node.get("acceptance_criteria")),
                    read_paths=_string_list(node.get("read_paths")),
                    write_paths=_string_list(node.get("write_paths")),
                    parallel_safe=(
                        node.get("parallel_safe") is True
                        and task_type not in {TaskType.FILE_WRITE, TaskType.COMMAND}
                    ),
                    retry_limit=max(0, min(3, _integer(node.get("retry_limit"), 1))),
                )
            )

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                continue
            task = plan.get_task(f"task_{index}")
            if not task:
                continue
            dependencies = node.get("dependencies") or []
            if not isinstance(dependencies, list):
                raise ValueError(f"dependencies for {task.id} must be an array")
            for raw_dep in dependencies:
                source_dep = str(raw_dep)
                if source_dep not in id_mapping:
                    raise ValueError(
                        f'{task.id} depends on unknown planner task "{source_dep}"'
                    )
                dep_id = id_mapping[source_dep]
                task.add_dependency(dep_id)
                plan.tasks[dep_id].add_dependent(task.id)

        errors = plan.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return plan


async def _collect_text(
    llm_client: LlmClient,
    messages: list[Message],
    *,
    system_prompt: str,
) -> tuple[str, int]:
    text = ""
    tokens = 0
    async for event in llm_client.chat(messages, [], system_prompt=system_prompt):
        event_type = event.get("type")
        if event_type == "text_delta":
            text += str(event.get("text") or "")
        elif event_type == "usage":
            usage = event.get("usage") or {}
            tokens += int(usage.get("input_tokens") or 0)
            tokens += int(usage.get("output_tokens") or 0)
        elif event_type == "error":
            raise event["error"]
    return text, tokens


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty planner output")
    return json.loads(cleaned)


def _parse_task_type(value: str) -> TaskType:
    normalized = value.upper()
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.ANALYSIS


def _is_simple_goal(goal: str | None) -> bool:
    normalized = (goal or "").strip()
    if not normalized or len(normalized) > 30:
        return False
    multi_step_cues = ["然后", "并且", "再", "最后", "同时", "先", "之后", "接着", "以及"]
    if any(cue in normalized for cue in multi_step_cues):
        return False
    simple_cues = ["列出", "查看", "读取", "显示", "执行", "运行", "搜索", "当前目录", "文件"]
    return any(cue in normalized for cue in simple_cues)


def _minimal_plan(goal: str) -> ExecutionPlan:
    normalized = goal.strip()
    plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=normalized)
    plan.summary = f"直接执行简单任务：{normalized}"
    plan.add_task(Task(id="task_1", description=normalized, type=_infer_simple_type(normalized)))
    plan.compute_execution_order()
    return plan


def _fallback_plan(goal: str) -> ExecutionPlan:
    normalized = goal.strip()
    plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=normalized)
    match = re.search(
        r"(?:并行)?(?:检查|分析|审查)\s*(.+?)\s*(?:和|与|、)\s*(.+?)(?:模块)?$",
        normalized,
        re.IGNORECASE,
    )
    if match:
        targets = [_clean_target(match.group(1)), _clean_target(match.group(2))]
        plan.summary = f"安全降级：并行检查 {targets[0]} 与 {targets[1]}"
        for index, target in enumerate(targets, start=1):
            plan.add_task(
                Task(
                    id=f"task_{index}",
                    description=(
                        f"检查 {target}，报告结构、关键实现、风险和可验证的改进建议"
                    ),
                    type=TaskType.ANALYSIS,
                    acceptance_criteria=[f"给出 {target} 的证据化检查结果"],
                    read_paths=_target_paths(target),
                    parallel_safe=True,
                    retry_limit=1,
                )
            )
        plan.add_task(
            Task(
                id="task_3",
                description="汇总两个模块的检查结果，标出共同问题和优先级",
                type=TaskType.ANALYSIS,
                dependencies=["task_1", "task_2"],
                acceptance_criteria=["形成去重、分级且可执行的汇总结论"],
                retry_limit=1,
            )
        )
    else:
        plan.summary = "安全降级：直接执行原始任务"
        plan.add_task(
            Task(
                id="task_1",
                description=normalized,
                type=_infer_simple_type(normalized),
                acceptance_criteria=["返回与原始请求直接对应的具体结果"],
                retry_limit=1,
            )
        )
    errors = plan.validate()
    if errors:  # pragma: no cover - constructed locally from fixed dependencies
        raise ValueError("invalid fallback plan: " + "; ".join(errors))
    return plan


def _clean_target(value: str) -> str:
    cleaned = value.strip(" ：:，,。.")
    return re.sub(r"\s*模块$", "", cleaned, flags=re.IGNORECASE).strip()


def _target_paths(target: str) -> list[str]:
    normalized = target.casefold()
    known = {
        "plan": "src/paicli/plan",
        "skill": "src/paicli/skill",
        "agent": "src/paicli/agent",
        "mcp": "src/paicli/mcp",
        "memory": "src/paicli/memory",
    }
    return [path for name, path in known.items() if name in normalized]


def _infer_simple_type(goal: str) -> TaskType:
    if any(token in goal for token in ["读取", "打开", "查看"]) and "文件" in goal:
        return TaskType.FILE_READ
    if any(token in goal for token in ["写入", "修改", "创建文件"]):
        return TaskType.FILE_WRITE
    if any(token in goal for token in ["分析", "总结", "解释"]):
        return TaskType.ANALYSIS
    if any(token in goal for token in ["验证", "检查"]):
        return TaskType.VERIFICATION
    return TaskType.COMMAND


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
