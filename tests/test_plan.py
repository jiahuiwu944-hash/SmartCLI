from __future__ import annotations

import asyncio
from typing import Any

import pytest

from paicli.agent import PlanExecuteAgent
from paicli.agent.plan_execute import (
    _build_plan_result,
    _parse_review,
    _planner_task_limit,
    _select_safe_batch,
    _workspace_inventory,
)
from paicli.config import load_config
from paicli.plan import ExecutionPlan, Planner, Task, TaskStatus, TaskType
from paicli.tools import ToolRegistry, get_builtin_tools


def test_execution_plan_exposes_dag_batches():
    plan = ExecutionPlan(id="plan_1", goal="demo")
    task_1 = Task("task_1", "read a", TaskType.FILE_READ)
    task_2 = Task("task_2", "read b", TaskType.FILE_READ)
    task_3 = Task("task_3", "summarize", TaskType.ANALYSIS, ["task_1", "task_2"])

    plan.add_task(task_1)
    plan.add_task(task_2)
    plan.add_task(task_3)

    assert plan.execution_order() == ["task_1", "task_2", "task_3"]
    assert plan.execution_batches() == [[task_1, task_2], [task_3]]
    assert plan.executable_tasks() == [task_1, task_2]
    task_1.mark_completed("done")
    assert plan.executable_tasks() == [task_2]


def test_planner_parses_tasks_and_dependencies():
    planner = Planner(FakeClient())

    plan = planner.parse_plan(
        "demo",
        """
        ```json
        {
          "summary": "demo plan",
          "tasks": [
            {"id": "a", "description": "A", "type": "COMMAND", "dependencies": []},
            {"id": "b", "description": "B", "type": "VERIFICATION", "dependencies": ["a"]}
          ]
        }
        ```
        """,
    )

    assert plan.summary == "demo plan"
    assert plan.get_task("task_2").dependencies == ["task_1"]
    assert plan.get_task("task_2").type == TaskType.VERIFICATION


@pytest.mark.parametrize(
    "tasks",
    [
        [
            {"id": "a", "description": "A", "dependencies": []},
            {"id": "a", "description": "B", "dependencies": []},
        ],
        [{"id": "a", "description": "A", "dependencies": ["missing"]}],
        [
            {"id": "a", "description": "A", "dependencies": ["b"]},
            {"id": "b", "description": "B", "dependencies": ["a"]},
        ],
    ],
)
def test_planner_rejects_invalid_graphs(tasks):
    planner = Planner(FakeClient())

    with pytest.raises(ValueError):
        planner.parse_plan("demo", __import__("json").dumps({"tasks": tasks}))


def test_resource_scheduler_only_batches_declared_non_conflicting_reads():
    read_a = Task(
        "a",
        "read a",
        TaskType.FILE_READ,
        parallel_safe=True,
        read_paths=["src/a.py"],
    )
    read_b = Task(
        "b",
        "read b",
        TaskType.FILE_READ,
        parallel_safe=True,
        read_paths=["src/b.py"],
    )
    write = Task(
        "write",
        "write a",
        TaskType.FILE_WRITE,
        parallel_safe=True,
        write_paths=["src/a.py"],
    )

    assert _select_safe_batch([read_a, read_b], 4) == [read_a, read_b]
    assert _select_safe_batch([write, read_b], 4) == [write]


def test_failed_dependencies_are_propagated_as_blocked():
    plan = ExecutionPlan(id="plan", goal="demo")
    failed = Task("a", "A")
    dependent = Task("b", "B", dependencies=["a"])
    plan.add_task(failed)
    plan.add_task(dependent)
    failed.mark_failed("boom")

    blocked = plan.propagate_blocked()

    assert blocked == [dependent]
    assert dependent.status == TaskStatus.BLOCKED
    assert "a" in dependent.error


def test_failed_team_report_does_not_claim_completion():
    plan = ExecutionPlan(id="plan", goal="demo")
    task = Task("a", "A")
    plan.add_task(task)
    task.mark_failed("boom")

    result = _build_plan_result(plan, "team")

    assert result.startswith("Multi-Agent task failed.")
    assert "task completed" not in result.lower()


def test_workspace_inventory_exposes_real_module_paths(tmp_path):
    (tmp_path / "src" / "paicli" / "plan").mkdir(parents=True)
    (tmp_path / "src" / "paicli" / "skill").mkdir(parents=True)
    (tmp_path / ".git").mkdir()

    inventory = _workspace_inventory(str(tmp_path))

    assert "src/paicli/plan/" in inventory
    assert "src/paicli/skill/" in inventory
    assert ".git" not in inventory
    assert _planner_task_limit(20, review_each_task=True) == 4


def test_review_parser_requires_a_real_json_boolean():
    assert _parse_review('{"approved": true, "issues": []}')[0]
    assert not _parse_review('{"approved": "false", "issues": []}')[0]


def test_planner_retries_an_empty_successful_stream():
    client = EmptyThenValidPlannerClient()
    planner = Planner(client)

    plan = asyncio.run(planner.create_plan("先检查 Plan，然后给出结论"))

    assert client.calls == 2
    assert planner.last_turns == 2
    assert plan.get_task("task_1").description == "Inspect Plan"
    assert planner.last_warning == ""


def test_planner_uses_parallel_fallback_after_repeated_empty_streams():
    planner = Planner(AlwaysEmptyPlannerClient())

    plan = asyncio.run(planner.create_plan("并行检查 Plan 和 Skill 模块"))

    assert planner.last_warning
    assert len(plan.all_tasks()) == 3
    assert plan.get_task("task_1").parallel_safe
    assert plan.get_task("task_2").parallel_safe
    assert plan.get_task("task_3").dependencies == ["task_1", "task_2"]


def test_plan_execute_runs_independent_tasks_in_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    client = ParallelPlanClient()
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    agent = PlanExecuteAgent(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run():
        text = ""
        events = []
        async for event in agent.run("先做 A 和 B，然后汇总"):
            events.append(event)
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text, events

    result, events = asyncio.run(run())

    assert "Completed [task_1]" in result
    assert "Completed [task_2]" in result
    assert client.peak_concurrency == 2
    assert any(event.get("type") == "task_started" for event in events)
    assert any(event.get("type") == "task_text_delta" for event in events)
    first_delta = next(
        i for i, event in enumerate(events) if event.get("type") == "task_text_delta"
    )
    first_done = next(
        i for i, event in enumerate(events) if event.get("type") == "plan_task_done"
    )
    assert first_delta < first_done


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "{}"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


class EmptyThenValidPlannerClient(FakeClient):
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 2:
            yield {
                "type": "text_delta",
                "text": (
                    '{"tasks":[{"id":"a","description":"Inspect Plan",'
                    '"type":"ANALYSIS","dependencies":[]}]}'
                ),
            }
        yield {"type": "message_end", "stop_reason": "end_turn"}


class AlwaysEmptyPlannerClient(FakeClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "message_end", "stop_reason": "end_turn"}


class ParallelPlanClient(FakeClient):
    def __init__(self):
        self.current_concurrency = 0
        self.peak_concurrency = 0
        self.ready = asyncio.Event()

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        body = _message_text(messages[-1].content)
        if "Please create an execution plan" in body:
            yield {
                "type": "text_delta",
                "text": (
                    '{"summary":"parallel","tasks":['
                    '{"id":"a","description":"Task A","type":"ANALYSIS",'
                    '"dependencies":[],"parallel_safe":true,"read_paths":["a"]},'
                    '{"id":"b","description":"Task B","type":"ANALYSIS",'
                    '"dependencies":[],"parallel_safe":true,"read_paths":["b"]}'
                    "]}"
                ),
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        if "Stop Hook reviewer" in system_prompt:
            yield {
                "type": "text_delta",
                "text": '{"approved": true, "feedback": "", "memories": []}',
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        if "Task A" in body or "Task B" in body:
            self.current_concurrency += 1
            self.peak_concurrency = max(self.peak_concurrency, self.current_concurrency)
            if self.current_concurrency == 2:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), timeout=2)
            self.current_concurrency -= 1
            text = "result for A" if "Task A" in body else "result for B"
            yield {"type": "text_delta", "text": text}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        yield {"type": "text_delta", "text": "fallback"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return str(content)
