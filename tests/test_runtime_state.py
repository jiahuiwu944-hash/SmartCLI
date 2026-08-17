from __future__ import annotations

import asyncio

from paicli.agent.verifier import CompletionVerifier
from paicli.runtime.budget import BudgetManager
from paicli.runtime.run_state import RunStateStore, resume_target
from paicli.types import Message


def test_run_state_store_round_trip_and_latest_paused(tmp_path):
    store = RunStateStore(tmp_path)
    state = store.create("plan", "demo")
    state.status = "PAUSED"
    state.turns = 4
    state.tokens = 123
    state.payload = {"plan": {"id": "plan_1"}}
    store.save(state)

    restored = store.latest_paused("plan")

    assert restored is not None
    assert restored.run_id == state.run_id
    assert restored.payload["plan"]["id"] == "plan_1"
    assert restored.turns == 4


def test_run_state_can_recover_running_checkpoint_by_id(tmp_path):
    store = RunStateStore(tmp_path)
    older = store.create("team", "older")
    newer = store.create("team", "newer")

    assert store.latest_resumable("team").run_id == newer.run_id
    assert store.latest_resumable("team", run_id=older.run_id).goal == "older"
    assert resume_target(f"resume {older.run_id}") == older.run_id
    assert resume_target("please continue the analysis") is False


def test_budget_manager_extends_once_at_orchestrator_boundary():
    requests = []

    def approve(request):
        requests.append(request)
        return {"continue": True, "additional_turns": 3, "additional_tokens": 50}

    budget = BudgetManager(turn_limit=2, token_limit=10, callback=approve)
    budget.consume(turns=2, tokens=10)
    approved, detail = asyncio.run(
        budget.request_extension(additional_turns=2, additional_tokens=20, mode="team")
    )

    assert approved
    assert len(requests) == 1
    assert budget.turn_limit == 5
    assert budget.token_limit == 60
    assert detail["context_preserved"] is True


def test_budget_reserves_turns_before_starting_a_reviewed_task():
    budget = BudgetManager(turn_limit=5, token_limit=0)
    budget.consume(turns=4)

    assert budget.reached() == []
    assert budget.reached(minimum_turns=2) == ["max_turns"]
    assert not budget.can_start(minimum_turns=2)


def test_completion_verifier_rejects_only_unrecovered_tool_failure():
    verifier = CompletionVerifier(object())
    failed = Message(role="tool", content='{"error_code":"FILE_NOT_FOUND"}')
    recovered = Message(role="tool", content="WRITE_OK")

    rejected = verifier.verify_task(description="write", result="done", messages=[failed])
    approved = verifier.verify_task(
        description="write",
        result="done",
        messages=[failed, recovered],
    )

    assert not rejected.approved
    assert approved.approved
