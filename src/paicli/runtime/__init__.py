from paicli.runtime.budget import BudgetManager
from paicli.runtime.run_state import RunState, RunStateStore, is_resume_request
from paicli.runtime.tasks import DurableTaskManager, TaskRecord

__all__ = [
    "BudgetManager",
    "DurableTaskManager",
    "RunState",
    "RunStateStore",
    "RuntimeApiServer",
    "TaskRecord",
    "is_resume_request",
]


def __getattr__(name: str):
    if name == "RuntimeApiServer":
        from paicli.runtime.api import RuntimeApiServer

        return RuntimeApiServer
    raise AttributeError(name)
