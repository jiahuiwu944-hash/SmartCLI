from __future__ import annotations

import json

from paicli.config import load_config


def test_config_precedence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".paicli").mkdir(parents=True)
    (project / ".paicli").mkdir(parents=True)
    (home / ".paicli" / "config.json").write_text(
        json.dumps({"llm": {"provider": "home", "model": "home-model"}}),
        encoding="utf-8",
    )
    (project / ".paicli" / "config.json").write_text(
        json.dumps({"llm": {"provider": "project", "model": "project-model"}}),
        encoding="utf-8",
    )
    (project / ".env").write_text("PAICLI_MODEL=env-file-model\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PAICLI_PROVIDER", "process")
    monkeypatch.setenv("PAICLI_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("PAICLI_LLM_RETRY_BASE_DELAY", "0.25")

    config = load_config(
        project_root=project,
        overrides={"llm": {"model": "cli-model"}},
    )

    assert config.llm.provider == "process"
    assert config.llm.model == "cli-model"
    assert config.llm.max_retries == 4
    assert config.llm.retry_base_delay == 0.25


def test_provider_specific_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PAICLI_PROVIDER", "deepseek")
    monkeypatch.delenv("PAICLI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    config = load_config(project_root=tmp_path)

    assert config.llm.api_key == "deepseek-key"


def test_agent_safety_budgets_can_be_configured_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PAICLI_AGENT_MAX_TURNS", "30")
    monkeypatch.setenv("PAICLI_AGENT_TOKEN_BUDGET", "120000")
    monkeypatch.setenv("PAICLI_AGENT_MAX_SECONDS", "600")
    monkeypatch.setenv("PAICLI_AGENT_REPEAT_LIMIT", "4")
    monkeypatch.setenv("PAICLI_AGENT_ERROR_LIMIT", "5")
    monkeypatch.setenv("PAICLI_STOP_HOOK", "false")
    monkeypatch.setenv("PAICLI_STOP_HOOK_RETRIES", "4")
    monkeypatch.setenv("PAICLI_AGENT_EXTENSION_TURNS", "25")
    monkeypatch.setenv("PAICLI_AGENT_EXTENSION_TOKENS", "150000")

    config = load_config(project_root=tmp_path)

    assert config.agent.max_turns == 30
    assert config.agent.max_total_tokens == 120000
    assert config.agent.max_runtime_seconds == 600
    assert config.agent.repeated_tool_call_limit == 4
    assert config.agent.consecutive_tool_error_limit == 5
    assert config.agent.stop_hook_enabled is False
    assert config.agent.stop_hook_max_retries == 4
    assert config.agent.budget_extension_turns == 25
    assert config.agent.budget_extension_tokens == 150000
