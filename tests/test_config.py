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
    monkeypatch.setenv("PAICLI_FILE_VERSION_CHECK", "enforce")
    monkeypatch.setenv("PAICLI_ATOMIC_FILE_WRITE", "false")
    monkeypatch.setenv("PAICLI_CODE_INDEX", "false")

    config = load_config(
        project_root=project,
        overrides={"llm": {"model": "cli-model"}},
    )

    assert config.llm.provider == "process"
    assert config.llm.model == "cli-model"
    assert config.llm.max_retries == 4
    assert config.llm.retry_base_delay == 0.25
    assert config.tools.file_version_check == "enforce"
    assert config.tools.atomic_file_write is False
    assert config.features.code_index is False


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
    monkeypatch.setenv("PAICLI_AGENT_EXPLORATION_LIMIT", "14")
    monkeypatch.setenv("PAICLI_STOP_HOOK", "false")
    monkeypatch.setenv("PAICLI_STOP_HOOK_RETRIES", "4")
    monkeypatch.setenv("PAICLI_AGENT_EXTENSION_TURNS", "25")
    monkeypatch.setenv("PAICLI_AGENT_EXTENSION_TOKENS", "150000")
    monkeypatch.setenv("PAICLI_SHELL_SOURCE_WRITE_GUARD", "false")

    config = load_config(project_root=tmp_path)

    assert config.agent.max_turns == 30
    assert config.agent.max_total_tokens == 120000
    assert config.agent.max_runtime_seconds == 600
    assert config.agent.repeated_tool_call_limit == 4
    assert config.agent.consecutive_tool_error_limit == 5
    assert config.agent.exploration_tool_call_limit == 14
    assert config.agent.stop_hook_enabled is False
    assert config.agent.stop_hook_max_retries == 4
    assert config.agent.budget_extension_turns == 25
    assert config.agent.budget_extension_tokens == 150000
    assert config.tools.shell_source_write_guard is False


def test_short_term_context_can_be_configured_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PAICLI_SHORT_TERM_MEMORY", "false")
    monkeypatch.setenv("PAICLI_CONTEXT_WARNING_THRESHOLD", "0.6")
    monkeypatch.setenv("PAICLI_CONTEXT_COMPRESSION_THRESHOLD", "0.75")
    monkeypatch.setenv("PAICLI_CONTEXT_TARGET_MIN", "0.45")
    monkeypatch.setenv("PAICLI_CONTEXT_TARGET_MAX", "0.55")
    monkeypatch.setenv("PAICLI_CONTEXT_EMERGENCY_TARGET", "0.4")

    config = load_config(project_root=tmp_path)

    assert config.memory.short_term_enabled is False
    assert config.memory.warning_threshold == 0.6
    assert config.memory.compression_threshold == 0.75
    assert config.memory.target_pressure_min == 0.45
    assert config.memory.target_pressure_max == 0.55
    assert config.memory.emergency_target == 0.4


def test_long_term_memory_recall_can_be_configured_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PAICLI_LONG_TERM_MEMORY", "false")
    monkeypatch.setenv("PAICLI_MEMORY_RECALL_LIMIT", "7")
    monkeypatch.setenv("PAICLI_MEMORY_MIN_SCORE", "1.5")
    monkeypatch.setenv("PAICLI_MEMORY_PROMPT_MAX_CHARS", "2400")
    monkeypatch.setenv("PAICLI_AUTO_MEMORY", "false")
    monkeypatch.setenv("PAICLI_AUTO_MEMORY_MIN_CONFIDENCE", "0.9")
    monkeypatch.setenv("PAICLI_AUTO_MEMORY_MAX_CANDIDATES", "2")

    config = load_config(project_root=tmp_path)

    assert config.memory.long_term_enabled is False
    assert config.memory.long_term_recall_limit == 7
    assert config.memory.long_term_min_score == 1.5
    assert config.memory.long_term_prompt_max_chars == 2400
    assert config.memory.auto_memory_enabled is False
    assert config.memory.auto_memory_min_confidence == 0.9
    assert config.memory.auto_memory_max_candidates == 2
