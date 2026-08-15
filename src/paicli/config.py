from __future__ import annotations

import json
import os
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _home() -> Path:
    return Path.home()


@dataclass(slots=True)
class LlmConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0
    max_retries: int = 2
    retry_base_delay: float = 0.5


@dataclass(slots=True)
class AgentConfig:
    # These are safety budgets, not a fixed execution plan. A run normally ends
    # as soon as the model returns a final response without requesting tools.
    max_turns: int = 20
    max_total_tokens: int = 100_000
    max_runtime_seconds: float = 900.0
    repeated_tool_call_limit: int = 3
    consecutive_tool_error_limit: int = 3
    stop_hook_enabled: bool = True
    stop_hook_max_retries: int = 2
    budget_extension_turns: int = 20
    budget_extension_tokens: int = 100_000


@dataclass(slots=True)
class ToolsConfig:
    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    timeout: float = 60.0
    batch_timeout: float = 90.0
    max_concurrent_read: int = 4
    file_version_check: str = "warn"
    atomic_file_write: bool = True


@dataclass(slots=True)
class McpConfig:
    servers: list[dict[str, Any]] = field(default_factory=list)
    auto_start: bool = True


@dataclass(slots=True)
class MemoryConfig:
    max_conversation_history: int = 100
    long_term_enabled: bool = True
    long_term_db_path: str = "~/.paicli/memory.db"
    token_budget_mode: str = "balanced"
    short_term_enabled: bool = True
    warning_threshold: float = 0.65
    compression_threshold: float = 0.8
    target_pressure_min: float = 0.5
    target_pressure_max: float = 0.6
    emergency_target: float = 0.45
    recent_turns_to_keep: int = 4
    tool_reserve_min: int = 4096
    tool_reserve_max_ratio: float = 0.15
    safety_margin_ratio: float = 0.03
    summary_max_tokens: int = 1200
    emergency_retry_limit: int = 1


@dataclass(slots=True)
class PolicyConfig:
    hitl_mode: str = "auto"
    path_guard_enabled: bool = True
    command_blacklist: list[str] = field(
        default_factory=lambda: [
            "sudo",
            "rm -rf /",
            "rm -rf ~",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",
            "chmod -R 777 /",
            "curl | sh",
            "curl|sh",
            "shutdown",
            "reboot",
        ]
    )
    audit_log_path: str = "~/.paicli/audit.jsonl"


@dataclass(slots=True)
class PromptConfig:
    personality: str = "default"
    agent_mode: str = "react"
    custom_prompt_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureConfig:
    mcp: bool = True
    skill: bool = True
    memory: bool = True
    audit_log: bool = True
    context_compression: bool = True
    code_index: bool = True


@dataclass(slots=True)
class PaiCliConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    render_mode: str = "inline"
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


def load_config(
    project_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str | None] | None = None,
) -> PaiCliConfig:
    env_map = env if env is not None else os.environ
    data = _config_to_dict(PaiCliConfig())

    user_config = _read_json(_home() / ".paicli" / "config.json")
    if user_config:
        data = _deep_merge(data, user_config)

    root = Path(project_root).resolve() if project_root else None
    if root:
        project_config = _read_json(root / ".paicli" / "config.json")
        if project_config:
            data = _deep_merge(data, project_config)
        project_env = _read_env(root / ".env")
        if project_env:
            data = _apply_env(data, project_env)

    if overrides:
        data = _deep_merge(data, overrides)

    data = _apply_env(data, env_map)
    config = _dict_to_config(data)
    config.memory.long_term_db_path = _expand_home(config.memory.long_term_db_path)
    config.policy.audit_log_path = _expand_home(config.policy.audit_log_path)
    return config


def get_config_paths(project_root: str | Path | None = None) -> list[Path]:
    paths = [_home() / ".paicli" / "config.json"]
    if project_root:
        paths.append(Path(project_root).resolve() / ".paicli" / "config.json")
    return paths


def config_to_public_dict(config: PaiCliConfig) -> dict[str, Any]:
    data = _config_to_dict(config)
    if data.get("llm", {}).get("api_key"):
        data["llm"]["api_key"] = "***"
    return data


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        result[key] = value
    return result


def _apply_env(data: dict[str, Any], env: dict[str, str | None]) -> dict[str, Any]:
    result = deepcopy(data)
    llm = result.setdefault("llm", {})
    agent = result.setdefault("agent", {})
    features = result.setdefault("features", {})
    memory = result.setdefault("memory", {})
    policy = result.setdefault("policy", {})
    tools = result.setdefault("tools", {})

    mappings: list[tuple[str, str, Any]] = [
        ("PAICLI_API_KEY", "api_key", str),
        ("PAICLI_PROVIDER", "provider", str),
        ("PAICLI_MODEL", "model", str),
        ("PAICLI_BASE_URL", "base_url", str),
        ("PAICLI_MAX_TOKENS", "max_tokens", int),
        ("PAICLI_TEMPERATURE", "temperature", float),
        ("PAICLI_LLM_MAX_RETRIES", "max_retries", int),
        ("PAICLI_LLM_RETRY_BASE_DELAY", "retry_base_delay", float),
    ]
    for env_key, config_key, caster in mappings:
        raw = env.get(env_key)
        if raw not in (None, ""):
            with suppress(TypeError, ValueError):
                llm[config_key] = caster(raw)

    agent_mappings: list[tuple[str, str, Any]] = [
        ("PAICLI_AGENT_MAX_TURNS", "max_turns", int),
        ("PAICLI_AGENT_TOKEN_BUDGET", "max_total_tokens", int),
        ("PAICLI_AGENT_MAX_SECONDS", "max_runtime_seconds", float),
        ("PAICLI_AGENT_REPEAT_LIMIT", "repeated_tool_call_limit", int),
        ("PAICLI_AGENT_ERROR_LIMIT", "consecutive_tool_error_limit", int),
        ("PAICLI_STOP_HOOK", "stop_hook_enabled", _parse_bool),
        ("PAICLI_STOP_HOOK_RETRIES", "stop_hook_max_retries", int),
        ("PAICLI_AGENT_EXTENSION_TURNS", "budget_extension_turns", int),
        ("PAICLI_AGENT_EXTENSION_TOKENS", "budget_extension_tokens", int),
    ]
    for env_key, config_key, caster in agent_mappings:
        raw = env.get(env_key)
        if raw not in (None, ""):
            with suppress(TypeError, ValueError):
                agent[config_key] = caster(raw)

    tool_mappings: list[tuple[str, str, Any]] = [
        ("PAICLI_FILE_VERSION_CHECK", "file_version_check", str),
        ("PAICLI_ATOMIC_FILE_WRITE", "atomic_file_write", _parse_bool),
    ]
    for env_key, config_key, caster in tool_mappings:
        raw = env.get(env_key)
        if raw not in (None, ""):
            with suppress(TypeError, ValueError):
                tools[config_key] = caster(raw)

    memory_mappings: list[tuple[str, str, Any]] = [
        ("PAICLI_SHORT_TERM_MEMORY", "short_term_enabled", _parse_bool),
        ("PAICLI_CONTEXT_WARNING_THRESHOLD", "warning_threshold", float),
        ("PAICLI_CONTEXT_COMPRESSION_THRESHOLD", "compression_threshold", float),
        ("PAICLI_CONTEXT_TARGET_MIN", "target_pressure_min", float),
        ("PAICLI_CONTEXT_TARGET_MAX", "target_pressure_max", float),
        ("PAICLI_CONTEXT_EMERGENCY_TARGET", "emergency_target", float),
    ]
    for env_key, config_key, caster in memory_mappings:
        raw = env.get(env_key)
        if raw not in (None, ""):
            with suppress(TypeError, ValueError):
                memory[config_key] = caster(raw)

    provider = str(llm.get("provider") or "").lower()
    if not llm.get("api_key"):
        provider_key_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "glm": "GLM_API_KEY",
            "zhipu": "GLM_API_KEY",
            "step": "STEP_API_KEY",
            "kimi": "KIMI_API_KEY",
            "moonshot": "KIMI_API_KEY",
            "freellmapi": "FREELLMAPI_API_KEY",
            "xfyun": "XFYUN_API_KEY",
            "agnes": "AGNES_API_KEY",
        }
        provider_key = provider_key_map.get(provider)
        if provider_key and env.get(provider_key):
            llm["api_key"] = env[provider_key]

    provider_model_key = f"{provider.upper()}_MODEL" if provider else ""
    provider_base_url_key = f"{provider.upper()}_BASE_URL" if provider else ""
    if provider_model_key and env.get(provider_model_key):
        llm["model"] = env[provider_model_key]
    if provider_base_url_key and env.get(provider_base_url_key):
        llm["base_url"] = env[provider_base_url_key]

    render_mode = env.get("PAICLI_RENDER_MODE") or env.get("PAICLI_RENDERER")
    if render_mode in {"plain", "inline"}:
        result["render_mode"] = render_mode

    if env.get("PAICLI_TUI") == "true":
        result["render_mode"] = "inline"

    for env_key, feature_key in [
        ("PAICLI_MCP", "mcp"),
        ("PAICLI_SKILL", "skill"),
        ("PAICLI_MEMORY", "memory"),
        ("PAICLI_CODE_INDEX", "code_index"),
    ]:
        raw = env.get(env_key)
        if raw == "false":
            features[feature_key] = False
        elif raw == "true":
            features[feature_key] = True

    hitl = env.get("PAICLI_HITL")
    if hitl in {"always", "auto", "never"}:
        policy["hitl_mode"] = hitl

    return result


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(target)
    for key, value in source.items():
        if value is None:
            continue
        old = result.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            result[key] = _deep_merge(old, value)
        else:
            result[key] = deepcopy(value)
    return result


def _config_to_dict(config: PaiCliConfig) -> dict[str, Any]:
    return asdict(config)


def _dict_to_config(data: dict[str, Any]) -> PaiCliConfig:
    return PaiCliConfig(
        llm=LlmConfig(**data.get("llm", {})),
        agent=AgentConfig(**data.get("agent", {})),
        render_mode=data.get("render_mode", "inline"),
        tools=ToolsConfig(**data.get("tools", {})),
        mcp=McpConfig(**data.get("mcp", {})),
        memory=MemoryConfig(**data.get("memory", {})),
        policy=PolicyConfig(**data.get("policy", {})),
        prompt=PromptConfig(**data.get("prompt", {})),
        features=FeatureConfig(**data.get("features", {})),
    )


def _expand_home(path: str) -> str:
    return str(Path(path).expanduser())


def _parse_bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean: {value}")
