from __future__ import annotations

from typing import Any

import httpx


def friendly_llm_error(error: Any, *, source: str = "model") -> str:
    label = "Stop Hook" if source == "stop_hook" else "模型服务"
    if isinstance(error, httpx.ConnectError):
        return (
            f"无法连接{label}。请检查网络、PAICLI_BASE_URL 和服务状态；"
            "当前上下文已保留，可以在连接恢复后直接输入“继续”。"
        )
    if isinstance(error, httpx.TimeoutException):
        return f"{label}请求超时。当前上下文已保留，可以稍后直接输入“继续”重试。"
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status in {401, 403}:
            return f"{label}鉴权失败（HTTP {status}），请检查 API Key；当前上下文已保留。"
        if status == 429:
            return f"{label}请求频率或额度受限（HTTP 429）；当前上下文已保留，请稍后继续。"
        return f"{label}返回 HTTP {status}；当前上下文已保留，可以修复配置后继续。"
    if isinstance(error, httpx.RequestError):
        return f"{label}网络请求失败；当前上下文已保留，可以稍后直接输入“继续”。"
    detail = str(error).strip()
    kind = type(error).__name__ if isinstance(error, BaseException) else "LLMError"
    suffix = f"：{detail}" if detail else ""
    return f"{label}调用失败（{kind}）{suffix}。当前上下文已保留。"


def llm_error_event(
    error: Any,
    *,
    messages: list[Any],
    source: str = "model",
) -> dict[str, Any]:
    exception = error if isinstance(error, BaseException) else RuntimeError(str(error))
    return {
        "type": "error",
        "error": exception,
        "message": friendly_llm_error(error, source=source),
        "source": source,
        "recoverable": True,
        "context_preserved": True,
        "messages": list(messages),
    }
