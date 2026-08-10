from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from paicli.types import Message


@dataclass(slots=True)
class OpenAICompatibleClient:
    provider_name: str
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0
    max_retries: int = 2
    retry_base_delay: float = 0.5
    max_context_window: int = 128_000
    prompt_cache: bool = False

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_images(self) -> bool:
        model = self.model.lower()
        provider = self.provider_name.lower()
        return any(marker in model for marker in ("vision", "image", "5v", "vl")) or (
            provider in {"glm", "zhipu"} and "5v" in model
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> AsyncIterator[dict[str, Any]]:
        if not self.api_key:
            yield {
                "type": "error",
                "error": RuntimeError(
                    "PAICLI_API_KEY is not configured. Set it in env, ~/.paicli/config.json, "
                    "or project .paicli/config.json."
                ),
            }
            return

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages, system_prompt),
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "user-agent": "SmartCLI/0.1.0",
        }
        url = self.base_url.rstrip("/") + "/chat/completions"

        yield {"type": "message_start", "model": self.model}
        retry_limit = max(0, int(self.max_retries))
        async with httpx.AsyncClient(timeout=self.timeout, http2=False) as client:
            for attempt in range(retry_limit + 1):
                streamed_model_event = False
                try:
                    async with client.stream(
                        "POST",
                        url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        async for event in _iter_sse(response):
                            if event == "[DONE]":
                                return
                            try:
                                chunk = json.loads(event)
                            except json.JSONDecodeError:
                                continue
                            async for parsed in self._parse_chunk(chunk):
                                streamed_model_event = True
                                yield parsed
                    return
                except Exception as exc:
                    if (
                        streamed_model_event
                        or attempt >= retry_limit
                        or not _is_retryable_api_error(exc)
                    ):
                        raise
                    delay = _retry_delay(
                        exc,
                        attempt=attempt,
                        base_delay=max(0.0, float(self.retry_base_delay)),
                    )
                    yield {
                        "type": "llm_retry",
                        "attempt": attempt + 1,
                        "max_retries": retry_limit,
                        "delay_seconds": delay,
                        "reason": _retry_reason(exc),
                    }
                    await asyncio.sleep(delay)

    def _format_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            if message.role == "tool":
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": str(message.content),
                    }
                )
            elif message.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    item["tool_calls"] = message.tool_calls
                formatted.append(item)
            else:
                formatted.append(
                    {"role": message.role, "content": self._format_content(message.content)}
                )
        return formatted

    def _format_content(self, content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        if self.supports_images:
            cleaned = []
            for part in content:
                item = {key: value for key, value in part.items() if key != "metadata"}
                cleaned.append(item)
            return cleaned
        text_parts = []
        for part in content:
            if part.get("type") == "text":
                text_parts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                metadata = part.get("metadata") or {}
                source = metadata.get("source", "remote image")
                width = metadata.get("width", "?")
                height = metadata.get("height", "?")
                text_parts.append(f"[Image omitted: {source}, {width}x{height}]")
        return "\n".join(text_parts)

    async def _parse_chunk(self, chunk: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta") or {}

        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            yield {"type": "thinking_delta", "thinking": reasoning}

        content = delta.get("content")
        if isinstance(content, str) and content:
            yield {"type": "text_delta", "text": content}

        tool_calls = delta.get("tool_calls") or []
        for tool_call in tool_calls:
            yield {"type": "tool_call_delta", "tool_call": tool_call}

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            yield {"type": "message_end", "stop_reason": _map_finish_reason(str(finish_reason))}

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            yield {
                "type": "usage",
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                },
            }


async def _iter_sse(response: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    async for text in response.aiter_text():
        buffer += text
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in event.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                yield "\n".join(data_lines)
    if buffer.strip():
        data_lines = []
        for line in buffer.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            yield "\n".join(data_lines)


def _map_finish_reason(reason: str) -> str:
    if reason in {"tool_calls", "tool_use"}:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    if reason == "content_filter":
        return "stop_sequence"
    return "end_turn"


def _is_retryable_api_error(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in {408, 409, 429} or status >= 500
    return False


def _retry_delay(error: Exception, *, attempt: int, base_delay: float) -> float:
    exponential = base_delay * (2**attempt)
    if not isinstance(error, httpx.HTTPStatusError):
        return exponential

    retry_after = error.response.headers.get("retry-after")
    if not retry_after:
        return exponential
    try:
        header_delay = max(0.0, float(retry_after))
    except ValueError:
        try:
            target = parsedate_to_datetime(retry_after)
            if target.tzinfo is None:
                target = target.replace(tzinfo=UTC)
            header_delay = max(0.0, (target - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return exponential
    return max(exponential, header_delay)


def _retry_reason(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    if isinstance(error, httpx.TimeoutException):
        return "request timeout"
    if isinstance(error, httpx.ConnectError):
        return "connection failed"
    if isinstance(error, httpx.TransportError):
        return "transport error"
    return "temporary API error"
