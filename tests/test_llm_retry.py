from __future__ import annotations

import asyncio

import httpx
import pytest

from paicli.llm.openai_compatible import OpenAICompatibleClient


def _client(**overrides) -> OpenAICompatibleClient:
    values = {
        "provider_name": "test",
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "max_retries": 2,
        "retry_base_delay": 0.5,
    }
    values.update(overrides)
    return OpenAICompatibleClient(**values)


async def _collect(client: OpenAICompatibleClient) -> list[dict]:
    return [event async for event in client.chat([], [], system_prompt="test system prompt")]


def test_transient_api_errors_retry_with_exponential_backoff(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, request=request)
        if attempts == 2:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            request=request,
            text=(
                'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "paicli.llm.openai_compatible.httpx.AsyncClient",
        lambda **kwargs: real_async_client(transport=transport, **kwargs),
    )
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("paicli.llm.openai_compatible.asyncio.sleep", fake_sleep)

    events = asyncio.run(_collect(_client()))

    assert attempts == 3
    assert delays == [0.5, 1.0]
    retry_events = [event for event in events if event.get("type") == "llm_retry"]
    assert [
        (event["attempt"], event["max_retries"], event["delay_seconds"], event["reason"])
        for event in retry_events
    ] == [
        (1, 2, 0.5, "HTTP 429"),
        (2, 2, 1.0, "HTTP 503"),
    ]
    assert any(event.get("text") == "recovered" for event in events)
    assert events[-1] == {"type": "message_end", "stop_reason": "end_turn"}


def test_non_transient_http_error_is_not_retried(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "paicli.llm.openai_compatible.httpx.AsyncClient",
        lambda **kwargs: real_async_client(transport=transport, **kwargs),
    )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_collect(_client()))

    assert attempts == 1


class _BrokenSseStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.ReadError("stream disconnected")

    async def aclose(self) -> None:
        return None


def test_stream_failure_after_model_output_is_not_retried(monkeypatch):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, request=request, stream=_BrokenSseStream())

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "paicli.llm.openai_compatible.httpx.AsyncClient",
        lambda **kwargs: real_async_client(transport=transport, **kwargs),
    )
    events: list[dict] = []

    async def run() -> None:
        with pytest.raises(httpx.ReadError):
            async for event in _client().chat([], [], system_prompt="test"):
                events.append(event)

    asyncio.run(run())

    assert attempts == 1
    assert any(event.get("text") == "partial" for event in events)
