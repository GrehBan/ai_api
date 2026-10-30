"""Anthropic generation adapter: shared requests to Messages and back.

The payloads are the reference examples of ``test_anthropic_messages``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_api.enums import Provider
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import ServiceUnavailableError
from ai_api.providers.anthropic import AnthropicClient
from ai_api.providers.anthropic.adapters import AnthropicGenerationAdapter
from ai_api.providers.anthropic.client import session_factory
from ai_api.providers.anthropic.methods import CreateMessage, StreamMessage
from ai_api.providers.anthropic.types import Message, MessageStreamEvent
from ai_api.resources import GenerationResource, IGenerationResource
from ai_api.types.requests import GenerationRequest
from ai_api.types.responses import (
    CompletedEvent,
    OtherEvent,
    TextDeltaEvent,
    TokenUsage,
)
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply
from tests.providers.test_anthropic_messages import (
    MESSAGE,
    MODEL,
    STREAM,
    frame,
)

EVENTS: TypeAdapter[MessageStreamEvent] = TypeAdapter(MessageStreamEvent)

REQUEST = GenerationRequest(
    model=MODEL,
    input="Hello, world",
    instructions="Be brief",
    max_output_tokens=256,
    metadata={"user_id": "u-1"},
)

adapter = AnthropicGenerationAdapter()


def message(**changes: Any) -> Message:
    """The reference message with some fields changed."""
    return Message.model_validate(MESSAGE | changes)


async def events(payloads: Iterable[dict[str, Any]]) -> AsyncIterator[Any]:
    """Parse payloads into Anthropic events, as the stream would."""
    for payload in payloads:
        yield EVENTS.validate_python(payload)


async def convert(payloads: Iterable[dict[str, Any]]) -> list[Any]:
    """Run ``to_events`` over parsed payloads."""
    return [
        event async for event in adapter.to_events(events(payloads), REQUEST)
    ]


def stream_with(**delta: Any) -> list[dict[str, Any]]:
    """The reference stream with another final ``message_delta``."""
    return [*STREAM[:-2], {"type": "message_delta", **delta}, STREAM[-1]]


def test_methods_take_the_request_fields() -> None:
    fields = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello, world"}],
        "max_tokens": 256,
        "system": "Be brief",
        "metadata": {"user_id": "u-1"},
    }

    created = adapter.to_method(REQUEST)
    streamed = adapter.to_stream_method(REQUEST)

    assert isinstance(created, CreateMessage)
    assert created.get_body_map() == fields
    assert isinstance(streamed, StreamMessage)
    assert streamed.get_body_map() == fields | {"stream": True}


def test_max_tokens_falls_back_to_the_default() -> None:
    request = GenerationRequest(model=MODEL, input="Hi")

    assert adapter.to_method(request).get_body_map() == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1024,
    }
    custom = AnthropicGenerationAdapter(default_max_tokens=8)
    assert custom.to_stream_method(request).max_tokens == 8


@pytest.mark.parametrize("value", [0, -1])
def test_default_max_tokens_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="default_max_tokens"):
        AnthropicGenerationAdapter(default_max_tokens=value)


def test_unsupported_metadata_keys_raise() -> None:
    request = REQUEST.model_copy(
        update={"metadata": {"user_id": "u", "team": "t", "app": "a"}}
    )

    with pytest.raises(ValueError, match="unsupported keys: app, team"):
        adapter.to_method(request)
    with pytest.raises(ValueError, match="only 'user_id'"):
        adapter.to_stream_method(request)


def test_too_long_user_id_is_rejected() -> None:
    request = REQUEST.model_copy(update={"metadata": {"user_id": "u" * 513}})

    with pytest.raises(ValidationError):
        adapter.to_method(request)


def test_finished_message_is_converted() -> None:
    raw = message()

    converted = adapter.to_response(raw, REQUEST)

    assert converted.id == MESSAGE["id"]
    assert converted.provider is Provider.ANTHROPIC
    assert converted.model == MODEL
    assert converted.text == "Hi! My name is Claude."
    # Input is 2095 + 2051 written to + 2051 read from the cache.
    assert converted.usage == TokenUsage(
        input_tokens=6197, output_tokens=503, total_tokens=6700
    )
    assert converted.incomplete_reason is None
    assert converted.raw is raw


def test_stop_sequence_is_a_finished_message() -> None:
    converted = adapter.to_response(
        message(stop_reason="stop_sequence", stop_sequence="END"), REQUEST
    )

    assert converted.incomplete_reason is None


@pytest.mark.parametrize(
    "reason",
    ["max_tokens", "refusal", "model_context_window_exceeded", "new_reason"],
)
def test_early_stop_keeps_the_text_and_reason(reason: str) -> None:
    converted = adapter.to_response(message(stop_reason=reason), REQUEST)

    assert converted.text == "Hi! My name is Claude."
    assert converted.incomplete_reason == reason


def test_message_without_text_or_cache_counts() -> None:
    usage = {"input_tokens": 5, "output_tokens": 0}
    converted = adapter.to_response(message(content=[], usage=usage), REQUEST)

    assert converted.text is None
    assert converted.usage == TokenUsage(
        input_tokens=5, output_tokens=0, total_tokens=5
    )


def test_message_without_a_stop_reason_raises() -> None:
    with pytest.raises(APIResponseError, match="without a stop_reason"):
        adapter.to_response(message(stop_reason=None), REQUEST)


async def test_stream_events_are_converted() -> None:
    converted = await convert(STREAM)

    assert [event.type for event in converted] == [
        "other",
        "other",
        "other",
        "text_delta",
        "text_delta",
        "other",
        "other",
        "completed",
    ]
    assert all(
        isinstance(event, OtherEvent)
        for event in converted
        if event.type == "other"
    )
    assert [event.text for event in converted[3:5]] == ["Hello", "!"]
    assert converted[3].raw.index == 0
    completed = converted[-1]
    # Input from message_start, cumulative output from message_delta.
    assert completed.usage == TokenUsage(
        input_tokens=25, output_tokens=15, total_tokens=40
    )
    assert completed.incomplete_reason is None
    assert completed.raw.type == "message_stop"


async def test_usage_takes_the_last_cumulative_counts() -> None:
    payloads = [
        *STREAM[:-2],
        {
            "type": "message_delta",
            "delta": {"stop_reason": None},
            "usage": {"output_tokens": 7},
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {
                "input_tokens": 30,
                "cache_creation_input_tokens": 4,
                "cache_read_input_tokens": 6,
                "output_tokens": 9,
            },
        },
        {"type": "message_delta", "delta": {}},
        STREAM[-1],
    ]

    completed = (await convert(payloads))[-1]

    assert isinstance(completed, CompletedEvent)
    assert completed.usage == TokenUsage(
        input_tokens=40, output_tokens=9, total_tokens=49
    )
    assert completed.incomplete_reason == "max_tokens"


async def test_stream_stopped_early_completes_with_its_reason() -> None:
    payloads = stream_with(
        delta={"stop_reason": "refusal"}, usage={"output_tokens": 2}
    )

    completed = (await convert(payloads))[-1]

    assert isinstance(completed, CompletedEvent)
    assert completed.incomplete_reason == "refusal"


async def test_stream_without_message_start_has_partial_usage() -> None:
    payloads = [
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        STREAM[-1],
    ]

    completed = (await convert(payloads))[-1]

    assert completed.usage == TokenUsage()


async def test_stream_stop_without_a_stop_reason_raises() -> None:
    with pytest.raises(APIResponseError, match="without a stop_reason"):
        await convert([STREAM[0], STREAM[-1]])


async def test_stream_cut_before_message_stop_raises() -> None:
    received: list[Any] = []

    with pytest.raises(APIResponseError, match="before message_stop"):
        async for event in adapter.to_events(events(STREAM[:-1]), REQUEST):
            received.append(event)  # noqa: PERF401 - kept before the error

    assert len(received) == len(STREAM) - 1


@pytest.fixture
def client(
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> AnthropicClient:
    """``AnthropicClient`` built from a token, sent to the local server."""
    monkeypatch.setattr(session_factory, "api", api)
    return AnthropicClient(TEST_TOKEN)


def test_client_generation_shares_the_client_session(
    client: AnthropicClient,
) -> None:
    resource: IGenerationResource[Message, MessageStreamEvent] = (
        client.generation
    )

    assert isinstance(client.generation, GenerationResource)
    assert resource.session is client.session


def test_client_checks_default_max_tokens_first() -> None:
    with pytest.raises(ValueError, match="default_max_tokens"):
        AnthropicClient(TEST_TOKEN, default_max_tokens=0)


async def test_client_generation_runs_on_messages(
    recorder: Recorder,
    client: AnthropicClient,
) -> None:
    recorder.reply(Reply(json=MESSAGE))
    recorder.reply(
        Reply(
            chunks=[frame(event) for event in STREAM],
            content_type="text/event-stream",
        )
    )

    async with client:
        created = await client.generation.create(REQUEST)
        async with client.generation.stream(REQUEST) as stream:
            texts = [
                event.text
                async for event in stream
                if isinstance(event, TextDeltaEvent)
            ]

    assert created.text == "Hi! My name is Claude."
    assert texts == ["Hello", "!"]
    first, second = recorder.requests
    assert first.raw_path == second.raw_path == "/v1/messages"
    assert first.headers["Authorization"] == f"Bearer {TEST_TOKEN}"
    assert first.headers["anthropic-version"] == "2026-10-21"
    assert json.loads(second.body)["stream"] is True


async def test_client_generation_raises_stream_errors(
    recorder: Recorder,
    client: AnthropicClient,
) -> None:
    error = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    recorder.reply(
        Reply(
            chunks=[frame(STREAM[0]), frame(error)],
            content_type="text/event-stream",
        )
    )

    async with client, client.generation.stream(REQUEST) as stream:
        with pytest.raises(ServiceUnavailableError, match="Overloaded"):
            async for _ in stream:
                pass
