"""Anthropic generation adapter: shared requests to Messages and back.

The payloads are the reference examples of ``test_anthropic_messages``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_api.enums import IncompleteReason, Provider
from ai_api.exceptions import RequestValidationError
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import ServiceUnavailableError
from ai_api.providers.anthropic import AnthropicClient
from ai_api.providers.anthropic.adapters import AnthropicGenerationAdapter
from ai_api.providers.anthropic.client import session_factory
from ai_api.providers.anthropic.methods import CreateMessage, StreamMessage
from ai_api.providers.anthropic.types import Message, MessageStreamEvent
from ai_api.types.requests import GenerationRequest, Turn
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

    with pytest.raises(
        RequestValidationError, match="unsupported keys: app, team"
    ) as caught:
        adapter.to_method(request)
    with pytest.raises(RequestValidationError, match="only 'user_id'"):
        adapter.to_stream_method(request)

    assert caught.value.provider is Provider.ANTHROPIC
    assert caught.value.field == "metadata"
    assert caught.value.__cause__ is None


def test_too_long_user_id_is_rejected() -> None:
    request = REQUEST.model_copy(update={"metadata": {"user_id": "u" * 513}})

    with pytest.raises(RequestValidationError) as caught:
        adapter.to_method(request)

    assert caught.value.field == "metadata"
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value.__cause__, ValidationError)
    assert "metadata.user_id: String should have at most 512" in str(
        caught.value
    )


def test_method_fields_are_reported_by_their_shared_name() -> None:
    request = REQUEST.model_copy(update={"max_output_tokens": -1})

    with pytest.raises(RequestValidationError) as caught:
        adapter.to_stream_method(request)

    assert caught.value.field == "max_output_tokens"
    assert "max_output_tokens: Input should be greater" in str(caught.value)
    assert "max_tokens" not in str(caught.value)


def test_every_error_is_listed_under_its_shared_name() -> None:
    # field names the first error; the message names all of them, by
    # the names the caller used.
    request = REQUEST.model_copy(
        update={"max_output_tokens": -1, "instructions": 5}
    )

    with pytest.raises(RequestValidationError) as caught:
        adapter.to_method(request)

    message = str(caught.value)
    assert caught.value.field == "max_output_tokens"
    assert "max_output_tokens: Input should be greater" in message
    # system is a str or a list of blocks: pydantic names the member.
    assert "instructions.str: Input should be a valid string" in message
    assert "system" not in message


def test_turns_are_sent_as_messages() -> None:
    request = REQUEST.model_copy(
        update={
            "input": [
                Turn.user("Hi"),
                Turn.assistant("Hello!"),
                Turn.user("How are you?"),
            ]
        }
    )

    method = adapter.to_method(request)

    assert method.get_body_map()["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]


def test_finished_message_is_converted() -> None:
    raw = message()

    converted = adapter.to_response(raw, REQUEST)

    assert converted.id == MESSAGE["id"]
    assert converted.provider is Provider.ANTHROPIC
    assert converted.model == MODEL
    assert converted.text == "Hi! My name is Claude."
    # Input is 2095 + 2051 written to + 2051 read from the cache.
    assert converted.usage == TokenUsage(
        input_tokens=6197,
        output_tokens=503,
        total_tokens=6700,
        cache_read_tokens=2051,
        cache_write_tokens=2051,
        reasoning_tokens=0,
    )
    assert converted.incomplete_reason is None
    assert converted.raw_incomplete_reason is None
    assert converted.refusal is None
    assert converted.raw is raw


def test_stop_sequence_is_a_finished_message() -> None:
    converted = adapter.to_response(
        message(stop_reason="stop_sequence", stop_sequence="END"), REQUEST
    )

    assert converted.incomplete_reason is None


@pytest.mark.parametrize(
    ("reason", "shared"),
    [
        ("max_tokens", IncompleteReason.MAX_TOKENS),
        ("refusal", IncompleteReason.REFUSAL),
        ("model_context_window_exceeded", IncompleteReason.OTHER),
        ("pause_turn", IncompleteReason.OTHER),
        ("tool_use", IncompleteReason.OTHER),
        ("new_reason", IncompleteReason.OTHER),
    ],
)
def test_early_stop_keeps_the_text_and_reason(
    reason: str,
    shared: IncompleteReason,
) -> None:
    raw = message(stop_reason=reason, stop_details=None)

    converted = adapter.to_response(raw, REQUEST)

    assert converted.text == "Hi! My name is Claude."
    assert converted.incomplete_reason is shared
    assert converted.raw_incomplete_reason == reason
    assert converted.refusal is None


@pytest.mark.parametrize(
    ("details", "refusal"),
    [
        (
            {"type": "refusal", "category": "cyber", "explanation": "No."},
            "No.",
        ),
        ({"type": "refusal", "category": "cyber"}, "cyber"),
        ({"type": "refusal"}, None),
        (None, None),
    ],
)
def test_refusal_is_a_field_not_an_error(
    details: dict[str, Any] | None,
    refusal: str | None,
) -> None:
    raw = message(stop_reason="refusal", stop_details=details)

    converted = adapter.to_response(raw, REQUEST)

    assert converted.incomplete_reason is IncompleteReason.REFUSAL
    assert converted.refusal == refusal


def test_stop_details_of_other_reasons_are_no_refusal() -> None:
    raw = message(
        stop_reason="max_tokens",
        stop_details={"type": "refusal", "explanation": "No."},
    )

    assert adapter.to_response(raw, REQUEST).refusal is None


def test_thinking_tokens_are_reasoning_tokens() -> None:
    usage = {
        "input_tokens": 5,
        "output_tokens": 9,
        "output_tokens_details": {"thinking_tokens": 7},
    }

    converted = adapter.to_response(message(usage=usage), REQUEST)

    assert converted.usage is not None
    assert converted.usage.reasoning_tokens == 7


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
    assert completed.raw_incomplete_reason is None
    assert completed.refusal is None
    assert completed.raw.type == "message_stop"
    # Id and model of message_start, text of the deltas.
    assert completed.id == STREAM[0]["message"]["id"]
    assert completed.model == MODEL
    assert completed.text == "Hello!"


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
                "output_tokens_details": {"thinking_tokens": 3},
            },
        },
        {"type": "message_delta", "delta": {}},
        STREAM[-1],
    ]

    completed = (await convert(payloads))[-1]

    assert isinstance(completed, CompletedEvent)
    assert completed.usage == TokenUsage(
        input_tokens=40,
        output_tokens=9,
        total_tokens=49,
        cache_read_tokens=6,
        cache_write_tokens=4,
        reasoning_tokens=3,
    )
    assert completed.incomplete_reason is IncompleteReason.MAX_TOKENS
    assert completed.raw_incomplete_reason == "max_tokens"


async def test_stream_stopped_early_completes_with_its_reason() -> None:
    payloads = stream_with(
        delta={
            "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "cyber"},
        },
        usage={"output_tokens": 2},
    )

    completed = (await convert(payloads))[-1]

    assert isinstance(completed, CompletedEvent)
    assert completed.incomplete_reason is IncompleteReason.REFUSAL
    assert completed.raw_incomplete_reason == "refusal"
    assert completed.refusal == "cyber"


async def test_stream_without_message_start_raises() -> None:
    payloads = [
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        STREAM[-1],
    ]

    with pytest.raises(APIResponseError, match="without message_start"):
        await convert(payloads)


async def test_counts_before_message_start_raise() -> None:
    payloads = [
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        STREAM[-1],
    ]

    with pytest.raises(APIResponseError, match="before message_start"):
        await convert(payloads)


async def test_stream_stop_without_a_stop_reason_raises() -> None:
    with pytest.raises(APIResponseError, match="without a stop_reason"):
        await convert([STREAM[0], STREAM[-1]])


async def test_adapter_leaves_the_end_to_the_parser_and_use_case() -> None:
    # StreamMessage and GenerationResource check message_stop.
    converted = await convert(STREAM[:-1])

    assert len(converted) == len(STREAM) - 1


@pytest.fixture
def client(
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> AnthropicClient:
    """``AnthropicClient`` built from a token, sent to the local server."""
    monkeypatch.setattr(session_factory, "api", api)
    return AnthropicClient(TEST_TOKEN)


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
