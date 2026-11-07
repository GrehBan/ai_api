"""Anthropic Messages API: types, methods and the stream error event.

The payloads follow the examples of Anthropic's API reference and
streaming guide (captured 2026-10-30).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from aiollm.http.exceptions.status import ServiceUnavailableError
from aiollm.providers.anthropic import API_VERSION, AnthropicErrorResolver
from aiollm.providers.anthropic.adapters.conversions import (
    message_text,
    message_thinking,
)
from aiollm.providers.anthropic.methods import (
    VERSION_HEADERS,
    CreateMessage,
    StreamMessage,
)
from aiollm.providers.anthropic.types import (
    InputJSONDelta,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    TextBlock,
    TextDelta,
    ThinkingBlock,
)
from aiollm.providers.anthropic.types.shared import UnknownVariant
from aiollm.session.aiohttp import AiohttpSession
from tests.conftest import (
    FakeStreamResponse,
    FakeTransport,
    Recorder,
    Reply,
)

MODEL = "claude-opus-5-5"

MESSAGE: dict[str, Any] = {
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "container": {
        "id": "container_011CpZohnwH4vuy7gazohgSP",
        "expires_at": "2019-12-27T18:11:19.117Z",
        "skills": [
            {"skill_id": "pdf", "type": "anthropic", "version": "latest"}
        ],
    },
    "content": [
        {
            "citations": [
                {
                    "cited_text": "The grass is green. The sky is blue.",
                    "document_index": 0,
                    "document_title": "My Document",
                    "end_char_index": 0,
                    "file_id": "file_011CNha8iCJcU1wXNR6q4V8w",
                    "start_char_index": 0,
                    "type": "char_location",
                }
            ],
            "text": "Hi! My name is Claude.",
            "type": "text",
        }
    ],
    "diagnostics": {
        "cache_miss_reason": {
            "cache_missed_input_tokens": 0,
            "type": "model_changed",
        }
    },
    "model": MODEL,
    "role": "assistant",
    "stop_details": {
        "category": "cyber",
        "explanation": "This request was declined because it conflicts "
        "with Anthropic's Usage Policy.",
        "type": "refusal",
    },
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "type": "message",
    "usage": {
        "cache_creation": {
            "ephemeral_1h_input_tokens": 0,
            "ephemeral_5m_input_tokens": 0,
        },
        "cache_creation_input_tokens": 2051,
        "cache_read_input_tokens": 2051,
        "inference_geo": "global",
        "input_tokens": 2095,
        "output_tokens": 503,
        "output_tokens_details": {"thinking_tokens": 0},
        "server_tool_use": {"web_fetch_requests": 2, "web_search_requests": 0},
        "service_tier": "standard",
        "speed": "standard",
    },
}

STREAM: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": MODEL,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 25, "output_tokens": 1},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {"type": "ping"},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "!"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 15},
    },
    {"type": "message_stop"},
]

SSE = {"Content-Type": "text/event-stream"}


def frame(data: dict[str, Any]) -> bytes:
    """One SSE frame as Anthropic sends it: ``event:`` and ``data:``."""
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def request(**changes: Any) -> dict[str, Any]:
    """Fields of a minimal valid request."""
    fields: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello, world"}],
    }
    return fields | changes


def test_message_reads_the_reference_example() -> None:
    message = Message.model_validate(MESSAGE)

    assert message.id == MESSAGE["id"]
    (block,) = message.content
    assert isinstance(block, TextBlock)
    assert block.text == "Hi! My name is Claude."
    assert message_text(message) == "Hi! My name is Claude."
    assert message.stop_reason == "end_turn"
    assert message.stop_details is not None
    assert message.stop_details.category == "cyber"
    assert message.usage.input_tokens == 2095
    assert message.usage.cache_read_input_tokens == 2051


def test_message_keeps_unknown_blocks_whole() -> None:
    redacted = {"type": "future_block", "data": "EmwK..."}
    message = Message.model_validate(
        MESSAGE
        | {
            "content": [
                {"type": "text", "text": "Hel"},
                redacted,
                {"type": "text", "text": "lo"},
            ]
        }
    )

    block = message.content[1]
    assert isinstance(block, UnknownVariant)
    assert block.model_dump() == redacted
    assert message_text(message) == "Hello"
    assert message_thinking(message) is None


def test_redacted_thinking_is_no_summary() -> None:
    message = Message.model_validate(
        MESSAGE
        | {
            "content": [
                {"type": "redacted_thinking", "data": "EmwK..."},
                {"type": "text", "text": "Hi"},
            ]
        }
    )

    assert isinstance(message.content[0], RedactedThinkingBlock)
    assert message_thinking(message) is None


def test_thinking_blocks_are_joined_and_empty_ones_dropped() -> None:
    # On the 5.x models a thinking block whose display is omitted comes
    # with empty thinking: it is no summary.
    message = Message.model_validate(
        MESSAGE
        | {
            "content": [
                {"type": "thinking", "thinking": "", "signature": "a"},
                {"type": "thinking", "thinking": "Let me", "signature": "b"},
                {"type": "thinking", "thinking": " think", "signature": "c"},
                {"type": "text", "text": "Hi"},
            ]
        }
    )

    assert isinstance(message.content[1], ThinkingBlock)
    assert message_thinking(message) == "Let me think"
    assert message_text(message) == "Hi"


def test_message_without_text_has_no_text() -> None:
    message = Message.model_validate(MESSAGE | {"content": []})

    assert message_text(message) is None


def test_a_block_without_a_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="type"):
        Message.model_validate(MESSAGE | {"content": [{"text": "x"}]})


async def test_create_message_sends_the_version_and_parses(
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    recorder.reply(Reply(json=MESSAGE))

    message = await session.send_request(
        CreateMessage(
            **request(
                system="Be brief",
                metadata={"user_id": "u-1"},
                stop_sequences=["END"],
            )
        )
    )

    assert message_text(message) == "Hi! My name is Claude."
    assert recorder.last.method == "POST"
    assert recorder.last.raw_path == "/v1/messages"
    assert recorder.last.headers["anthropic-version"] == API_VERSION
    assert json.loads(recorder.last.body) == request(
        system="Be brief",
        metadata={"user_id": "u-1"},
        stop_sequences=["END"],
    )


def test_version_headers_are_shared_by_the_methods() -> None:
    created = CreateMessage(**request())
    streamed = StreamMessage(**request())

    assert created.request_headers.build() == VERSION_HEADERS
    assert streamed.request_headers.build() == VERSION_HEADERS
    assert API_VERSION == "2026-10-21"


async def test_stream_message_yields_typed_events(
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    recorder.reply(
        Reply(
            chunks=[frame(event) for event in STREAM],
            content_type="text/event-stream",
        )
    )

    async with session.stream(StreamMessage(**request())) as events:
        received = [event async for event in events]

    assert [type(event) for event in received] == [
        RawMessageStartEvent,
        RawContentBlockStartEvent,
        UnknownVariant,
        RawContentBlockDeltaEvent,
        RawContentBlockDeltaEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStopEvent,
    ]
    assert json.loads(recorder.last.body) == request(stream=True)
    assert recorder.last.headers["anthropic-version"] == API_VERSION

    start = received[0]
    assert isinstance(start, RawMessageStartEvent)
    assert start.message.usage.input_tokens == 25
    delta = received[3]
    assert isinstance(delta, RawContentBlockDeltaEvent)
    assert isinstance(delta.delta, TextDelta)
    assert delta.delta.text == "Hello"
    end = received[6]
    assert isinstance(end, RawMessageDeltaEvent)
    assert end.delta.stop_reason == "end_turn"
    assert end.usage.output_tokens == 15


async def test_unknown_deltas_and_events_are_kept_whole() -> None:
    tool_delta = {
        "type": "content_block_delta",
        "index": 1,
        "delta": {
            "type": "input_json_delta",
            "partial_json": '{"location": "San Fra',
        },
    }
    future_delta = {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "future_delta", "value": 1},
    }
    new_event = {"type": "message_checkpoint", "id": "c-1"}
    transport = FakeTransport(
        stream_responses=[
            FakeStreamResponse(
                [
                    frame(tool_delta),
                    frame(future_delta),
                    frame(new_event),
                    frame({"type": "message_stop"}),
                ],
                headers=SSE,
            )
        ],
        error_resolver=AnthropicErrorResolver(),
    )

    async with transport.stream(StreamMessage(**request())) as events:
        delta, future, unknown, _ = [event async for event in events]

    assert isinstance(delta, RawContentBlockDeltaEvent)
    assert isinstance(delta.delta, InputJSONDelta)
    assert isinstance(future, RawContentBlockDeltaEvent)
    assert isinstance(future.delta, UnknownVariant)
    assert future.delta.model_dump() == future_delta["delta"]
    assert isinstance(unknown, UnknownVariant)
    assert unknown.model_dump() == new_event


async def test_error_event_raises_by_its_type() -> None:
    error = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    transport = FakeTransport(
        stream_responses=[
            FakeStreamResponse([frame(STREAM[0]), frame(error)], headers=SSE)
        ],
        error_resolver=AnthropicErrorResolver(),
    )

    async with transport.stream(StreamMessage(**request())) as events:
        with pytest.raises(ServiceUnavailableError) as caught:
            async for _ in events:
                pass

    assert caught.value.error_type == "overloaded_error"
    assert caught.value.transient is False
    assert "Overloaded" in str(caught.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 1},
        {"metadata": {"user_id": "u", "team": "t"}},
        {"metadata": {"user_id": "u" * 513}},
        {"messages": []},
        {"messages": [{"role": "user", "content": "x", "name": "n"}]},
        {"max_tokens": -1},
    ],
    ids=[
        "temperature",
        "metadata-key",
        "user-id-length",
        "no-messages",
        "message-field",
        "max-tokens",
    ],
)
def test_bad_requests_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CreateMessage(**request(**changes))


def test_max_tokens_is_required() -> None:
    fields = request()
    del fields["max_tokens"]

    with pytest.raises(ValidationError, match="max_tokens"):
        StreamMessage(**fields)
