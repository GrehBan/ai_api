"""Anthropic Messages API: types, methods and the stream error event.

The payloads follow the examples of Anthropic's API reference and
streaming guide (captured 2026-10-30).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ai_api.http.exceptions.status import ServiceUnavailableError
from ai_api.providers.anthropic import API_VERSION, AnthropicErrorResolver
from ai_api.providers.anthropic.methods import (
    VERSION_HEADERS,
    CreateMessage,
    StreamMessage,
)
from ai_api.providers.anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    TextBlock,
    TextDelta,
    UnknownBlock,
    UnknownDelta,
    UnknownEvent,
)
from ai_api.session.aiohttp import AiohttpSession
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
    assert message.content == [
        TextBlock(
            text="Hi! My name is Claude.",
            citations=MESSAGE["content"][0]["citations"],
        )
    ]
    assert message.text == "Hi! My name is Claude."
    assert message.stop_reason == "end_turn"
    assert message.stop_details is not None
    assert message.stop_details.category == "cyber"
    assert message.usage.input_tokens == 2095
    assert message.usage.cache_read_input_tokens == 2051


def test_message_keeps_unknown_blocks_whole() -> None:
    thinking = {
        "type": "thinking",
        "thinking": "Let me think",
        "signature": "Eq...",
    }
    message = Message.model_validate(
        MESSAGE
        | {
            "content": [
                {"type": "text", "text": "Hel"},
                thinking,
                {"type": "text", "text": "lo"},
            ]
        }
    )

    block = message.content[1]
    assert isinstance(block, UnknownBlock)
    assert block.model_dump() == thinking
    assert message.text == "Hello"


def test_message_without_text_has_no_text() -> None:
    message = Message.model_validate(MESSAGE | {"content": []})

    assert message.text is None


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

    assert message.text == "Hi! My name is Claude."
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
        MessageStartEvent,
        ContentBlockStartEvent,
        PingEvent,
        ContentBlockDeltaEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageDeltaEvent,
        MessageStopEvent,
    ]
    assert json.loads(recorder.last.body) == request(stream=True)
    assert recorder.last.headers["anthropic-version"] == API_VERSION

    start = received[0]
    assert isinstance(start, MessageStartEvent)
    assert start.message.usage.input_tokens == 25
    delta = received[3]
    assert isinstance(delta, ContentBlockDeltaEvent)
    assert delta.delta == TextDelta(text="Hello")
    end = received[6]
    assert isinstance(end, MessageDeltaEvent)
    assert end.delta.stop_reason == "end_turn"
    assert end.usage is not None
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
    new_event = {"type": "message_checkpoint", "id": "c-1"}
    no_usage = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
    }
    transport = FakeTransport(
        stream_responses=[
            FakeStreamResponse(
                [frame(tool_delta), frame(new_event), frame(no_usage)],
                headers=SSE,
            )
        ],
        error_resolver=AnthropicErrorResolver(),
    )

    async with transport.stream(StreamMessage(**request())) as events:
        delta, unknown, end = [event async for event in events]

    assert isinstance(delta, ContentBlockDeltaEvent)
    assert isinstance(delta.delta, UnknownDelta)
    assert delta.delta.model_dump() == tool_delta["delta"]
    assert isinstance(unknown, UnknownEvent)
    assert unknown.model_dump() == new_event
    assert isinstance(end, MessageDeltaEvent)
    assert end.usage is None


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
    assert caught.value.transient is True
    assert "Overloaded" in str(caught.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 1},
        {"metadata": {"user_id": "u", "team": "t"}},
        {"metadata": {"user_id": "u" * 513}},
        {"messages": []},
        {"messages": [{"role": "system", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x", "name": "n"}]},
        {"max_tokens": -1},
    ],
    ids=[
        "temperature",
        "metadata-key",
        "user-id-length",
        "no-messages",
        "role",
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
