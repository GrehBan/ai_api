"""OpenAI Responses API: types, methods and the stream error event.

The payloads follow the examples of OpenAI's API reference (captured
2026-10-29); the abbreviated ``response`` objects of the streaming
example are filled with the fields of the non-streaming one.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ai_api.enums import Provider
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import (
    APIStatusError,
    InternalServerError,
    RateLimitError,
)
from ai_api.http.exceptions.transport import APIStreamInterruptedError
from ai_api.providers.openai.methods import CreateResponse, StreamResponse
from ai_api.providers.openai.resolver import OpenAIErrorResolver
from ai_api.providers.openai.types import (
    ContentPartEvent,
    OutputItemEvent,
    OutputMessage,
    OutputRefusal,
    OutputText,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    RefusalDeltaEvent,
    RefusalDoneEvent,
    Response,
    ResponseStateEvent,
    UnknownContent,
    UnknownEvent,
    UnknownOutputItem,
)
from ai_api.response.buffered import BufferedResponse
from ai_api.session.aiohttp import AiohttpSession
from tests.conftest import (
    FakeStreamResponse,
    FakeTransport,
    Recorder,
    Reply,
)

MESSAGE: dict[str, Any] = {
    "type": "message",
    "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
    "status": "completed",
    "role": "assistant",
    "content": [
        {
            "type": "output_text",
            "text": "The image depicts a scenic landscape ...",
            "annotations": [],
            "logprobs": [],
        }
    ],
}

USAGE: dict[str, Any] = {
    "input_tokens": 328,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 52,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 380,
}

RESPONSE: dict[str, Any] = {
    "id": "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",
    "object": "response",
    "access_programs": None,
    "created_at": 1741476777,
    "status": "completed",
    "completed_at": 1741476778,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "model": "gpt-6-astra",
    "output": [MESSAGE],
    "parallel_tool_calls": True,
    "previous_response_id": None,
    "reasoning": {"effort": None, "summary": None, "context": None},
    "store": True,
    "temperature": 1,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": USAGE,
    "user": None,
    "metadata": {},
    "conversation": {"id": "id"},
    "service_tier": "auto",
    "output_text": "The image depicts a scenic landscape ...",
}

SSE = {"Content-Type": "text/event-stream"}


def frame(data: dict[str, Any]) -> bytes:
    """One SSE frame as OpenAI sends it: ``event:`` and ``data:``."""
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def state(kind: str, number: int, **changes: Any) -> dict[str, Any]:
    """A response state event carrying the example response."""
    response = RESPONSE | {"output": []} | changes
    return {"type": kind, "response": response, "sequence_number": number}


ITEM_ID = "msg_67c9"

PART = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}

STREAM: list[dict[str, Any]] = [
    state("response.created", 0, status="in_progress", usage=None),
    state("response.in_progress", 1, status="in_progress", usage=None),
    {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "id": ITEM_ID,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
        "sequence_number": 2,
    },
    {
        "type": "response.content_part.added",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "part": PART,
        "sequence_number": 3,
    },
    {
        "type": "response.output_text.delta",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "delta": "Hi",
        "sequence_number": 4,
        "logprobs": [],
    },
    {
        "type": "response.output_text.annotation.added",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "annotation_index": 0,
        "annotation": None,
        "sequence_number": 5,
    },
    {
        "type": "response.output_text.done",
        "item_id": ITEM_ID,
        "output_index": 0,
        "content_index": 0,
        "text": "Hi there!",
        "sequence_number": 6,
        "logprobs": [],
    },
    state("response.completed", 7),
]


def test_response_reads_the_reference_example() -> None:
    response = Response.model_validate(RESPONSE)

    assert response.id == RESPONSE["id"]
    assert response.status == "completed"
    assert response.created_at == 1741476777
    assert response.output == [
        OutputMessage(
            id=MESSAGE["id"],
            status="completed",
            content=[
                OutputText(text=MESSAGE["content"][0]["text"], logprobs=[])
            ],
        )
    ]
    assert response.output_text == MESSAGE["content"][0]["text"]
    assert response.usage is not None
    assert response.usage.total_tokens == 380
    assert response.usage.input_tokens_details is not None
    assert response.usage.input_tokens_details.cache_write_tokens == 0


def test_response_keeps_unknown_items_and_parts_whole() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Thinking"}],
    }
    message = MESSAGE | {
        "content": [
            {"type": "output_text", "text": "Hel"},
            {"type": "refusal", "refusal": "No"},
            {"type": "reasoning_text", "text": "hidden"},
            {"type": "output_text", "text": "lo"},
        ]
    }
    response = Response.model_validate(
        RESPONSE | {"output": [reasoning, message]}
    )

    first, second = response.output
    assert isinstance(first, UnknownOutputItem)
    assert first.model_dump() == reasoning
    assert isinstance(second, OutputMessage)
    assert isinstance(second.content[1], OutputRefusal)
    assert isinstance(second.content[2], UnknownContent)
    assert second.content[2].model_dump() == message["content"][2]
    assert response.output_text == "Hello"


def test_an_item_without_a_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="type"):
        Response.model_validate(RESPONSE | {"output": [{"id": "x"}]})


def test_text_without_logprobs_is_read() -> None:
    # The schema allows null; the documentation examples always send [].
    message = MESSAGE | {
        "content": [{**MESSAGE["content"][0], "logprobs": None}]
    }

    response = Response.model_validate(RESPONSE | {"output": [message]})

    assert response.output_text == MESSAGE["content"][0]["text"]


def test_response_without_text_has_no_output_text() -> None:
    response = Response.model_validate(RESPONSE | {"output": []})

    assert response.output_text is None


def test_failed_response_carries_its_error() -> None:
    response = Response.model_validate(
        RESPONSE
        | {
            "status": "failed",
            "output": [],
            "usage": None,
            "error": {
                "code": "server_error",
                "message": "The model failed to generate a response.",
            },
        }
    )

    assert response.error is not None
    assert response.error.code == "server_error"
    assert response.usage is None


async def test_create_response_sends_and_parses(
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    recorder.reply(Reply(json=RESPONSE))

    response = await session.send_request(
        CreateResponse(
            model="gpt-6-astra",
            input="Hi",
            instructions="Be brief",
            max_output_tokens=16,
            metadata={"user": "u-1"},
        )
    )

    assert response.output_text == MESSAGE["content"][0]["text"]
    assert recorder.last.method == "POST"
    assert recorder.last.raw_path == "/v1/responses"
    assert json.loads(recorder.last.body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "instructions": "Be brief",
        "max_output_tokens": 16,
        "metadata": {"user": "u-1"},
    }


async def test_stream_response_yields_typed_events(
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    recorder.reply(
        Reply(
            chunks=[frame(event) for event in STREAM],
            content_type="text/event-stream",
        )
    )

    async with session.stream(
        StreamResponse(model="gpt-6-astra", input="Hi")
    ) as events:
        received = [event async for event in events]

    assert [type(event) for event in received] == [
        ResponseStateEvent,
        ResponseStateEvent,
        OutputItemEvent,
        ContentPartEvent,
        OutputTextDeltaEvent,
        UnknownEvent,
        OutputTextDoneEvent,
        ResponseStateEvent,
    ]
    assert json.loads(recorder.last.body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "stream": True,
    }

    delta = received[4]
    assert isinstance(delta, OutputTextDeltaEvent)
    assert (delta.delta, delta.item_id, delta.sequence_number) == (
        "Hi",
        ITEM_ID,
        4,
    )

    unknown = received[5]
    assert isinstance(unknown, UnknownEvent)
    assert unknown.model_dump() == STREAM[5]

    completed = received[-1]
    assert isinstance(completed, ResponseStateEvent)
    assert completed.type == "response.completed"
    assert completed.response.usage is not None
    assert completed.response.usage.total_tokens == 380


async def test_refusal_events_are_regular_events() -> None:
    part = {"item_id": ITEM_ID, "output_index": 0, "content_index": 0}
    chunks = [
        frame(
            part
            | {
                "type": "response.refusal.delta",
                "delta": "No",
                "sequence_number": 1,
            }
        ),
        frame(
            part
            | {
                "type": "response.refusal.done",
                "refusal": "No",
                "sequence_number": 2,
            }
        ),
        frame(state("response.completed", 3)),
    ]
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(chunks, headers=SSE)],
        error_resolver=OpenAIErrorResolver(),
    )

    async with transport.stream(
        StreamResponse(model="m", input="Hi")
    ) as events:
        received = [event async for event in events]

    delta, done, completed = received
    assert isinstance(delta, RefusalDeltaEvent)
    assert isinstance(done, RefusalDoneEvent)
    assert done.refusal == "No"
    assert isinstance(completed, ResponseStateEvent)


async def test_failed_event_raises_by_its_code() -> None:
    failed = state(
        "response.failed",
        2,
        status="failed",
        usage=None,
        error={"code": "server_error", "message": "Failed"},
    )
    chunks = [frame(state("response.created", 1)), frame(failed)]
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(chunks, headers=SSE)],
        error_resolver=OpenAIErrorResolver(),
    )
    received: list[Any] = []

    async with transport.stream(
        StreamResponse(model="m", input="Hi")
    ) as events:
        with pytest.raises(InternalServerError) as caught:
            async for event in events:
                received.append(event)  # noqa: PERF401 - fails mid-loop

    assert len(received) == 1
    assert caught.value.code == "server_error"
    assert caught.value.transient is False
    assert caught.value.provider is Provider.OPENAI
    assert "Failed" in str(caught.value)
    raw = caught.value.raw
    assert isinstance(raw, ResponseStateEvent)
    assert raw.type == "response.failed"
    assert raw.response.status == "failed"


async def test_stream_cut_before_its_terminal_event_raises() -> None:
    chunks = [frame(state("response.created", 1))]
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(chunks, headers=SSE)],
        error_resolver=OpenAIErrorResolver(),
    )

    async with transport.stream(
        StreamResponse(model="m", input="Hi")
    ) as events:
        with pytest.raises(APIStreamInterruptedError) as caught:
            async for _ in events:
                pass

    assert caught.value.items_received == 1


async def test_event_after_the_terminal_event_raises() -> None:
    chunks = [
        frame(state("response.completed", 1)),
        frame(state("response.completed", 2)),
    ]
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(chunks, headers=SSE)],
        error_resolver=OpenAIErrorResolver(),
    )

    async with transport.stream(
        StreamResponse(model="m", input="Hi")
    ) as events:
        first = await anext(events)
        with pytest.raises(APIResponseError, match="after its terminal"):
            await anext(events)

    assert isinstance(first, ResponseStateEvent)


FAILED = RESPONSE | {
    "status": "failed",
    "output": [],
    "usage": None,
    "error": {"code": "server_error", "message": "The model failed."},
}


@pytest.mark.parametrize(
    ("error", "error_class", "message"),
    [
        (
            {"code": "server_error", "message": "The model failed."},
            InternalServerError,
            "The model failed.",
        ),
        (
            {"code": "invalid_prompt", "message": "Flagged."},
            APIStatusError,
            "Flagged.",
        ),
        (None, APIStatusError, "failed without an error"),
    ],
    ids=["server-error", "unmapped-code", "no-error"],
)
async def test_failed_response_raises_with_the_response(
    error: dict[str, Any] | None,
    error_class: type[APIStatusError],
    message: str,
) -> None:
    body = json.dumps(FAILED | {"error": error}).encode()
    transport = FakeTransport(
        [BufferedResponse(status=200, headers={}, body=body)],
        error_resolver=OpenAIErrorResolver(),
    )

    with pytest.raises(APIStatusError) as caught:
        await transport.send_request(CreateResponse(model="m", input="Hi"))

    assert type(caught.value) is error_class
    assert caught.value.status == 200
    assert caught.value.transient is False
    assert message in str(caught.value)
    assert isinstance(caught.value.raw, Response)
    assert caught.value.raw.status == "failed"


async def test_error_event_raises_with_its_message() -> None:
    error = {
        "type": "error",
        "code": "ERR_SOMETHING",
        "message": "Something went wrong",
        "param": None,
        "sequence_number": 1,
    }
    transport = FakeTransport(
        stream_responses=[
            FakeStreamResponse([frame(STREAM[0]), frame(error)], headers=SSE)
        ],
        error_resolver=OpenAIErrorResolver(),
    )

    async with transport.stream(
        StreamResponse(model="m", input="Hi")
    ) as events:
        with pytest.raises(APIStatusError) as caught:
            async for _ in events:
                pass

    assert caught.value.code == "ERR_SOMETHING"
    assert "Something went wrong" in str(caught.value)
    assert caught.value.provider is Provider.OPENAI
    assert caught.value.error_type is None


def test_resolver_keeps_the_nested_error_shape() -> None:
    error = OpenAIErrorResolver().resolve(
        status=429,
        body={"error": {"code": "rate_limit_exceeded", "message": "Slow"}},
        headers={"Retry-After": "1"},
    )

    assert isinstance(error, RateLimitError)
    assert error.code == "rate_limit_exceeded"
    assert error.transient

    not_json = OpenAIErrorResolver().resolve(status=503, body=None, headers={})

    assert not_json.code is None
    assert not_json.transient


def test_create_response_rejects_a_bad_request() -> None:
    with pytest.raises(ValidationError, match="greater_than_equal"):
        CreateResponse(model="m", input="i", max_output_tokens=15)

    with pytest.raises(ValidationError, match="too_long"):
        CreateResponse(
            model="m",
            input="i",
            metadata={str(key): "v" for key in range(17)},
        )

    with pytest.raises(ValidationError, match="string_too_long"):
        CreateResponse(model="m", input="i", metadata={"k" * 65: "v"})

    with pytest.raises(ValidationError, match="string_too_long"):
        CreateResponse(model="m", input="i", metadata={"k": "v" * 513})

    with pytest.raises(ValidationError, match="less_than_equal"):
        CreateResponse(model="m", input="i", temperature=2.5)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        CreateResponse.model_validate(
            {"model": "m", "input": "i", "max_tokens": 16}
        )
