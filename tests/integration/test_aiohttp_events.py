"""SSE stage 4: SSE and NDJSON streams of ``AiohttpSession``."""

from __future__ import annotations

import asyncio
import itertools
import json
from typing import Annotated, Any, Literal

import pytest
from pydantic import Field

from ai_api.enums import BodyFormat, ResponseFormat
from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import APIStatusError
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.response.parse.sse import ServerSentEvent
from ai_api.session.aiohttp import AiohttpSession
from ai_api.types.base import BaseAPIType
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply

pytestmark = pytest.mark.integration

SSE_TYPE = "text/event-stream; charset=utf-8"


class MessageStart(BaseAPIType):
    type: Literal["message_start"]


class TextDelta(BaseAPIType):
    type: Literal["text_delta"]
    text: str


class Ping(BaseAPIType):
    type: Literal["ping"]


class MessageStop(BaseAPIType):
    type: Literal["message_stop"]


type MessageEvent = Annotated[
    MessageStart | TextDelta | Ping | MessageStop,
    Field(discriminator="type"),
]


class CreateMessage(BaseStreamMethod[MessageEvent]):
    __request_path__ = "/messages"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __response_format__ = ResponseFormat.SSE

    stream: bool = True


class Chunk(BaseAPIType):
    content: str


class Completions(BaseStreamMethod[Chunk]):
    __request_path__ = "/chat/completions"
    __request_method__ = "POST"
    __response_format__ = ResponseFormat.SSE


class RawEvents(BaseStreamMethod[ServerSentEvent]):
    __request_path__ = "/events"
    __request_method__ = "GET"


class Pull(BaseStreamMethod[dict[str, Any]]):
    __request_path__ = "/pull"
    __request_method__ = "POST"
    __response_format__ = ResponseFormat.NDJSON


class Health(BaseAPIMethod[bytes]):
    __request_path__ = "/health"
    __request_method__ = "GET"


def anthropic_event(data: dict[str, Any]) -> bytes:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


async def test_anthropic_style_events(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(
            chunks=[
                anthropic_event({"type": "message_start"}),
                anthropic_event({"type": "ping"}),
                anthropic_event({"type": "text_delta", "text": "При"}),
                anthropic_event({"type": "text_delta", "text": "вет"}),
                anthropic_event({"type": "message_stop"}),
            ],
            content_type=SSE_TYPE,
            chunk_delay=0.001,
        )
    )

    async with session.stream(CreateMessage()) as events:
        received = [event async for event in events]

    assert [type(event) for event in received] == [
        MessageStart,
        Ping,
        TextDelta,
        TextDelta,
        MessageStop,
    ]
    assert (
        "".join(
            event.text for event in received if isinstance(event, TextDelta)
        )
        == "Привет"
    )
    assert recorder.last.body == b'{"stream":true}'


async def test_openai_style_events_end_with_done(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    body = (
        b'data: {"content": "Hel"}\r\n\r\n'
        b'data: {"content": "lo"}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )
    recorder.reply(
        Reply(
            chunks=[body[i : i + 7] for i in range(0, len(body), 7)],
            content_type=SSE_TYPE,
        )
    )

    async with session.stream(Completions()) as chunks:
        received = [chunk.content async for chunk in chunks]

    assert received == ["Hel", "lo"]


async def test_events_arrive_before_the_body_ends(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    first_seen = asyncio.Event()
    recorder.reply(
        Reply(
            chunks=[
                b'data: {"content": "a"}\n\n',
                first_seen,
                b'data: {"content": "b"}\n\n',
            ],
            content_type=SSE_TYPE,
        )
    )

    async with asyncio.timeout(5), session.stream(Completions()) as chunks:
        items = aiter(chunks)
        assert (await anext(items)).content == "a"
        first_seen.set()
        assert [chunk.content async for chunk in items] == ["b"]


async def test_raw_events(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(
            chunks=[b": hello\n\nid: 5\nevent: tick\ndata: 1\ndata: 2\n\n"],
            content_type=SSE_TYPE,
        )
    )

    async with session.stream(RawEvents()) as events:
        received = [event async for event in events]

    assert received == [ServerSentEvent(data="1\n2", event="tick", id="5")]


async def test_error_event_is_raised_mid_stream(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    error = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    recorder.reply(
        Reply(
            chunks=[
                anthropic_event({"type": "message_start"}),
                f"event: error\ndata: {json.dumps(error)}\n\n".encode(),
            ],
            content_type=SSE_TYPE,
            headers={"request-id": "req-9"},
        )
    )

    with pytest.raises(APIStatusError) as info:
        async with session.stream(CreateMessage()) as events:
            async for _ in events:
                pass

    assert info.value.error_type == "overloaded_error"
    assert info.value.request_id == "req-9"


async def test_ndjson_lines(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    body = b'{"status": "pulling"}\n{"status": "done", "size": 3}'
    recorder.reply(
        Reply(
            chunks=[body[:10], body[10:30], body[30:]],
            content_type="application/x-ndjson",
        )
    )

    async with session.stream(Pull()) as lines:
        received = [line async for line in lines]

    assert received == [
        {"status": "pulling"},
        {"status": "done", "size": 3},
    ]


async def test_break_from_endless_events_releases_connection(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(
            chunks=itertools.repeat(b'data: {"content": "x"}\n\n'),
            content_type=SSE_TYPE,
            chunk_delay=0.001,
        ),
        Reply(body=b"ok"),
    )

    # With max_connections=1 a leaked connection would block the next request.
    async with AiohttpSession(
        api,
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
        max_connections=1,
    ) as session:
        async with session.stream(Completions()) as chunks:
            async for _ in chunks:
                break

        async with asyncio.timeout(2):
            assert await session.send_request(Health()) == b"ok"
