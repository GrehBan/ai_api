"""S2: a body delimited by closing the connection, cut mid-stream.

Without ``Content-Length`` and chunked encoding, HTTP/1.1 ends a body
when the server closes the connection (RFC 9112, section 6.3). A cut is
then indistinguishable from a normal end, so only the stream's terminal
event can tell them apart. The local aiohttp server always frames its
bodies, so a raw socket server sends these.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal

import pytest
from yarl import URL

from ai_api.enums import ResponseFormat
from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.transport import APIStreamInterruptedError
from ai_api.methods.base import BaseStreamMethod
from ai_api.session.aiohttp import AiohttpSession
from ai_api.types.base import BaseAPIType
from ai_api.types.matchers import EventTypes
from tests.conftest import TEST_TOKEN, LocalAPI

pytestmark = pytest.mark.integration


class Event(BaseAPIType):
    type: Literal["delta", "stop"]


class Events(BaseStreamMethod[Event]):
    __request_path__ = "/events"
    __request_method__ = "GET"
    __response_format__ = ResponseFormat.SSE


class TerminatedEvents(Events):
    __stream_terminal__ = EventTypes("stop")


HEAD = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/event-stream\r\n"
    b"Connection: close\r\n"
    b"\r\n"
)


def frame(kind: str) -> bytes:
    return f"data: {json.dumps({'type': kind})}\n\n".encode()


@pytest.fixture
async def cut_server() -> AsyncIterator[LocalAPI]:
    """Answers with two events and a half, then closes the connection."""

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(HEAD + frame("delta") + frame("delta") + b'data: {"ty')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    try:
        yield LocalAPI(URL(f"http://{host}:{port}/"))
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def session(cut_server: LocalAPI) -> AsyncIterator[AiohttpSession]:
    async with AiohttpSession(
        cut_server, BearerAuth(TEST_TOKEN), BaseErrorResolver()
    ) as session:
        yield session


async def test_the_cut_looks_like_a_normal_end_without_a_terminal(
    session: AiohttpSession,
) -> None:
    # This is what the standards allow, and why terminals exist.
    async with session.stream(Events()) as events:
        received = [event async for event in events]

    assert [event.type for event in received] == ["delta", "delta"]


async def test_a_declared_terminal_catches_the_cut(
    session: AiohttpSession,
) -> None:
    received: list[Event] = []

    async with session.stream(TerminatedEvents()) as events:
        with pytest.raises(APIStreamInterruptedError) as info:
            async for event in events:
                received.append(event)  # noqa: PERF401 - fails mid-loop

    assert len(received) == 2
    assert info.value.items_received == 2
    assert info.value.__cause__ is None
