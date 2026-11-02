"""An attempt lets go of its uploads, however it ends.

aiohttp does not close the iterator of a body it stops writing: when the
connection breaks mid-upload, or the server answers before it has read
the body, the reader of the first attempt is left suspended. The session
closes it when the attempt ends, so the retry can read the file again.
The local aiohttp server always reads whole bodies, so a raw socket
server breaks the first attempt.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import MISSING
from yarl import URL

from aiollm.enums import BodyFormat, ResponseFormat
from aiollm.http.auth import BearerAuth
from aiollm.http.exceptions.resolver import BaseErrorResolver
from aiollm.http.exceptions.response import APIResponseError
from aiollm.http.exceptions.status import ServiceUnavailableError
from aiollm.http.exceptions.transport import (
    APIConnectionError,
    APIRequestError,
)
from aiollm.methods.base import BaseAPIMethod, BaseStreamMethod
from aiollm.session.aiohttp import AiohttpSession
from aiollm.types.base import BaseAPIType
from aiollm.types.files import IOFile, PathFile, StreamFile
from tests.conftest import TEST_TOKEN, LocalAPI

pytestmark = pytest.mark.integration

CONTENT = bytes(range(256)) * 32 * 1024
"""8 MiB: more than the socket buffers hold, so the cut is mid-upload."""

type Failure = Literal["cut", "early"]


class Upload(BaseAPIMethod[bytes]):
    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __retryable__ = True

    file: Any = MISSING


async def read_chunked(reader: asyncio.StreamReader) -> bytes:
    """Read a body sent with chunked transfer encoding."""
    body = bytearray()

    while True:
        size = int((await reader.readline()).strip(), 16)
        body += await reader.readexactly(size)
        await reader.readexactly(2)

        if size == 0:
            return bytes(body)


class FlakyServer:
    """Breaks the first request as told, then reads and answers 200."""

    def __init__(self, failure: Failure) -> None:
        self.failure = failure
        self.attempts = 0
        self.bodies: list[bytes] = []
        self.url = URL()

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.attempts += 1
        await reader.readuntil(b"\r\n\r\n")

        if self.attempts == 1:
            if self.failure == "cut":
                await reader.readexactly(64 * 1024)
                writer.transport.abort()
                return

            # Answer before reading the body, then hang up.
            writer.write(
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            return

        self.bodies.append(await read_chunked(reader))
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 2\r\nConnection: close\r\n\r\nok"
        )
        await writer.drain()
        writer.close()


@pytest.fixture(params=["cut", "early"])
async def flaky(request: pytest.FixtureRequest) -> AsyncIterator[FlakyServer]:
    flaky = FlakyServer(request.param)
    server = await asyncio.start_server(flaky.handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    flaky.url = URL(f"http://{host}:{port}/")

    try:
        yield flaky
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
async def session(flaky: FlakyServer) -> AsyncIterator[AiohttpSession]:
    async with AiohttpSession(
        LocalAPI(flaky.url), BearerAuth(TEST_TOKEN), BaseErrorResolver()
    ) as session:
        yield session


def make_io_file(_: Path) -> IOFile:
    return IOFile(io.BytesIO(CONTENT), filename="a.bin")


def make_path_file(tmp_path: Path) -> PathFile:
    path = tmp_path / "a.bin"
    path.write_bytes(CONTENT)
    return PathFile(path)


def make_raw_stream(_: Path) -> io.BytesIO:
    return io.BytesIO(CONTENT)


@pytest.mark.parametrize(
    "make_file",
    [make_io_file, make_path_file, make_raw_stream],
    ids=["io-file", "path-file", "raw-stream"],
)
async def test_the_retry_sends_the_whole_file_again(
    session: AiohttpSession,
    flaky: FlakyServer,
    tmp_path: Path,
    make_file: Callable[[Path], Any],
) -> None:
    file = make_file(tmp_path)

    result = await session.send_request(Upload(file=file))

    assert result == b"ok"
    assert flaky.attempts == 2
    (body,) = flaky.bodies
    assert CONTENT in body

    # Nothing is left reading the stream: the first reader was closed.
    stream = file.file if isinstance(file, IOFile) else file
    if isinstance(stream, io.BytesIO):
        stream.seek(0)
        chunks = [chunk async for chunk in IOFile(stream).open()]
        assert b"".join(chunks) == CONTENT


class Produce:
    """The 8 MiB content in pieces; it records its close."""

    def __init__(self, *, close_fails: bool = False) -> None:
        self.pieces = [
            CONTENT[start : start + 64 * 1024]
            for start in range(0, len(CONTENT), 64 * 1024)
        ]
        self.close_fails = close_fails
        self.closed = 0

    def __aiter__(self) -> Produce:
        return self

    async def __anext__(self) -> bytes:
        if not self.pieces:
            raise StopAsyncIteration
        return self.pieces.pop(0)

    async def aclose(self) -> None:
        self.closed += 1
        if self.close_fails:
            raise OSError("close failed")


async def test_a_stream_file_source_is_closed_when_the_attempt_breaks(
    session: AiohttpSession,
    flaky: FlakyServer,
) -> None:
    # A stream file is never retried: the first attempt is the last.
    source = Produce()

    with pytest.raises((APIConnectionError, ServiceUnavailableError)):
        await session.send_request(Upload(file=StreamFile(source)))

    assert flaky.attempts == 1
    assert source.closed == 1


async def answer_before_the_body(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Answer 200 before reading the body, then hang up."""
    await reader.readuntil(b"\r\n\r\n")
    writer.write(
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
    )
    await writer.drain()
    writer.close()


@pytest.fixture
async def early_url() -> AsyncIterator[URL]:
    server = await asyncio.start_server(answer_before_the_body, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    try:
        yield URL(f"http://{host}:{port}/")
    finally:
        server.close()
        await server.wait_closed()


async def test_a_source_that_fails_to_close_after_an_answer_is_raised(
    early_url: URL,
) -> None:
    # The answer came, but the upload was left unread and its source
    # could not be closed: that is never dropped.
    source = Produce(close_fails=True)

    async with AiohttpSession(
        LocalAPI(early_url), BearerAuth(TEST_TOKEN), BaseErrorResolver()
    ) as session:
        with pytest.raises(APIRequestError, match="close failed"):
            await session.send_request(Upload(file=StreamFile(source)))

    assert source.closed == 1


async def test_a_failing_close_is_noted_on_the_attempts_error(
    early_url: URL,
) -> None:
    source = Produce(close_fails=True)

    async with AiohttpSession(
        LocalAPI(early_url),
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
        max_body_size=1,
    ) as session:
        with pytest.raises(APIResponseError) as info:
            await session.send_request(
                Upload(file=StreamFile(source, filename="a.bin"))
            )

    (note,) = info.value.__notes__
    assert note.startswith("Closing upload 'a.bin' also failed")
    assert source.closed == 1


class Event(BaseAPIType):
    text: str


class StreamUpload(BaseStreamMethod[Event]):
    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __response_format__ = ResponseFormat.SSE

    file: Any = MISSING


async def stream_before_the_body(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Answer an SSE stream the parser rejects, before reading the body."""
    await reader.readuntil(b"\r\n\r\n")
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\nConnection: close\r\n\r\n"
        b"data: not json\n\n"
    )
    await writer.drain()
    writer.close()


async def test_a_failing_close_never_hides_the_streams_error() -> None:
    # The parser fails while the upload is still open, and its source
    # fails to close when the response is: the parser's error is raised,
    # with the close failure as a note.
    server = await asyncio.start_server(stream_before_the_body, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    source = Produce(close_fails=True)
    method = StreamUpload(file=StreamFile(source, filename="a.bin"))

    try:
        async with AiohttpSession(
            LocalAPI(URL(f"http://{host}:{port}/")),
            BearerAuth(TEST_TOKEN),
            BaseErrorResolver(),
        ) as session:
            with pytest.raises(APIResponseError, match="decode") as info:
                async with session.stream(method) as events:
                    async for _ in events:
                        pass
    finally:
        server.close()
        await server.wait_closed()

    (note,) = info.value.__notes__
    assert note.startswith("Closing the response also failed: APIRequestError")
    assert "close failed" in note
    assert source.closed == 1


async def refuse_before_the_body(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Answer 503 before reading the body, then hang up."""
    await reader.readuntil(b"\r\n\r\n")
    writer.write(
        b"HTTP/1.1 503 Service Unavailable\r\n"
        b"Content-Length: 4\r\nConnection: close\r\n\r\nbusy"
    )
    await writer.drain()
    writer.close()


async def test_a_failing_close_never_hides_the_error_status() -> None:
    # The stream is refused while its upload is still open, and the
    # upload's source fails to close: the status error is raised, with
    # the close failure as a note, not the close failure instead of it.
    server = await asyncio.start_server(refuse_before_the_body, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    source = Produce(close_fails=True)
    method = StreamUpload(file=StreamFile(source, filename="a.bin"))

    try:
        async with AiohttpSession(
            LocalAPI(URL(f"http://{host}:{port}/")),
            BearerAuth(TEST_TOKEN),
            BaseErrorResolver(),
        ) as session:
            with pytest.raises(ServiceUnavailableError) as info:
                async with session.stream(method):
                    pass
    finally:
        server.close()
        await server.wait_closed()

    (note,) = info.value.__notes__
    assert note.startswith("Closing the response also failed: APIRequestError")
    assert source.closed == 1
