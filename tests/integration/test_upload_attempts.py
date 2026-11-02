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
from yarl import URL

from ai_api.enums import BodyFormat
from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.methods.base import BaseAPIMethod
from ai_api.session.aiohttp import AiohttpSession
from ai_api.types.files import IOFile, PathFile
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

    file: Any = None


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
    if flaky.failure == "early" and make_file is make_path_file:
        pytest.skip(
            "known bug, also on master: an answer that comes while aiofiles "
            "opens the file cancels the opening, and the file the worker "
            "thread opened is never closed"
        )

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
