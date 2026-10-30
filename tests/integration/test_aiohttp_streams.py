"""Block K: streamed responses of ``AiohttpSession`` against a real server."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from typing import Any

import pytest
from aiohttp import ClientTimeout

from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import NotFoundError
from ai_api.http.exceptions.transport import (
    APIConnectionError,
    APITimeoutError,
)
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.session.aiohttp import AiohttpSession
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply

pytestmark = pytest.mark.integration


class Download(BaseStreamMethod[bytes]):
    __request_path__ = "/files/{file_id}/content"
    __request_method__ = "GET"

    file_id: str = "f1"


class RetryableDownload(Download):
    __retryable__ = True


class Generate(BaseStreamMethod[str]):
    __request_path__ = "/generate"
    __request_method__ = "POST"


class Ping(BaseAPIMethod[bytes]):
    __request_path__ = "/ping"
    __request_method__ = "GET"


def make_session(api: LocalAPI, **kwargs: Any) -> AiohttpSession:
    return AiohttpSession(
        api,
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
        **kwargs,
    )


async def test_chunks_arrive_before_the_body_ends(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    # The server only sends the last chunk after the client has seen the
    # first one; a buffered client would deadlock (and time out).
    first_seen = asyncio.Event()
    recorder.reply(Reply(chunks=[b"first", first_seen, b"last"]))

    async with asyncio.timeout(5), session.stream(Download()) as stream:
        chunks = aiter(stream)
        assert await anext(chunks) == b"first"
        first_seen.set()
        rest = [chunk async for chunk in chunks]

    assert b"".join(rest) == b"last"
    assert recorder.last.raw_path == "/v1/files/f1/content"


async def test_text_stream_with_split_character(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    encoded = "Привет, 😀!".encode()
    cut = encoded.index("😀".encode()) + 2
    recorder.reply(
        Reply(
            chunks=[encoded[:cut], encoded[cut:]],
            content_type="text/plain; charset=utf-8",
        )
    )

    async with session.stream(Generate()) as stream:
        pieces = [piece async for piece in stream]

    assert "".join(pieces) == "Привет, 😀!"
    assert all(pieces)


async def test_save_large_stream(
    session: AiohttpSession,
    recorder: Recorder,
    tmp_path: Path,
) -> None:
    block = bytes(range(256)) * 64
    recorder.reply(Reply(chunks=[block] * 64))
    target = tmp_path / "download.bin"

    async with session.stream(Download()) as stream:
        written = await stream.save(target)

    assert written == 1024 * 1024
    assert target.read_bytes() == block * 64


async def test_error_status_on_enter(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=404, json={"error": {"code": "no_file"}}))
    entered = False

    with pytest.raises(NotFoundError) as info:
        async with session.stream(Download()):
            entered = True

    assert not entered
    assert info.value.code == "no_file"


async def test_error_body_cut_short(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(status=500, chunks=[b"{"], content_length=50, abort=True)
    )

    with pytest.raises(APIConnectionError, match="Connection lost"):
        async with session.stream(Download()):
            pass

    assert len(recorder.requests) == 1


async def test_opening_is_retried(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=503), Reply(chunks=[b"data"]))

    async with session.stream(RetryableDownload()) as stream:
        assert [chunk async for chunk in stream] == [b"data"]

    assert len(recorder.requests) == 2


async def test_connection_dropped_mid_stream(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    # A retryable stream is not reopened once iteration has started.
    recorder.reply(Reply(chunks=[b"part"], abort=True))
    received: list[bytes] = []

    with pytest.raises(APIConnectionError, match="Connection lost"):
        async with session.stream(RetryableDownload()) as stream:
            async for chunk in stream:
                received.append(chunk)  # noqa: PERF401 - fails mid-loop

    assert received == [b"part"]
    assert len(recorder.requests) == 1


async def test_silence_between_chunks_times_out(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    resume = asyncio.Event()
    recorder.reply(Reply(chunks=[b"first", resume, b"never"]))
    received: list[bytes] = []

    async with make_session(
        api,
        stream_timeout=ClientTimeout(sock_read=0.3),
    ) as session:
        with pytest.raises(APITimeoutError):
            async with session.stream(Download()) as stream:
                async for chunk in stream:
                    received.append(chunk)  # noqa: PERF401 - fails mid-loop

        resume.set()

    assert received == [b"first"]


async def test_session_timeout_does_not_limit_streams(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    # ~0.6 s in total, three times the buffered timeout.
    recorder.reply(Reply(chunks=[b"x"] * 6, chunk_delay=0.1))

    async with (
        make_session(api, timeout=ClientTimeout(total=0.2)) as session,
        session.stream(Download()) as stream,
    ):
        chunks = [chunk async for chunk in stream]

    assert chunks == [b"x"] * 6


async def test_call_timeout_limits_silence(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    resume = asyncio.Event()
    recorder.reply(Reply(chunks=[b"first", resume, b"never"]))

    with pytest.raises(APITimeoutError):
        async with session.stream(Download(), timeout=0.3) as stream:
            async for _ in stream:
                pass

    resume.set()


async def test_call_timeout_does_not_limit_the_stream_length(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    # ~0.6 s in total, twice the call timeout, but never silent for long.
    recorder.reply(Reply(chunks=[b"x"] * 6, chunk_delay=0.1))

    async with session.stream(Download(), timeout=0.3) as stream:
        chunks = [chunk async for chunk in stream]

    assert chunks == [b"x"] * 6


async def test_stream_sends_call_headers(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(chunks=[b"x"]))

    async with session.stream(
        Download(),
        headers={"Last-Event-ID": "7"},
    ) as stream:
        assert [chunk async for chunk in stream] == [b"x"]

    assert recorder.last.headers["Last-Event-ID"] == "7"


async def test_break_from_endless_stream_releases_connection(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(chunks=itertools.repeat(b"x" * 1024), chunk_delay=0.001),
        Reply(body=b"pong"),
    )

    # With max_connections=1 a leaked connection would block the next request.
    async with make_session(api, max_connections=1) as session:
        async with session.stream(Download()) as stream:
            async for _ in stream:
                break

        async with asyncio.timeout(2):
            assert await session.send_request(Ping()) == b"pong"
