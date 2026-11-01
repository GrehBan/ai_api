"""Audit fixes: failures of ``AiohttpSession`` that used to be mishandled."""

from __future__ import annotations

import asyncio
import io
import math
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from pytest_aiohttp import AiohttpServer
from pytest_mock import MockerFixture
from yarl import URL

from ai_api.enums import BodyFormat
from ai_api.http.auth import APIKeyAuth, BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import ServiceUnavailableError
from ai_api.http.exceptions.transport import (
    APIRedirectError,
    APIRequestError,
)
from ai_api.http.headers import Headers
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.session.aiohttp import AiohttpSession
from ai_api.session.loop import bind_loop
from ai_api.types.files import IOFile, PathFile, StreamFile
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply

pytestmark = pytest.mark.integration


class Ping(BaseAPIMethod[bytes]):
    __request_path__ = "/ping"
    __request_method__ = "GET"
    __retryable__ = True


class Search(BaseAPIMethod[bytes]):
    __request_path__ = "/search"
    __request_method__ = "GET"
    __request_query__ = {"ids", "limit"}

    ids: list[str]
    limit: int | None = None


class Create(BaseAPIMethod[bytes]):
    __request_path__ = "/items"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __retryable__ = True

    payload: Any = None


class Upload(BaseAPIMethod[bytes]):
    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __retryable__ = True

    file: Any = None


class Download(BaseStreamMethod[bytes]):
    __request_path__ = "/download"
    __request_method__ = "GET"


class RotatingAuth:
    """``IAuth`` whose token changes after every request."""

    header_name = "Authorization"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self) -> str:
        self.calls += 1
        return f"Bearer token-{self.calls}"


class BrokenAuth:
    """``IAuth`` that tries to inject a header."""

    header_name = "Authorization"

    def encode(self) -> str:
        return "Bearer x\r\nX-Injected: 1"


class FailingReads(io.BytesIO):
    """A seekable stream whose reads fail after the first chunk."""

    def __init__(self) -> None:
        super().__init__(b"x" * 10)
        self.reads = 0

    def read(self, size: int | None = -1, /) -> bytes:
        self.reads += 1

        if self.reads > 1:
            raise OSError("disk failure")

        return super().read(1)


def make_session(api: LocalAPI, **kwargs: Any) -> AiohttpSession:
    return AiohttpSession(
        api,
        kwargs.pop("auth", BearerAuth(TEST_TOKEN)),
        BaseErrorResolver(),
        **kwargs,
    )


async def test_query_lists_are_repeated_keys(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply())
    await session.send_request(Search(ids=["a", "b c"], limit=2))

    assert recorder.last.query.getall("ids") == ["a", "b c"]
    assert recorder.last.query.getall("limit") == ["2"]


async def test_json_body_is_sent_as_json(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply())
    await session.send_request(Create(payload={"a": [1, None]}))

    assert recorder.last.headers["Content-Type"] == "application/json"
    assert recorder.last.body == b'{"payload":{"a":[1,null]}}'


@pytest.mark.parametrize(
    ("payload", "error"),
    [(math.nan, ValueError), ({1, 2}, TypeError)],
    ids=["nan", "set"],
)
async def test_invalid_json_is_rejected_before_sending(
    session: AiohttpSession,
    recorder: Recorder,
    payload: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        await session.send_request(Create(payload=payload))

    assert recorder.requests == []


async def test_missing_upload_is_not_sent(
    session: AiohttpSession,
    recorder: Recorder,
    no_retry_wait: MagicMock,
    tmp_path: Path,
) -> None:
    with pytest.raises(APIRequestError, match="missing.bin"):
        await session.send_request(
            Upload(file=PathFile(tmp_path / "missing.bin"))
        )

    assert recorder.requests == []
    assert no_retry_wait.call_count == 0


async def test_failing_upload_source_is_not_retried(
    session: AiohttpSession,
    no_retry_wait: MagicMock,
) -> None:
    source = FailingReads()
    method = Upload(file=IOFile(source, filename="a.bin"))
    assert method.retryable

    with pytest.raises(APIRequestError, match="a.bin") as info:
        await session.send_request(method)

    assert isinstance(info.value.__cause__, OSError)
    assert info.value.transient is False
    assert no_retry_wait.call_count == 0


async def test_failing_stream_file_is_a_request_error(
    session: AiohttpSession,
) -> None:
    async def produce() -> AsyncIterator[bytes]:
        yield b"x"
        raise RuntimeError("producer failed")

    with pytest.raises(APIRequestError) as info:
        await session.send_request(Upload(file=StreamFile(produce())))

    assert str(info.value.__cause__) == "producer failed"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_redirects_are_not_followed_by_default(
    session: AiohttpSession,
    recorder: Recorder,
    no_retry_wait: MagicMock,
    status: int,
) -> None:
    recorder.default = Reply(status=status, headers={"Location": "/v1/x"})

    with pytest.raises(APIRedirectError, match="not followed: /v1/x") as info:
        await session.send_request(Ping())

    assert info.value.transient is False
    assert len(recorder.requests) == 1
    assert no_retry_wait.call_count == 0


async def test_same_origin_redirect_is_followed(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=302, headers={"Location": "/v1/moved"}))
    recorder.default = Reply(body=b"moved")

    async with make_session(api, follow_redirects=True) as session:
        assert await session.send_request(Ping()) == b"moved"

    assert [r.raw_path for r in recorder.requests] == ["/v1/ping", "/v1/moved"]
    assert recorder.last.headers["Authorization"] == f"Bearer {TEST_TOKEN}"


async def test_redirect_loop_is_a_redirect_error(
    api: LocalAPI,
    recorder: Recorder,
    no_retry_wait: MagicMock,
) -> None:
    recorder.default = Reply(status=302, headers={"Location": "/v1/ping"})

    async with make_session(api, follow_redirects=True) as session:
        with pytest.raises(APIRedirectError, match="not followed") as info:
            await session.send_request(Ping())

    assert info.value.transient is False
    assert no_retry_wait.call_count == 0


async def test_redirect_without_location_is_a_redirect_error(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=302))

    async with make_session(api, follow_redirects=True) as session:
        with pytest.raises(APIRedirectError, match="not followed"):
            await session.send_request(Ping())


async def test_redirect_with_a_consumed_upload_is_a_request_error(
    api: LocalAPI,
    recorder: Recorder,
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"data")
    recorder.reply(Reply(status=307, headers={"Location": "/v1/again"}))

    async with make_session(api, follow_redirects=True) as session:
        with pytest.raises(APIRequestError, match="Cannot follow"):
            await session.send_request(Upload(file=PathFile(path)))

    assert len(recorder.requests) == 1


async def test_header_injection_is_a_request_error(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    async with make_session(api, auth=BrokenAuth()) as session:
        with pytest.raises(APIRequestError, match="Invalid request"):
            await session.send_request(Ping())

    assert recorder.requests == []


async def test_authorization_is_built_for_every_request(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(*[Reply()] * 2)
    async with make_session(api, auth=RotatingAuth()) as session:
        await session.send_request(Ping())
        await session.send_request(Ping())

    assert [r.headers["Authorization"] for r in recorder.requests] == [
        "Bearer token-1",
        "Bearer token-2",
    ]


async def test_cross_origin_redirect_is_not_followed(
    api: LocalAPI,
    recorder: Recorder,
    aiohttp_server: AiohttpServer,
) -> None:
    seen: list[str | None] = []

    async def other(request: web.Request) -> web.Response:
        seen.append(request.headers.get("x-api-key"))
        return web.Response(body=b"moved")

    app = web.Application()
    app.router.add_get("/elsewhere", other)
    elsewhere: TestServer = await aiohttp_server(app)
    target = str(elsewhere.make_url("/elsewhere"))
    recorder.reply(Reply(status=302, headers={"Location": target}))

    async with make_session(
        api,
        headers=Headers({"x-api-key": "SECRET"}),
        follow_redirects=True,
    ) as session:
        with pytest.raises(APIRedirectError, match="another origin"):
            await session.send_request(Ping())

    assert recorder.last.headers["x-api-key"] == "SECRET"
    assert seen == []


async def test_api_key_is_sent_in_its_header(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply())
    async with make_session(api, auth=APIKeyAuth("SECRET")) as session:
        await session.send_request(Ping())

    assert recorder.last.headers["x-api-key"] == "SECRET"
    assert "Authorization" not in recorder.last.headers


@pytest.mark.parametrize("trust_env", [False, True])
async def test_trust_env_is_passed_to_aiohttp(
    api: LocalAPI,
    mocker: MockerFixture,
    trust_env: bool,
) -> None:
    created = mocker.patch(
        "ai_api.session.aiohttp.ClientSession",
        wraps=ClientSession,
    )

    async with make_session(api, trust_env=trust_env):
        pass

    assert created.call_args.kwargs["trust_env"] is trust_env


async def test_trust_env_is_off_by_default(
    api: LocalAPI,
    mocker: MockerFixture,
) -> None:
    created = mocker.patch(
        "ai_api.session.aiohttp.ClientSession",
        wraps=ClientSession,
    )

    async with AiohttpSession(
        api, BearerAuth(TEST_TOKEN), BaseErrorResolver()
    ):
        pass

    assert created.call_args.kwargs["trust_env"] is False


@pytest.mark.parametrize(
    "reply",
    [
        Reply(body=b"x" * 11),
        Reply(chunks=[b"x" * 6, b"x" * 5]),
    ],
    ids=["content-length", "chunked"],
)
async def test_buffered_body_over_the_limit(
    api: LocalAPI,
    recorder: Recorder,
    reply: Reply,
) -> None:
    recorder.reply(reply)

    async with make_session(api, max_body_size=10) as session:
        with pytest.raises(APIResponseError, match="exceeds 10 bytes"):
            await session.send_request(Ping())


async def test_buffered_body_at_the_limit(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(chunks=[b"x" * 6, b"x" * 4]))

    async with make_session(api, max_body_size=10) as session:
        assert await session.send_request(Ping()) == b"x" * 10


async def test_stream_error_body_over_the_limit(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=503, body=b"x" * 11))

    async with make_session(api, max_body_size=10) as session:
        with pytest.raises(APIResponseError, match="exceeds 10 bytes"):
            async with session.stream(Download()):
                pass


async def test_stream_body_is_not_limited(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(chunks=[b"x" * 8, b"x" * 8]))

    async with (
        make_session(api, max_body_size=10) as session,
        session.stream(Download()) as chunks,
    ):
        assert b"".join([chunk async for chunk in chunks]) == b"x" * 16


async def test_stream_error_body_within_the_limit(
    api: LocalAPI,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=503, body=b"busy"))

    async with make_session(api, max_body_size=10) as session:
        with pytest.raises(ServiceUnavailableError):
            async with session.stream(Download()):
                pass


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_session_outside_a_coroutine_binds_on_first_use() -> None:
    port = free_port()
    session = make_session(LocalAPI(URL(f"http://127.0.0.1:{port}/")))

    assert session.loop is None

    async def pong(_: web.Request) -> web.Response:
        return web.Response(body=b"pong")

    async def run_once() -> tuple[bytes, asyncio.AbstractEventLoop | None]:
        app = web.Application()
        app.router.add_get("/v1/ping", pong)

        async with TestServer(app, port=port), session:
            return await session.send_request(Ping()), session.loop

    loop = asyncio.new_event_loop()

    try:
        body, bound = loop.run_until_complete(run_once())
    finally:
        loop.close()

    assert body == b"pong"
    assert bound is loop
    assert session.loop is loop


def test_reading_the_loop_does_not_bind_it() -> None:
    port = free_port()
    session = make_session(LocalAPI(URL(f"http://127.0.0.1:{port}/")))

    async def peek() -> asyncio.AbstractEventLoop | None:
        return session.loop

    assert asyncio.run(peek()) is None

    async def pong(_: web.Request) -> web.Response:
        return web.Response(body=b"pong")

    async def run_once() -> bytes:
        app = web.Application()
        app.router.add_get("/v1/ping", pong)

        async with TestServer(app, port=port), session:
            return await session.send_request(Ping())

    # A second loop can still use the session: peeking bound nothing.
    assert asyncio.run(run_once()) == b"pong"


def test_bind_loop_outside_a_coroutine_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="^Owner can only be used inside"):
        bind_loop(None, "Owner")


async def test_bind_loop_binds_and_checks_the_running_loop() -> None:
    running = asyncio.get_running_loop()
    other = asyncio.new_event_loop()

    try:
        assert bind_loop(None, "Owner") is running
        assert bind_loop(running, "Owner") is running

        with pytest.raises(RuntimeError, match="^Owner is bound to another"):
            bind_loop(other, "Owner")
    finally:
        other.close()


def test_session_from_another_loop_is_an_error() -> None:
    async def create() -> AiohttpSession:
        async with make_session(LocalAPI(URL("http://127.0.0.1/"))) as s:
            return s

    session = asyncio.run(create())

    with pytest.raises(RuntimeError, match="another event loop"):
        asyncio.run(session.send_request(Ping()))

    with pytest.raises(RuntimeError, match="another event loop"):
        asyncio.run(session.close())


def test_session_bound_to_an_explicit_loop() -> None:
    port = free_port()
    loop = asyncio.new_event_loop()
    session = make_session(
        LocalAPI(URL(f"http://127.0.0.1:{port}/")),
        loop=loop,
    )

    async def pong(_: web.Request) -> web.Response:
        return web.Response(body=b"pong")

    async def run_once() -> bytes:
        app = web.Application()
        app.router.add_get("/v1/ping", pong)

        async with TestServer(app, port=port), session:
            return await session.send_request(Ping())

    try:
        with pytest.raises(RuntimeError, match="another event loop"):
            asyncio.run(session.send_request(Ping()))

        assert loop.run_until_complete(run_once()) == b"pong"
    finally:
        loop.close()


async def test_malformed_http_response_is_a_response_error() -> None:
    async def garbage(
        _: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        writer.write(b"NOT-HTTP\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(garbage, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async with (
        server,
        make_session(LocalAPI(URL(f"http://127.0.0.1:{port}/"))) as session,
    ):
        with pytest.raises(APIResponseError, match="Invalid HTTP response"):
            await session.send_request(Ping())
