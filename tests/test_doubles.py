"""Self-checks of the shared doubles in ``conftest.py``.

They only prove the infrastructure works; behavior of the library is
tested by the stage suites.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from pytest_aiohttp import AiohttpServer

from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import (
    InternalServerError,
    ServiceUnavailableError,
)
from ai_api.http.exceptions.transport import APIConnectionError
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.response.buffered import BufferedResponse
from ai_api.session.aiohttp import AiohttpSession
from tests.conftest import (
    TEST_TOKEN,
    FakeStreamResponse,
    FakeTransport,
    LocalAPI,
    Recorder,
    Reply,
    make_method,
)


async def test_retry_wait_is_instant_and_records_outcomes(
    no_retry_wait: MagicMock,
) -> None:
    method = make_method(BaseAPIMethod[bytes], __retryable__=True)()
    unavailable = BufferedResponse(503, {}, b"")
    transport = FakeTransport(
        [unavailable, unavailable, BufferedResponse(200, {}, b"ok")]
    )

    async with asyncio.timeout(0.5):
        assert await transport.send_request(method) == b"ok"

    assert transport.attempts == 3
    assert [type(exc) for exc in no_retry_wait.outcomes] == [
        ServiceUnavailableError,
        ServiceUnavailableError,
    ]


async def test_fake_transport_replays_queue() -> None:
    method = make_method(BaseAPIMethod[bytes])()
    transport = FakeTransport(
        [APIConnectionError(), BufferedResponse(200, {}, b"ok")]
    )

    with pytest.raises(APIConnectionError):
        await transport.send_request(method)

    assert await transport.send_request(method) == b"ok"
    assert transport.attempts == 2

    with pytest.raises(AssertionError, match="queue is empty"):
        await transport.send_request(method)


async def test_fake_stream_response_fails_at_chunk() -> None:
    response = FakeStreamResponse(
        [b"a", b"b", b"c"],
        fail_at=1,
        error=APIConnectionError(),
    )
    chunks = response.iter_chunks()

    assert await anext(chunks) == b"a"

    with pytest.raises(APIConnectionError):
        await anext(chunks)

    assert not response.closed

    await response.aclose()
    assert response.closed


def test_make_method_declares_fields() -> None:
    method_cls = make_method(
        BaseAPIMethod[str],
        path="/items/{item_id}",
        fields={"item_id": str, "q": (int | None, None)},
        __request_query__={"q"},
    )

    method = method_cls.model_validate({"item_id": "a/b", "q": 1})

    assert method.request_path == "/items/a%2Fb"
    assert method.get_query_map() == {"q": "1"}


async def test_session_talks_to_recorder(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(body=b"pong"))
    method_cls = make_method(
        BaseAPIMethod[bytes],
        path="/ping/{name}",
        fields={"name": str},
    )

    assert (
        await session.send_request(method_cls.model_validate({"name": "../x"}))
        == b"pong"
    )
    assert recorder.last.raw_path == "/v1/ping/..%2Fx"
    assert recorder.last.headers["Authorization"] == f"Bearer {TEST_TOKEN}"


async def test_recorder_streams_and_waits_for_events(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    release = asyncio.Event()
    recorder.reply(Reply(chunks=[b"first", release, b"last"]))
    method = make_method(BaseStreamMethod[bytes], http_method="GET")()

    async with session.stream(method) as stream:
        chunks = aiter(stream)
        received = [await anext(chunks)]
        release.set()
        received.extend([chunk async for chunk in chunks])

    assert b"".join(received) == b"firstlast"


async def test_recorder_abort_breaks_body(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(chunks=[b"part"], content_length=100, abort=True))
    method = make_method(BaseAPIMethod[bytes])()

    with pytest.raises(APIConnectionError):
        await session.send_request(method)


async def test_close_is_instant_and_repeatable(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply())
    await session.send_request(make_method(BaseAPIMethod[None])())

    async with asyncio.timeout(0.2):
        await session.close()
        await session.close()


async def test_unscripted_request_gets_500_and_is_recorded(
    aiohttp_server: AiohttpServer,
) -> None:
    # A request nobody scripted, such as a hidden retry, must not pass:
    # it gets a 500 and the recorder fixture fails the test at teardown.
    own = Recorder()
    server = await aiohttp_server(own.app)

    async with AiohttpSession(
        LocalAPI(server.make_url("/")),
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
    ) as session:
        with pytest.raises(InternalServerError):
            await session.send_request(make_method(BaseAPIMethod[None])())

    assert [request.raw_path for request in own.unscripted] == ["/v1/test"]


async def test_default_reply_answers_once_the_script_is_exhausted(
    aiohttp_server: AiohttpServer,
) -> None:
    own = Recorder()
    own.reply(Reply(body=b"first"))
    own.default = Reply(body=b"default")
    server = await aiohttp_server(own.app)

    async with AiohttpSession(
        LocalAPI(server.make_url("/")),
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
    ) as session:
        method = make_method(BaseAPIMethod[bytes])()
        bodies = [await session.send_request(method) for _ in range(3)]

    assert bodies == [b"first", b"default", b"default"]
    assert own.unscripted == []
