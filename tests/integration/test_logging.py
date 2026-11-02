"""What the library logs, at which level, and what it never logs.

Each layer logs only its own part, so a message is never repeated: the
session logs a call's start, end and retries, the transport each request
it sends. Tokens, headers, bodies, query strings, the values of path
parameters, exception messages and notes, and file names never reach a
log: any of them may hold a key or a user's data.
"""

from __future__ import annotations

import logging
import re
from typing import cast

import pytest
from aiohttp import ClientResponse

from ai_api.enums import BodyFormat
from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import RateLimitError
from ai_api.http.exceptions.transport import APIConnectionError
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.openai import OpenAIClient
from ai_api.response.aiohttp import wait_for_writer
from ai_api.response.buffered import BufferedResponse
from ai_api.session.aiohttp import AiohttpSession
from ai_api.session.retry import should_retry
from ai_api.types.unknown import UNKNOWN, by_type
from tests.conftest import (
    TEST_TOKEN,
    FakeTransport,
    LocalAPI,
    Recorder,
    Reply,
)

pytestmark = pytest.mark.integration


class Ask(BaseAPIMethod[bytes]):
    __request_path__ = "/ask"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __request_query__ = frozenset({"secret"})
    __retryable__ = True

    prompt: str
    secret: str


class Fetch(BaseAPIMethod[bytes]):
    __request_path__ = "/files/{file_id}"
    __request_method__ = "GET"

    file_id: str


class Chunks(BaseStreamMethod[bytes]):
    __request_path__ = "/chunks"
    __request_method__ = "GET"


def records(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str, str]]:
    """The library's records as (logger, level, message)."""
    return [
        (record.name, record.levelname, record.getMessage())
        for record in caplog.records
        if record.name.startswith("ai_api")
    ]


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG, logger="ai_api")
    return caplog


async def test_a_retried_call_is_logged_once_per_layer(
    session: AiohttpSession,
    recorder: Recorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder.reply(Reply(status=503), Reply(body=b"ok"))

    await session.send_request(Ask(prompt="hi", secret="q-secret"))

    flow = [
        (name, level, re.sub(r"duration=[\d.]+s", "duration=…", message))
        for name, level, message in records(caplog)
        if name in {"ai_api.session.base", "ai_api.session.aiohttp"}
        and not message.startswith(("Opened", "Closed"))
    ]
    finished = "HTTP attempt finished: status={} duration=…"
    assert flow == [
        ("ai_api.session.base", "DEBUG", "Starting Ask"),
        ("ai_api.session.aiohttp", "DEBUG", "Sending POST /ask"),
        ("ai_api.session.aiohttp", "DEBUG", finished.format(503)),
        (
            "ai_api.session.base",
            "WARNING",
            "Retrying request after ServiceUnavailableError: attempt=2, "
            "delay=0.00s",
        ),
        ("ai_api.session.aiohttp", "DEBUG", "Sending POST /ask"),
        ("ai_api.session.aiohttp", "DEBUG", finished.format(200)),
        ("ai_api.session.base", "DEBUG", "Completed Ask"),
    ]


async def test_secrets_never_reach_a_log(
    session: AiohttpSession,
    recorder: Recorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder.reply(Reply(status=503), Reply(body=b"ok"), Reply(body=b"f"))

    await session.send_request(Ask(prompt="the prompt", secret="q-secret"))
    await session.send_request(Fetch(file_id="file-of-a-user"))
    await session.close()

    text = caplog.text
    assert (
        "ai_api.session.aiohttp",
        "DEBUG",
        "Sending GET /files/{file_id}",
    ) in (records(caplog))
    for secret in (
        TEST_TOKEN,
        "q-secret",
        "the prompt",
        "Authorization",
        "file-of-a-user",
    ):
        assert secret not in text


async def test_a_stream_logs_its_start_and_its_end(
    session: AiohttpSession,
    recorder: Recorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder.reply(Reply(chunks=[b"a", b"b"]))

    async with session.stream(Chunks()) as chunks:
        assert [chunk async for chunk in chunks]

    messages = [message for _, _, message in records(caplog)]
    assert messages[0] == "Starting Chunks"
    assert messages[-1].startswith("Completed Chunks items=")


async def test_the_pool_and_the_loop_are_logged(
    api: LocalAPI,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = AiohttpSession(api, BearerAuth(TEST_TOKEN), BaseErrorResolver())

    await session.get_client()
    await session.close()

    messages = [message for _, _, message in records(caplog)]
    assert messages[0].startswith("AiohttpSession is bound to the event loop")
    assert messages[1:] == [
        f"Opened a connection pool for {api.base_url}",
        f"Closed the connection pool of {api.base_url}",
    ]


async def test_a_followed_redirect_is_logged(
    api: LocalAPI,
    recorder: Recorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder.reply(Reply(status=302, headers={"Location": "/v1/moved?x=1"}))
    recorder.default = Reply(body=b"moved")

    async with AiohttpSession(
        api,
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
        follow_redirects=True,
    ) as session:
        await session.send_request(Ask(prompt="hi", secret="s"))

    assert (
        "ai_api.session.aiohttp",
        "DEBUG",
        "Following a redirect",
    ) in records(caplog)
    # The server's path is not logged: it may name a user's data.
    assert "moved" not in caplog.text


async def test_a_retry_counts_the_notes_of_the_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The failed attempt is not raised, so the log says it had notes (a
    # failing close of an upload, say), but not what they say: a note or
    # the error's message may hold a file path.
    error = APIConnectionError("reset at /home/user/private.bin")
    error.add_note("Closing upload 'passport-scan.pdf' also failed")
    session = FakeTransport([error, BufferedResponse(200, {}, b"ok")])

    await session.send_request(Ask(prompt="hi", secret="s"))

    (warning,) = [m for _, level, m in records(caplog) if level == "WARNING"]
    assert warning == (
        "Retrying request after APIConnectionError: attempt=2, delay=0.00s, "
        "notes=1"
    )
    assert "passport-scan" not in caplog.text
    assert "private.bin" not in caplog.text


def test_a_retry_after_above_the_limit_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = RateLimitError(status=429, retry_after=120.0, transient=True)

    assert should_retry(error, max_retry_after=60) is False

    assert records(caplog) == [
        (
            "ai_api.session.retry",
            "DEBUG",
            "Not retrying RateLimitError: the API asks to wait 120.0 s, "
            "more than max_retry_after (60 s)",
        )
    ]


async def test_the_key_from_the_environment_is_logged_by_name(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    async with OpenAIClient():
        pass

    assert (
        "ai_api.client.validators",
        "DEBUG",
        "Using the API key from OPENAI_API_KEY",
    ) in records(caplog)
    assert "sk-from-env" not in caplog.text


def test_an_unknown_type_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    discriminate = by_type({"known": "known"})

    assert discriminate({"type": "known"}) == "known"
    assert discriminate({"type": "brand_new"}) == UNKNOWN
    assert discriminate({}) == UNKNOWN

    assert records(caplog) == [
        (
            "ai_api.types.unknown",
            "DEBUG",
            "Unknown type 'brand_new' is kept as an unknown item",
        )
    ]


async def test_nothing_is_printed_without_logging_set_up(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The library adds only a NullHandler: without the application's
    # handlers, not even a warning reaches stderr through lastResort.
    logger = logging.getLogger("ai_api.session.base")
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()

    try:
        logger.warning("Retrying attempt=2")
    finally:
        root.handlers[:] = saved

    assert capsys.readouterr().err == ""
    assert any(
        isinstance(handler, logging.NullHandler)
        for handler in logging.getLogger("ai_api").handlers
    )


class BrokenWriter:
    """A response whose body writer failed after the answer came."""

    async def wait_for_close(self) -> None:
        raise OSError("writer failed on /home/user/private.bin")


async def test_a_writer_failure_after_the_answer_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # aiohttp drops it: the response is already there. It is not raised,
    # but it is not lost either: its type is logged.
    await wait_for_writer(cast("ClientResponse", BrokenWriter()))

    assert records(caplog) == [
        (
            "ai_api.response.aiohttp",
            "DEBUG",
            "The request body writer failed after the response came: OSError",
        )
    ]
    # Not its message or traceback, which may hold a file path.
    (record,) = caplog.records
    assert record.exc_info is None
    assert "private.bin" not in caplog.text
