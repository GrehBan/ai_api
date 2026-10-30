"""Shared fixtures and test doubles.

The doubles are importable from tests as ``tests.conftest``; see
``docs/testing-plan.md`` (stage 0) for what each stage uses.
"""

from __future__ import annotations

import asyncio
import datetime
import email.policy
from collections import deque
from collections.abc import (
    AsyncIterator,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.parser import BytesParser
from types import SimpleNamespace
from typing import Any, Self, cast
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict
from pytest_aiohttp import AiohttpServer
from pytest_mock import MockerFixture
from tenacity import RetryCallState
from yarl import URL

from ai_api.http.auth import BearerAuth
from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.headers import CONTENT_TYPE
from ai_api.http.protocols import IAPI, HeadersType, IAuth, Query
from ai_api.methods.base import BaseMethod
from ai_api.methods.protocols import IAPIMethod, IStreamMethod
from ai_api.response.buffered import BufferedResponse
from ai_api.session.aiohttp import AiohttpSession
from ai_api.session.base import BaseSession
from ai_api.session.protocols import Timeout

TEST_TOKEN = "test-token"

NOW = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
"""The current time while ``frozen_now`` is active."""


@pytest.fixture(autouse=True)
def no_retry_wait(mocker: MockerFixture) -> MagicMock:
    """Make retries instant.

    Replaces ``calculate_wait`` where ``BaseSession`` looks it up, so the
    retry loop waits 0 seconds. Tenacity reuses one mutable state object,
    so the mock's ``call_args`` only show the final state; the error each
    wait was computed for is kept in ``mock.outcomes`` instead.
    """
    outcomes: list[BaseException | None] = []

    def wait(state: RetryCallState) -> float:
        outcomes.append(
            None if state.outcome is None else state.outcome.exception()
        )
        return 0.0

    mock = mocker.patch(
        "ai_api.session.base.calculate_wait",
        autospec=True,
        side_effect=wait,
    )
    mock.outcomes = outcomes
    return mock


@pytest.fixture(autouse=True)
def no_close_delay(mocker: MockerFixture) -> None:
    """Drop the ``asyncio.sleep(0.25)`` from ``AiohttpSession.close()``.

    Only the ``asyncio`` name inside the session module is replaced, so
    sleeps elsewhere (server handlers, tests) keep their real duration.
    """
    real_sleep = asyncio.sleep

    async def instant_sleep(delay: float, result: Any = None) -> Any:
        return await real_sleep(0, result)

    mocker.patch(
        "ai_api.session.aiohttp.asyncio",
        SimpleNamespace(
            sleep=instant_sleep,
            get_running_loop=asyncio.get_running_loop,
        ),
    )


class _FrozenDatetimeMeta(type):
    """Lets real datetimes pass ``isinstance`` checks against the fake."""

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, datetime.datetime)


class _FrozenDatetime(datetime.datetime, metaclass=_FrozenDatetimeMeta):
    """``datetime`` whose ``now()`` always returns ``NOW``."""

    @classmethod
    def now(cls, tz: datetime.tzinfo | None = None) -> Self:
        moment = NOW if tz is None else NOW.astimezone(tz)
        return cls.combine(moment.date(), moment.timetz())


@pytest.fixture
def frozen_now(mocker: MockerFixture) -> datetime.datetime:
    """Fix the current time at ``NOW`` for the library's clock reads.

    Covers ``Retry-After`` dates in the error resolver and ``timedelta``
    values in ``BodyNormalizer``.

    Returns:
        ``NOW``.
    """
    mocker.patch(
        "ai_api.http.headers.datetime",
        _FrozenDatetime,
    )
    mocker.patch(
        "ai_api.serialization.normalize.datetime",
        SimpleNamespace(
            datetime=_FrozenDatetime,
            timedelta=datetime.timedelta,
            UTC=datetime.UTC,
        ),
    )
    return NOW


@dataclass
class LocalAPI:
    """``IAPI`` that builds URLs the way the README does."""

    base_url: URL
    version: str = "v1"

    def build_url(
        self,
        method: str,
        query: Query | None = None,
    ) -> str:
        # `method` is already percent-encoded: do not encode it again.
        url = self.base_url.joinpath(
            self.version,
            method.lstrip("/"),
            encoded=True,
        )
        return str(url.with_query(query or {}))


class FakeStreamResponse:
    """``IStreamResponse`` that yields a fixed list of chunks.

    Args:
        chunks: The body, chunk by chunk.
        status: HTTP status code.
        headers: Response headers.
        charset: Charset from the ``Content-Type`` header.
        fail_at: Index of the chunk at which ``error`` is raised instead of
            yielding it; ``None`` never fails.
        error: The exception raised at ``fail_at``.
    """

    def __init__(
        self,
        chunks: Iterable[bytes] = (),
        *,
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        charset: str | None = None,
        fail_at: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        if (fail_at is None) != (error is None):
            raise ValueError("fail_at and error go together")

        self.chunks = list(chunks)
        self._status = status
        self._headers = CIMultiDictProxy(CIMultiDict(headers or {}))
        self._charset = charset
        self.fail_at = fail_at
        self.error = error
        self.aclose_calls = 0
        self.read_calls = 0

    @property
    def status(self) -> int:
        return self._status

    @property
    def headers(self) -> Mapping[str, str]:
        return self._headers

    @property
    def charset(self) -> str | None:
        return self._charset

    @property
    def closed(self) -> bool:
        """Whether ``aclose()`` has been called at least once."""
        return self.aclose_calls > 0

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            if index == self.fail_at:
                assert self.error is not None
                raise self.error
            yield chunk

    async def read(self) -> bytes:
        self.read_calls += 1
        return b"".join(self.chunks)

    async def aclose(self) -> None:
        self.aclose_calls += 1


type Outcome[R] = R | BaseException
"""A queued transport result: returned as is, or raised if an exception."""


class FakeTransport(BaseSession):
    """``BaseSession`` whose transport replays queued outcomes.

    ``_execute`` pops from ``responses`` and ``_open_stream`` from
    ``stream_responses``; an exception in the queue is raised instead of
    returned. Every call is recorded, so tests can assert exact attempt
    counts and which stream responses were closed, and the headers and
    timeout of each attempt go to ``call_options``.

    Args:
        responses: Outcomes of ``_execute``, in order.
        stream_responses: Outcomes of ``_open_stream``, in order.
        api: URL builder; never used by the fake transport itself.
        auth: Authorization header source.
        error_resolver: Turns error responses into exceptions.
        **kwargs: Forwarded to ``BaseSession`` (headers, JSON codecs).
    """

    def __init__(
        self,
        responses: Iterable[Outcome[BufferedResponse]] = (),
        stream_responses: Iterable[Outcome[FakeStreamResponse]] = (),
        *,
        api: IAPI | None = None,
        auth: IAuth | None = None,
        error_resolver: IErrorResolver | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api or LocalAPI(URL("http://fake.invalid")),
            auth or BearerAuth(TEST_TOKEN),
            error_resolver or BaseErrorResolver(),
            **kwargs,
        )
        self.responses: deque[Outcome[BufferedResponse]] = deque(responses)
        self.stream_responses: deque[Outcome[FakeStreamResponse]] = deque(
            stream_responses
        )
        self.execute_calls: list[IAPIMethod[Any]] = []
        self.stream_calls: list[IStreamMethod[Any]] = []
        self.opened: list[FakeStreamResponse] = []
        self.call_options: list[tuple[HeadersType | None, Timeout | None]] = []
        self.close_calls = 0

    @property
    def attempts(self) -> int:
        """Number of ``_execute`` and ``_open_stream`` calls."""
        return len(self.execute_calls) + len(self.stream_calls)

    @property
    def closed_responses(self) -> int:
        """Number of opened stream responses that were closed."""
        return sum(response.closed for response in self.opened)

    async def close(self) -> None:
        self.close_calls += 1

    async def _execute(
        self,
        method: IAPIMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> BufferedResponse:
        self.execute_calls.append(method)
        self.call_options.append((headers, timeout))
        return self._next(self.responses, "_execute")

    async def _open_stream(
        self,
        method: IStreamMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> FakeStreamResponse:
        self.stream_calls.append(method)
        self.call_options.append((headers, timeout))
        response = self._next(self.stream_responses, "_open_stream")
        self.opened.append(response)
        return response

    @staticmethod
    def _next[R](queue: deque[Outcome[R]], name: str) -> R:
        if not queue:
            raise AssertionError(f"FakeTransport.{name}: queue is empty")

        outcome = queue.popleft()

        if isinstance(outcome, BaseException):
            raise outcome

        return outcome


@pytest.fixture
def fake_transport() -> FakeTransport:
    """A ``FakeTransport`` with empty queues; fill them in the test."""
    return FakeTransport()


def make_method[M: BaseMethod[Any]](
    base: type[M],
    *,
    path: str | None = "/test",
    fields: Mapping[str, Any] | None = None,
    name: str = "Method",
    http_method: str = "POST",
    **class_vars: Any,
) -> type[M]:
    """Declare a method class with ``type(...)``.

    Meant for parametrized class-definition checks; wrap the call in
    ``pytest.raises`` when the definition must fail. Type checkers only
    see ``base``, so create instances with dynamic fields through
    ``model_validate({...})``.

    Args:
        base: Base class, e.g. ``BaseAPIMethod[str]``.
        path: ``__request_path__``; ``None`` inherits it from ``base``.
        fields: Field annotations; a ``(type, default)`` tuple also sets
            the default.
        name: Class name, shown in error messages.
        http_method: ``__request_method__``.
        **class_vars: Other class attributes, e.g.
            ``__request_query__={"q"}``.

    Returns:
        The new method class.
    """
    annotations: dict[str, Any] = {}
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": name,
        "__request_method__": http_method,
        **class_vars,
    }

    if path is not None:
        namespace["__request_path__"] = path

    for field_name, spec in (fields or {}).items():
        if isinstance(spec, tuple):
            annotations[field_name], namespace[field_name] = spec
        else:
            annotations[field_name] = spec

    namespace["__annotations__"] = annotations
    return cast("type[M]", type(name, (base,), namespace))


@dataclass
class RecordedPart:
    """One multipart part as the server received it."""

    name: str | None
    filename: str | None
    content_type: str | None
    data: bytes


@dataclass
class RecordedRequest:
    """A request as the server received it.

    ``raw_path`` is the path exactly as sent, still percent-encoded and
    without the query string. Multipart bodies are split into ``parts``
    and leave ``body`` empty.
    """

    method: str
    raw_path: str
    query_string: str
    query: MultiDict[str]
    headers: CIMultiDictProxy[str]
    body: bytes
    parts: list[RecordedPart]


type Chunk = bytes | asyncio.Event
"""A streamed chunk, or an event the handler waits on before going on."""


@dataclass
class Reply:
    """What the recorder answers to one request.

    Args:
        status: HTTP status code.
        body: Whole body, sent with ``Content-Length``.
        json: Body serialized as JSON; overrides ``body``.
        headers: Extra response headers.
        content_type: ``Content-Type``, charset included.
        chunks: Streamed body (chunked encoding); an ``asyncio.Event``
            item makes the handler wait for it. May be infinite.
        delay: Seconds to wait before answering.
        chunk_delay: Seconds to wait before each chunk.
        content_length: Declared ``Content-Length`` for ``chunks``; with
            ``abort`` it makes the body end before it is complete.
        abort: Drop the connection after the chunks are sent.
    """

    status: int = 200
    body: bytes = b""
    json: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)
    content_type: str | None = None
    chunks: Iterable[Chunk] | None = None
    delay: float = 0.0
    chunk_delay: float = 0.0
    content_length: int | None = None
    abort: bool = False


def _parse_multipart(content_type: str, body: bytes) -> list[RecordedPart]:
    """Split a multipart body into its parts.

    Args:
        content_type: ``Content-Type`` of the request, with the boundary.
        body: The whole request body.

    Returns:
        The parts, in order.
    """
    message = BytesParser(policy=email.policy.HTTP).parsebytes(
        f"{CONTENT_TYPE}: {content_type}\r\n\r\n".encode() + body
    )
    assert isinstance(message, EmailMessage)
    assert message.is_multipart()

    parts: list[RecordedPart] = []

    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True)
        assert isinstance(name, str | None)
        assert isinstance(data, bytes)
        parts.append(
            RecordedPart(
                name=name,
                filename=part.get_filename(),
                content_type=part.get(CONTENT_TYPE),
                data=data,
            )
        )

    return parts


class Recorder:
    """aiohttp app that records requests and answers from a script.

    Replies are taken from ``replies`` in order; ``default`` answers once
    the script is exhausted. Every path and HTTP method is routed here.
    """

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self.replies: deque[Reply] = deque()
        self.default = Reply()
        self.app = web.Application()
        self.app.router.add_route("*", "/{tail:.*}", self._handle)

    def reply(self, *replies: Reply) -> None:
        """Queue replies for the next requests."""
        self.replies.extend(replies)

    @property
    def last(self) -> RecordedRequest:
        """The most recent request."""
        assert self.requests, "no request was received"
        return self.requests[-1]

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(await self._record(request))
        reply = self.replies.popleft() if self.replies else self.default

        if reply.delay:
            await asyncio.sleep(reply.delay)

        if reply.chunks is None:
            return self._buffered(reply)

        return await self._streamed(request, reply)

    @staticmethod
    async def _record(request: web.Request) -> RecordedRequest:
        parts: list[RecordedPart] = []
        body = b""

        if request.content_type.startswith("multipart/"):
            # Parsed from the whole body: aiohttp's streaming multipart
            # reader calls the deprecated ``unread_data`` for parts without
            # a length, which fails the test under ``-W error`` whenever a
            # boundary falls inside a received chunk.
            parts = _parse_multipart(
                request.headers[CONTENT_TYPE],
                await request.read(),
            )
        else:
            body = await request.read()

        return RecordedRequest(
            method=request.method,
            raw_path=request.raw_path.partition("?")[0],
            query_string=request.query_string,
            query=MultiDict(request.query),
            headers=request.headers,
            body=body,
            parts=parts,
        )

    @staticmethod
    def _buffered(reply: Reply) -> web.Response:
        if reply.json is not None:
            return web.json_response(
                reply.json,
                status=reply.status,
                headers=reply.headers,
            )

        response = web.Response(
            status=reply.status,
            body=reply.body,
            headers=reply.headers,
        )

        if reply.content_type is not None:
            response.headers["Content-Type"] = reply.content_type

        return response

    @staticmethod
    async def _streamed(
        request: web.Request,
        reply: Reply,
    ) -> web.StreamResponse:
        assert reply.chunks is not None
        response = web.StreamResponse(
            status=reply.status,
            headers=reply.headers,
        )

        if reply.content_type is not None:
            response.headers["Content-Type"] = reply.content_type

        if reply.content_length is not None:
            response.content_length = reply.content_length
        else:
            response.enable_chunked_encoding()

        await response.prepare(request)

        for chunk in reply.chunks:
            if isinstance(chunk, asyncio.Event):
                await chunk.wait()
                continue

            if reply.chunk_delay:
                await asyncio.sleep(reply.chunk_delay)

            await response.write(chunk)

        if reply.abort:
            assert request.transport is not None
            request.transport.close()
            return response

        await response.write_eof()
        return response


@pytest.fixture
def recorder() -> Recorder:
    """A fresh request recorder; ``api`` serves it."""
    return Recorder()


@pytest.fixture
async def server(
    aiohttp_server: AiohttpServer,
    recorder: Recorder,
) -> TestServer:
    """The recorder running on a local port."""
    return await aiohttp_server(recorder.app)


@pytest.fixture
def api(server: TestServer) -> LocalAPI:
    """``IAPI`` pointing at the local server."""
    return LocalAPI(server.make_url("/"))


@pytest.fixture
async def session(api: LocalAPI) -> AsyncIterator[AiohttpSession]:
    """``AiohttpSession`` against the local server, closed on teardown."""
    session = AiohttpSession(
        api,
        BearerAuth(TEST_TOKEN),
        BaseErrorResolver(),
    )

    try:
        yield session
    finally:
        await session.close()
