"""Block J: ``BaseSession`` logic on the fake transport."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from tenacity import RetryCallState

from ai_api.client.exceptions.errror_resolver.base import BaseErrorResolver
from ai_api.client.exceptions.errror_resolver.protocols import (
    IErrorResolver,
)
from ai_api.client.exceptions.http import (
    APIError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
)
from ai_api.client.exceptions.transport import (
    APIConnectionError,
    APIResponseError,
    APITLSError,
)
from ai_api.client.response.base import BufferedResponse
from ai_api.client.response.protocols import IStreamResponse
from ai_api.client.serialization.json import JsonLoads, loads
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.methods.enums import ResponseFormat
from ai_api.methods.protocols import IAPIMethod, IStreamMethod
from ai_api.types.base import BaseAPIType
from ai_api.types.files import StreamFile
from tests.conftest import FakeStreamResponse, FakeTransport, make_method


class Model(BaseAPIType):
    value: int


def ok(body: bytes = b"ok", **headers: str) -> BufferedResponse:
    return BufferedResponse(200, headers, body)


def failed(
    status: int,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> BufferedResponse:
    return BufferedResponse(status, headers or {}, body)


def buffered_method(
    argument: Any = bytes,
    *,
    retryable: bool = False,
    **class_vars: Any,
) -> IAPIMethod[Any]:
    method_cls = make_method(
        BaseAPIMethod[argument],
        __retryable__=retryable,
        **class_vars,
    )
    return method_cls()


def stream_method(
    argument: Any = bytes,
    *,
    retryable: bool = False,
    **class_vars: Any,
) -> IStreamMethod[Any]:
    method_cls = make_method(
        BaseStreamMethod[argument],
        http_method="GET",
        __retryable__=retryable,
        **class_vars,
    )
    return method_cls()


class RaisingParser:
    """Response parser that fails with the given error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def parse(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error


class TrackingStreamParser:
    """Stream parser whose generator records how it ended."""

    def __init__(self) -> None:
        self.json_loads: JsonLoads | None = None
        self.error_resolver: IErrorResolver | None = None
        self.closed = False

    def parse(
        self,
        method: IStreamMethod[Any],
        response: IStreamResponse,
        json_loads: JsonLoads | None = None,
        error_resolver: IErrorResolver | None = None,
    ) -> AsyncIterator[Any]:
        self.json_loads = json_loads
        self.error_resolver = error_resolver
        return self._items(response)

    async def _items(self, response: IStreamResponse) -> AsyncIterator[Any]:
        try:
            async for chunk in response.iter_chunks():
                yield chunk
        finally:
            self.closed = True


class HangingStreamResponse(FakeStreamResponse):
    """Yields its chunks, then waits forever for more."""

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

        await asyncio.Event().wait()


async def test_successful_response_is_parsed() -> None:
    transport = FakeTransport([ok(b'{"value": 3}')])

    result = await transport.send_request(buffered_method(Model))

    assert result == Model(value=3)


async def test_parser_receives_session_json_loads() -> None:
    custom = MagicMock(wraps=loads)
    transport = FakeTransport([ok(b'{"value": 3}')], json_loads=custom)

    assert await transport.send_request(buffered_method(Model)) == Model(
        value=3
    )
    custom.assert_called_once_with(b'{"value": 3}')


async def test_error_status_is_resolved() -> None:
    body = b'{"error": {"code": "missing", "message": "No such thing"}}'
    transport = FakeTransport([failed(404, body, {"x-request-id": "req-1"})])

    with pytest.raises(NotFoundError) as info:
        await transport.send_request(buffered_method())

    assert info.value.status == 404
    assert info.value.code == "missing"
    assert info.value.request_id == "req-1"


@pytest.mark.parametrize(
    ("body", "json_loads"),
    [
        (b"<html>bad gateway</html>", loads),
        (b"{}", MagicMock(side_effect=TypeError("not decodable"))),
    ],
    ids=["not-json", "decoder-type-error"],
)
async def test_undecodable_error_body_reaches_resolver_as_none(
    body: bytes,
    json_loads: JsonLoads,
) -> None:
    resolver = MagicMock()
    resolver.resolve.return_value = APIError(status=502)
    transport = FakeTransport(
        [failed(502, body, {"a": "b"})],
        error_resolver=resolver,
        json_loads=json_loads,
    )

    with pytest.raises(APIError):
        await transport.send_request(buffered_method())

    resolver.resolve.assert_called_once_with(
        status=502,
        body=None,
        headers={"a": "b"},
    )


@pytest.mark.parametrize(
    "error",
    [ValueError("bad value"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "x")],
    ids=["value-error", "unicode-error"],
)
async def test_parser_errors_become_response_errors(error: Exception) -> None:
    method = buffered_method(__response_parser__=RaisingParser(error))
    transport = FakeTransport([ok(b"body")])

    with pytest.raises(APIResponseError, match="Failed to parse") as info:
        await transport.send_request(method)

    assert info.value.status == 200
    assert info.value.__cause__ is error


async def test_other_parser_errors_propagate() -> None:
    error = TypeError("bug")
    method = buffered_method(__response_parser__=RaisingParser(error))
    transport = FakeTransport([ok()])

    with pytest.raises(TypeError, match="bug"):
        await transport.send_request(method)


async def test_non_retryable_method_makes_one_attempt() -> None:
    transport = FakeTransport([failed(503), ok()])

    with pytest.raises(ServiceUnavailableError):
        await transport.send_request(buffered_method())

    assert transport.attempts == 1


async def test_transient_errors_are_retried_until_success(
    no_retry_wait: MagicMock,
) -> None:
    transport = FakeTransport([failed(503), failed(503), ok(b"done")])

    assert await transport.send_request(buffered_method(retryable=True)) == (
        b"done"
    )
    assert transport.attempts == 3
    assert no_retry_wait.call_count == 2


async def test_retry_that_would_end_after_the_deadline_is_not_started(
    no_retry_wait: MagicMock,
) -> None:
    def long_wait(_: RetryCallState) -> float:
        return 30.0

    no_retry_wait.side_effect = long_wait
    transport = FakeTransport([failed(503), ok()], retry_deadline=10)

    with pytest.raises(ServiceUnavailableError):
        await transport.send_request(buffered_method(retryable=True))

    assert transport.attempts == 1


async def test_retry_deadline_can_be_disabled(
    no_retry_wait: MagicMock,
) -> None:
    transport = FakeTransport([failed(503)] * 4, retry_deadline=None)

    with pytest.raises(ServiceUnavailableError):
        await transport.send_request(buffered_method(retryable=True))

    assert transport.attempts == 4
    assert transport.retry_deadline is None


def test_retry_deadline_defaults_to_15_minutes() -> None:
    assert FakeTransport().retry_deadline == 900


async def test_retries_stop_after_four_attempts() -> None:
    transport = FakeTransport([failed(503)] * 4 + [ok()])

    with pytest.raises(ServiceUnavailableError):
        await transport.send_request(buffered_method(retryable=True))

    assert transport.attempts == 4
    assert len(transport.responses) == 1


@pytest.mark.parametrize(
    ("status", "error_cls"),
    [(404, NotFoundError), (429, RateLimitError)],
    ids=["404", "429"],
)
async def test_permanent_errors_are_not_retried(
    status: int,
    error_cls: type[APIError],
) -> None:
    transport = FakeTransport([failed(status), ok()])

    with pytest.raises(error_cls):
        await transport.send_request(buffered_method(retryable=True))

    assert transport.attempts == 1


async def test_transient_transport_error_is_retried() -> None:
    transport = FakeTransport([APIConnectionError(), ok(b"done")])

    assert await transport.send_request(buffered_method(retryable=True)) == (
        b"done"
    )
    assert transport.attempts == 2


async def test_tls_error_is_not_retried() -> None:
    transport = FakeTransport([APITLSError(), ok()])

    with pytest.raises(APITLSError):
        await transport.send_request(buffered_method(retryable=True))

    assert transport.attempts == 1


async def test_response_errors_are_not_retried() -> None:
    method = buffered_method(
        Model,
        retryable=True,
    )
    transport = FakeTransport([ok(b"not json"), ok(b'{"value": 1}')])

    with pytest.raises(APIResponseError):
        await transport.send_request(method)

    assert transport.attempts == 1


async def test_wait_is_computed_from_previous_error(
    no_retry_wait: MagicMock,
) -> None:
    transport = FakeTransport(
        [
            failed(503, headers={"Retry-After": "7"}),
            failed(502),
            ok(),
        ]
    )

    await transport.send_request(buffered_method(retryable=True))

    first, second = no_retry_wait.outcomes
    assert isinstance(first, ServiceUnavailableError)
    assert first.retry_after == 7.0
    assert isinstance(second, APIError)
    assert second.status == 502
    assert second.retry_after is None


async def test_one_shot_file_disables_retries() -> None:
    async def produce() -> AsyncIterator[bytes]:
        yield b"x"

    method_cls = make_method(
        BaseAPIMethod[bytes],
        fields={"file": (Any, None)},
        __retryable__=True,
    )
    upload = method_cls.model_validate({"file": StreamFile(produce())})
    transport = FakeTransport([failed(503), ok()])

    with pytest.raises(ServiceUnavailableError):
        await transport.send_request(upload)

    assert transport.attempts == 1


async def test_exhausted_retry_loop_is_an_internal_error(
    mocker: Any,
) -> None:
    # Defensive guard: tenacity either returns or raises with
    # reraise=True, so an empty loop means it was misconfigured.
    class NoAttempts:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __aiter__(self) -> NoAttempts:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    mocker.patch("ai_api.client.session.base.AsyncRetrying", NoAttempts)
    transport = FakeTransport([ok()])

    with pytest.raises(RuntimeError, match="Unexpected retry state"):
        await transport.send_request(buffered_method(retryable=True))


async def test_stream_yields_items_and_closes_response_once() -> None:
    response = FakeStreamResponse([b"a", b"b"])
    transport = FakeTransport(stream_responses=[response])

    async with transport.stream(stream_method()) as stream:
        items = [item async for item in stream]
        assert not response.closed

    assert items == [b"a", b"b"]
    assert response.aclose_calls == 1


async def test_stream_parser_receives_session_codecs() -> None:
    custom = MagicMock(wraps=loads)
    parser = TrackingStreamParser()
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse([b"x"])],
        json_loads=custom,
    )

    async with transport.stream(stream_method(__stream_parser__=parser)):
        pass

    assert parser.json_loads is custom
    assert parser.error_resolver is transport.error_resolver


async def test_stream_error_status_is_raised_on_enter() -> None:
    response = FakeStreamResponse(
        [b'{"error": {"code": "gone"}}'],
        status=404,
    )
    transport = FakeTransport(stream_responses=[response])
    entered = False

    with pytest.raises(NotFoundError) as info:
        async with transport.stream(stream_method()):
            entered = True

    assert not entered
    assert info.value.code == "gone"
    assert response.read_calls == 1
    assert response.aclose_calls == 1


async def test_stream_opening_is_retried_and_failures_are_closed() -> None:
    unavailable = FakeStreamResponse([b"busy"], status=503)
    success = FakeStreamResponse([b"data"])
    transport = FakeTransport(stream_responses=[unavailable, success])

    async with transport.stream(stream_method(retryable=True)) as stream:
        assert unavailable.aclose_calls == 1
        assert [item async for item in stream] == [b"data"]

    assert transport.attempts == 2
    assert transport.closed_responses == 2


async def test_stream_transport_error_on_open_is_retried() -> None:
    transport = FakeTransport(
        stream_responses=[APIConnectionError(), FakeStreamResponse([b"x"])]
    )

    async with transport.stream(stream_method(retryable=True)) as stream:
        assert [item async for item in stream] == [b"x"]

    assert transport.attempts == 2


async def test_failure_mid_stream_is_not_retried() -> None:
    response = FakeStreamResponse(
        [b"a", b"b"],
        fail_at=1,
        error=APIConnectionError(),
    )
    transport = FakeTransport(
        stream_responses=[response, FakeStreamResponse([b"again"])]
    )
    received: list[bytes] = []

    with pytest.raises(APIConnectionError):
        async with transport.stream(stream_method(retryable=True)) as stream:
            async for item in stream:
                received.append(item)  # noqa: PERF401 - fails mid-loop

    assert received == [b"a"]
    assert transport.attempts == 1
    assert response.aclose_calls == 1


async def test_break_closes_parser_and_response() -> None:
    parser = TrackingStreamParser()
    response = FakeStreamResponse([b"a", b"b", b"c"])
    transport = FakeTransport(stream_responses=[response])

    async with transport.stream(
        stream_method(__stream_parser__=parser)
    ) as stream:
        async for _ in stream:
            break

    assert parser.closed
    assert response.aclose_calls == 1


async def test_exception_in_block_closes_and_propagates() -> None:
    parser = TrackingStreamParser()
    response = FakeStreamResponse([b"a", b"b"])
    transport = FakeTransport(stream_responses=[response])
    error = LookupError("from the caller")

    with pytest.raises(LookupError) as info:
        async with transport.stream(
            stream_method(__stream_parser__=parser)
        ) as stream:
            async for _ in stream:
                raise error

    assert info.value is error
    assert parser.closed
    assert response.aclose_calls == 1


async def test_unsupported_format_still_closes_response() -> None:
    response = FakeStreamResponse([b"x"])
    transport = FakeTransport(stream_responses=[response])
    method = stream_method(__response_format__=ResponseFormat.JSON)

    with pytest.raises(ValueError, match="Unsupported stream"):
        async with transport.stream(method):
            pass  # pragma: no cover - the block must not run

    assert response.aclose_calls == 1


async def test_cancellation_while_iterating_closes_response() -> None:
    response = HangingStreamResponse([b"first"])
    transport = FakeTransport(stream_responses=[response])
    first = asyncio.Event()

    async def consume() -> None:
        async with transport.stream(stream_method()) as stream:
            async for _ in stream:
                first.set()

    task = asyncio.create_task(consume())
    await first.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert response.aclose_calls == 1


async def test_context_manager_closes_on_exit() -> None:
    transport = FakeTransport()

    async with transport as entered:
        assert entered is transport
        assert transport.close_calls == 0

    assert transport.close_calls == 1


async def test_context_manager_closes_on_exception() -> None:
    transport = FakeTransport()

    with pytest.raises(LookupError):
        async with transport:
            raise LookupError

    assert transport.close_calls == 1


class Delta(BaseAPIType):
    text: str


class TransientErrors(BaseErrorResolver):
    """Marks every error as transient."""

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIError], bool]:
        return APIError, True


def sse_method(**class_vars: Any) -> IStreamMethod[Any]:
    return stream_method(
        Delta,
        __response_format__=ResponseFormat.SSE,
        **class_vars,
    )


async def test_error_event_uses_session_resolver_and_is_not_retried(
    no_retry_wait: MagicMock,
) -> None:
    response = FakeStreamResponse(
        [b'data: {"text": "a"}\n\nevent: error\ndata: {}\n\n'],
        headers={"Content-Type": "text/event-stream"},
    )
    transport = FakeTransport(
        stream_responses=[response],
        error_resolver=TransientErrors(),
    )
    with pytest.raises(APIError) as info:
        async with transport.stream(sse_method(retryable=True)) as stream:
            items = aiter(stream)
            assert await anext(items) == Delta(text="a")
            await anext(items)

    assert info.value.transient is True
    assert transport.attempts == 1
    assert response.aclose_calls == 1


async def test_break_in_event_stream_closes_response() -> None:
    response = FakeStreamResponse(
        [b'data: {"text": "a"}\n\n', b'data: {"text": "b"}\n\n'],
        headers={"Content-Type": "text/event-stream"},
    )
    transport = FakeTransport(stream_responses=[response])

    async with transport.stream(sse_method()) as stream:
        async for _ in stream:
            break

    assert response.aclose_calls == 1
