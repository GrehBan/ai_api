"""SSE stages 2-3: SSE and NDJSON in the stream parser."""

import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Annotated, Any, Literal

import pytest
from pydantic import Field, ValidationError

from ai_api.client.exceptions.error_resolver.base import BaseErrorResolver
from ai_api.client.exceptions.http import APIError
from ai_api.client.exceptions.transport import APIResponseError
from ai_api.client.response.parse.sse import ServerSentEvent
from ai_api.client.response.parse.stream import StreamResponseParser
from ai_api.client.serialization.json import JsonLoads
from ai_api.methods.base import BaseStreamMethod
from ai_api.methods.enums import ResponseFormat
from ai_api.types.base import BaseAPIType
from tests.conftest import FakeStreamResponse, make_method


class Start(BaseAPIType):
    type: Literal["start"]
    id: str


class Delta(BaseAPIType):
    type: Literal["delta"]
    text: str


class Tagged(BaseAPIType):
    tag: Annotated[uuid.UUID, Field(strict=True)]


type Event = Annotated[Start | Delta, Field(discriminator="type")]

TAG = uuid.UUID(int=1)


class OverloadedIsTransient(BaseErrorResolver):
    """Treats ``overloaded_error`` events as transient."""

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIError], bool]:
        if error_type == "overloaded_error":
            return APIError, True

        return super().classify(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )


def parse(
    item: Any,
    response_format: ResponseFormat | None,
    chunks: list[bytes],
    *,
    headers: Mapping[str, str] | None = None,
    json_loads: JsonLoads | None = None,
    error_resolver: BaseErrorResolver | None = None,
) -> AsyncIterator[Any]:
    class_vars: dict[str, Any] = {}

    if response_format is not None:
        class_vars["__response_format__"] = response_format

    method = make_method(BaseStreamMethod[item], **class_vars)()
    return StreamResponseParser().parse(
        method,
        FakeStreamResponse(chunks, headers={**SSE_HEADERS, **(headers or {})}),
        json_loads,
        error_resolver,
    )


SSE_HEADERS = {"Content-Type": "text/event-stream"}


async def collect(items: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in items]


def sse(*events: str) -> bytes:
    return "".join(f"{event}\n\n" for event in events).encode()


def ndjson(*items: Any) -> bytes:
    return "".join(f"{json.dumps(item)}\n" for item in items).encode()


START = {"type": "start", "id": "m-1"}
DELTA = {"type": "delta", "text": "Hi 😀"}


async def test_sse_events_are_validated_into_the_union() -> None:
    body = sse(
        f"event: start\ndata: {json.dumps(START)}",
        f"event: delta\ndata: {json.dumps(DELTA)}",
    )

    items = await collect(parse(Event, ResponseFormat.SSE, [body]))

    assert items == [
        Start(type="start", id="m-1"),
        Delta(type="delta", text="Hi 😀"),
    ]


async def test_sse_survives_one_byte_chunks() -> None:
    body = sse(f"data: {json.dumps(DELTA)}")
    chunks = [body[i : i + 1] for i in range(len(body))]

    items = await collect(parse(Delta, ResponseFormat.SSE, chunks))

    assert items == [Delta(type="delta", text="Hi 😀")]


async def test_server_sent_event_items_are_raw() -> None:
    body = sse("event: ping\ndata: not json", "id: 1\ndata: {}")

    items = await collect(parse(ServerSentEvent, None, [body]))

    assert items == [
        ServerSentEvent(data="not json", event="ping"),
        ServerSentEvent(data="{}", id="1"),
    ]


@pytest.mark.parametrize(
    "item",
    [Delta, ServerSentEvent],
    ids=["validated", "raw"],
)
async def test_done_ends_the_stream(item: Any) -> None:
    body = sse(f"data: {json.dumps(DELTA)}", "data: [DONE]", "data: late")

    items = await collect(parse(item, ResponseFormat.SSE, [body]))

    assert len(items) == 1


async def test_done_with_another_event_type_still_ends_the_stream() -> None:
    body = sse("event: end\ndata: [DONE]", f"data: {json.dumps(DELTA)}")

    assert await collect(parse(Delta, ResponseFormat.SSE, [body])) == []


async def test_comments_and_keep_alives_are_skipped() -> None:
    body = sse(": ping", "event: ping", f"data: {json.dumps(DELTA)}")

    items = await collect(parse(Delta, ResponseFormat.SSE, [body]))

    assert items == [Delta(type="delta", text="Hi 😀")]


async def test_unknown_event_is_an_error() -> None:
    body = sse('event: ping\ndata: {"type": "ping"}')

    with pytest.raises(APIResponseError) as info:
        await collect(parse(Event, ResponseFormat.SSE, [body]))

    assert info.value.status == 200
    assert isinstance(info.value.__cause__, ValidationError)


@pytest.mark.parametrize(
    "response_format",
    [ResponseFormat.SSE, ResponseFormat.NDJSON],
    ids=["sse", "ndjson"],
)
async def test_invalid_json_is_a_response_error(
    response_format: ResponseFormat,
) -> None:
    body = (
        b"data: {oops\n\n"
        if response_format is ResponseFormat.SSE
        else b"{oops\n"
    )

    with pytest.raises(APIResponseError, match="Failed to validate"):
        await collect(parse(Delta, response_format, [body]))


async def test_items_before_a_bad_event_are_delivered() -> None:
    body = sse(f"data: {json.dumps(DELTA)}", "data: {}")
    items = parse(Delta, ResponseFormat.SSE, [body])

    assert await anext(items) == Delta(type="delta", text="Hi 😀")

    with pytest.raises(APIResponseError):
        await anext(items)


async def test_sse_error_event_goes_through_the_resolver() -> None:
    error = {
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    }
    body = sse(
        f"data: {json.dumps(DELTA)}",
        f"event: error\ndata: {json.dumps(error)}",
    )
    items = parse(
        Delta,
        ResponseFormat.SSE,
        [body],
        headers={"request-id": "req-1"},
        error_resolver=OverloadedIsTransient(),
    )

    assert await anext(items) == Delta(type="delta", text="Hi 😀")

    with pytest.raises(APIError) as info:
        await anext(items)

    assert type(info.value) is APIError
    assert info.value.status == 200
    assert info.value.error_type == "overloaded_error"
    assert info.value.message == "detail: Overloaded"
    assert info.value.request_id == "req-1"
    assert info.value.transient is True


async def test_error_event_with_raw_items_is_raised_too() -> None:
    body = sse('event: error\ndata: {"error": {"code": "boom"}}')

    with pytest.raises(APIError) as info:
        await collect(parse(ServerSentEvent, None, [body]))

    assert info.value.code == "boom"


async def test_error_event_with_invalid_json_has_no_details() -> None:
    body = sse("event: error\ndata: Internal error")

    with pytest.raises(APIError) as info:
        await collect(parse(Delta, ResponseFormat.SSE, [body]))

    assert info.value.code is None
    assert info.value.message == ""


@pytest.mark.parametrize(
    ("response_format", "body"),
    [
        (
            ResponseFormat.SSE,
            sse('data: {"error": {"code": "rate_limit", "message": "Slow"}}'),
        ),
        (
            ResponseFormat.NDJSON,
            ndjson({"error": {"code": "rate_limit", "message": "Slow"}}),
        ),
    ],
    ids=["sse", "ndjson"],
)
async def test_top_level_error_object_is_raised(
    response_format: ResponseFormat,
    body: bytes,
) -> None:
    with pytest.raises(APIError) as info:
        await collect(parse(Delta, response_format, [body]))

    assert info.value.code == "rate_limit"
    assert info.value.message == "detail: Slow"
    assert info.value.transient is False


async def test_error_as_a_string_is_raised() -> None:
    with pytest.raises(APIError) as info:
        await collect(
            parse(
                dict[str, Any], ResponseFormat.NDJSON, [ndjson({"error": "x"})]
            )
        )

    assert info.value.message == "detail: x"


@pytest.mark.parametrize(
    "item",
    [
        {"result": {"error": None}},
        {"text": '"error"'},
        ["error"],
        {"id": "batch_1", "error": None},
        {"error": ""},
        {"error": {}},
        {"error": False},
        {"type": "tool_result", "error": "file not found"},
        {"type": "x", "error": True},
    ],
    ids=[
        "nested-key",
        "in-a-string",
        "not-an-object",
        "null",
        "empty-string",
        "empty-object",
        "false",
        "typed-event-string",
        "typed-event-true",
    ],
)
async def test_error_word_elsewhere_is_data(item: Any) -> None:
    items = await collect(parse(Any, ResponseFormat.NDJSON, [ndjson(item)]))

    assert items == [item]


@pytest.mark.parametrize(
    "item",
    [
        {"error": True},
        {"type": "error", "message": "boom"},
        {"type": "error", "error": {"message": "boom"}},
    ],
    ids=["true-without-type", "error-type", "error-type-with-object"],
)
async def test_error_events_are_raised(item: Any) -> None:
    with pytest.raises(APIError):
        await collect(parse(Any, ResponseFormat.NDJSON, [ndjson(item)]))


async def test_is_error_event_can_be_overridden() -> None:
    class StatusParser(StreamResponseParser):
        def is_error_event(self, body: Any) -> bool:
            return isinstance(body, dict) and body.get("status") == "error"

    method = make_method(
        BaseStreamMethod[dict[str, Any]],
        __response_format__=ResponseFormat.NDJSON,
    )()
    body = ndjson({"error": "kept"}) + ndjson({"status": "error"})
    items = StatusParser().parse(method, FakeStreamResponse([body]))

    assert await anext(items) == {"error": "kept"}

    with pytest.raises(APIError):
        await anext(items)


@pytest.mark.parametrize(
    "content_type",
    ["text/html; charset=utf-8", ""],
    ids=["html", "missing"],
)
async def test_sse_needs_an_event_stream(content_type: str) -> None:
    body = [b"<html>Proxy login required</html>"]

    with pytest.raises(APIResponseError, match="Expected an event stream"):
        await collect(
            parse(
                Delta,
                ResponseFormat.SSE,
                body,
                headers={"Content-Type": content_type},
            )
        )


async def test_event_stream_check_can_be_turned_off() -> None:
    method = make_method(
        BaseStreamMethod[Delta],
        __response_format__=ResponseFormat.SSE,
    )()
    parser = StreamResponseParser(require_event_stream=False)
    body = sse(f"data: {json.dumps(DELTA)}")
    response = FakeStreamResponse(
        [body], headers={"Content-Type": "text/plain"}
    )

    assert await collect(parser.parse(method, response)) == [
        Delta.model_validate(DELTA)
    ]


async def test_event_over_the_limit_is_a_response_error() -> None:
    method = make_method(BaseStreamMethod[ServerSentEvent])()
    parser = StreamResponseParser(max_event_size=100)
    body = b"data: " + b"x" * 80 + b"\n" + b"data: " + b"x" * 80 + b"\n"
    response = FakeStreamResponse([body], headers=SSE_HEADERS)

    with pytest.raises(APIResponseError, match="Event exceeds 100"):
        await collect(parser.parse(method, response))


@pytest.mark.parametrize("charset", ["base64", "zlib_codec", "rot13"])
async def test_text_stream_rejects_non_text_charsets(charset: str) -> None:
    method = make_method(BaseStreamMethod[str])()
    response = FakeStreamResponse([b"aGVsbG8="], charset=charset)

    with pytest.raises(APIResponseError, match="Unknown response charset"):
        await collect(StreamResponseParser().parse(method, response))


async def test_ndjson_lines() -> None:
    body = (
        b'{"type": "start", "id": "m-1"}\r\n\n  \n'
        + json.dumps(DELTA).encode()
    )
    chunks = [body[:5], body[5:40], body[40:]]

    items = await collect(parse(Event, ResponseFormat.NDJSON, chunks))

    assert items == [
        Start(type="start", id="m-1"),
        Delta(type="delta", text="Hi 😀"),
    ]


async def test_ndjson_without_item_type_yields_decoded_json() -> None:
    method = make_method(
        BaseStreamMethod,
        __response_format__=ResponseFormat.NDJSON,
    )()
    response = FakeStreamResponse([ndjson({"a": 1}, [2])])

    items = await collect(StreamResponseParser().parse(method, response))

    assert items == [{"a": 1}, [2]]


async def test_default_decoder_validates_in_json_mode() -> None:
    body = ndjson({"tag": str(TAG)})

    items = await collect(parse(Tagged, ResponseFormat.NDJSON, [body]))

    assert items == [Tagged(tag=TAG)]


async def test_custom_decoder_validates_in_python_mode() -> None:
    body = ndjson({"tag": str(TAG)})
    calls: list[str | bytes] = []

    def custom(data: str | bytes) -> Any:
        calls.append(data)
        return json.loads(data)

    with pytest.raises(APIResponseError):
        await collect(
            parse(Tagged, ResponseFormat.NDJSON, [body], json_loads=custom)
        )

    assert calls == [body.decode().strip()]


async def test_custom_decoder_errors_are_response_errors() -> None:
    def failing(data: str | bytes) -> Any:
        raise ValueError("bad")

    with pytest.raises(APIResponseError) as info:
        await collect(
            parse(Delta, ResponseFormat.NDJSON, [b"{}\n"], json_loads=failing)
        )

    assert str(info.value.__cause__) == "bad"


@pytest.mark.parametrize(
    "response_format",
    [ResponseFormat.SSE, ResponseFormat.NDJSON],
    ids=["sse", "ndjson"],
)
async def test_invalid_utf8_is_a_response_error(
    response_format: ResponseFormat,
) -> None:
    with pytest.raises(APIResponseError, match="decode event stream") as info:
        await collect(parse(Delta, response_format, [b"data: \xff\n\n"]))

    assert isinstance(info.value.__cause__, UnicodeError)


async def test_parser_uses_its_own_resolver_by_default() -> None:
    method = make_method(
        BaseStreamMethod[Delta],
        __response_format__=ResponseFormat.SSE,
    )()
    parser = StreamResponseParser(error_resolver=OverloadedIsTransient())
    body = sse('event: error\ndata: {"error": {"type": "overloaded_error"}}')

    with pytest.raises(APIError) as info:
        await collect(
            parser.parse(
                method, FakeStreamResponse([body], headers=SSE_HEADERS)
            )
        )

    assert info.value.transient is True


@pytest.mark.parametrize(
    "response_format",
    [ResponseFormat.SSE, ResponseFormat.NDJSON],
    ids=["sse", "ndjson"],
)
async def test_line_over_the_limit_is_a_response_error(
    response_format: ResponseFormat,
) -> None:
    method = make_method(
        BaseStreamMethod[dict[str, Any]],
        __response_format__=response_format,
    )()
    body = b"x" * 40
    parser = StreamResponseParser(max_line_size=20)
    response = FakeStreamResponse([body[:20], body[20:]], headers=SSE_HEADERS)
    items = parser.parse(method, response)

    with pytest.raises(APIResponseError, match="exceeds 20") as info:
        await collect(items)

    assert info.value.status == 200


async def test_closing_early_stops_reading() -> None:
    body = sse(*[f"data: {json.dumps(DELTA)}"] * 3)
    response = FakeStreamResponse([body[:30], body[30:]], headers=SSE_HEADERS)
    method = make_method(
        BaseStreamMethod[Delta],
        __response_format__=ResponseFormat.SSE,
    )()
    items = StreamResponseParser().parse(method, response)

    assert isinstance(items, AsyncGenerator)
    await anext(items)
    await items.aclose()

    with pytest.raises(StopAsyncIteration):
        await anext(items)
