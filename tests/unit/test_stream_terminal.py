"""S1-S5: the terminal event of SSE and NDJSON streams.

By the HTTP and SSE standards the end of a body is a normal end, so a
stream cut between two events looks complete. A method that declares
``__stream_terminal__`` has the parser check it: the body must end after
it, and nothing may follow it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Any, Literal

import pytest
from pydantic import Field

from ai_api.enums import ResponseFormat
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import APIStatusError
from ai_api.http.exceptions.transport import APIStreamInterruptedError
from ai_api.methods.base import BaseStreamMethod
from ai_api.response.parse.sse import ServerSentEvent
from ai_api.response.parse.stream import StreamResponseParser
from ai_api.serialization.json import loads
from ai_api.types.base import BaseAPIType
from ai_api.types.matchers import DONE, BaseMatcher, EventTypes
from ai_api.types.protocols import IEventMatcher
from tests.conftest import FakeStreamResponse, make_method


class Delta(BaseAPIType):
    type: Literal["delta"]
    text: str


class Stop(BaseAPIType):
    type: Literal["stop"]


class Failed(BaseAPIType):
    type: Literal["failed"]
    reason: str


type Event = Annotated[Delta | Stop | Failed, Field(discriminator="type")]

SSE_HEADERS = {"Content-Type": "text/event-stream"}
DELTA = {"type": "delta", "text": "Hi"}
STOP = {"type": "stop"}
STOPS = EventTypes("stop")


def parse(
    chunks: list[bytes],
    *,
    terminal: IEventMatcher | None = STOPS,
    response_format: ResponseFormat = ResponseFormat.SSE,
    item: Any = Event,
    parser: StreamResponseParser | None = None,
) -> AsyncIterator[Any]:
    method = make_method(
        BaseStreamMethod[item],
        __response_format__=response_format,
        __stream_terminal__=terminal,
    )()
    return (parser or StreamResponseParser()).parse(
        method,
        FakeStreamResponse(chunks, headers=SSE_HEADERS),
        loads,
        BaseErrorResolver(),
    )


async def collect(items: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in items]


def sse(*events: str) -> bytes:
    return "".join(f"{event}\n\n" for event in events).encode()


def data(value: Any) -> str:
    return f"data: {json.dumps(value)}"


def ndjson(*items: Any) -> bytes:
    return "".join(f"{json.dumps(item)}\n" for item in items).encode()


async def test_a_complete_stream_ends_after_its_terminal() -> None:
    items = await collect(parse([sse(data(DELTA), data(STOP))]))

    assert items == [Delta(type="delta", text="Hi"), Stop(type="stop")]


async def test_a_stream_cut_between_events_is_interrupted() -> None:
    items = parse([sse(data(DELTA), data(DELTA))])
    received = [await anext(items), await anext(items)]

    with pytest.raises(APIStreamInterruptedError) as info:
        await anext(items)

    assert len(received) == 2
    assert info.value.items_received == 2
    assert info.value.__cause__ is None
    assert info.value.transient is False


async def test_an_empty_body_is_interrupted() -> None:
    with pytest.raises(APIStreamInterruptedError) as info:
        await collect(parse([]))

    assert info.value.items_received == 0


async def test_an_unfinished_last_event_is_dropped() -> None:
    # S1: by the HTML standard an event without its blank line is
    # discarded at the end of the body, also when it is the terminal.
    body = sse(data(DELTA)) + data(STOP).encode()

    without_terminal = await collect(parse([body], terminal=None))

    assert without_terminal == [Delta(type="delta", text="Hi")]

    with pytest.raises(APIStreamInterruptedError) as info:
        await collect(parse([body]))

    assert info.value.items_received == 1


async def test_an_item_after_the_terminal_is_an_error() -> None:
    items = parse([sse(data(STOP), data(DELTA))])

    assert await anext(items) == Stop(type="stop")

    with pytest.raises(APIResponseError, match="after its terminal") as info:
        await anext(items)

    assert info.value.status == 200


async def test_a_second_terminal_is_an_error() -> None:
    with pytest.raises(APIResponseError, match="after its terminal"):
        await collect(parse([sse(data(STOP), data(STOP))]))


async def test_comments_and_done_after_the_terminal_are_not_items() -> None:
    body = sse(data(STOP), ": keep-alive", "data: [DONE]", "event: x")

    assert await collect(parse([body])) == [Stop(type="stop")]


async def test_an_error_event_after_the_terminal_is_raised() -> None:
    error = {"type": "error", "error": {"message": "late"}}
    body = sse(data(STOP), data(error))

    with pytest.raises(APIStatusError, match="late"):
        await collect(parse([body]))


async def test_done_is_ignored_before_another_terminal() -> None:
    # [DONE] is one kind of terminal, not the only end of a stream.
    body = sse(data(DELTA), "data: [DONE]", data(STOP))

    items = await collect(parse([body]))

    assert items == [Delta(type="delta", text="Hi"), Stop(type="stop")]


async def test_done_alone_does_not_complete_another_terminal() -> None:
    with pytest.raises(APIStreamInterruptedError):
        await collect(parse([sse(data(DELTA), "data: [DONE]")]))


async def test_done_as_the_terminal() -> None:
    # OpenAI Chat Completions: a usage chunk follows finish_reason, so
    # only [DONE] ends the stream.
    body = sse(data(DELTA), data(STOP), "data: [DONE]")

    items = await collect(parse([body], terminal=DONE))

    assert len(items) == 2

    with pytest.raises(APIStreamInterruptedError) as info:
        await collect(parse([sse(data(DELTA))], terminal=DONE))

    assert info.value.items_received == 1


async def test_done_terminal_reads_the_body_to_its_end() -> None:
    body = sse(data(DELTA), "data: [DONE]", data(DELTA))

    with pytest.raises(APIResponseError, match="after its terminal"):
        await collect(parse([body], terminal=DONE))


async def test_without_a_terminal_done_still_ends_the_stream() -> None:
    body = sse(data(DELTA), "data: [DONE]", data(DELTA))

    items = await collect(parse([body], terminal=None))

    assert items == [Delta(type="delta", text="Hi")]


async def test_closing_early_is_never_checked() -> None:
    items = parse([sse(data(DELTA), data(DELTA))])

    assert isinstance(items, AsyncGenerator)
    await anext(items)
    await items.aclose()

    with pytest.raises(StopAsyncIteration):
        await anext(items)


async def test_ndjson_terminal() -> None:
    complete = await collect(
        parse(
            [ndjson(DELTA, STOP)],
            response_format=ResponseFormat.NDJSON,
        )
    )

    assert complete == [Delta(type="delta", text="Hi"), Stop(type="stop")]

    with pytest.raises(APIStreamInterruptedError):
        await collect(
            parse([ndjson(DELTA)], response_format=ResponseFormat.NDJSON)
        )

    with pytest.raises(APIResponseError, match="after its terminal"):
        await collect(
            parse(
                [ndjson(STOP, DELTA)],
                response_format=ResponseFormat.NDJSON,
            )
        )


class Done(BaseMatcher):
    """Ollama's NDJSON terminal: a chunk with ``"done": true``."""

    def matches(self, item: object) -> bool:
        return isinstance(item, dict) and item.get("done") is True


async def test_a_provider_writes_its_own_terminal() -> None:
    body = ndjson({"message": "Hi", "done": False}, {"done": True})

    items = await collect(
        parse(
            [body],
            terminal=Done(),
            response_format=ResponseFormat.NDJSON,
            item=dict[str, Any],
        )
    )

    assert items[-1] == {"done": True}


class Named(BaseMatcher):
    """Recognizes a raw event by its SSE event name."""

    def matches(self, item: object) -> bool:
        return isinstance(item, ServerSentEvent) and item.event == "end"


async def test_raw_events_take_a_terminal_too() -> None:
    body = sse("data: a", "event: end\ndata: b")

    items = await collect(
        parse([body], terminal=Named(), item=ServerSentEvent)
    )

    assert [event.data for event in items] == ["a", "b"]

    with pytest.raises(APIResponseError, match="after its terminal"):
        await collect(
            parse(
                [body + sse("data: c")],
                terminal=Named(),
                item=ServerSentEvent,
            )
        )

    with pytest.raises(APIStreamInterruptedError):
        await collect(
            parse([sse("data: a")], terminal=Named(), item=ServerSentEvent)
        )


async def test_an_error_event_that_fits_the_item_type_is_its_raw() -> None:
    parser = StreamResponseParser(error_events=EventTypes("failed"))
    body = sse(data({"type": "failed", "reason": "boom"}))

    with pytest.raises(APIStatusError) as info:
        await collect(parse([body], parser=parser))

    assert info.value.raw == Failed(type="failed", reason="boom")


async def test_an_error_event_that_does_not_fit_has_no_raw() -> None:
    body = sse(data({"type": "error", "message": "boom"}))

    with pytest.raises(APIStatusError) as info:
        await collect(parse([body]))

    assert info.value.raw is None


@pytest.mark.parametrize(
    ("item", "response_format"),
    [(bytes, ResponseFormat.BYTES), (str, ResponseFormat.TEXT)],
    ids=["bytes", "text"],
)
def test_terminal_of_a_raw_stream_is_rejected(
    item: type[Any], response_format: ResponseFormat
) -> None:
    # Only SSE and NDJSON read the terminal; a raw stream would ignore it.
    with pytest.raises(TypeError, match="checked only for SSE and NDJSON"):
        make_method(
            BaseStreamMethod[item],
            __response_format__=response_format,
            __stream_terminal__=STOPS,
        )


def test_terminal_with_own_parser_is_kept() -> None:
    method_cls = make_method(
        BaseStreamMethod[bytes],
        __stream_terminal__=STOPS,
        __stream_parser__=StreamResponseParser(),
    )

    assert method_cls().stream_terminal is STOPS
