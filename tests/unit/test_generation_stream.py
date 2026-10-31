"""``GenerationStream``: the ``text()`` and ``final()`` helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

import pytest

from ai_api.http.exceptions.transport import APIStreamInterruptedError
from ai_api.resources.stream import GenerationStream, generation_stream
from ai_api.response.stream import Stream
from ai_api.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)

COMPLETED = CompletedEvent[Any](id="r-1", model="m", text="Hello!", raw={})

EVENTS: list[GenerationEvent[Any]] = [
    OtherEvent[Any](raw={}),
    TextDeltaEvent[Any](text="Hello", raw={}),
    OtherEvent[Any](raw={}),
    TextDeltaEvent[Any](text="!", raw={}),
    COMPLETED,
]


async def items(
    events: Iterable[GenerationEvent[Any]],
) -> AsyncIterator[GenerationEvent[Any]]:
    """Yield the events one by one."""
    for event in events:
        yield event


@asynccontextmanager
async def opened(
    events: Iterable[GenerationEvent[Any]],
) -> AsyncGenerator[Stream[GenerationEvent[Any]]]:
    """A source stream of the events, closed on exit."""
    stream = Stream(items(events))
    try:
        yield stream
    finally:
        await stream.aclose()


async def test_text_gives_only_the_text_deltas() -> None:
    async with generation_stream(opened(EVENTS)) as events:
        texts = [text async for text in events.text()]

    assert texts == ["Hello", "!"]


async def test_final_after_text_keeps_the_completed_event() -> None:
    async with generation_stream(opened(EVENTS)) as events:
        _ = [text async for text in events.text()]
        completed = await events.final()

    assert completed is COMPLETED
    assert completed.text == "Hello!"


async def test_final_reads_the_rest_of_the_stream() -> None:
    async with generation_stream(opened(EVENTS)) as events:
        first = await anext(events)
        completed = await events.final()

    assert isinstance(first, OtherEvent)
    assert completed is COMPLETED


async def test_final_keeps_what_a_plain_loop_read() -> None:
    async with generation_stream(opened(EVENTS)) as events:
        received = [event async for event in events]
        completed = await events.final()

    assert received == EVENTS
    assert completed is COMPLETED


async def test_final_without_a_completed_event_raises() -> None:
    stream = GenerationStream(items(EVENTS[:-1]))

    with pytest.raises(
        APIStreamInterruptedError, match="without a CompletedEvent"
    ) as caught:
        await stream.final()

    assert caught.value.items_received == len(EVENTS) - 1
    assert not caught.value.transient


async def test_helpers_refuse_a_closed_stream() -> None:
    async with generation_stream(opened(EVENTS)) as events:
        pass

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await events.final()
    with pytest.raises(RuntimeError, match="Stream is closed"):
        await anext(events.text())


async def test_closing_closes_the_source() -> None:
    source = Stream(items(EVENTS))

    @asynccontextmanager
    async def wrap() -> AsyncGenerator[Stream[GenerationEvent[Any]]]:
        try:
            yield source
        finally:
            await source.aclose()

    async with generation_stream(wrap()) as events:
        await anext(events)

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await anext(source)
