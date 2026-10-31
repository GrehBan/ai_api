"""The stream of a generation, with helpers for its text and its end."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from ai_api.http.exceptions.transport import APIStreamInterruptedError
from ai_api.response.stream import Stream
from ai_api.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    TextDeltaEvent,
)


class GenerationStream[RawItemT](Stream[GenerationEvent[RawItemT]]):
    """The events of a generation, with helpers for the common cases.

    It is a ``Stream`` of shared events, so it can be iterated as it is;
    ``text()`` gives only the text, and ``final()`` reads to the end and
    gives the ``CompletedEvent``. Whatever consumed the events, the
    ``CompletedEvent`` is kept, so ``final()`` after ``text()`` (or after
    a loop over the events) returns it without reading again.

    Example:
        >>> async with client.generation.stream(request) as events:
        ...     async for text in events.text():
        ...         print(text, end="")
        ...     completed = await events.final()
    """

    def __init__(
        self,
        items: AsyncIterator[GenerationEvent[RawItemT]],
    ) -> None:
        """Wrap the shared events.

        Args:
            items: The events, usually a ``Stream`` of a use case.
        """
        super().__init__(items)
        self._completed: CompletedEvent[RawItemT] | None = None

    async def __anext__(self) -> GenerationEvent[RawItemT]:
        """Return the next event, keeping the ``CompletedEvent``.

        Returns:
            The next event.
        """
        event = await super().__anext__()
        if isinstance(event, CompletedEvent):
            self._completed = event
        return event

    async def text(self) -> AsyncGenerator[str]:
        """Give the generated text as it arrives.

        Other events are read and skipped; the ``CompletedEvent`` is kept
        for ``final()``.

        Yields:
            The text of each ``TextDeltaEvent``.

        Raises:
            RuntimeError: If the stream is closed.
        """
        async for event in self:
            if isinstance(event, TextDeltaEvent):
                yield event.text

    async def final(self) -> CompletedEvent[RawItemT]:
        """Read the stream to its end and give its ``CompletedEvent``.

        Returns:
            The last event of a complete generation, with its id, model,
            whole text, token counts and stop.

        Raises:
            RuntimeError: If the stream is closed.
            APIStreamInterruptedError: The stream ended without a
                ``CompletedEvent``. Through ``GenerationResource`` this
                is raised while reading, before ``final()`` gets here.
        """
        async for _ in self:
            pass
        if self._completed is None:
            message = "Generation stream ended without a CompletedEvent"
            raise APIStreamInterruptedError(
                message, items_received=self._received
            )
        return self._completed


@asynccontextmanager
async def generation_stream[RawItemT](
    source: AbstractAsyncContextManager[Stream[GenerationEvent[RawItemT]]],
) -> AsyncGenerator[GenerationStream[RawItemT]]:
    """Open a stream of shared events as a ``GenerationStream``.

    On exit the ``GenerationStream`` is closed first and the source after
    it, as ``transform_stream`` does.

    Args:
        source: The stream to open, e.g. a use case's ``stream(request)``.

    Yields:
        The events of the source, with the generation helpers.
    """
    async with source as events:
        stream = GenerationStream(events)
        try:
            yield stream
        finally:
            await stream.aclose()
