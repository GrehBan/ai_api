"""The stream of a generation, with helpers for its text and its end."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator

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
    a loop over the events) returns it without reading again. An error
    raised while reading is kept too: reading again, ``text()`` or
    ``final()`` raise the same error instead of ending as if the stream
    were cut.

    ``GenerationResource.stream`` builds it on the converted events
    through ``transform_stream(..., stream_type=GenerationStream)``.

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
            items: The converted events.
        """
        super().__init__(items)
        self._completed: CompletedEvent[RawItemT] | None = None
        self._failure: Exception | None = None

    async def __anext__(self) -> GenerationEvent[RawItemT]:
        """Return the next event, keeping the ``CompletedEvent``.

        Returns:
            The next event.

        Raises:
            Exception: The error a previous read raised, again: the
                events behind it are gone, so ending quietly would look
                like a stream that was cut.
        """
        if self._failure is not None:
            raise self._failure

        try:
            event = await super().__anext__()
        except StopAsyncIteration:
            raise
        except Exception as exc:
            self._failure = exc
            raise

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
            Exception: The error an earlier read raised, such as the
                provider's error event, again.
        """
        async for _ in self:
            pass
        if self._completed is None:
            message = "Generation stream ended without a CompletedEvent"
            raise APIStreamInterruptedError(
                message, items_received=self.items_received
            )
        return self._completed
