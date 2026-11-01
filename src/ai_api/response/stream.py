"""The object returned to callers of ``session.stream()``."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, Self, overload

import aiofiles

from ai_api.http.exceptions.transport import (
    APIStreamInterruptedError,
    APITransportError,
)


class Stream[T]:
    """Async iterator over the items of a streamed response.

    Obtained from ``async with session.stream(method) as stream``; the
    session closes it when the ``async with`` block exits, and iterating
    it after that raises ``RuntimeError``.

    A transport failure while iterating (a lost connection, a silent
    stream timing out) is raised as ``APIStreamInterruptedError``, which
    is never retried, with the original error as its ``__cause__``.

    Example:
        >>> async with session.stream(Download()) as stream:
        ...     async for chunk in stream:
        ...         handle(chunk)
    """

    def __init__(self, items: AsyncIterator[T]) -> None:
        """Wrap the parser's item iterator.

        Args:
            items: Items produced by the method's stream parser.
        """
        self._items = items
        self._closed = False
        self._received = 0

    @property
    def items_received(self) -> int:
        """How many items the stream has given so far."""
        return self._received

    def __aiter__(self) -> Self:
        """Return the stream itself.

        Returns:
            The stream; its items can be consumed only once.

        Raises:
            RuntimeError: If the stream is closed.
        """
        self._check_open()
        return self

    async def __anext__(self) -> T:
        """Return the next item.

        The check runs on every item, so an iterator taken inside the
        ``async with`` block also fails after it instead of ending
        silently.

        Returns:
            The next item.

        Raises:
            RuntimeError: If the stream is closed.
            StopAsyncIteration: When the items are exhausted.
            APIStreamInterruptedError: If the transport fails or the body
                ends before the method's terminal event. One raised by
                an inner stream (``transform_stream``) or by the parser
                passes through as it is.
        """
        self._check_open()

        try:
            item = await anext(self._items)
        except APIStreamInterruptedError:
            raise
        except APITransportError as exc:
            raise APIStreamInterruptedError(
                f"Stream interrupted by {type(exc).__name__} after "
                f"{self._received} items",
                items_received=self._received,
            ) from exc

        self._received += 1
        return item

    def _check_open(self) -> None:
        """Ensure the stream has not been closed.

        Raises:
            RuntimeError: If the stream is closed.
        """
        if self._closed:
            raise RuntimeError(
                "Stream is closed: iterate it inside the "
                "'async with session.stream(...)' block"
            )

    async def aclose(self) -> None:
        """Close the underlying parser generator, if it is one."""
        self._closed = True

        if isinstance(self._items, AsyncGenerator):
            await self._items.aclose()

    async def save(
        self: Stream[bytes],
        path: str | os.PathLike[str],
    ) -> int:
        """Write a byte stream to a file without buffering it in memory.

        Only available on ``Stream[bytes]``.

        Args:
            path: Destination file; it is created or overwritten.

        Returns:
            Number of bytes written.

        Raises:
            RuntimeError: If the stream is closed; the file is not
                touched.
            APIStreamInterruptedError: If the stream breaks mid-way; the
                file keeps what was written.
        """
        self._check_open()
        written = 0

        async with aiofiles.open(path, "wb") as file:
            async for chunk in self:
                await file.write(chunk)
                written += len(chunk)

        return written


@overload
def transform_stream[T, R](
    source: AbstractAsyncContextManager[Stream[T]],
    transform: Callable[[AsyncIterator[T]], AsyncIterator[R]],
) -> AbstractAsyncContextManager[Stream[R]]: ...


@overload
def transform_stream[T, R, S: Stream[Any]](
    source: AbstractAsyncContextManager[Stream[T]],
    transform: Callable[[AsyncIterator[T]], AsyncIterator[R]],
    *,
    stream_type: Callable[[AsyncIterator[R]], S],
) -> AbstractAsyncContextManager[S]: ...


def transform_stream[T, R](
    source: AbstractAsyncContextManager[Stream[T]],
    transform: Callable[[AsyncIterator[T]], AsyncIterator[R]],
    *,
    stream_type: Callable[[AsyncIterator[R]], Stream[Any]] = Stream,
) -> AbstractAsyncContextManager[Stream[Any]]:
    """Open a stream and pass its items through ``transform``.

    ``transform`` sees the whole source, so it can keep state between
    items, skip some and turn one item into several, e.g. collect text
    deltas into a final event. On exit, including ``break`` and
    exceptions, the transformed stream is closed first and the source
    after it; iterating the result after the block raises
    ``RuntimeError``.

    Args:
        source: The stream to open, e.g. ``session.stream(method)``.
        transform: Turns the source items into result items; usually an
            async generator function.
        stream_type: The ``Stream`` class to yield, such as
            ``GenerationStream``; built on the transformed items once,
            with no second wrapper.

    Returns:
        A context manager that yields a stream of transformed items.

    Example:
        >>> async def texts(events):
        ...     async for event in events:
        ...         if event.text:
        ...             yield event.text
        >>> async with transform_stream(session.stream(Deltas()), texts) as s:
        ...     async for text in s:
        ...         handle(text)
    """
    return _transformed(source, transform, stream_type)


@asynccontextmanager
async def _transformed[T, R](
    source: AbstractAsyncContextManager[Stream[T]],
    transform: Callable[[AsyncIterator[T]], AsyncIterator[R]],
    stream_type: Callable[[AsyncIterator[R]], Stream[Any]],
) -> AsyncGenerator[Stream[Any]]:
    """Open the source and yield its transformed stream.

    Args:
        source: The stream to open.
        transform: Turns the source items into result items.
        stream_type: The class of the yielded stream.

    Yields:
        The transformed stream; closed before the source on exit.
    """
    async with source as stream:
        transformed = stream_type(transform(stream))

        try:
            yield transformed
        finally:
            await transformed.aclose()


def map_stream[T, R](
    source: AbstractAsyncContextManager[Stream[T]],
    convert: Callable[[T], R],
) -> AbstractAsyncContextManager[Stream[R]]:
    """Open a stream and convert each of its items.

    The one-to-one case of ``transform_stream``, with the same closing
    order.

    Args:
        source: The stream to open, e.g. ``session.stream(method)``.
        convert: Turns one source item into one result item.

    Returns:
        A context manager that yields a stream of converted items.

    Example:
        >>> async with map_stream(session.stream(Deltas()), to_event) as s:
        ...     async for event in s:
        ...         handle(event)
    """

    async def items(stream: AsyncIterator[T]) -> AsyncGenerator[R]:
        """Convert the source items one by one.

        Args:
            stream: The source items.

        Yields:
            The converted items.
        """
        async for item in stream:
            yield convert(item)

    return transform_stream(source, items)
