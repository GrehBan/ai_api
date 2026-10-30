"""The object returned to callers of ``session.stream()``."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Self

import aiofiles


class Stream[T]:
    """Async iterator over the items of a streamed response.

    Obtained from ``async with session.stream(method) as stream``; the
    session closes it when the ``async with`` block exits, and iterating
    it after that raises ``RuntimeError``.

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
        """
        self._check_open()
        return await anext(self._items)

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
        """
        self._check_open()
        written = 0

        async with aiofiles.open(path, "wb") as file:
            async for chunk in self:
                await file.write(chunk)
                written += len(chunk)

        return written


@asynccontextmanager
async def map_stream[T, R](
    source: AbstractAsyncContextManager[Stream[T]],
    convert: Callable[[T], R],
) -> AsyncGenerator[Stream[R]]:
    """Open a stream and convert each of its items.

    Meant for adapters that turn a provider's stream into another item
    type. On exit, including ``break`` and exceptions, the converted
    stream is closed first and the source after it; iterating the result
    after the block raises ``RuntimeError``.

    Args:
        source: The stream to open, e.g. ``session.stream(method)``.
        convert: Turns one source item into one result item.

    Yields:
        A stream of converted items.

    Example:
        >>> async with map_stream(session.stream(Deltas()), to_event) as s:
        ...     async for event in s:
        ...         handle(event)
    """
    async with source as stream:

        async def items() -> AsyncGenerator[R]:
            """Convert the source items one by one.

            Yields:
                The converted items.
            """
            async for item in stream:
                yield convert(item)

        mapped = Stream(items())

        try:
            yield mapped
        finally:
            await mapped.aclose()
