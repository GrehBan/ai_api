"""The object returned to callers of ``session.stream()``."""

import os
from collections.abc import AsyncGenerator, AsyncIterator

import aiofiles


class Stream[T]:
    """Async iterator over the items of a streamed response.

    Obtained from ``async with session.stream(method) as stream``; the
    session closes it when the ``async with`` block exits, so it must not
    be used after that.

    Example:
        >>> async with session.stream(Download()) as stream:
        ...     async for chunk in stream:
        ...         handle(chunk)
    """

    def __init__(self, events: AsyncIterator[T]) -> None:
        """Wrap the parser's item iterator.

        Args:
            events: Items produced by the method's stream parser.
        """
        self._events = events

    def __aiter__(self) -> AsyncIterator[T]:
        """Return the item iterator.

        Returns:
            The parser's iterator. It can be consumed only once.
        """
        return self._events

    async def aclose(self) -> None:
        """Close the underlying parser generator, if it is one."""
        if isinstance(self._events, AsyncGenerator):
            await self._events.aclose()

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
        """
        written = 0

        async with aiofiles.open(path, "wb") as file:
            async for chunk in self:
                await file.write(chunk)
                written += len(chunk)

        return written
