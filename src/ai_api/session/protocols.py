"""Interface of an API session."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Protocol, Self

from ai_api.methods.protocols import IAPIMethod, IStreamMethod
from ai_api.response.stream import Stream

if TYPE_CHECKING:
    from types import TracebackType


class ISession(Protocol):
    """Sends methods to an API and returns their typed results."""

    async def close(self) -> None:
        """Release the connection pool."""
        ...

    async def __aenter__(self) -> Self:
        """Open the session.

        Returns:
            The session itself.
        """
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the session.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc_value: The exception raised in the block, if any.
            traceback: Its traceback, if any.
        """
        ...

    async def send_request[T](
        self,
        method: IAPIMethod[T],
    ) -> T:
        """Send a method and return its parsed response.

        Args:
            method: The method to send.

        Returns:
            The parsed response.
        """
        ...

    def stream[T](
        self,
        method: IStreamMethod[T],
    ) -> AbstractAsyncContextManager[Stream[T]]:
        """Send a method and stream its response.

        Args:
            method: The stream method to send.

        Returns:
            A context manager that yields the stream and closes it on
            exit.
        """
        ...
