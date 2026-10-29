"""Interface of a provider client."""

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Protocol, Self

from ai_api.client.response.stream import Stream
from ai_api.client.session.protocols import ISession
from ai_api.methods.protocols import IAPIMethod, IStreamMethod

if TYPE_CHECKING:
    from types import TracebackType


class IClient(Protocol):
    """Sends methods through a session it owns."""

    @property
    def session(self) -> ISession:
        """The session that sends the requests."""
        ...

    async def __aenter__(self) -> Self:
        """Open the session.

        Returns:
            The client itself.
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

    async def close(self) -> None:
        """Close the session and its connections."""
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
        """Send a stream method; use the result with ``async with``.

        Args:
            method: The stream method to send.

        Returns:
            A context manager that yields the stream and closes it on
            exit.
        """
        ...
