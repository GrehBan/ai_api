"""Base class for provider clients."""

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Self

from ai_api.client.response.stream import Stream
from ai_api.client.session.protocols import ISession
from ai_api.methods.protocols import IAPIMethod, IStreamMethod

if TYPE_CHECKING:
    from types import TracebackType


class BaseClient[SessionT: ISession]:
    """Sends methods through a session; a provider client subclasses it.

    The client owns its session: ``close()`` and leaving ``async with``
    close it. ``SessionT`` keeps the session's own type, so a client
    built on ``AiohttpSession`` exposes its settings, such as ``loop``.

    Attributes:
        session: The session that sends the requests.

    Example:
        >>> async with OpenAIClient("sk-...") as client:
        ...     models = await client.send_request(ListModels())
    """

    def __init__(self, session: SessionT) -> None:
        """Wrap a session.

        Args:
            session: The session that sends the requests.
        """
        self.session: SessionT = session

    async def __aenter__(self) -> Self:
        """Open the session.

        Returns:
            The client itself.
        """
        await self.session.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the session on leaving the context.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc_value: The exception raised in the block, if any.
            traceback: Its traceback, if any.
        """
        await self.close()

    async def close(self) -> None:
        """Close the session and its connections."""
        await self.session.close()

    async def send_request[T](
        self,
        method: IAPIMethod[T],
    ) -> T:
        """Send a method and return its parsed response.

        Args:
            method: The method to send.

        Returns:
            The parsed response, typed by the method's generic argument.
        """
        return await self.session.send_request(method)

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
        return self.session.stream(method)
