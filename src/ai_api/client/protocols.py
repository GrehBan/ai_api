"""Interfaces of a provider client and of its session factory."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Protocol, Self

from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.protocols import IAuth
from ai_api.methods.protocols import IAPIMethod, IStreamMethod
from ai_api.response.stream import Stream
from ai_api.session.protocols import ISession, Timeout

if TYPE_CHECKING:
    from types import TracebackType

    from ai_api.http.protocols import HeadersType


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
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> T:
        """Send a method and return its parsed response.

        Args:
            method: The method to send.
            headers: Headers for this request only.
            timeout: Seconds the whole request may take; ``None`` for
                the session's timeout.

        Returns:
            The parsed response.
        """
        ...

    def stream[T](
        self,
        method: IStreamMethod[T],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> AbstractAsyncContextManager[Stream[T]]:
        """Send a stream method; use the result with ``async with``.

        Args:
            method: The stream method to send.
            headers: Headers for this request only.
            timeout: Longest silence between two chunks, in seconds;
                ``None`` for the session's stream timeout.

        Returns:
            A context manager that yields the stream and closes it on
            exit.
        """
        ...


class ISessionFactory[SessionT: ISession, AuthT: IAuth, **P](Protocol):
    """Builds sessions for one provider."""

    @property
    def error_resolver(self) -> IErrorResolver:
        """The provider's error resolver, given to every session."""
        ...

    def __call__(
        self,
        auth: AuthT | str,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> SessionT:
        """Build a session.

        Args:
            auth: An ``IAuth``, or a token.
            *args: Further positional arguments of the session.
            **kwargs: Further keyword arguments of the session.

        Returns:
            The new session.
        """
        ...
