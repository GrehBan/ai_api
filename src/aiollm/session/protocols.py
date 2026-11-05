"""Interface of an API session."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Protocol, Self

from typing_extensions import TypedDict

from aiollm.http.exceptions.protocols import IErrorResolver
from aiollm.http.protocols import HeadersType, IAuth
from aiollm.methods.protocols import IAPIMethod, IStreamMethod
from aiollm.response.stream import Stream
from aiollm.session.retry.policy import RetryPolicy

if TYPE_CHECKING:
    from types import TracebackType

type Timeout = float
"""Seconds; for a buffered request the limit of the whole request, for
a stream the longest silence between two chunks."""


class CallOptions(TypedDict, total=False, closed=True):
    """The options of one call, as ``send_request`` and ``stream`` take.

    The shared layer (use cases, ``client.generation``) takes them as
    ``**options`` and passes them on to the session unchanged, so type
    checkers check them at the call. Each key is a keyword-only argument
    of ``ISession.send_request`` with the same type; a test keeps the
    two in step.

    It is closed (PEP 728, from ``typing_extensions`` until Python
    3.15): no other key can ride in ``**options``. So a method that
    takes them may have keyword parameters of its own beside them, such
    as ``GenerationResource.create``'s ``provider_options``, and still
    match a protocol whose method has none.

    Attributes:
        headers: Headers for this request only; they override the
            session's and the method's headers.
        timeout: Seconds for this request instead of the session's
            limit; ``None`` to keep it.
        retry_policy: Retry policy of this request; ``None`` for the
            session's.
    """

    headers: HeadersType | None
    timeout: Timeout | None
    retry_policy: RetryPolicy | None


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
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> T:
        """Send a method and return its parsed response.

        Args:
            method: The method to send.
            headers: Headers for this request only; they override the
                session's and the method's headers.
            timeout: Seconds the whole request may take; ``None`` for
                the session's timeout.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.

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
        retry_policy: RetryPolicy | None = None,
    ) -> AbstractAsyncContextManager[Stream[T]]:
        """Send a method and stream its response.

        Args:
            method: The stream method to send.
            headers: Headers for this request only; they override the
                session's and the method's headers.
            timeout: Longest silence between two chunks, in seconds;
                ``None`` for the session's stream timeout.
            retry_policy: Retry policy of opening the stream; ``None``
                for the session's.

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
