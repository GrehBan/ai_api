"""Transport-agnostic session logic: retries, errors and parsing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Self

from aiollm.http.exceptions.protocols import IErrorResolver
from aiollm.http.headers import Headers
from aiollm.http.protocols import IAPI, HeadersType, IAuth, IHeaders
from aiollm.methods.protocols import IAPIMethod, IStreamMethod
from aiollm.response.buffered import BufferedResponse
from aiollm.response.protocols import IStreamResponse
from aiollm.response.stream import Stream
from aiollm.serialization.json import JsonDumps, JsonLoads, dumps, loads
from aiollm.session.protocols import Timeout
from aiollm.session.retry.policy import DEFAULT_RETRY_POLICY, RetryPolicy
from aiollm.session.retry.runner import build_retrying
from aiollm.session.retry.state import RetryState
from aiollm.types.files import closing_source

if TYPE_CHECKING:
    from types import TracebackType

logger = logging.getLogger(__name__)


def _log_retry(state: RetryState, delay: float) -> None:
    """Log a retry before its wait: the next attempt, the wait, the cause.

    The failed attempt's error is not logged elsewhere, since it is not
    raised, so its type is named here, and the number of its notes (a
    failing close of an upload, say). Not their text, nor the error's
    message: either may hold a file path or a value of the request.

    Args:
        state: The failed attempt.
        delay: Seconds to wait before the next one.
    """
    notes = len(getattr(state.error, "__notes__", ()))
    logger.warning(
        "Retrying request after %s: attempt=%d, delay=%.2fs%s",
        type(state.error).__name__,
        state.attempt + 1,
        delay,
        f", notes={notes}" if notes else "",
    )


class BaseSession(ABC):
    """Base class for sessions; a transport only does the HTTP I/O.

    Retries, turning error statuses into ``APIStatusError`` and parsing
    are done here, so every transport behaves the same. A subclass implements
    ``_execute``, ``_open_stream`` and ``close``.

    Attributes:
        api: Builds request URLs.
        auth: Produces the authentication header.
        error_resolver: Turns error responses into exceptions.
        headers: Default headers sent with every request.
        json_loads: JSON decoder for response bodies.
        json_dumps: JSON encoder for request bodies.
        retry_policy: When and how long a retryable request is retried,
            unless a call gives its own policy.
    """

    def __init__(
        self,
        api: IAPI,
        auth: IAuth,
        error_resolver: IErrorResolver,
        *,
        headers: HeadersType | None = None,
        json_loads: JsonLoads = loads,
        json_dumps: JsonDumps = dumps,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    ) -> None:
        """Store the session's collaborators.

        Args:
            api: Builds request URLs.
            auth: Produces the authentication header.
            error_resolver: Turns error responses into exceptions.
            headers: Default headers. An ``IHeaders`` collection is used
                as it is; a mapping is copied into ``Headers``, and an
                empty set is used when ``None``.
            json_loads: JSON decoder for response bodies.
            json_dumps: JSON encoder for request bodies.
            retry_policy: When and how long a retryable request is
                retried: four attempts at most, none started after 15
                minutes, a ``Retry-After`` above 60 s not waited for.
        """
        self.api = api
        self.auth = auth
        self.error_resolver = error_resolver
        self.headers: IHeaders = (
            headers if isinstance(headers, IHeaders) else Headers(headers)
        )
        self.json_loads = json_loads
        self.json_dumps = json_dumps
        self.retry_policy = retry_policy

    @abstractmethod
    async def close(self) -> None:
        """Release the transport's resources."""
        raise NotImplementedError

    async def __aenter__(self) -> Self:
        """Enter the session context.

        Returns:
            The session itself.
        """
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

    async def send_request[T](
        self,
        method: IAPIMethod[T],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> T:
        """Send a method and return its parsed response.

        If ``method.retryable`` is true, transient failures are retried as
        the retry policy allows. ``headers`` and ``timeout`` apply to
        every attempt.

        Args:
            method: The method to send.
            headers: Headers for this request only; they override the
                session's and the method's headers, but not the
                authentication header.
            timeout: Seconds the whole request may take, instead of the
                total of the session's timeout; ``None`` to keep it.
            retry_policy: Retry policy of this request only, instead of
                the session's; ``None`` to keep it. It does not make a
                method retryable.

        Returns:
            The parsed response, typed by the method's generic argument.

        Raises:
            APIStatusError: If the API answered with an error status, or
                the method's parser found a failure in a successful one.
            APITransportError: If the request could not be completed.
            APIResponseError: If the response body could not be parsed.
        """
        name = type(method).__name__
        logger.debug("Starting %s", name)
        result = await self._with_retry(
            method.retryable,
            lambda: self._send_once(
                method,
                headers=headers,
                timeout=timeout,
            ),
            retry_policy=retry_policy,
        )
        logger.debug("Completed %s", name)
        return result

    @asynccontextmanager
    async def stream[T](
        self,
        method: IStreamMethod[T],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> AsyncGenerator[Stream[T]]:
        """Send a method and stream its response.

        Only opening the stream is retried: once the block has started, a
        failure is raised to the caller instead of resending the request.
        The connection is released when the block exits, including on
        ``break`` and exceptions.

        Args:
            method: The stream method to send.
            headers: Headers for this request only; they override the
                session's and the method's headers, but not the
                authentication header.
            timeout: Longest silence between two chunks, in seconds,
                instead of the session's stream timeout; ``None`` to keep
                it. A stream has no total limit, so a long generation is
                not cut off.
            retry_policy: Retry policy of opening this stream only,
                instead of the session's; ``None`` to keep it. It does
                not make a method retryable.

        Yields:
            The stream of items, typed by the method's generic argument.

        Raises:
            APIStatusError: If the API answered with an error status; raised on
                entering the block. Also, while iterating, for an error
                event in the stream.
            APITransportError: If the connection fails on entering the
                block.
            APIStreamInterruptedError: While iterating, if the connection
                fails or goes silent for too long (the transport error is
                its ``__cause__``), or the body ends before the method's
                terminal event. Never retried.
            APIResponseError: If the body cannot be decoded or validated,
                or an item comes after the terminal event.

        Example:
            >>> async with session.stream(Download()) as stream:
            ...     await stream.save("out.bin")
        """
        name = type(method).__name__
        logger.debug("Starting %s", name)
        response = await self._with_retry(
            method.retryable,
            lambda: self._open_stream_once(
                method,
                headers=headers,
                timeout=timeout,
            ),
            retry_policy=retry_policy,
        )

        # A failing close (of an upload left open, say) is noted on the
        # error that ended the block instead of replacing it.
        async with closing_source(response, what="the response"):
            stream = Stream(
                method.stream_parser.parse(
                    method,
                    response,
                    self.json_loads,
                    self.error_resolver,
                )
            )

            async with closing_source(stream, what="the stream"):
                yield stream

        logger.debug("Completed %s items=%d", name, stream.items_received)

    async def _with_retry[R](
        self,
        retryable: bool,
        call: Callable[[], Awaitable[R]],
        *,
        retry_policy: RetryPolicy | None,
    ) -> R:
        """Run ``call``, retrying transient failures if allowed.

        Args:
            retryable: Whether retries are allowed for this request.
            call: Performs one attempt.
            retry_policy: The call's own policy; ``None`` for the
                session's.

        Returns:
            The result of the first successful attempt.
        """
        policy = self.retry_policy if retry_policy is None else retry_policy

        if not retryable or policy.max_attempts == 1:
            return await call()

        async def attempt() -> R:
            """Perform one attempt.

            Tenacity awaits only a coroutine function; ``call`` is any
            callable that returns an awaitable.

            Returns:
                The result of the attempt.
            """
            return await call()

        retrying = build_retrying(policy, before_sleep=_log_retry)
        return await retrying(attempt)

    async def _send_once[T](
        self,
        method: IAPIMethod[T],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> T:
        """Perform one attempt of ``send_request``.

        Args:
            method: The method to send.
            headers: Headers for this request only.
            timeout: Total timeout of this request, in seconds.

        Returns:
            The parsed response.
        """
        # Status check stays inside the retried unit: retries are driven
        # by the transient flag of the APIStatusError raised here.
        response = await self._execute(
            method,
            headers=headers,
            timeout=timeout,
        )
        self._raise_for_status(response)

        # The parser raises APIResponseError for a body it cannot read,
        # as the stream parser does; anything else is a bug of the
        # parser and is not disguised as a bad response.
        return method.response_parser.parse(
            method,
            response,
            self.json_loads,
            self.error_resolver,
        )

    async def _open_stream_once(
        self,
        method: IStreamMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> IStreamResponse:
        """Perform one attempt of opening a stream.

        An error response is read, closed and raised here, so it is still
        inside the retry loop.

        Args:
            method: The stream method to send.
            headers: Headers for this request only.
            timeout: Longest silence between two chunks, in seconds.

        Returns:
            The open response with a status below 400.
        """
        response = await self._open_stream(
            method,
            headers=headers,
            timeout=timeout,
        )

        if response.status >= HTTPStatus.BAD_REQUEST:
            # `response.aclose()` may raise an `APIRequestError` before
            # `_raise_for_status()` is called. Consequently,
            # the original HTTP status will
            # not be included in the `APIStatusError`.
            # so closing_source() can capture an APIStatusError
            # and attach a cleanup error using add_note().
            async with closing_source(response, what="the response"):
                body = await response.read()
                self._raise_for_status(
                    BufferedResponse(
                        status=response.status,
                        headers=response.headers,
                        body=body,
                        charset=response.charset,
                    )
                )

        return response

    def _raise_for_status(self, response: BufferedResponse) -> None:
        """Raise the error the resolver builds for an error status.

        Args:
            response: The response to check.

        Raises:
            APIStatusError: If the status is 400 or higher.
        """
        if response.status < HTTPStatus.BAD_REQUEST:
            return

        try:
            body = self.json_loads(response.body)
        except (ValueError, TypeError):
            body = None

        raise self.error_resolver.resolve(
            status=response.status,
            body=body,
            headers=response.headers,
        )

    @abstractmethod
    async def _execute(
        self,
        method: IAPIMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> BufferedResponse:
        """Send the request and read the whole body.

        Args:
            method: The method to send.
            headers: Headers for this request only, sent over the
                session's and the method's headers and under the
                authentication header.
            timeout: Total timeout of this request, in seconds; ``None``
                for the session's.

        Returns:
            The response, including responses with an error status.

        Raises:
            APITransportError: If the request could not be completed.
        """
        raise NotImplementedError

    @abstractmethod
    async def _open_stream(
        self,
        method: IStreamMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> IStreamResponse:
        """Send the request without reading the body.

        Args:
            method: The stream method to send.
            headers: Headers for this request only, sent over the
                session's and the method's headers and under the
                authentication header.
            timeout: Longest silence between two chunks, in seconds;
                ``None`` for the session's stream timeout.

        Returns:
            The open response, including responses with an error status.
            The caller is responsible for closing it.

        Raises:
            APITransportError: If the request could not be sent.
        """
        raise NotImplementedError
