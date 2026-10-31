"""Transport-agnostic session logic: retries, errors and parsing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Self

from tenacity import AsyncRetrying, stop_after_attempt, stop_before_delay
from tenacity.stop import stop_base

from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.headers import Headers
from ai_api.http.protocols import IAPI, HeadersType, IAuth, IHeaders
from ai_api.methods.protocols import IAPIMethod, IStreamMethod
from ai_api.response.buffered import BufferedResponse
from ai_api.response.protocols import IStreamResponse
from ai_api.response.stream import Stream
from ai_api.serialization.json import JsonDumps, JsonLoads, dumps, loads
from ai_api.session.protocols import Timeout
from ai_api.session.retry import (
    DEFAULT_MAX_RETRY_AFTER,
    DEFAULT_RETRY_DEADLINE,
    calculate_wait,
    retry_condition,
)

if TYPE_CHECKING:
    from types import TracebackType


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
        max_retry_after: Longest ``Retry-After`` the session waits for,
            in seconds; a transient error asking for a longer wait is
            raised without retrying.
        retry_deadline: Seconds since the first attempt after which no
            retry starts; ``None`` for no limit.
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
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
        retry_deadline: float | None = DEFAULT_RETRY_DEADLINE,
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
            max_retry_after: Longest ``Retry-After`` to wait for, in
                seconds.
            retry_deadline: Seconds since the first attempt after which
                no retry starts; a wait that would end later is not
                started. An attempt in progress is not interrupted (its
                own timeout limits it). ``None`` for no limit.
        """
        self.api = api
        self.auth = auth
        self.error_resolver = error_resolver
        self.headers: IHeaders = (
            headers if isinstance(headers, IHeaders) else Headers(headers)
        )
        self.json_loads = json_loads
        self.json_dumps = json_dumps
        self.max_retry_after = max_retry_after
        self.retry_deadline = retry_deadline

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
    ) -> T:
        """Send a method and return its parsed response.

        If ``method.retryable`` is true, transient failures are retried up
        to four attempts in total, unless the API asks to wait longer than
        ``max_retry_after`` or the retry would start after
        ``retry_deadline``. ``headers`` and ``timeout`` apply to every
        attempt.

        Args:
            method: The method to send.
            headers: Headers for this request only; they override the
                session's and the method's headers, but not the
                authentication header.
            timeout: Seconds the whole request may take, instead of the
                total of the session's timeout; ``None`` to keep it.

        Returns:
            The parsed response, typed by the method's generic argument.

        Raises:
            APIStatusError: If the API answered with an error status, or
                the method's parser found a failure in a successful one.
            APITransportError: If the request could not be completed.
            APIResponseError: If the response body could not be parsed.
        """
        return await self._with_retry(
            method.retryable,
            lambda: self._send_once(
                method,
                headers=headers,
                timeout=timeout,
            ),
        )

    @asynccontextmanager
    async def stream[T](
        self,
        method: IStreamMethod[T],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
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
        response = await self._with_retry(
            method.retryable,
            lambda: self._open_stream_once(
                method,
                headers=headers,
                timeout=timeout,
            ),
        )

        try:
            stream = Stream(
                method.stream_parser.parse(
                    method,
                    response,
                    self.json_loads,
                    self.error_resolver,
                )
            )

            try:
                yield stream
            finally:
                await stream.aclose()
        finally:
            await response.aclose()

    async def _with_retry[R](
        self,
        retryable: bool,
        call: Callable[[], Awaitable[R]],
    ) -> R:
        """Run ``call``, retrying transient failures if allowed.

        Args:
            retryable: Whether retries are allowed for this request.
            call: Performs one attempt.

        Returns:
            The result of the first successful attempt.
        """
        if not retryable:
            return await call()

        stop: stop_base = stop_after_attempt(4)

        if self.retry_deadline is not None:
            stop |= stop_before_delay(self.retry_deadline)

        async for attempt in AsyncRetrying(
            retry=retry_condition(self.max_retry_after),
            wait=calculate_wait,
            stop=stop,
            reraise=True,
        ):
            with attempt:
                return await call()

        raise RuntimeError("Unexpected retry state")

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

        try:
            return method.response_parser.parse(
                method,
                response,
                self.json_loads,
                self.error_resolver,
            )
        except (ValueError, UnicodeError) as exc:
            raise APIResponseError(
                "Failed to parse API response",
                status=response.status,
            ) from exc

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
            try:
                body = await response.read()
            finally:
                await response.aclose()

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
