"""Transport-agnostic session logic: retries, errors and parsing."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, NoReturn, Self

from tenacity import AsyncRetrying, stop_after_attempt

from ai_api.client.api.protocols import IAPI
from ai_api.client.auth.protocols import IAuth
from ai_api.client.exceptions.errror_resolver.protocols import IErrorResolver
from ai_api.client.exceptions.transport import APIResponseError
from ai_api.client.headers.headers import Headers
from ai_api.client.headers.protocols import IHeaders
from ai_api.client.response.base import BufferedResponse
from ai_api.client.response.protocols import IStreamResponse
from ai_api.client.response.stream import Stream
from ai_api.client.retry.startegy import RETRY_CONDITION, calculate_wait
from ai_api.client.serialization.json import JsonDumps, JsonLoads, dumps, loads
from ai_api.methods.protocols import IAPIMethod, IStreamMethod

if TYPE_CHECKING:
    from types import TracebackType


class BaseSession(ABC):
    """Base class for sessions; a transport only does the HTTP I/O.

    Retries, turning error statuses into ``APIError`` and parsing are done
    here, so every transport behaves the same. A subclass implements
    ``_execute``, ``_open_stream`` and ``close``.

    Attributes:
        api: Builds request URLs.
        auth: Produces the ``Authorization`` header.
        error_resolver: Turns error responses into exceptions.
        headers: Default headers sent with every request.
        json_loads: JSON decoder for response bodies.
        json_dumps: JSON encoder for request bodies.
    """

    def __init__(
        self,
        api: IAPI,
        auth: IAuth,
        error_resolver: IErrorResolver,
        headers: IHeaders | None = None,
        json_loads: JsonLoads = loads,
        json_dumps: JsonDumps = dumps
    ) -> None:
        """Store the session's collaborators.

        Args:
            api: Builds request URLs.
            auth: Produces the ``Authorization`` header.
            error_resolver: Turns error responses into exceptions.
            headers: Default headers; an empty set is used when ``None``.
            json_loads: JSON decoder for response bodies.
            json_dumps: JSON encoder for request bodies.
        """
        self.api = api
        self.auth = auth
        self.error_resolver = error_resolver
        self.headers = headers or Headers()
        self.json_loads = json_loads
        self.json_dumps = json_dumps

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
    ) -> T:
        """Send a method and return its parsed response.

        If ``method.retryable`` is true, transient failures are retried up
        to four attempts in total.

        Args:
            method: The method to send.

        Returns:
            The parsed response, typed by the method's generic argument.

        Raises:
            APIError: If the API answered with an error status.
            APITransportError: If the request could not be completed.
            APIResponseError: If the response body could not be parsed.
        """
        return await self._with_retry(
            method.retryable,
            lambda: self._send_once(method),
        )

    @asynccontextmanager
    async def stream[T](
        self,
        method: IStreamMethod[T],
    ) -> AsyncGenerator[Stream[T]]:
        """Send a method and stream its response.

        Only opening the stream is retried: once the block has started, a
        failure is raised to the caller instead of resending the request.
        The connection is released when the block exits, including on
        ``break`` and exceptions.

        Args:
            method: The stream method to send.

        Yields:
            The stream of items, typed by the method's generic argument.

        Raises:
            APIError: If the API answered with an error status; raised on
                entering the block.
            APITransportError: If the connection fails, either on entering
                the block or while iterating.
            APIResponseError: If the body ends early or cannot be decoded.

        Example:
            >>> async with session.stream(Download()) as stream:
            ...     await stream.save("out.bin")
        """
        response = await self._with_retry(
            method.retryable,
            lambda: self._open_stream_once(method),
        )

        try:
            stream = Stream(
                method.stream_parser.parse(
                    method,
                    response,
                    self.json_loads,
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

        async for attempt in AsyncRetrying(
            retry=RETRY_CONDITION,
            wait=calculate_wait,
            stop=stop_after_attempt(4),
            reraise=True,
        ):
            with attempt:
                return await call()

        raise RuntimeError("Unexpected retry state")

    async def _send_once[T](
        self,
        method: IAPIMethod[T],
    ) -> T:
        """Perform one attempt of ``send_request``.

        Args:
            method: The method to send.

        Returns:
            The parsed response.
        """
        # Status check stays inside the retried unit: retries are driven
        # by the transient flag of the APIError raised here.
        buffered = await self._execute(method)
        self._raise_for_status(buffered)

        try:
            return method.response_parser.parse(
                method,
                buffered,
                self.json_loads,
            )
        except (ValueError, UnicodeError) as exc:
            raise APIResponseError(
                "Failed to parse API response",
                status=buffered.status,
            ) from exc

    async def _open_stream_once(
        self,
        method: IStreamMethod[Any],
    ) -> IStreamResponse:
        """Perform one attempt of opening a stream.

        An error response is read, closed and raised here, so it is still
        inside the retry loop.

        Args:
            method: The stream method to send.

        Returns:
            The open response with a status below 400.
        """
        response = await self._open_stream(method)

        if response.status >= HTTPStatus.BAD_REQUEST:
            try:
                body = await response.read()
            finally:
                await response.aclose()

            self._raise_api_error(
                BufferedResponse(
                    status=response.status,
                    headers=response.headers,
                    body=body,
                    charset=response.charset,
                )
            )

        return response

    def _raise_for_status(self, buffered: BufferedResponse) -> None:
        """Raise the API error if the response has an error status.

        Args:
            buffered: The response to check.
        """
        if buffered.status >= HTTPStatus.BAD_REQUEST:
            self._raise_api_error(buffered)

    def _raise_api_error(self, buffered: BufferedResponse) -> NoReturn:
        """Raise the error the resolver builds for an error response.

        Args:
            buffered: The error response.

        Raises:
            APIError: Always.
        """
        try:
            body = self.json_loads(buffered.body)
        except (ValueError, TypeError):
            body = None

        raise self.error_resolver.resolve(
            status=buffered.status,
            body=body,
            headers=buffered.headers,
        )

    @abstractmethod
    async def _execute(
        self,
        method: IAPIMethod[Any],
    ) -> BufferedResponse:
        """Send the request and read the whole body.

        Args:
            method: The method to send.

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
    ) -> IStreamResponse:
        """Send the request without reading the body.

        Args:
            method: The stream method to send.

        Returns:
            The open response, including responses with an error status.
            The caller is responsible for closing it.

        Raises:
            APITransportError: If the request could not be sent.
        """
        raise NotImplementedError
