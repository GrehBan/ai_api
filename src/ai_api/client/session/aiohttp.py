"""Session implementation on top of ``aiohttp``."""

import asyncio
import ssl
from collections.abc import AsyncIterator, Generator, Mapping
from contextlib import contextmanager
from typing import Any, Final, Self

import certifi
from aiohttp import (
    ClientConnectionError,
    ClientHandlerType,
    ClientMiddlewareType,
    ClientPayloadError,
    ClientRequest,
    ClientResponse,
    ClientResponseError,
    ClientSession,
    ClientSSLError,
    ClientTimeout,
    ServerFingerprintMismatch,
    TCPConnector,
    TooManyRedirects,
)
from yarl import URL

from ai_api.client.api.protocols import IAPI
from ai_api.client.auth.protocols import IAuth
from ai_api.client.exceptions.errror_resolver.protocols import IErrorResolver
from ai_api.client.exceptions.transport import (
    APIConnectionError,
    APIRedirectError,
    APIRequestError,
    APIResponseError,
    APITimeoutError,
    APITLSError,
)
from ai_api.client.headers.names import (
    LOCATION,
    SERVER_SOFTWARE,
    USER_AGENT,
)
from ai_api.client.headers.protocols import IHeaders
from ai_api.client.response.base import BufferedResponse
from ai_api.client.response.protocols import IStreamResponse
from ai_api.client.retry.startegy import (
    DEFAULT_MAX_RETRY_AFTER,
    DEFAULT_RETRY_DEADLINE,
)
from ai_api.client.serialization.json import JsonDumps, JsonLoads, dumps, loads
from ai_api.client.session.base import BaseSession
from ai_api.client.session.builders.aiohttp import AiohttpBodyBuilder
from ai_api.methods.protocols import IAPIMethod, IBaseMethod, IStreamMethod

DEFAULT_TIMEOUT: Final[ClientTimeout] = ClientTimeout(
    total=600,
    sock_connect=30,
)
"""Default timeout of buffered requests."""

DEFAULT_STREAM_TIMEOUT: Final[ClientTimeout] = ClientTimeout(
    total=None,
    sock_connect=30,
    sock_read=600,
)
"""Default timeout of streamed requests."""

REDIRECT_STATUSES: Final[frozenset[int]] = frozenset({301, 302, 303, 307, 308})
"""Statuses that redirect the request to the ``Location`` header."""


def _find_request_error(exc: BaseException) -> APIRequestError | None:
    """Find a failed upload source among the causes of an exception.

    Args:
        exc: The exception raised by aiohttp.

    Returns:
        The ``APIRequestError`` raised while reading an upload, if any.
    """
    seen: set[int] = set()
    current = exc.__cause__ or exc.__context__

    while current is not None and id(current) not in seen:
        if isinstance(current, APIRequestError):
            return current

        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return None


def _same_origin(url: URL) -> ClientMiddlewareType:
    """Build a middleware that refuses redirects to another origin.

    aiohttp drops only ``Authorization`` on a cross-origin redirect, so
    other secret headers (``x-api-key``, default headers) would reach
    the other host. The middleware runs before each hop is sent.

    Args:
        url: The URL of the original request.

    Returns:
        The middleware for ``ClientSession.request``.
    """
    origin = url.origin()

    async def middleware(
        request: ClientRequest,
        handler: ClientHandlerType,
    ) -> ClientResponse:
        """Send the hop only if it stays on the original origin.

        Args:
            request: The request of this hop.
            handler: Sends the request.

        Returns:
            The response of this hop.

        Raises:
            APIRedirectError: If the hop leaves the original origin.
        """
        if request.url.origin() != origin:
            raise APIRedirectError(
                f"Redirect to another origin is not followed: {request.url}"
            )

        return await handler(request)

    return middleware


@contextmanager
def _transport_errors() -> Generator[None]:
    """Translate aiohttp exceptions raised in the block into library ones.

    Yields:
        Control to the wrapped block.

    Raises:
        APITimeoutError: On any timeout.
        APITLSError: If TLS verification fails.
        APIRedirectError: If the API redirects too many times.
        APIRequestError: If an upload source fails while it is sent.
        APIConnectionError: On other connection failures.
        APIResponseError: If the response is not valid HTTP.
        APIRequestError: If aiohttp rejects a header or the URL.
    """
    try:
        yield
    except TimeoutError as exc:
        raise APITimeoutError() from exc

    except (
        ClientSSLError,
        ServerFingerprintMismatch,
    ) as exc:
        raise APITLSError() from exc

    except TooManyRedirects as exc:
        raise APIRedirectError() from exc

    except ClientConnectionError as exc:
        # aiohttp reports a failing upload as a connection error, which
        # would be retried: re-raise the source's own error instead.
        if (source := _find_request_error(exc)) is not None:
            raise source from source.__cause__

        raise APIConnectionError() from exc

    except ClientResponseError as exc:
        raise APIResponseError(
            "Invalid HTTP response",
            status=exc.status or None,
        ) from exc

    # Last: certificate errors are ValueErrors too.
    except ValueError as exc:
        raise APIRequestError(f"Invalid request: {exc}") from exc


async def _read_body(response: ClientResponse, limit: int | None) -> bytes:
    """Read a whole body, refusing to buffer more than ``limit`` bytes.

    Args:
        response: The response to read.
        limit: Largest body accepted, in bytes; ``None`` for no limit.

    Returns:
        The body.

    Raises:
        APIResponseError: If the body is larger than ``limit``.
    """
    if limit is None:
        return await response.read()

    too_large = APIResponseError(
        f"Response body exceeds {limit} bytes",
        status=response.status,
    )

    if (response.content_length or 0) > limit:
        raise too_large

    body = bytearray()

    async for chunk in response.content.iter_any():
        body += chunk

        if len(body) > limit:
            raise too_large

    return bytes(body)


class _AiohttpStreamResponse:
    """``IStreamResponse`` over an open ``aiohttp.ClientResponse``."""

    def __init__(
        self,
        response: ClientResponse,
        max_body_size: int | None = None,
    ) -> None:
        """Wrap a response whose body has not been read.

        Args:
            response: The open aiohttp response.
            max_body_size: Largest body ``read()`` accepts, in bytes;
                ``None`` for no limit.
        """
        self._response = response
        self._headers = response.headers.copy()
        self._max_body_size = max_body_size

    @property
    def status(self) -> int:
        """HTTP status code."""
        return self._response.status

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""
        return self._headers

    @property
    def charset(self) -> str | None:
        """Charset from the ``Content-Type`` header, if any."""
        return self._response.charset

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        """Yield the body as soon as data arrives.

        Yields:
            Chunks of the body, of arbitrary size.

        Raises:
            APITransportError: If the connection fails mid-stream.
            APIResponseError: If the body ends before it is complete.
        """
        try:
            with _transport_errors():
                async for chunk in self._response.content.iter_any():
                    yield chunk
        except ClientPayloadError as exc:
            raise APIResponseError(
                "HTTP response stream ended unexpectedly",
                status=self.status,
            ) from exc

    async def read(self) -> bytes:
        """Read the whole remaining body.

        Returns:
            The body.

        Raises:
            APITransportError: If the connection fails.
            APIResponseError: If the body ends before it is complete or
                is larger than the limit.
        """
        try:
            with _transport_errors():
                return await _read_body(self._response, self._max_body_size)
        except ClientPayloadError as exc:
            raise APIResponseError(
                "Failed to read HTTP response body",
                status=self.status,
            ) from exc

    async def aclose(self) -> None:
        """Release the connection.

        A partially read body makes aiohttp close the connection instead
        of returning it to the pool.
        """
        self._response.release()


class AiohttpSession(BaseSession):
    """Session that sends requests with ``aiohttp``.

    The session is bound to one event loop: the running loop at creation,
    or ``loop``. Create it inside a coroutine (for example, in ``main()``)
    and use it only from that loop. The underlying ``ClientSession`` is
    created lazily on first use and re-created after ``close()``. Use it
    as an async context manager to open and close it deterministically;
    close it before its event loop ends.

    Example:
        >>> async with AiohttpSession(api, BearerAuth(key), resolver) as s:
        ...     pong = await s.send_request(Ping())
    """

    def __init__(
        self,
        api: IAPI,
        auth: IAuth,
        error_resolver: IErrorResolver,
        headers: IHeaders | None = None,
        json_loads: JsonLoads = loads,
        json_dumps: JsonDumps = dumps,
        limit: int = 100,
        timeout: ClientTimeout = DEFAULT_TIMEOUT,
        stream_timeout: ClientTimeout = DEFAULT_STREAM_TIMEOUT,
        *,
        max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
        retry_deadline: float | None = DEFAULT_RETRY_DEADLINE,
        max_body_size: int | None = None,
        trust_env: bool = False,
        follow_redirects: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Configure the session; no connection is opened yet.

        Args:
            api: Builds request URLs.
            auth: Produces the authentication header.
            error_resolver: Turns error responses into exceptions.
            headers: Default headers sent with every request.
            json_loads: JSON decoder for response bodies.
            json_dumps: JSON encoder for request bodies.
            limit: Maximum number of simultaneous connections.
            timeout: Timeout of buffered requests. aiohttp's own default
                caps a request at 5 minutes, which long model generations
                exceed; 10 minutes matches the official AI SDKs.
            stream_timeout: Timeout of streamed requests. It has no total
                limit, because a stream may last indefinitely; only the
                silence between chunks (``sock_read``) is limited.
            max_retry_after: Longest ``Retry-After`` to wait for, in
                seconds.
            retry_deadline: Seconds since the first attempt after which
                no retry starts; ``None`` for no limit.
            max_body_size: Largest buffered response or error body
                accepted, in bytes; ``None`` for no limit. Streamed
                bodies are not limited.
            trust_env: Whether to use proxy settings and ``.netrc``
                credentials from the environment.
            follow_redirects: Whether to follow redirects within the
                origin of the request. Redirects to another origin are
                never followed, so no header leaks to another host;
                a redirect that is not followed raises
                ``APIRedirectError``.
            loop: Event loop the session is bound to; ``None`` for the
                running loop.

        Raises:
            RuntimeError: If ``loop`` is ``None`` and no event loop is
                running.
        """
        super().__init__(
            api,
            auth,
            error_resolver,
            headers,
            json_loads,
            json_dumps,
            max_retry_after=max_retry_after,
            retry_deadline=retry_deadline,
        )

        self._connector_type: type[TCPConnector] = TCPConnector
        self._connector_init: dict[str, Any] = {
            "ssl": ssl.create_default_context(cafile=certifi.where()),
            "limit": limit,
            "ttl_dns_cache": 3600,
        }

        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._max_body_size = max_body_size
        self._trust_env = trust_env
        self._follow_redirects = follow_redirects
        self._session: ClientSession | None = None

        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._body_builder = AiohttpBodyBuilder(json_dumps=self.json_dumps)

    async def __aenter__(self) -> Self:
        """Open the underlying ``ClientSession``.

        Returns:
            The session itself.
        """
        await self.create_session()
        return self

    async def close(self) -> None:
        """Close the underlying ``ClientSession`` and its connections.

        Raises:
            RuntimeError: If called from another event loop.
        """
        self._check_loop()
        session = self._session
        self._session = None

        if session is not None and not session.closed:
            await session.close()
            await asyncio.sleep(0.25)

    async def create_session(self) -> ClientSession:
        """Return the ``ClientSession``, creating it if needed.

        Returns:
            An open ``ClientSession`` with the default headers set.

        Raises:
            RuntimeError: If called from another event loop.
        """
        self._check_loop()

        if self._session is None or self._session.closed:
            self._session = ClientSession(
                connector=self._connector_type(**self._connector_init),
                json_serialize_bytes=self.json_dumps,
                timeout=self._timeout,
                headers=self.headers.build({USER_AGENT: SERVER_SOFTWARE}),
                trust_env=self._trust_env,
                loop=self._loop,
            )

        return self._session

    def _check_loop(self) -> None:
        """Ensure the session is used from the loop it is bound to.

        A ``ClientSession`` and its connections belong to one event loop;
        used from another, aiohttp fails with unclear errors such as
        ``Event loop is closed``.

        Raises:
            RuntimeError: If the running loop is not the session's loop.
        """
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError(
                "AiohttpSession is bound to another event loop; create "
                "one session per loop, inside a coroutine"
            )

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
            APIResponseError: If the body ends before it is complete or
                is larger than ``max_body_size``.
        """
        response = await self._request(method)

        async with response:
            try:
                with _transport_errors():
                    raw = await _read_body(response, self._max_body_size)
            except ClientPayloadError as exc:
                raise APIResponseError(
                    "Failed to read HTTP response body",
                    status=response.status,
                ) from exc

            return BufferedResponse(
                status=response.status,
                headers=response.headers.copy(),
                body=raw,
                charset=response.charset,
            )

    async def _open_stream(
        self,
        method: IStreamMethod[Any],
    ) -> IStreamResponse:
        """Send the request with ``stream_timeout`` and leave the body unread.

        Args:
            method: The stream method to send.

        Returns:
            The open response; the caller closes it.

        Raises:
            APITransportError: If the request could not be sent.
        """
        response = await self._request(
            method,
            timeout=self._stream_timeout,
        )
        return _AiohttpStreamResponse(response, self._max_body_size)

    async def _request(
        self,
        method: IBaseMethod[Any],
        **kwargs: Any,
    ) -> ClientResponse:
        """Send the HTTP request for a method.

        The authentication header is produced for every request, so an
        ``IAuth`` that rotates its token takes effect immediately.

        Args:
            method: The method whose path, query and body are sent.
            **kwargs: Extra arguments for ``ClientSession.request``, such
                as ``timeout``.

        Returns:
            The response with its body not yet read.

        Raises:
            APITransportError: If the request could not be sent.
            APIRedirectError: If the response is a redirect that is not
                followed.
            APIRequestError: If a redirect needs a body that cannot be
                sent again.
        """
        session = await self.create_session()

        url = self.api.build_url(
            method.request_path,
            method.get_query_map(),
        )
        body = self._body_builder.build(
            method.get_body_map(),
            method.body_format,
        )

        redirects: dict[str, Any] = (
            {"middlewares": (_same_origin(URL(url)),)}
            if self._follow_redirects
            else {"allow_redirects": False}
        )

        try:
            with _transport_errors():
                response = await session.request(
                    method=method.request_method,
                    url=url,
                    headers={self.auth.header_name: self.auth.encode()},
                    **redirects,
                    **body,
                    **kwargs,
                )
        except ClientPayloadError as exc:
            # A 307/308 needs the body again, and a stream was consumed.
            raise APIRequestError(
                f"Cannot follow the redirect: {exc}"
            ) from exc

        if response.status in REDIRECT_STATUSES:
            location = response.headers.get(LOCATION, "")
            response.release()
            raise APIRedirectError(f"Redirect is not followed: {location}")

        return response
