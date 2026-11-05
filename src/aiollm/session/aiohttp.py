"""Session implementation on top of ``aiohttp``."""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
from contextlib import AsyncExitStack
from typing import Any, Final, Self, TypedDict

import certifi
from aiohttp import (
    ClientHandlerType,
    ClientMiddlewareType,
    ClientPayloadError,
    ClientRequest,
    ClientResponse,
    ClientSession,
    ClientTimeout,
    TCPConnector,
)
from yarl import URL

from aiollm.http.exceptions.aiohttp import body_errors, transport_errors
from aiollm.http.exceptions.protocols import IErrorResolver
from aiollm.http.exceptions.transport import (
    APIRedirectError,
    APIRequestError,
)
from aiollm.http.headers import (
    LOCATION,
    SERVER_SOFTWARE,
    USER_AGENT,
)
from aiollm.http.protocols import IAPI, HeadersType, IAuth
from aiollm.methods.protocols import IAPIMethod, IBaseMethod, IStreamMethod
from aiollm.response.aiohttp import (
    AiohttpStreamResponse,
    read_body,
    wait_for_writer,
)
from aiollm.response.buffered import BufferedResponse
from aiollm.response.protocols import IStreamResponse
from aiollm.serialization.json import JsonDumps, JsonLoads, dumps, loads
from aiollm.session.base import BaseSession
from aiollm.session.body.aiohttp import AiohttpBodyBuilder
from aiollm.session.loop import bind_loop
from aiollm.session.protocols import Timeout
from aiollm.session.retry.policy import DEFAULT_RETRY_POLICY, RetryPolicy

logger = logging.getLogger(__name__)

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

_CLOSE_DELAY: Final[float] = 0.25
"""Seconds to wait after closing, so aiohttp can close the SSL transports."""

_DNS_CACHE_TTL: Final[int] = 3600
"""Seconds a resolved host name is cached."""


def _replace_timeout(
    timeout: ClientTimeout,
    *,
    total: float | None = None,
    sock_read: float | None = None,
) -> ClientTimeout:
    """Copy a timeout with its total or its read limit replaced.

    Args:
        timeout: The timeout to copy.
        total: New total limit, in seconds; ``None`` keeps the old one.
        sock_read: New limit of the silence between two chunks, in
            seconds; ``None`` keeps the old one.

    Returns:
        A new timeout; the other limits are those of ``timeout``.
    """
    return ClientTimeout(
        total=timeout.total if total is None else total,
        connect=timeout.connect,
        sock_read=timeout.sock_read if sock_read is None else sock_read,
        sock_connect=timeout.sock_connect,
        ceil_threshold=timeout.ceil_threshold,
    )


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

        if request.url != url:
            # Not the target: the server's path may name a user's data.
            logger.debug("Following a redirect")

        return await handler(request)

    return middleware


class AiohttpSessionOptions(TypedDict, total=False):
    """The optional arguments of ``AiohttpSession``, for ``**kwargs``.

    A provider client takes them as ``**session_kwargs`` and passes them
    on, so type checkers check them at the client's call. Each key is a
    keyword-only argument of ``AiohttpSession.__init__`` with the same
    type; a test keeps the two in step.

    Attributes:
        headers: Default headers sent with every request.
        json_loads: JSON decoder for response bodies.
        json_dumps: JSON encoder for request bodies.
        max_connections: Maximum number of simultaneous connections.
        timeout: Timeout of buffered requests.
        stream_timeout: Timeout of streamed requests.
        retry_policy: When and how long a retryable request is retried.
        max_body_size: Largest buffered body accepted, in bytes.
        trust_env: Whether to use proxy settings from the environment.
        follow_redirects: Whether to follow same-origin redirects.
        loop: The event loop the session is bound to.
    """

    headers: HeadersType | None
    json_loads: JsonLoads
    json_dumps: JsonDumps
    max_connections: int
    timeout: ClientTimeout
    stream_timeout: ClientTimeout
    retry_policy: RetryPolicy
    max_body_size: int | None
    trust_env: bool
    follow_redirects: bool
    loop: asyncio.AbstractEventLoop | None


class AiohttpSession(BaseSession):
    """Session that sends requests with ``aiohttp``.

    The session is bound to one event loop: ``loop`` if given, otherwise
    the loop of its first use. It can be created anywhere, even outside a
    coroutine, but is then usable only from that loop. The underlying
    ``ClientSession`` is created lazily on first use and re-created after
    ``close()``. Use it as an async context manager to open and close it
    deterministically; close it before its event loop ends.

    Example:
        >>> async with AiohttpSession(api, BearerAuth(key), resolver) as s:
        ...     pong = await s.send_request(Ping())
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
        max_connections: int = 100,
        timeout: ClientTimeout = DEFAULT_TIMEOUT,
        stream_timeout: ClientTimeout = DEFAULT_STREAM_TIMEOUT,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
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
            max_connections: Maximum number of simultaneous connections.
            timeout: Timeout of buffered requests. aiohttp's own default
                caps a request at 5 minutes, which long model generations
                exceed; 10 minutes matches the official AI SDKs.
            stream_timeout: Timeout of streamed requests. It has no total
                limit, because a stream may last indefinitely; only the
                silence between chunks (``sock_read``) is limited.
            retry_policy: When and how long a retryable request is
                retried; see ``RetryPolicy``.
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
            loop: Event loop the session is bound to; ``None`` to bind
                the loop of its first use.
        """
        super().__init__(
            api,
            auth,
            error_resolver,
            headers=headers,
            json_loads=json_loads,
            json_dumps=json_dumps,
            retry_policy=retry_policy,
        )

        self._connector_kwargs: dict[str, Any] = {
            "ssl": ssl.create_default_context(cafile=certifi.where()),
            "limit": max_connections,
            "ttl_dns_cache": _DNS_CACHE_TTL,
        }

        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._max_body_size = max_body_size
        self._trust_env = trust_env
        self._follow_redirects = follow_redirects
        self._client: ClientSession | None = None
        self._loop = loop
        self._body_builder = AiohttpBodyBuilder(json_dumps=self.json_dumps)

    async def __aenter__(self) -> Self:
        """Open the underlying ``ClientSession``.

        Returns:
            The session itself.
        """
        await self.get_client()
        return self

    async def close(self) -> None:
        """Close the underlying ``ClientSession`` and its connections.

        Raises:
            RuntimeError: If called from another event loop.
        """
        self._loop = bind_loop(self._loop, type(self).__name__)
        client = self._client
        self._client = None

        if client is not None and not client.closed:
            await client.close()
            await asyncio.sleep(_CLOSE_DELAY)
            logger.debug("Closed the connection pool of %s", self.api.base_url)

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop the session is bound to; ``None`` until first use.

        Reading it never binds a loop.
        """
        return self._loop

    async def get_client(self) -> ClientSession:
        """Return the ``ClientSession``, creating it if needed.

        Returns:
            An open ``ClientSession``. It holds no headers: they are
            built for every request, so later changes take effect.

        Raises:
            RuntimeError: If called from another event loop.
        """
        # Checked before the connector is created, so a wrong loop does
        # not leave an unclosed connector behind.
        self._loop = bind_loop(self._loop, type(self).__name__)

        if self._client is None or self._client.closed:
            self._client = ClientSession(
                connector=TCPConnector(**self._connector_kwargs),
                json_serialize_bytes=self.json_dumps,
                timeout=self._timeout,
                trust_env=self._trust_env,
                loop=self._loop,
            )
            logger.debug("Opened a connection pool for %s", self.api.base_url)

        return self._client

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
            headers: Headers for this request only.
            timeout: Total timeout of this request, in seconds, in place
                of the total of ``timeout``; its other limits are kept.

        Returns:
            The response, including responses with an error status.

        Raises:
            APITransportError: If the request could not be completed,
                including a body that ends before it is complete.
            APIResponseError: If the body is larger than
                ``max_body_size``.
        """
        # Closed after the response: the upload readers of this attempt
        # are closed before a retry opens the files again.
        async with AsyncExitStack() as uploads:
            response = await self._request(
                method,
                headers=headers,
                timeout=(
                    self._timeout
                    if timeout is None
                    else _replace_timeout(self._timeout, total=timeout)
                ),
                uploads=uploads,
            )

            async with response:
                with body_errors():
                    raw = await read_body(response, self._max_body_size)

                return BufferedResponse(
                    status=response.status,
                    headers=response.headers.copy(),
                    body=raw,
                    charset=response.charset,
                )

    async def _open_stream(
        self,
        method: IStreamMethod[Any],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
    ) -> IStreamResponse:
        """Send the request with ``stream_timeout`` and leave the body unread.

        Args:
            method: The stream method to send.
            headers: Headers for this request only.
            timeout: Longest silence between two chunks, in seconds, in
                place of the ``sock_read`` of ``stream_timeout``; its
                other limits are kept.

        Returns:
            The open response; the caller closes it.

        Raises:
            APITransportError: If the request could not be sent.
        """
        # Closed with the stream, or here if the request fails.
        uploads = AsyncExitStack()

        try:
            response = await self._request(
                method,
                headers=headers,
                timeout=(
                    self._stream_timeout
                    if timeout is None
                    else _replace_timeout(
                        self._stream_timeout, sock_read=timeout
                    )
                ),
                uploads=uploads,
            )
        except BaseException as exc:
            # The readers see why the attempt ended, so a failing close
            # is noted on that exception instead of replacing it.
            await uploads.__aexit__(type(exc), exc, exc.__traceback__)
            raise

        return AiohttpStreamResponse(
            response, self._max_body_size, uploads=uploads
        )

    async def _request(
        self,
        method: IBaseMethod[Any],
        *,
        headers: HeadersType | None,
        timeout: ClientTimeout,
        uploads: AsyncExitStack,
    ) -> ClientResponse:
        """Send the HTTP request for a method.

        Every request gets the session's headers as they are now (with
        the library's ``User-Agent``), so a header added later is sent;
        the method's headers override them, the headers of the call
        override those, and the authentication header, produced for every
        request so that an ``IAuth`` that rotates its token takes effect
        immediately, overrides all of them.

        Args:
            method: The method whose path, query, headers and body are
                sent.
            headers: Headers for this request only.
            timeout: Timeout of this request.
            uploads: The stack of the attempt. The upload readers are
                registered on it, and once there is a response, waiting
                for aiohttp to stop writing the body, which runs first.

        Returns:
            The response with its body not yet read.

        Raises:
            APITransportError: If the request could not be sent.
            APIRedirectError: If the response is a redirect that is not
                followed.
            APIRequestError: If a redirect needs a body that cannot be
                sent again.
        """
        client = await self.get_client()

        url = self.api.build_url(
            method.request_path,
            method.get_query_map(),
        )
        body = self._body_builder.build(
            method.get_body_map(),
            method.request_body_format,
            method.value_formatter,
            uploads=uploads,
        )

        redirects: dict[str, Any] = (
            {"middlewares": (_same_origin(URL(url)),)}
            if self._follow_redirects
            else {"allow_redirects": False}
        )

        request_headers = self.headers.build({USER_AGENT: SERVER_SOFTWARE})
        request_headers.update(method.request_headers.build(headers))
        request_headers[self.auth.header_name] = self.auth.encode()

        # The declared path only: its values may name a user's data, a
        # query may carry a key; never headers or bodies.
        logger.debug(
            "Sending %s %s",
            method.request_method,
            method.request_path_template,
        )
        started = time.perf_counter()

        try:
            with transport_errors():
                response = await client.request(
                    method=method.request_method,
                    url=url,
                    headers=request_headers,
                    timeout=timeout,
                    **redirects,
                    **body,
                )
        except ClientPayloadError as exc:
            # A 307/308 needs the body again, and a stream was consumed.
            raise APIRequestError(
                f"Cannot follow the redirect: {exc}"
            ) from exc

        uploads.push_async_callback(wait_for_writer, response)
        # A diagnostic, whatever the status: an error status is raised, and
        # so reported, a layer above. A stream's body is not read yet.
        logger.debug(
            "HTTP attempt finished: status=%d duration=%.3fs",
            response.status,
            time.perf_counter() - started,
        )

        if response.status in REDIRECT_STATUSES:
            location = response.headers.get(LOCATION, "")
            response.release()
            raise APIRedirectError(f"Redirect is not followed: {location}")

        return response
