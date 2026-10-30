"""Session implementation on top of ``aiohttp``."""

from __future__ import annotations

import asyncio
import ssl
from typing import Any, Final, Self

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

from ai_api.http.exceptions.aiohttp import body_errors, transport_errors
from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.exceptions.transport import (
    APIRedirectError,
    APIRequestError,
)
from ai_api.http.headers import (
    LOCATION,
    SERVER_SOFTWARE,
    USER_AGENT,
)
from ai_api.http.protocols import IAPI, IAuth, IHeaders
from ai_api.methods.protocols import IAPIMethod, IBaseMethod, IStreamMethod
from ai_api.response.aiohttp import AiohttpStreamResponse, read_body
from ai_api.response.buffered import BufferedResponse
from ai_api.response.protocols import IStreamResponse
from ai_api.serialization.json import JsonDumps, JsonLoads, dumps, loads
from ai_api.session.base import BaseSession
from ai_api.session.body.aiohttp import AiohttpBodyBuilder
from ai_api.session.loop import bind_loop
from ai_api.session.retry import (
    DEFAULT_MAX_RETRY_AFTER,
    DEFAULT_RETRY_DEADLINE,
)

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
        headers: IHeaders | None = None,
        json_loads: JsonLoads = loads,
        json_dumps: JsonDumps = dumps,
        max_connections: int = 100,
        timeout: ClientTimeout = DEFAULT_TIMEOUT,
        stream_timeout: ClientTimeout = DEFAULT_STREAM_TIMEOUT,
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
            max_connections: Maximum number of simultaneous connections.
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
            max_retry_after=max_retry_after,
            retry_deadline=retry_deadline,
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

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop the session is bound to; ``None`` until first use.

        Reading it never binds a loop.
        """
        return self._loop

    async def get_client(self) -> ClientSession:
        """Return the ``ClientSession``, creating it if needed.

        Returns:
            An open ``ClientSession`` with the default headers set.

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
                headers=self.headers.build({USER_AGENT: SERVER_SOFTWARE}),
                trust_env=self._trust_env,
                loop=self._loop,
            )

        return self._client

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
            APITransportError: If the request could not be completed,
                including a body that ends before it is complete.
            APIResponseError: If the body is larger than
                ``max_body_size``.
        """
        response = await self._request(method)

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
        return AiohttpStreamResponse(response, self._max_body_size)

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
        client = await self.get_client()

        url = self.api.build_url(
            method.request_path,
            method.get_query_map(),
        )
        body = self._body_builder.build(
            method.get_body_map(),
            method.request_body_format,
        )

        redirects: dict[str, Any] = (
            {"middlewares": (_same_origin(URL(url)),)}
            if self._follow_redirects
            else {"allow_redirects": False}
        )

        try:
            with transport_errors():
                response = await client.request(
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
