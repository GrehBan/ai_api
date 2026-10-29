"""Session implementation on top of ``aiohttp``."""

import asyncio
import ssl
from collections.abc import AsyncIterator, Generator, Mapping
from contextlib import contextmanager
from typing import Any, Final, Self

import certifi
from aiohttp import (
    ClientConnectionError,
    ClientPayloadError,
    ClientResponse,
    ClientSession,
    ClientSSLError,
    ClientTimeout,
    ServerFingerprintMismatch,
    TCPConnector,
)

from ai_api.client.api.protocols import IAPI
from ai_api.client.auth.protocols import IAuth
from ai_api.client.exceptions.errror_resolver.protocols import IErrorResolver
from ai_api.client.exceptions.transport import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    APITLSError,
)
from ai_api.client.headers.names import (
    AUTHORIZATION,
    SERVER_SOFTWARE,
    USER_AGENT,
)
from ai_api.client.headers.protocols import IHeaders
from ai_api.client.response.base import BufferedResponse
from ai_api.client.response.protocols import IStreamResponse
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


@contextmanager
def _transport_errors() -> Generator[None]:
    """Translate aiohttp exceptions raised in the block into library ones.

    Yields:
        Control to the wrapped block.

    Raises:
        APITimeoutError: On any timeout.
        APITLSError: If TLS verification fails.
        APIConnectionError: On other connection failures.
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

    except ClientConnectionError as exc:
        raise APIConnectionError() from exc


class _AiohttpStreamResponse:
    """``IStreamResponse`` over an open ``aiohttp.ClientResponse``."""

    def __init__(self, response: ClientResponse) -> None:
        """Wrap a response whose body has not been read.

        Args:
            response: The open aiohttp response.
        """
        self._response = response
        self._headers = response.headers.copy()

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
            APIResponseError: If the body ends before it is complete.
        """
        try:
            with _transport_errors():
                return await self._response.read()
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

    The underlying ``ClientSession`` is created lazily on first use and
    re-created after ``close()``. Use it as an async context manager to
    open and close it deterministically.

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
    ) -> None:
        """Configure the session; no connection is opened yet.

        Args:
            api: Builds request URLs.
            auth: Produces the ``Authorization`` header.
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
        """
        super().__init__(
            api,
            auth,
            error_resolver,
            headers,
            json_loads,
            json_dumps,
        )

        self._connector_type: type[TCPConnector] = TCPConnector
        self._connector_init: dict[str, Any] = {
            "ssl": ssl.create_default_context(
                cafile=certifi.where()
            ),
            "limit": limit,
            "ttl_dns_cache": 3600,
        }

        self._timeout = timeout
        self._stream_timeout = stream_timeout
        self._session: ClientSession | None = None
        self._body_builder = AiohttpBodyBuilder(
            json_dumps=self.json_dumps
        )

    async def __aenter__(self) -> Self:
        """Open the underlying ``ClientSession``.

        Returns:
            The session itself.
        """
        await self.create_session()
        return self

    async def close(self) -> None:
        """Close the underlying ``ClientSession`` and its connections."""
        session = self._session
        self._session = None

        if session is not None and not session.closed:
            await session.close()
            await asyncio.sleep(0.25)

    async def create_session(self) -> ClientSession:
        """Return the ``ClientSession``, creating it if needed.

        Returns:
            An open ``ClientSession`` with the default headers set.
        """
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                connector=self._connector_type(
                    **self._connector_init
                ),
                json_serialize_bytes=self.json_dumps,
                timeout=self._timeout,
                headers=self.headers.build(
                    {
                        USER_AGENT: SERVER_SOFTWARE,
                        AUTHORIZATION: self.auth.encode(),
                    }
                ),
            )

        return self._session

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
            APIResponseError: If the body ends before it is complete.
        """
        response = await self._request(method)

        async with response:
            try:
                with _transport_errors():
                    raw = await response.read()
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
        return _AiohttpStreamResponse(response)

    async def _request(
        self,
        method: IBaseMethod[Any],
        **kwargs: Any,
    ) -> ClientResponse:
        """Send the HTTP request for a method.

        Args:
            method: The method whose path, query and body are sent.
            **kwargs: Extra arguments for ``ClientSession.request``, such
                as ``timeout``.

        Returns:
            The response with its body not yet read.

        Raises:
            APITransportError: If the request could not be sent.
        """
        session = await self.create_session()

        url = self.api.build_url(
            method.request_path,
            method.get_query_map(),
        )

        with _transport_errors():
            return await session.request(
                method=method.request_method,
                url=url,
                **self._body_builder.build(
                    method.get_body_map(),
                    method.body_format,
                ),
                **kwargs,
            )
