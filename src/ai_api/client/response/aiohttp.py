"""``IStreamResponse`` and body reading over ``aiohttp`` responses."""

from collections.abc import AsyncIterator, Mapping

from aiohttp import ClientPayloadError, ClientResponse

from ai_api.client.exceptions.aiohttp import transport_errors
from ai_api.client.exceptions.transport import APIResponseError


async def read_body(response: ClientResponse, limit: int | None) -> bytes:
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


class AiohttpStreamResponse:
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
            with transport_errors():
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
            with transport_errors():
                return await read_body(self._response, self._max_body_size)
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
