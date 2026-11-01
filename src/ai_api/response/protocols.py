"""Interface for HTTP responses consumed as a stream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Protocol


class IStreamResponse(Protocol):
    """An HTTP response whose body has not been read yet.

    The session owns it and closes it once the stream is done; a stream
    parser only consumes ``iter_chunks()``.
    """

    @property
    def status(self) -> int:
        """HTTP status code."""
        ...

    @property
    def headers(self) -> Mapping[str, str]:
        """Response headers."""
        ...

    @property
    def charset(self) -> str | None:
        """Charset from the ``Content-Type`` header, if any."""
        ...

    def iter_chunks(self) -> AsyncIterator[bytes]:
        """Yield the body as it arrives.

        Yields:
            Chunks of the body, of arbitrary size.

        Raises:
            APITransportError: If the connection fails mid-stream or the
                body ends before it is complete (``APIConnectionError``),
                which ``Stream`` raises as ``APIStreamInterruptedError``.
        """
        ...

    async def read(self) -> bytes:
        """Read the whole remaining body.

        Returns:
            The body; used for error responses.
        """
        ...

    async def aclose(self) -> None:
        """Release the connection.

        Safe to call before the body is fully read; the connection is then
        closed instead of being reused.
        """
        ...
