"""Parser for streamed responses."""

import codecs
from collections.abc import AsyncIterator
from typing import cast

from ai_api.client.exceptions.transport import APIResponseError
from ai_api.client.response.protocols import IStreamResponse
from ai_api.client.serialization.json import JsonLoads, loads
from ai_api.methods.enums import ResponseFormat
from ai_api.methods.protocols import IStreamMethod


class StreamResponseParser:
    """Turns a streamed body into byte chunks or text pieces."""

    def __init__(self, json_loads: JsonLoads = loads) -> None:
        """Create the parser.

        Args:
            json_loads: JSON decoder used when ``parse`` is not given one.
        """
        self._json_loads = json_loads

    def parse[T](
        self,
        method: IStreamMethod[T],
        response: IStreamResponse,
        json_loads: JsonLoads | None = None
    ) -> AsyncIterator[T]:
        """Start parsing according to ``method.response_format``.

        Args:
            method: The method that was sent.
            response: The successful streamed response.
            json_loads: JSON decoder to use instead of the parser's own.

        Returns:
            Raw chunks for BYTES, decoded text pieces for TEXT.

        Raises:
            ValueError: If the format is not supported for streams.
        """
        match method.response_format:
            case ResponseFormat.BYTES:
                return cast(AsyncIterator[T], response.iter_chunks())

            case ResponseFormat.TEXT:
                return cast(AsyncIterator[T], self._iter_text(response))

            case _:
                raise ValueError(
                    "Unsupported stream response format: "
                    f"{method.response_format}"
                )

    async def _iter_text(
        self,
        response: IStreamResponse,
    ) -> AsyncIterator[str]:
        """Decode the body incrementally.

        A multi-byte character split between two chunks is held back until
        the rest of it arrives.

        Args:
            response: The streamed response.

        Yields:
            Non-empty pieces of decoded text.

        Raises:
            APIResponseError: If the charset is unknown or the body is not
                valid in it.
        """
        try:
            decoder = codecs.getincrementaldecoder(
                response.charset or "utf-8"
            )()
        except LookupError as exc:
            raise APIResponseError(
                "Unknown response charset",
                status=response.status,
            ) from exc

        try:
            async for chunk in response.iter_chunks():
                if text := decoder.decode(chunk):
                    yield text

            if text := decoder.decode(b"", final=True):
                yield text
        except UnicodeError as exc:
            raise APIResponseError(
                "Failed to decode text stream",
                status=response.status,
            ) from exc
