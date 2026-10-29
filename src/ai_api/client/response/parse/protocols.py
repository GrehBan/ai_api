"""Interfaces for turning responses into typed results."""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

from ai_api.client.exceptions.error_resolver.protocols import (
    IErrorResolver,
)
from ai_api.client.response.base import BufferedResponse
from ai_api.client.response.protocols import IStreamResponse
from ai_api.client.serialization.json import JsonLoads

if TYPE_CHECKING:
    from ai_api.methods.protocols import IAPIMethod, IStreamMethod


class IResponseParser(Protocol):
    """Parses a fully read response into the method's result type."""

    def parse[T](
        self,
        method: IAPIMethod[T],
        buffered: BufferedResponse,
        json_loads: JsonLoads | None = None,
    ) -> T:
        """Parse a successful response.

        Args:
            method: The method that was sent; its response format and
                type decide how the body is parsed.
            buffered: The response, with a status below 400.
            json_loads: JSON decoder of the session. The parser's own
                default is used when ``None``.

        Returns:
            The parsed result.
        """
        ...


class IStreamParser(Protocol):
    """Parses a streamed response into an iterator of items."""

    def parse[T](
        self,
        method: IStreamMethod[T],
        response: IStreamResponse,
        json_loads: JsonLoads | None = None,
        error_resolver: IErrorResolver | None = None,
    ) -> AsyncIterator[T]:
        """Start parsing a successful streamed response.

        Args:
            method: The method that was sent; its response format decides
                how chunks are turned into items.
            response: The response, with a status below 400.
            json_loads: JSON decoder of the session. The parser's own
                default is used when ``None``.
            error_resolver: Error resolver of the session, for errors
                reported inside the stream. The parser's own default is
                used when ``None``.

        Returns:
            An iterator over the stream's items.
        """
        ...
