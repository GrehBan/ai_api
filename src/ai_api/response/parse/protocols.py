"""Interfaces for turning responses into typed results."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.response.buffered import BufferedResponse
from ai_api.response.protocols import IStreamResponse
from ai_api.serialization.json import JsonLoads

if TYPE_CHECKING:
    from ai_api.methods.protocols import IAPIMethod, IStreamMethod


class IResponseParser(Protocol):
    """Parses a fully read response into the method's result type."""

    def parse[T](
        self,
        method: IAPIMethod[T],
        response: BufferedResponse,
        json_loads: JsonLoads,
        error_resolver: IErrorResolver,
    ) -> T:
        """Parse a successful response.

        Args:
            method: The method that was sent; its response format and
                type decide how the body is parsed.
            response: The response, with a status below 400.
            json_loads: JSON decoder of the session.
            error_resolver: Error resolver of the session, for a result
                that reports a failure although its status is below 400.

        Returns:
            The parsed result.

        Raises:
            APIResponseError: If the body cannot be decoded or validated.
                The session does not wrap other errors, so a parser
                reports a bad body this way itself.
            APIStatusError: If the result reports a failure.
        """
        ...


class IStreamParser(Protocol):
    """Parses a streamed response into an iterator of items."""

    def parse[T](
        self,
        method: IStreamMethod[T],
        response: IStreamResponse,
        json_loads: JsonLoads,
        error_resolver: IErrorResolver,
    ) -> AsyncIterator[T]:
        """Start parsing a successful streamed response.

        Args:
            method: The method that was sent; its response format decides
                how chunks are turned into items.
            response: The response, with a status below 400.
            json_loads: JSON decoder of the session.
            error_resolver: Error resolver of the session, for errors
                reported inside the stream.

        Returns:
            An iterator over the stream's items. It raises
            ``APIResponseError`` for an item that cannot be decoded or
            validated, and ``APIStatusError`` for an error event.
        """
        ...
