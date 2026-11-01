"""Interfaces of API methods as seen by sessions and parsers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ai_api.enums import BodyFormat, ResponseFormat

if TYPE_CHECKING:
    from pydantic import TypeAdapter

    from ai_api.http.protocols import IHeaders, Query
    from ai_api.response.parse.protocols import (
        IResponseParser,
        IStreamParser,
    )
    from ai_api.serialization.protocols import IValueFormatter
    from ai_api.types.protocols import IEventMatcher


class IBaseMethod[ResponseT](Protocol):
    """What a session needs to know to send any method.

    Only regular properties are declared, never the dunder class
    attributes of ``BaseMethod``: pyright does not accept unannotated
    class attributes of subclasses as implementing protocol members
    declared as ``ClassVar``.
    """

    @property
    def request_headers(self) -> IHeaders:
        """Headers sent with every request of the method."""
        ...

    @property
    def request_path(self) -> str:
        """Path appended to the API's base URL, parameters filled in."""
        ...

    @property
    def request_method(self) -> str:
        """HTTP method."""
        ...

    @property
    def request_body_format(self) -> BodyFormat:
        """Encoding of the request body."""
        ...

    @property
    def value_formatter(self) -> IValueFormatter:
        """Writes the dates, times and durations of the request."""
        ...

    @property
    def retryable(self) -> bool:
        """Whether transient failures may be retried."""
        ...

    @property
    def response_format(self) -> ResponseFormat:
        """How the response body is parsed."""
        ...

    @property
    def response_type(self) -> type[ResponseT] | None:
        """Runtime class of the result; ties ``ResponseT`` to the method."""
        ...

    def get_query_map(self) -> Query | None:
        """Return the query parameters.

        Returns:
            Parameters as strings or lists of strings, or ``None`` if
            there are none.
        """
        ...

    def get_body_map(self) -> Mapping[str, Any]:
        """Return the body fields.

        Returns:
            Body fields keyed by their wire names.
        """
        ...


class IAPIMethod[ResponseT](IBaseMethod[ResponseT], Protocol):
    """A method whose response is read at once.

    ``ResponseT`` is the type of the whole parsed response.
    """

    @property
    def response_parser(self) -> IResponseParser:
        """Parser of the response."""
        ...


class IStreamMethod[ItemT](IBaseMethod[ItemT], Protocol):
    """A method whose response is consumed as a stream.

    ``ItemT`` is the type of a single item yielded by the stream.
    """

    @property
    def stream_parser(self) -> IStreamParser:
        """Parser of the stream."""
        ...

    @property
    def stream_terminal(self) -> IEventMatcher | None:
        """Recognizes the item that ends a complete stream, if declared."""
        ...

    @property
    def response_adapter(self) -> TypeAdapter[ItemT] | None:
        """Validator of one SSE or NDJSON item; ``None`` when not used."""
        ...
