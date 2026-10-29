"""Interfaces of API methods as seen by sessions and parsers."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ai_api.methods.enums import BodyFormat, ResponseFormat, ResponseMode

if TYPE_CHECKING:
    from ai_api.client.response.parse.protocols import (
        IResponseParser,
        IStreamParser,
    )


class IBaseMethod[ResponseT](Protocol):
    """What a session needs to know to send any method.

    Only regular properties are declared, never the dunder class
    attributes of ``BaseMethod``: pyright does not accept unannotated
    class attributes of subclasses as implementing protocol members
    declared as ``ClassVar``.
    """

    @property
    def request_path(self) -> str:
        """Path appended to the API's base URL."""
        ...

    @property
    def request_method(self) -> str:
        """HTTP method."""
        ...

    @property
    def body_format(self) -> BodyFormat:
        """Encoding of the request body."""
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
    def response_mode(self) -> ResponseMode:
        """Whether the response is buffered or streamed."""
        ...

    @property
    def response_type(self) -> type[ResponseT] | None:
        """Runtime class of the result; ties ``ResponseT`` to the method."""
        ...

    def get_query_map(self) -> Mapping[str, str] | None:
        """Return the query parameters.

        Returns:
            Parameters as strings, or ``None`` if there are none.
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


class IStreamMethod[ResponseT](IBaseMethod[ResponseT], Protocol):
    """A method whose response is consumed as a stream.

    ``ResponseT`` is the type of a single item yielded by the stream.
    """

    @property
    def stream_parser(self) -> IStreamParser:
        """Parser of the stream."""
        ...
