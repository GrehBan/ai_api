"""Interfaces of adapters between shared types and provider methods."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

from ai_api.methods.protocols import IAPIMethod, IStreamMethod
from ai_api.types.requests.base import BaseRequest
from ai_api.types.responses.base import BaseResponse
from ai_api.types.responses.events import BaseResponseEvent

if TYPE_CHECKING:
    from ai_api.enums import Provider


class IConversion[SourceT, TargetT](Protocol):
    """Converts one provider value into a shared one, or back.

    An adapter takes its conversions as keyword arguments, so one can be
    swapped without subclassing the adapter. A plain function matches
    too: the source is positional-only, so its name does not matter.
    """

    def __call__(self, source: SourceT, /) -> TargetT:
        """Convert a value.

        Args:
            source: The value to convert.

        Returns:
            The converted value.
        """
        ...


class IAdapter[
    ReqT: BaseRequest,
    RawT,
    RespT: BaseResponse,
](Protocol):
    """Converts one shared request and its provider's response.

    Both halves name the same ``RawT``, so a type checker sees at the
    adapter itself whether the method it builds returns what its response
    conversion expects. An adapter only converts models: it sends nothing
    and holds no session.
    """

    @property
    def provider(self) -> Provider:
        """The provider whose methods the adapter builds."""
        ...

    def to_method(self, request: ReqT) -> IAPIMethod[RawT]:
        """Build the provider's method for a shared request.

        Args:
            request: The request in shared types.

        Returns:
            The provider's method, ready to send.
        """
        ...

    def to_response(self, raw: RawT, request: ReqT) -> RespT:
        """Convert the provider's response into a shared response.

        Args:
            raw: The parsed result of the provider's method.
            request: The request the response answers, for what the
                provider does not send back, e.g. an output schema.

        Returns:
            The response in shared types.
        """
        ...


class IStreamAdapter[
    ReqT: BaseRequest,
    RawItemT,
    EventT: BaseResponseEvent,
](Protocol):
    """Converts one shared request and its provider's stream."""

    @property
    def provider(self) -> Provider:
        """The provider whose methods the adapter builds."""
        ...

    def to_stream_method(self, request: ReqT) -> IStreamMethod[RawItemT]:
        """Build the provider's stream method for a shared request.

        Args:
            request: The request in shared types.

        Returns:
            The provider's stream method, ready to send.
        """
        ...

    def to_events(
        self,
        items: AsyncIterator[RawItemT],
        request: ReqT,
    ) -> AsyncIterator[EventT]:
        """Convert the provider's stream items into shared events.

        It sees the whole stream, so it may keep state, skip items and
        turn one item into several events.

        Args:
            items: The items of the provider's stream.
            request: The request the stream answers.

        Returns:
            The events in shared types; usually an async generator.
        """
        ...


class ICompositeAdapter[
    ReqT: BaseRequest,
    RawT,
    RespT: BaseResponse,
    RawItemT,
    EventT: BaseResponseEvent,
](
    IAdapter[ReqT, RawT, RespT],
    IStreamAdapter[ReqT, RawItemT, EventT],
    Protocol,
):
    """An adapter for an endpoint that can both respond and stream."""
