"""Base classes of adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ai_api.methods.protocols import IAPIMethod, IStreamMethod
from ai_api.types.requests.base import BaseRequest
from ai_api.types.responses.base import BaseResponse
from ai_api.types.responses.events import BaseResponseEvent


class BaseConversion[SourceT, TargetT](ABC):
    """Base class of conversions between provider and shared values.

    A conversion is pure: it converts one value and does no I/O. It
    names what it converts in its type, ``source → target``, and an
    adapter takes it as a keyword argument, so a caller can swap it.

    Example:
        >>> class UsageConversion(BaseConversion[Usage, TokenUsage]):
        ...     def __call__(self, source):
        ...         return TokenUsage(input_tokens=source.input_tokens)
    """

    @abstractmethod
    def __call__(self, source: SourceT, /) -> TargetT:
        """Convert a value.

        Args:
            source: The value to convert.

        Returns:
            The converted value.
        """
        raise NotImplementedError


class BaseAdapter[
    ReqT: BaseRequest,
    RawT,
    RespT: BaseResponse,
](ABC):
    """Base class of adapters of requests answered at once.

    Example:
        >>> class ChatAdapter(BaseAdapter[GenerationRequest, Chat, Reply]):
        ...     def to_method(self, request):
        ...         return CreateChat(model=request.model, ...)
        ...
        ...     def to_response(self, raw, request):
        ...         return Reply(text=raw.choices[0].text, raw=raw)
    """

    @abstractmethod
    def to_method(self, request: ReqT) -> IAPIMethod[RawT]:
        """Build the provider's method for a shared request.

        Args:
            request: The request in shared types.

        Returns:
            The provider's method, ready to send.
        """
        raise NotImplementedError

    @abstractmethod
    def to_response(self, raw: RawT, request: ReqT) -> RespT:
        """Convert the provider's response into a shared response.

        Args:
            raw: The parsed result of the provider's method.
            request: The request the response answers.

        Returns:
            The response in shared types.
        """
        raise NotImplementedError


class BaseStreamAdapter[
    ReqT: BaseRequest,
    RawItemT,
    EventT: BaseResponseEvent,
](ABC):
    """Base class of adapters of streamed requests.

    ``to_events`` is usually written as an async generator.
    """

    @abstractmethod
    def to_stream_method(self, request: ReqT) -> IStreamMethod[RawItemT]:
        """Build the provider's stream method for a shared request.

        Args:
            request: The request in shared types.

        Returns:
            The provider's stream method, ready to send.
        """
        raise NotImplementedError

    @abstractmethod
    def to_events(
        self,
        items: AsyncIterator[RawItemT],
        request: ReqT,
    ) -> AsyncIterator[EventT]:
        """Convert the provider's stream items into shared events.

        Args:
            items: The items of the provider's stream.
            request: The request the stream answers.

        Returns:
            The events in shared types.
        """
        raise NotImplementedError


class BaseCompositeAdapter[
    ReqT: BaseRequest,
    RawT,
    RespT: BaseResponse,
    RawItemT,
    EventT: BaseResponseEvent,
](
    BaseAdapter[ReqT, RawT, RespT],
    BaseStreamAdapter[ReqT, RawItemT, EventT],
):
    """Base class of adapters of endpoints that respond and stream.

    A subclass usually builds both methods from one private helper, since
    a provider's buffered and streamed requests mostly differ by one flag.
    """
