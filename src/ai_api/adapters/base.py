"""Base classes of adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from ai_api.enums import Provider
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


def _is_abstract(cls: type) -> bool:
    """Tell whether a class still has abstract methods.

    ``ABCMeta`` sets ``__abstractmethods__`` only after
    ``__init_subclass__`` has run, so it is computed here.

    Args:
        cls: The class being defined.

    Returns:
        ``True`` if a method of the class is still abstract.
    """
    names = set[str]().union(
        *(getattr(base, "__abstractmethods__", ()) for base in cls.__mro__),
        cls.__dict__,
    )
    return any(
        getattr(getattr(cls, name, None), "__isabstractmethod__", False)
        for name in names
    )


class _ProviderAdapter(ABC):  # noqa: B024 - its subclasses are abstract
    """What every adapter has: the provider whose methods it builds.

    A concrete adapter must set ``provider``, so the shared results it
    builds, and the errors of the requests it refuses, name it; an
    adapter outside the library, whose provider ``Provider`` does not
    list, sets ``Provider.GENERIC``.

    Attributes:
        provider: The provider whose methods the adapter builds.
    """

    provider: ClassVar[Provider]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Require ``provider`` on a concrete adapter.

        Args:
            **kwargs: Class keyword arguments.

        Raises:
            TypeError: If a class with no abstract method left does not
                set ``provider`` to a ``Provider``.
        """
        super().__init_subclass__(**kwargs)

        if _is_abstract(cls):
            return

        if not isinstance(getattr(cls, "provider", None), Provider):
            message = (
                f"{cls.__name__} must set provider to a Provider "
                "(Provider.GENERIC for one the library does not list)"
            )
            raise TypeError(message)


class BaseAdapter[
    ReqT: BaseRequest,
    RawT,
    RespT: BaseResponse,
](_ProviderAdapter):
    """Base class of adapters of requests answered at once.

    Example:
        >>> class ChatAdapter(BaseAdapter[GenerationRequest, Chat, Reply]):
        ...     provider = Provider.GENERIC
        ...
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
](_ProviderAdapter):
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
