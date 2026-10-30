"""Base classes of use cases and the orchestrators built on adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager

from ai_api.adapters.protocols import (
    IAdapter,
    ICompositeAdapter,
    IStreamAdapter,
)
from ai_api.resources.base import BaseSessionAccess
from ai_api.response.stream import Stream, transform_stream
from ai_api.session.protocols import ISession
from ai_api.types.requests.base import BaseRequest
from ai_api.types.responses.base import BaseResponse
from ai_api.types.responses.events import BaseResponseEvent


class BaseUseCase[
    ReqT: BaseRequest,
    RespT: BaseResponse,
    SessionT: ISession = ISession,
](BaseSessionAccess[SessionT], ABC):
    """Turns a shared request into a shared response.

    A subclass builds the provider's method from the request, sends it
    with ``self.session.send_request`` and converts the result.
    """

    @abstractmethod
    async def __call__(self, request: ReqT) -> RespT:
        """Send the request and convert the provider's response.

        Args:
            request: The request in shared types.

        Returns:
            The response in shared types.
        """
        raise NotImplementedError


class BaseStreamUseCase[
    ReqT: BaseRequest,
    EventT: BaseResponseEvent,
    SessionT: ISession = ISession,
](BaseSessionAccess[SessionT], ABC):
    """Turns a shared request into a stream of shared events.

    A subclass usually returns
    ``map_stream(self.session.stream(method), convert)``, which closes
    the converted and the provider's stream in the right order.
    """

    @abstractmethod
    def stream(
        self,
        request: ReqT,
    ) -> AbstractAsyncContextManager[Stream[EventT]]:
        """Open the stream; use the result with ``async with``.

        Args:
            request: The request in shared types.

        Returns:
            A context manager that yields the events and closes the
            stream on exit.
        """
        raise NotImplementedError


class BaseCompositeUseCase[
    ReqT: BaseRequest,
    RespT: BaseResponse,
    EventT: BaseResponseEvent,
    SessionT: ISession = ISession,
](
    BaseUseCase[ReqT, RespT, SessionT],
    BaseStreamUseCase[ReqT, EventT, SessionT],
):
    """A use case that can both respond at once and stream."""


async def _respond[ReqT: BaseRequest, RawT, RespT: BaseResponse](
    session: ISession,
    adapter: IAdapter[ReqT, RawT, RespT],
    request: ReqT,
) -> RespT:
    """Send the adapted method and convert its result.

    Args:
        session: The session that sends the method.
        adapter: Converts the request and the response.
        request: The request in shared types.

    Returns:
        The response in shared types.
    """
    raw = await session.send_request(adapter.to_method(request))
    return adapter.to_response(raw, request)


def _open[ReqT: BaseRequest, RawItemT, EventT: BaseResponseEvent](
    session: ISession,
    adapter: IStreamAdapter[ReqT, RawItemT, EventT],
    request: ReqT,
) -> AbstractAsyncContextManager[Stream[EventT]]:
    """Open the adapted stream method and convert its items.

    Args:
        session: The session that sends the method.
        adapter: Converts the request and the stream items.
        request: The request in shared types.

    Returns:
        A context manager that yields the events and closes both streams
        on exit.
    """

    def events(items: AsyncIterator[RawItemT]) -> AsyncIterator[EventT]:
        """Convert the items of this request's stream.

        Args:
            items: The items of the provider's stream.

        Returns:
            The events in shared types.
        """
        return adapter.to_events(items, request)

    return transform_stream(
        session.stream(adapter.to_stream_method(request)),
        events,
    )


class AdapterUseCase[
    ReqT: BaseRequest,
    RawT,
    RespT: BaseResponse,
    SessionT: ISession = ISession,
](BaseUseCase[ReqT, RespT, SessionT]):
    """A use case that only orchestrates: adapter, session, adapter.

    The request becomes the provider's method, the session sends it and
    the result becomes the shared response; all conversion is in the
    adapter.

    Attributes:
        adapter: Converts the request and the response.

    Example:
        >>> generate = AdapterUseCase(session, ChatAdapter())
        >>> response = await generate(GenerationRequest(...))
    """

    def __init__(
        self,
        session: SessionT,
        adapter: IAdapter[ReqT, RawT, RespT],
    ) -> None:
        """Keep the session and the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the request and the response.
        """
        super().__init__(session)
        self.adapter = adapter

    async def __call__(self, request: ReqT) -> RespT:
        """Send the request and convert the provider's response.

        Args:
            request: The request in shared types.

        Returns:
            The response in shared types.
        """
        return await _respond(self.session, self.adapter, request)


class AdapterStreamUseCase[
    ReqT: BaseRequest,
    RawItemT,
    EventT: BaseResponseEvent,
    SessionT: ISession = ISession,
](BaseStreamUseCase[ReqT, EventT, SessionT]):
    """A stream use case that only orchestrates an adapter and a session.

    Attributes:
        adapter: Converts the request and the stream items.
    """

    def __init__(
        self,
        session: SessionT,
        adapter: IStreamAdapter[ReqT, RawItemT, EventT],
    ) -> None:
        """Keep the session and the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the request and the stream items.
        """
        super().__init__(session)
        self.adapter = adapter

    def stream(
        self,
        request: ReqT,
    ) -> AbstractAsyncContextManager[Stream[EventT]]:
        """Open the stream; use the result with ``async with``.

        Args:
            request: The request in shared types.

        Returns:
            A context manager that yields the events and closes the
            converted and the provider's stream on exit.
        """
        return _open(self.session, self.adapter, request)


class AdapterCompositeUseCase[
    ReqT: BaseRequest,
    RawT,
    RespT: BaseResponse,
    RawItemT,
    EventT: BaseResponseEvent,
    SessionT: ISession = ISession,
](BaseCompositeUseCase[ReqT, RespT, EventT, SessionT]):
    """A use case that orchestrates one adapter both ways.

    Attributes:
        adapter: Converts the request, the response and the stream items.
    """

    def __init__(
        self,
        session: SessionT,
        adapter: ICompositeAdapter[ReqT, RawT, RespT, RawItemT, EventT],
    ) -> None:
        """Keep the session and the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the request, the response and the stream
                items.
        """
        super().__init__(session)
        self.adapter = adapter

    async def __call__(self, request: ReqT) -> RespT:
        """Send the request and convert the provider's response.

        Args:
            request: The request in shared types.

        Returns:
            The response in shared types.
        """
        return await _respond(self.session, self.adapter, request)

    def stream(
        self,
        request: ReqT,
    ) -> AbstractAsyncContextManager[Stream[EventT]]:
        """Open the stream; use the result with ``async with``.

        Args:
            request: The request in shared types.

        Returns:
            A context manager that yields the events and closes the
            converted and the provider's stream on exit.
        """
        return _open(self.session, self.adapter, request)
