"""Base classes of use cases and the orchestrators built on adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from ai_api.adapters.protocols import (
    IAdapter,
    ICompositeAdapter,
    IStreamAdapter,
)
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.transport import APIStreamInterruptedError
from ai_api.resources.base import BaseSessionAccess
from ai_api.response.stream import Stream, transform_stream
from ai_api.session.protocols import ISession
from ai_api.types.protocols import IEventMatcher
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


async def _terminated[EventT](
    events: AsyncIterator[EventT],
    terminal: IEventMatcher,
    *,
    source_checked: bool,
    adapter: object,
) -> AsyncGenerator[EventT]:
    """Enforce the shared contract of a converted stream.

    Exactly one event matches ``terminal``, and it is the last one. The
    provider's own protocol is checked by its parser; this catches an
    adapter that drops, repeats or misplaces the final event.

    Args:
        events: The converted events.
        terminal: Recognizes the final event.
        source_checked: Whether the provider's method declares a
            ``__stream_terminal__``: its parser then proved the source
            complete, so a missing final event is the adapter's fault.
        adapter: The adapter that converted the events, named in the
            error.

    Yields:
        The events, unchanged.

    Raises:
        APIResponseError: If an event comes after the final one.
        RuntimeError: If the events end without it although the source
            was complete: the adapter lost it.
        APIStreamInterruptedError: If the events end without it and the
            source has no terminal to prove it complete, so it may have
            been cut.
    """
    ended = False
    received = 0

    try:
        async for event in events:
            if ended:
                raise APIResponseError(
                    "Stream produced an event after its final event"
                )

            ended = terminal.matches(event)
            received += 1
            yield event

        if not ended and source_checked:
            name = type(adapter).__name__
            message = f"{name} ended a complete stream without its final event"
            raise RuntimeError(message)

        if not ended:
            raise APIStreamInterruptedError(
                "Stream ended before its final event",
                items_received=received,
            )
    finally:
        # Closed here, or it would stay suspended until the loop ends.
        if isinstance(events, AsyncGenerator):
            await events.aclose()


def open_adapted_stream[
    ReqT: BaseRequest,
    RawItemT,
    EventT: BaseResponseEvent,
    S: Stream[Any],
](
    session: ISession,
    adapter: IStreamAdapter[ReqT, RawItemT, EventT],
    request: ReqT,
    terminal: IEventMatcher | None,
    stream_type: Callable[[AsyncIterator[EventT]], S],
) -> AbstractAsyncContextManager[S]:
    """Open the adapted stream method and convert its items.

    The stream use cases open their streams with it, and so does
    ``GenerationResource``, which asks for a ``GenerationStream``.

    Args:
        session: The session that sends the method.
        adapter: Converts the request and the stream items.
        request: The request in shared types.
        terminal: Recognizes the final event of the converted stream;
            ``None`` checks nothing.
        stream_type: The ``Stream`` class of the converted events.

    Returns:
        A context manager that yields the events and closes both streams
        on exit.
    """
    method = adapter.to_stream_method(request)

    def events(items: AsyncIterator[RawItemT]) -> AsyncIterator[EventT]:
        """Convert the items of this request's stream.

        Args:
            items: The items of the provider's stream.

        Returns:
            The events in shared types, checked against ``terminal``.
        """
        converted = adapter.to_events(items, request)

        if terminal is None:
            return converted

        return _terminated(
            converted,
            terminal,
            source_checked=method.stream_terminal is not None,
            adapter=adapter,
        )

    return transform_stream(
        session.stream(method),
        events,
        stream_type=stream_type,
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
        terminal: Recognizes the final event of the shared stream, such
            as ``GENERATION_TERMINAL``; ``None`` checks nothing.
    """

    def __init__(
        self,
        session: SessionT,
        adapter: IStreamAdapter[ReqT, RawItemT, EventT],
        *,
        terminal: IEventMatcher | None = None,
    ) -> None:
        """Keep the session and the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the request and the stream items.
            terminal: Recognizes the final event of the shared stream.
                When set, exactly one event must match it and be the
                last: an event after it raises ``APIResponseError``, and
                a stream without it ``APIStreamInterruptedError``.
        """
        super().__init__(session)
        self.adapter = adapter
        self.terminal = terminal

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
        return open_adapted_stream(
            self.session, self.adapter, request, self.terminal, Stream
        )


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
        terminal: Recognizes the final event of the shared stream;
            ``None`` checks nothing.
    """

    def __init__(
        self,
        session: SessionT,
        adapter: ICompositeAdapter[ReqT, RawT, RespT, RawItemT, EventT],
        *,
        terminal: IEventMatcher | None = None,
    ) -> None:
        """Keep the session and the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the request, the response and the stream
                items.
            terminal: Recognizes the final event of the shared stream,
                checked as in ``AdapterStreamUseCase``.
        """
        super().__init__(session)
        self.adapter = adapter
        self.terminal = terminal

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
        return open_adapted_stream(
            self.session, self.adapter, request, self.terminal, Stream
        )
