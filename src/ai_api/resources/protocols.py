"""Interfaces of resources and use cases."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from ai_api.response.stream import Stream
from ai_api.session.protocols import ISession
from ai_api.types.requests.base import BaseRequest
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.base import BaseResponse
from ai_api.types.responses.events import BaseResponseEvent, GenerationEvent
from ai_api.types.responses.generation import GenerationResponse


class ISessionAccess(Protocol):
    """Gives access to the session that sends the requests."""

    @property
    def session(self) -> ISession:
        """The session that sends the requests."""
        ...


class IResource(ISessionAccess, Protocol):
    """A group of use cases of one API area, such as responses."""


class IUseCase[
    ReqT: BaseRequest,
    RespT: BaseResponse,
](ISessionAccess, Protocol):
    """Turns a shared request into a shared response."""

    async def __call__(self, request: ReqT) -> RespT:
        """Send the request and convert the provider's response.

        Args:
            request: The request in shared types.

        Returns:
            The response in shared types.
        """
        ...


class IStreamUseCase[
    ReqT: BaseRequest,
    EventT: BaseResponseEvent,
](ISessionAccess, Protocol):
    """Turns a shared request into a stream of shared events."""

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
        ...


class ICompositeUseCase[
    ReqT: BaseRequest,
    RespT: BaseResponse,
    EventT: BaseResponseEvent,
](IUseCase[ReqT, RespT], IStreamUseCase[ReqT, EventT], Protocol):
    """A use case that can both respond at once and stream."""


class IGenerationResource[RawT, RawItemT](IResource, Protocol):
    """Generates text; the same interface for every provider."""

    async def create(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse[RawT]:
        """Generate the whole response at once.

        Args:
            request: What to generate.

        Returns:
            The generated text with the provider's own response.
        """
        ...

    def stream(
        self,
        request: GenerationRequest,
    ) -> AbstractAsyncContextManager[Stream[GenerationEvent[RawItemT]]]:
        """Open a stream of generation events; use with ``async with``.

        Args:
            request: What to generate.

        Returns:
            A context manager that yields the events and closes the
            stream on exit.
        """
        ...
