"""Interfaces of use cases."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol

from ai_api.client.resources.protocols import ISessionAccess
from ai_api.client.response.stream import Stream
from ai_api.types.requests.base import BaseRequest
from ai_api.types.responses.base import BaseResponse
from ai_api.types.responses.events import BaseResponseEvent


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
