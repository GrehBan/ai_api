"""Base classes of use cases."""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from ai_api.client.resources.base import BaseSessionAccess
from ai_api.client.response.stream import Stream
from ai_api.client.session.protocols import ISession
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
