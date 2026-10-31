"""Interfaces of resources and use cases."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Protocol, overload

from ai_api.client.protocols import IClient
from ai_api.response.stream import Stream
from ai_api.session.protocols import ISession
from ai_api.types.requests.base import BaseRequest
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.base import BaseResponse
from ai_api.types.responses.events import BaseResponseEvent
from ai_api.types.responses.generation import GenerationResponse

if TYPE_CHECKING:
    from ai_api.resources.stream import GenerationStream
    from ai_api.types.requests.generation import Turn


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
    """Generates text; the same interface for every provider.

    Both methods take the request's fields as keyword arguments, as the
    providers' SDKs do, or a ready ``GenerationRequest``.
    """

    @overload
    async def create(
        self,
        /,
        *,
        model: str,
        input: str | list[Turn],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> GenerationResponse[RawT]: ...

    @overload
    async def create(
        self,
        request: GenerationRequest,
        /,
    ) -> GenerationResponse[RawT]: ...

    async def create(
        self,
        request: GenerationRequest | None = None,
        /,
        **fields: Any,
    ) -> GenerationResponse[RawT]:
        """Generate the whole response at once.

        Args:
            request: What to generate, as a ready request.
            **fields: What to generate, as the fields of a
                ``GenerationRequest``.

        Returns:
            The generated text with the provider's own response.
        """
        ...

    @overload
    def stream(
        self,
        /,
        *,
        model: str,
        input: str | list[Turn],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]: ...

    @overload
    def stream(
        self,
        request: GenerationRequest,
        /,
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]: ...

    def stream(
        self,
        request: GenerationRequest | None = None,
        /,
        **fields: Any,
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]:
        """Open a stream of generation events; use with ``async with``.

        Args:
            request: What to generate, as a ready request.
            **fields: What to generate, as the fields of a
                ``GenerationRequest``.

        Returns:
            A context manager that yields the events, with the
            ``text()`` and ``final()`` helpers, and closes the stream on
            exit.
        """
        ...


class IGenerationClient[RawT = Any, RawItemT = Any](IClient, Protocol):
    """A client that generates text; every provider's client is one.

    ``RawT`` is the provider's response and ``RawItemT`` one item of its
    stream; both default to ``Any`` for code that works with any
    provider.

    Example:
        >>> async def ask(client: IGenerationClient, question: str) -> str:
        ...     response = await client.generation.create(
        ...         model=MODEL, input=question
        ...     )
        ...     return response.text or ""
    """

    @property
    def generation(self) -> IGenerationResource[RawT, RawItemT]:
        """Shared text generation over the provider's API."""
        ...
