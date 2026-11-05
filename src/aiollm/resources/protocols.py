"""Interfaces of resources and use cases."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Protocol, Self, Unpack, overload

from pydantic import MISSING

from aiollm.adapters.protocols import ICompositeAdapter
from aiollm.client.protocols import IClient
from aiollm.response.stream import Stream
from aiollm.session.protocols import CallOptions, ISession, Timeout
from aiollm.types.requests.base import BaseRequest
from aiollm.types.requests.generation import (
    GenerationRequest,
    NoProviderOptions,
)
from aiollm.types.responses.base import BaseResponse
from aiollm.types.responses.events import BaseResponseEvent, GenerationEvent
from aiollm.types.responses.generation import GenerationResponse

if TYPE_CHECKING:
    from aiollm.http.protocols import HeadersType
    from aiollm.resources.stream import GenerationStream
    from aiollm.session.retry.policy import RetryPolicy
    from aiollm.types.requests.generation import Turn


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

    async def __call__(
        self,
        request: ReqT,
        **options: Unpack[CallOptions],
    ) -> RespT:
        """Send the request and convert the provider's response.

        Args:
            request: The request in shared types.
            **options: Options of the call (``headers``, ``timeout``,
                ``retry_policy``), passed on to the session.

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
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[Stream[EventT]]:
        """Open the stream; use the result with ``async with``.

        Args:
            request: The request in shared types.
            **options: Options of the call (``headers``, ``timeout``,
                ``retry_policy``), passed on to the session.

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


class IGenerationAdapter[
    RawT,
    RawItemT,
    OptionsT: Mapping[str, object] = NoProviderOptions,
](
    ICompositeAdapter[
        GenerationRequest,
        RawT,
        GenerationResponse[RawT],
        RawItemT,
        GenerationEvent[RawItemT],
    ],
    Protocol,
):
    """The adapter a ``GenerationResource`` runs on.

    Besides converting, it takes the provider's own options of one call
    (``provider_options``): fields that only its provider has, such as
    OpenAI's ``store``. ``OptionsT`` is a ``TypedDict`` of them.
    """

    def with_options(self, options: OptionsT, /) -> Self:
        """Return a copy that adds the options to every method it builds.

        The adapter itself is left as it is, so concurrent calls with
        other options do not meet.

        Args:
            options: The provider's options of one call.

        Returns:
            A copy of the adapter with the options.

        Raises:
            RequestValidationError: A key is not one of the provider's
                options; the shared fields are set in the request.
        """
        ...


class IGenerationResource[RawT, RawItemT](IResource, Protocol):
    """Generates text; the same interface for every provider.

    Both methods take the request's fields as keyword arguments, as the
    providers' SDKs do, or a ready ``GenerationRequest``, and the options
    of the call (``headers``, ``timeout``, ``retry_policy``) as
    ``send_request`` does; no request field is named like one of them.
    """

    @overload
    async def create(
        self,
        /,
        *,
        model: str,
        input: str | list[Turn],
        instructions: str | MISSING = MISSING,
        max_output_tokens: int | MISSING = MISSING,
        metadata: dict[str, str] | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> GenerationResponse[RawT]: ...

    @overload
    async def create(
        self,
        request: GenerationRequest,
        /,
        **options: Unpack[CallOptions],
    ) -> GenerationResponse[RawT]: ...

    async def create(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
        **fields: Any,
    ) -> GenerationResponse[RawT]:
        """Generate the whole response at once.

        Args:
            request: What to generate, as a ready request.
            headers: Headers for this request only.
            timeout: Seconds for this request instead of the session's
                limit; ``None`` to keep it.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.
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
        instructions: str | MISSING = MISSING,
        max_output_tokens: int | MISSING = MISSING,
        metadata: dict[str, str] | MISSING = MISSING,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]: ...

    @overload
    def stream(
        self,
        request: GenerationRequest,
        /,
        **options: Unpack[CallOptions],
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]: ...

    def stream(
        self,
        request: GenerationRequest | None = None,
        /,
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
        **fields: Any,
    ) -> AbstractAsyncContextManager[GenerationStream[RawItemT]]:
        """Open a stream of generation events; use with ``async with``.

        Args:
            request: What to generate, as a ready request.
            headers: Headers for this request only.
            timeout: Seconds for this request instead of the session's
                limit; ``None`` to keep it.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.
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
    provider. It lives with the resources, not with the clients, since
    it names a resource protocol and the client layer is below this one.

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
