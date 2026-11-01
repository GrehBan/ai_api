"""The shared resource of text generation."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, overload

from ai_api.adapters.protocols import ICompositeAdapter
from ai_api.resources.base import BaseResource
from ai_api.resources.stream import GenerationStream
from ai_api.resources.usecases import (
    AdapterCompositeUseCase,
    open_adapted_stream,
)
from ai_api.session.protocols import ISession
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import (
    GENERATION_TERMINAL,
    GenerationEvent,
)
from ai_api.types.responses.generation import GenerationResponse

if TYPE_CHECKING:
    from ai_api.types.requests.generation import Turn


class GenerationResource[RawT, RawItemT, SessionT: ISession = ISession](
    BaseResource[SessionT],
):
    """Generates text with any provider through that provider's adapter.

    The class is the same for every provider: a provider's client builds
    it with its own adapter, so calling code reads the same for all of
    them and only the type of ``raw`` differs. ``RawT`` is the provider's
    response, ``RawItemT`` one item of its stream.

    Both methods take the request's fields as keyword arguments, as the
    providers' SDKs do, or a ready ``GenerationRequest``; not both.

    Example:
        >>> generation = GenerationResource(session, ResponsesAdapter())
        >>> response = await generation.create(model="gpt-5", input="Hi")
        >>> async with generation.stream(model="gpt-5", input="Hi") as s:
        ...     async for text in s.text():
        ...         ...
        ...     completed = await s.final()
    """

    def __init__(
        self,
        session: SessionT,
        adapter: ICompositeAdapter[
            GenerationRequest,
            RawT,
            GenerationResponse[RawT],
            RawItemT,
            GenerationEvent[RawItemT],
        ],
    ) -> None:
        """Keep the session and build the use case on the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the shared request, the provider's response
                and its stream items.
        """
        super().__init__(session)
        self._generate = AdapterCompositeUseCase(
            session, adapter, terminal=GENERATION_TERMINAL
        )

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

        Raises:
            TypeError: Both a request and fields are given.
            ValidationError: The fields do not make a valid request.
        """
        return await self._generate(_request(request, fields))

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
            A context manager that yields a ``GenerationStream`` (the
            events, with ``text()`` and ``final()``) and closes it and
            the provider's stream on exit. A complete stream ends with
            exactly one ``CompletedEvent``; a stream cut before it
            raises ``APIStreamInterruptedError``, and a failed
            generation an ``APIStatusError``.

        Raises:
            TypeError: Both a request and fields are given.
            ValidationError: The fields do not make a valid request.
        """
        # The use case's own stream would be a Stream that a
        # GenerationStream wraps once more; this builds it directly.
        return open_adapted_stream(
            self.session,
            self._generate.adapter,
            _request(request, fields),
            self._generate.terminal,
            GenerationStream,
        )


def _request(
    request: GenerationRequest | None,
    fields: dict[str, Any],
) -> GenerationRequest:
    """Take the request of a call, given ready or as fields.

    Args:
        request: The ready request, if one was given.
        fields: The keyword arguments of the call.

    Returns:
        The request to send.

    Raises:
        TypeError: Both a request and fields are given.
        ValidationError: The fields do not make a valid request, e.g. a
            field name is misspelled.
    """
    if request is None:
        return GenerationRequest(**fields)
    if fields:
        names = ", ".join(sorted(fields))
        message = (
            "Pass either a GenerationRequest or its fields as keyword "
            f"arguments, not both (got the request and {names})"
        )
        raise TypeError(message)
    return request
