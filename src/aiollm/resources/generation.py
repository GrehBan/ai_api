"""The shared resource of text generation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Self, Unpack, overload

from pydantic import MISSING

from aiollm.resources.base import BaseResource
from aiollm.resources.protocols import IGenerationAdapter
from aiollm.resources.stream import GenerationStream
from aiollm.resources.usecases import (
    AdapterCompositeUseCase,
    open_adapted_stream,
)
from aiollm.session.protocols import CallOptions, ISession, Timeout
from aiollm.types.requests.generation import (
    GenerationRequest,
    NoProviderOptions,
)
from aiollm.types.responses.events import (
    GENERATION_TERMINAL,
)
from aiollm.types.responses.generation import GenerationResponse

if TYPE_CHECKING:
    from aiollm.http.protocols import HeadersType
    from aiollm.session.retry.policy import RetryPolicy
    from aiollm.types.requests.generation import Turn


class GenerationResource[
    RawT,
    RawItemT,
    OptionsT: Mapping[str, object] = NoProviderOptions,
    SessionT: ISession = ISession,
](
    BaseResource[SessionT],
):
    """Generates text with any provider through that provider's adapter.

    The class is the same for every provider: a provider's client builds
    it with its own adapter, so calling code reads the same for all of
    them and only the type of ``raw`` differs. ``RawT`` is the provider's
    response, ``RawItemT`` one item of its stream, ``OptionsT`` the
    ``TypedDict`` of its own options (``NoProviderOptions`` for none).

    Both methods take the request's fields as keyword arguments, as the
    providers' SDKs do, or a ready ``GenerationRequest``; not both. The
    options of the call (``headers``, ``timeout``, ``retry_policy``) are
    passed on to the session, as ``send_request`` takes them. The
    provider's own fields, which the shared request does not hold, are
    set with ``with_options``, which code that works with any provider
    (``IGenerationResource``) does not have.

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
        adapter: IGenerationAdapter[RawT, RawItemT, OptionsT],
    ) -> None:
        """Keep the session and build the use case on the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the shared request, the provider's response
                and its stream items.
        """
        super().__init__(session)
        self._adapter = adapter
        self._generate = AdapterCompositeUseCase(
            session, adapter, terminal=GENERATION_TERMINAL
        )

    def with_options(self, options: OptionsT, /) -> Self:
        """Return a copy that sends the provider's own options.

        The options are fields that only this provider has, such as
        OpenAI's ``store``; a key left out is not sent. The copy shares
        the session, and this resource is left as it is, so it can be
        kept for many calls.

        Args:
            options: The provider's options, a ``TypedDict`` such as
                ``OpenAIGenerationOptions``.

        Returns:
            A resource whose requests carry the options.

        Raises:
            RequestValidationError: A key is not one of the provider's
                options; the shared fields are set in the request. An
                invalid value is raised when a request is built.

        Example:
            >>> unstored = client.generation.with_options({"store": False})
            >>> response = await unstored.create(model="gpt-5", input="Hi")
        """
        adapter = self._adapter.with_options(options)
        resource = copy.copy(self)
        resource._adapter = adapter
        resource._generate = AdapterCompositeUseCase(
            self.session, adapter, terminal=GENERATION_TERMINAL
        )
        return resource

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

        Raises:
            TypeError: Both a request and fields are given.
            ValidationError: The fields do not make a valid request.
        """
        return await self._generate(
            _request(request, fields),
            headers=headers,
            timeout=timeout,
            retry_policy=retry_policy,
        )

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
            headers=headers,
            timeout=timeout,
            retry_policy=retry_policy,
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
