"""Adapter of shared generation to OpenAI's Responses API."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ai_api.adapters.base import BaseCompositeAdapter
from ai_api.adapters.validation import check_last_turn, request_errors
from ai_api.enums import Provider
from ai_api.providers.openai.adapters.conversions import (
    check_status,
    incomplete_reason,
    raw_incomplete_reason,
    refusal,
    to_event,
    to_usage,
)
from ai_api.providers.openai.methods.responses import (
    CreateResponse,
    StreamResponse,
)
from ai_api.providers.openai.types.events import ResponseStreamEvent
from ai_api.providers.openai.types.responses import Response
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import GenerationEvent
from ai_api.types.responses.generation import GenerationResponse


class OpenAIGenerationAdapter(
    BaseCompositeAdapter[
        GenerationRequest,
        Response,
        GenerationResponse[Response],
        ResponseStreamEvent,
        GenerationEvent[ResponseStreamEvent],
    ],
):
    """Generates text with ``POST /responses``.

    Request fields map one to one; a list of turns becomes a list of
    ``EasyInputMessage`` items. OpenAI's own limits are checked when the
    method is built, so a request that OpenAI would reject, such as
    ``max_output_tokens`` below 16, more than 16 metadata pairs or a
    conversation that does not end with a ``user`` turn, raises
    ``RequestValidationError`` before anything is sent.

    Example:
        >>> generation = GenerationResource(session, OpenAIGenerationAdapter())
        >>> response = await generation.create(
        ...     GenerationRequest(model="gpt-5", input="Hi")
        ... )
    """

    def to_method(self, request: GenerationRequest) -> CreateResponse:
        """Build the method that creates a response at once.

        Args:
            request: The shared request.

        Returns:
            ``CreateResponse`` with the request's fields.

        Raises:
            RequestValidationError: OpenAI cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(Provider.OPENAI):
            return CreateResponse(**fields)

    def to_response(
        self,
        raw: Response,
        request: GenerationRequest,
    ) -> GenerationResponse[Response]:
        """Convert a finished response.

        Args:
            raw: OpenAI's response.
            request: The request it answers.

        Returns:
            The generated text with its token counts; an incomplete
            response keeps its text and gets ``incomplete_reason``, and
            a refusal gets ``IncompleteReason.REFUSAL`` and ``refusal``.

        Raises:
            APIResponseError: The response did not finish. A failed one
                is raised by ``CreateResponse`` before it gets here.
        """
        check_status(raw)
        return GenerationResponse[Response](
            id=raw.id,
            provider=Provider.OPENAI,
            model=raw.model,
            text=raw.output_text,
            usage=to_usage(raw.usage),
            incomplete_reason=incomplete_reason(raw),
            raw_incomplete_reason=raw_incomplete_reason(raw),
            refusal=refusal(raw),
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> StreamResponse:
        """Build the method that streams a response.

        Args:
            request: The shared request.

        Returns:
            ``StreamResponse`` with the request's fields.

        Raises:
            RequestValidationError: OpenAI cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(Provider.OPENAI):
            return StreamResponse(**fields)

    async def to_events(
        self,
        items: AsyncIterator[ResponseStreamEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[ResponseStreamEvent]]:
        """Convert the events of a Responses stream.

        Args:
            items: The events OpenAI sent.
            request: The request the stream answers.

        Yields:
            One shared event per OpenAI event. The stream's end is
            checked by ``StreamResponse`` and by the use case.
        """
        async for item in items:
            yield to_event(item)

    @staticmethod
    def _fields(request: GenerationRequest) -> dict[str, Any]:
        """Take the method fields of a request.

        Args:
            request: The shared request.

        Returns:
            The fields that both methods take.

        Raises:
            RequestValidationError: The conversation does not end with a
                ``user`` turn.
        """
        check_last_turn(request, Provider.OPENAI)
        return {
            "model": request.model,
            "input": (
                request.input
                if isinstance(request.input, str)
                else [
                    {"role": turn.role.value, "content": turn.content}
                    for turn in request.input
                ]
            ),
            "instructions": request.instructions,
            "max_output_tokens": request.max_output_tokens,
            "metadata": request.metadata,
        }
