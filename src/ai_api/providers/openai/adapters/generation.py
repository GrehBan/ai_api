"""Adapter of shared generation to OpenAI's Responses API."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from ai_api.adapters.base import BaseCompositeAdapter
from ai_api.adapters.validation import check_last_turn, request_errors
from ai_api.enums import Provider
from ai_api.providers.openai.adapters.conversions import (
    INPUT_CONVERSION,
    STOP_CONVERSION,
    USAGE_CONVERSION,
    check_status,
)
from ai_api.providers.openai.methods.responses import (
    CreateResponse,
    StreamResponse,
)
from ai_api.providers.openai.types.events import (
    OutputTextDeltaEvent,
    ResponseStateEvent,
    ResponseStreamEvent,
)
from ai_api.providers.openai.types.responses import Response
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)
from ai_api.types.responses.generation import GenerationResponse

if TYPE_CHECKING:
    from ai_api.adapters.protocols import IConversion
    from ai_api.adapters.stop import Stop
    from ai_api.providers.openai.types.responses import ResponseUsage
    from ai_api.types.requests.generation import Turn
    from ai_api.types.responses.generation import TokenUsage


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

    The conversions of the input, the token counts and the stop are
    keyword arguments, so any of them can be swapped without a subclass;
    a stream's terminal event goes through the same ones as a response.

    Attributes:
        input_conversion: Turns the shared ``input`` into OpenAI's.
        usage_conversion: Turns OpenAI's ``ResponseUsage`` into shared
            counts.
        stop_conversion: Tells how a finished response stopped.

    Example:
        >>> generation = GenerationResource(session, OpenAIGenerationAdapter())
        >>> response = await generation.create(
        ...     GenerationRequest(model="gpt-5", input="Hi")
        ... )
    """

    def __init__(
        self,
        *,
        input_conversion: IConversion[
            str | list[Turn], str | list[dict[str, Any]]
        ] = INPUT_CONVERSION,
        usage_conversion: IConversion[
            ResponseUsage, TokenUsage
        ] = USAGE_CONVERSION,
        stop_conversion: IConversion[Response, Stop] = STOP_CONVERSION,
    ) -> None:
        """Store the conversions.

        Args:
            input_conversion: Turns the shared ``input`` into OpenAI's.
            usage_conversion: Turns OpenAI's ``ResponseUsage`` into
                shared counts.
            stop_conversion: Tells how a finished response stopped.
        """
        self.input_conversion = input_conversion
        self.usage_conversion = usage_conversion
        self.stop_conversion = stop_conversion

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
        stop = self.stop_conversion(raw)
        return GenerationResponse[Response](
            id=raw.id,
            provider=Provider.OPENAI,
            model=raw.model,
            text=raw.output_text,
            usage=self._usage(raw.usage),
            incomplete_reason=stop.incomplete_reason,
            raw_incomplete_reason=stop.raw_incomplete_reason,
            refusal=stop.refusal,
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
            yield self._to_event(item)

    def _to_event(
        self,
        event: ResponseStreamEvent,
    ) -> GenerationEvent[ResponseStreamEvent]:
        """Convert one event of a Responses stream.

        Args:
            event: The event OpenAI sent.

        Returns:
            A text delta, the completion (also of an incomplete
            response), or ``OtherEvent`` for every other event.
            ``response.failed`` never gets here: ``StreamResponse``
            raises it as an ``APIStatusError``.
        """
        match event:
            case OutputTextDeltaEvent():
                return TextDeltaEvent[ResponseStreamEvent](
                    text=event.delta, raw=event
                )
            case ResponseStateEvent(
                type="response.completed" | "response.incomplete"
            ):
                stop = self.stop_conversion(event.response)
                return CompletedEvent[ResponseStreamEvent](
                    usage=self._usage(event.response.usage),
                    incomplete_reason=stop.incomplete_reason,
                    raw_incomplete_reason=stop.raw_incomplete_reason,
                    refusal=stop.refusal,
                    raw=event,
                )
            case _:
                return OtherEvent[ResponseStreamEvent](raw=event)

    def _usage(self, usage: ResponseUsage | None) -> TokenUsage | None:
        """Convert the token counts, if OpenAI reported them.

        Args:
            usage: Token counts of a response, if sent.

        Returns:
            The shared counts, or ``None`` without counts.
        """
        return None if usage is None else self.usage_conversion(usage)

    def _fields(self, request: GenerationRequest) -> dict[str, Any]:
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
            "input": self.input_conversion(request.input),
            "instructions": request.instructions,
            "max_output_tokens": request.max_output_tokens,
            "metadata": request.metadata,
        }
