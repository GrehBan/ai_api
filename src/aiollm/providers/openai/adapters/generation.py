"""Adapter of shared generation to OpenAI's Responses API."""

from __future__ import annotations

import copy
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Self

from pydantic import MISSING

from aiollm.adapters.base import BaseCompositeAdapter
from aiollm.adapters.validation import (
    check_conversation,
    checked_options,
    merge_options,
    request_errors,
)
from aiollm.enums import ToolChoice
from aiollm.providers.openai.adapters.conversions import (
    INPUT_CONVERSION,
    STOP_CONVERSION,
    USAGE_CONVERSION,
    check_status,
    native_output,
    output_text,
    reasoning_summary,
    tool_calls,
)
from aiollm.providers.openai.api import PROVIDER
from aiollm.providers.openai.methods.responses import (
    CreateResponse,
    StreamResponse,
)
from aiollm.providers.openai.types import (
    Response,
    ResponseCompletedEvent,
    ResponseIncompleteEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from aiollm.providers.openai.types.options import OpenAIGenerationOptions
from aiollm.types.requests.generation import (
    ForcedTool,
    GenerationRequest,
    InputItem,
    Tool,
)
from aiollm.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
)
from aiollm.types.responses.generation import GenerationResponse

if TYPE_CHECKING:
    from aiollm.adapters.protocols import IConversion
    from aiollm.adapters.stop import Stop
    from aiollm.providers.openai.types import ResponseUsage
    from aiollm.types.responses.generation import TokenUsage


SHARED_FIELDS: Final[Mapping[str, str]] = {
    "safety_identifier": "end_user_id",
}
"""Shared request field of each method field with another name."""

DEFAULT_SCHEMA_NAME: Final = "output"
"""``name`` of a JSON Schema format when the adapter is given none."""


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

    provider = PROVIDER

    def __init__(
        self,
        *,
        input_conversion: IConversion[
            str | Sequence[InputItem], str | list[dict[str, Any]]
        ] = INPUT_CONVERSION,
        usage_conversion: IConversion[
            ResponseUsage, TokenUsage
        ] = USAGE_CONVERSION,
        stop_conversion: IConversion[Response, Stop] = STOP_CONVERSION,
        schema_name: str = DEFAULT_SCHEMA_NAME,
    ) -> None:
        """Store the conversions and the name of an output schema.

        Args:
            input_conversion: Turns the shared ``input`` into OpenAI's.
            usage_conversion: Turns OpenAI's ``ResponseUsage`` into
                shared counts.
            stop_conversion: Tells how a finished response stopped.
            schema_name: The ``name`` OpenAI requires with a JSON Schema
                format; the shared ``output_schema`` has none.
        """
        self.input_conversion = input_conversion
        self.usage_conversion = usage_conversion
        self.stop_conversion = stop_conversion
        self.schema_name = schema_name
        self.provider_options: dict[str, Any] = {}

    def with_options(self, options: OpenAIGenerationOptions, /) -> Self:
        """Return a copy that sends the options with every method.

        Args:
            options: Fields of ``CreateResponse`` that the shared request
                does not hold, such as ``{"store": False}``.

        Returns:
            A copy of the adapter; this one is left as it is.

        Raises:
            RequestValidationError: A key is not one of the options.
        """
        adapter = copy.copy(self)
        adapter.provider_options = checked_options(
            options, OpenAIGenerationOptions, self.provider
        )
        return adapter

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
        with request_errors(self.provider, SHARED_FIELDS):
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
            provider=self.provider,
            model=raw.model,
            text=output_text(raw),
            usage=self._usage(raw.usage),
            incomplete_reason=stop.incomplete_reason,
            raw_incomplete_reason=stop.raw_incomplete_reason,
            refusal=stop.refusal,
            reasoning=reasoning_summary(raw),
            tool_calls=tool_calls(raw),
            native=native_output(raw),
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
        with request_errors(self.provider, SHARED_FIELDS):
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
            A text or reasoning summary delta, the completion (also of
            an incomplete response), or ``OtherEvent`` for every other
            event.
            ``response.failed`` never gets here: ``StreamResponse``
            raises it as an ``APIStatusError``.
        """
        match event:
            case ResponseTextDeltaEvent():
                return TextDeltaEvent[ResponseStreamEvent](
                    text=event.delta, raw=event
                )
            case ResponseReasoningSummaryTextDeltaEvent():
                return ReasoningDeltaEvent[ResponseStreamEvent](
                    text=event.delta, raw=event
                )
            case ResponseCompletedEvent() | ResponseIncompleteEvent():
                response = event.response
                stop = self.stop_conversion(response)
                return CompletedEvent[ResponseStreamEvent](
                    id=response.id,
                    provider=self.provider,
                    model=response.model,
                    text=output_text(response),
                    usage=self._usage(response.usage),
                    incomplete_reason=stop.incomplete_reason,
                    raw_incomplete_reason=stop.raw_incomplete_reason,
                    refusal=stop.refusal,
                    reasoning=reasoning_summary(response),
                    tool_calls=tool_calls(response),
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
            The fields that both methods take, the provider options of
            ``with_options`` included.

        Raises:
            RequestValidationError: The conversation does not end with a
                ``user`` turn.
        """
        check_conversation(request, self.provider)
        fields = {
            "model": request.model,
            "input": self.input_conversion(request.input),
            "instructions": request.instructions,
            "max_output_tokens": request.max_output_tokens,
            "safety_identifier": request.end_user_id,
            "reasoning": (
                MISSING
                if request.effort is MISSING
                else {"effort": request.effort.value}
            ),
            "text": (
                MISSING
                if request.output_schema is MISSING
                else {
                    "format": {
                        "type": "json_schema",
                        "name": self.schema_name,
                        "schema": request.output_schema,
                        "strict": True,
                    }
                }
            ),
            "tools": (
                MISSING
                if request.tools is MISSING
                else [_tool(tool) for tool in request.tools]
            ),
            "tool_choice": _tool_choice(request.tool_choice),
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        return merge_options(fields, self.provider_options, self.provider)


TOOL_CHOICES: Final[Mapping[ToolChoice, str]] = {
    ToolChoice.AUTO: "auto",
    ToolChoice.NONE: "none",
    ToolChoice.REQUIRED: "required",
}
"""OpenAI's ``tool_choice`` of each shared mode."""


def _tool(tool: Tool) -> dict[str, Any]:
    """Build OpenAI's function tool from a shared tool.

    Args:
        tool: The shared tool.

    Returns:
        A ``function`` tool. Its ``strict`` is required, so a tool that
        leaves it out sends ``null``, which OpenAI takes as not strict
        (seen in a live check, 2026-11-06), as Anthropic's default.
    """
    function: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": None if tool.strict is MISSING else tool.strict,
    }
    if tool.description is not MISSING:
        function["description"] = tool.description
    return function


def _tool_choice(
    choice: ToolChoice | ForcedTool | MISSING,
) -> str | dict[str, Any] | MISSING:
    """Build OpenAI's ``tool_choice``.

    Args:
        choice: The shared choice, if given.

    Returns:
        The mode, ``{"type": "function", "name": ...}`` for a forced
        tool, or ``MISSING`` when no choice is given.
    """
    if choice is MISSING:
        return MISSING
    if isinstance(choice, ForcedTool):
        return {"type": "function", "name": choice.name}
    return TOOL_CHOICES[choice]
