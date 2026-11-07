"""Shared events of a streamed generation."""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import ConfigDict, Field

from aiollm.enums import GenerationEventType, IncompleteReason, Provider
from aiollm.types.base import BaseAPIType
from aiollm.types.matchers import EventTypes
from aiollm.types.requests.generation import (
    NativeOutput,
    ToolCall,
    ToolCallTurn,
    Turn,
)
from aiollm.types.responses.generation import TokenUsage, continuation


class BaseResponseEvent(BaseAPIType):
    """Base class of streamed events in shared types.

    As for ``BaseResponse``, the provider's event in ``raw`` may be of
    any type, so any type is allowed here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RawResponseEvent[RawT](BaseResponseEvent):
    """An event that keeps the provider's own event.

    Attributes:
        raw: The provider's event.
    """

    raw: RawT


class TextDeltaEvent[RawT](RawResponseEvent[RawT]):
    """A piece of generated text.

    Attributes:
        type: Always ``GenerationEventType.TEXT_DELTA``.
        text: The new text.
    """

    type: Literal[GenerationEventType.TEXT_DELTA] = (
        GenerationEventType.TEXT_DELTA
    )
    text: str


class ReasoningDeltaEvent[RawT](RawResponseEvent[RawT]):
    """A piece of a summary of the model's reasoning.

    Only sent when the request asked for a summary through
    ``provider_options``; ``GenerationStream.text()`` skips it.

    Attributes:
        type: Always ``GenerationEventType.REASONING_DELTA``.
        text: The new text of the summary.
    """

    type: Literal[GenerationEventType.REASONING_DELTA] = (
        GenerationEventType.REASONING_DELTA
    )
    text: str


class ToolCallStartEvent[RawT](RawResponseEvent[RawT]):
    """The model begins a call of one of the request's tools.

    Attributes:
        type: Always ``GenerationEventType.TOOL_CALL_START``.
        call_id: The call's id, as ``ToolCall.id``.
        name: The tool called.
    """

    type: Literal[GenerationEventType.TOOL_CALL_START] = (
        GenerationEventType.TOOL_CALL_START
    )
    call_id: str
    name: str


class ToolCallDeltaEvent[RawT](RawResponseEvent[RawT]):
    """A piece of a tool call's arguments.

    The pieces of one call, joined, are its JSON text; a piece alone is
    rarely valid JSON.

    Attributes:
        type: Always ``GenerationEventType.TOOL_CALL_DELTA``.
        call_id: The call's id, as on its ``ToolCallStartEvent``.
        arguments: The new piece of the arguments.
    """

    type: Literal[GenerationEventType.TOOL_CALL_DELTA] = (
        GenerationEventType.TOOL_CALL_DELTA
    )
    call_id: str
    arguments: str


class ToolCallDoneEvent[RawT](RawResponseEvent[RawT]):
    """A tool call is complete and may be run.

    It comes before the end of the response, so a tool can start while
    the model writes on. A response that stops early (see
    ``CompletedEvent.incomplete_reason``) cuts its last call before this
    event.

    Attributes:
        type: Always ``GenerationEventType.TOOL_CALL_DONE``.
        call: The whole call.
    """

    type: Literal[GenerationEventType.TOOL_CALL_DONE] = (
        GenerationEventType.TOOL_CALL_DONE
    )
    call: ToolCall


class CompletedEvent[RawT](RawResponseEvent[RawT]):
    """The generation finished, or stopped early.

    Every complete generation stream has exactly one, and it is the last
    event. A failed generation raises an ``APIStatusError`` instead, in
    a stream as in ``create()``; a refusal is not a failure and ends
    with this event too.

    It also carries what identifies the whole response and its whole
    text, so a stream's caller does not have to collect them.

    Attributes:
        type: Always ``GenerationEventType.COMPLETED``.
        id: The provider's id of the response, if it has one.
        provider: Provider that generated the response, as on
            ``GenerationResponse``.
        model: The model that generated the response.
        text: The whole generated text, or ``None`` without text.
        usage: Token counts, if reported.
        incomplete_reason: Why the provider stopped before the end, in
            terms every provider has; ``None`` if the generation
            finished.
        raw_incomplete_reason: The provider's own reason, if it gave
            one.
        refusal: Why the model refused, if it did and the provider says
            why.
        reasoning: A summary of the model's reasoning, when the
            provider sent one (OpenAI's ``reasoning.summary``,
            Anthropic's ``thinking.display`` of ``summarized``, both
            set through ``provider_options``); ``None`` otherwise.
        tool_calls: The calls of the request's tools, as on
            ``GenerationResponse``.
        native: The stream's output as its provider's next request takes
            it, as ``GenerationResponse.native``.
    """

    type: Literal[GenerationEventType.COMPLETED] = (
        GenerationEventType.COMPLETED
    )
    id: str | None = None
    provider: Provider
    model: str
    text: str | None = None
    usage: TokenUsage | None = None
    incomplete_reason: IncompleteReason | None = None
    raw_incomplete_reason: str | None = None
    refusal: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    native: NativeOutput | None = None

    def to_turn(self) -> Turn | ToolCallTurn:
        """Turn the streamed generation into the next conversation item.

        The same rule as ``GenerationResponse.to_turn()``.

        Returns:
            A ``ToolCallTurn`` when the model called tools, else an
            ``assistant`` turn with ``text``.

        Raises:
            ValueError: There are neither calls nor text, or there are
                calls but the generation stopped early.
        """
        return continuation(
            "the stream" if self.id is None else f"stream {self.id}",
            text=self.text,
            tool_calls=self.tool_calls,
            incomplete_reason=self.incomplete_reason,
            native=self.native,
        )


class OtherEvent[RawT](RawResponseEvent[RawT]):
    """A provider event without a shared counterpart.

    Attributes:
        type: Always ``GenerationEventType.OTHER``.
    """

    type: Literal[GenerationEventType.OTHER] = GenerationEventType.OTHER


type GenerationEvent[RawT] = Annotated[
    TextDeltaEvent[RawT]
    | ReasoningDeltaEvent[RawT]
    | ToolCallStartEvent[RawT]
    | ToolCallDeltaEvent[RawT]
    | ToolCallDoneEvent[RawT]
    | CompletedEvent[RawT]
    | OtherEvent[RawT],
    Field(discriminator="type"),
]
"""Any shared event of a generation stream, told apart by ``type``."""

GENERATION_TERMINAL: Final = EventTypes(GenerationEventType.COMPLETED)
"""The last event of a complete generation stream, ``CompletedEvent``.

``GenerationResource`` checks it: an event after it raises
``APIResponseError`` and a stream without it raises
``APIStreamInterruptedError``.
"""
