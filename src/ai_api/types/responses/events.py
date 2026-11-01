"""Shared events of a streamed generation."""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import Field

from ai_api.enums import GenerationEventType, IncompleteReason, Provider
from ai_api.types.base import BaseAPIType
from ai_api.types.matchers import EventTypes
from ai_api.types.responses.generation import TokenUsage


class BaseResponseEvent(BaseAPIType):
    """Base class of streamed events in shared types."""


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


class OtherEvent[RawT](RawResponseEvent[RawT]):
    """A provider event without a shared counterpart.

    Attributes:
        type: Always ``GenerationEventType.OTHER``.
    """

    type: Literal[GenerationEventType.OTHER] = GenerationEventType.OTHER


type GenerationEvent[RawT] = Annotated[
    TextDeltaEvent[RawT] | CompletedEvent[RawT] | OtherEvent[RawT],
    Field(discriminator="type"),
]
"""Any shared event of a generation stream, told apart by ``type``."""

GENERATION_TERMINAL: Final = EventTypes(GenerationEventType.COMPLETED)
"""The last event of a complete generation stream, ``CompletedEvent``.

``GenerationResource`` checks it: an event after it raises
``APIResponseError`` and a stream without it raises
``APIStreamInterruptedError``.
"""
