"""Shared events of a streamed generation."""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import Field

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
        type: Always ``"text_delta"``.
        text: The new text.
    """

    type: Literal["text_delta"] = "text_delta"
    text: str


class CompletedEvent[RawT](RawResponseEvent[RawT]):
    """The generation finished, or stopped early.

    Every complete generation stream has exactly one, and it is the last
    event. A failed generation raises an ``APIStatusError`` instead, in
    a stream as in ``create()``.

    Attributes:
        type: Always ``"completed"``.
        usage: Token counts, if reported.
        incomplete_reason: Why the provider stopped before the end, such
            as a token limit, in the provider's own words; ``None`` if
            the generation finished.
    """

    type: Literal["completed"] = "completed"
    usage: TokenUsage | None = None
    incomplete_reason: str | None = None


class OtherEvent[RawT](RawResponseEvent[RawT]):
    """A provider event without a shared counterpart.

    Attributes:
        type: Always ``"other"``.
    """

    type: Literal["other"] = "other"


type GenerationEvent[RawT] = Annotated[
    TextDeltaEvent[RawT] | CompletedEvent[RawT] | OtherEvent[RawT],
    Field(discriminator="type"),
]
"""Any shared event of a generation stream, told apart by ``type``."""

GENERATION_TERMINAL: Final = EventTypes("completed")
"""The last event of a complete generation stream, ``CompletedEvent``.

``GenerationResource`` checks it: an event after it raises
``APIResponseError`` and a stream without it raises
``APIStreamInterruptedError``.
"""
