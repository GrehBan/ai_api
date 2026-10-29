"""Shared events of a streamed generation."""

from typing import Annotated, Literal

from pydantic import Field

from ai_api.types.base import BaseAPIType
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
    """The generation finished.

    Attributes:
        type: Always ``"completed"``.
        usage: Token counts, if reported.
    """

    type: Literal["completed"] = "completed"
    usage: TokenUsage | None = None


class FailedEvent[RawT](RawResponseEvent[RawT]):
    """The provider reported, as a regular event, that generation failed.

    Only for a failure the provider sends as part of the stream, such as
    OpenAI's ``response.failed``. Errors of the protocol itself (an SSE
    ``error`` event, an error status, a broken connection) are raised as
    exceptions by the session and never become this event.

    Attributes:
        type: Always ``"failed"``.
        message: The provider's description of the failure.
    """

    type: Literal["failed"] = "failed"
    message: str


class OtherEvent[RawT](RawResponseEvent[RawT]):
    """A provider event without a shared counterpart.

    Attributes:
        type: Always ``"other"``.
    """

    type: Literal["other"] = "other"


type ResponseEvent[RawT] = Annotated[
    TextDeltaEvent[RawT]
    | CompletedEvent[RawT]
    | FailedEvent[RawT]
    | OtherEvent[RawT],
    Field(discriminator="type"),
]
"""Any shared event, told apart by ``type``."""
