"""Server-sent events of a streamed OpenAI response."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal

from pydantic import Discriminator, Tag

from ai_api.providers.openai.types.responses import (
    OutputContent,
    OutputItem,
    Response,
)
from ai_api.types.base import BaseAPIType
from ai_api.types.unknown import UNKNOWN, BaseUnknown, by_type


class BaseResponseStreamEvent(BaseAPIType):
    """Fields every event of a response stream has.

    Attributes:
        sequence_number: Position of the event in the stream.
    """

    sequence_number: int


class ResponseStateEvent(BaseResponseStreamEvent):
    """The response changed state; carries the whole response.

    ``response.completed`` is the last event of a successful stream and
    has the usage. ``response.failed`` (with ``response.error``) and
    ``response.incomplete`` (with ``response.incomplete_details``) end it
    otherwise; they are regular events, not errors of the stream.

    Attributes:
        type: The new state.
        response: The response as of this event.
    """

    type: Literal[
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.completed",
        "response.failed",
        "response.incomplete",
    ]
    response: Response


class OutputItemEvent(BaseResponseStreamEvent):
    """An output item was added or finished.

    Attributes:
        type: ``"response.output_item.added"`` or ``".done"``.
        output_index: Position of the item in ``Response.output``.
        item: The item as of this event.
    """

    type: Literal[
        "response.output_item.added",
        "response.output_item.done",
    ]
    output_index: int
    item: OutputItem


class ContentPartEvent(BaseResponseStreamEvent):
    """A content part of a message was added or finished.

    Attributes:
        type: ``"response.content_part.added"`` or ``".done"``.
        item_id: The message the part belongs to.
        output_index: Position of the message in ``Response.output``.
        content_index: Position of the part in the message.
        part: The part as of this event.
    """

    type: Literal[
        "response.content_part.added",
        "response.content_part.done",
    ]
    item_id: str
    output_index: int
    content_index: int
    part: OutputContent


class BaseContentEvent(BaseResponseStreamEvent):
    """Fields of the events about one content part.

    Attributes:
        item_id: The message the part belongs to.
        output_index: Position of the message in ``Response.output``.
        content_index: Position of the part in the message.
    """

    item_id: str
    output_index: int
    content_index: int


class OutputTextDeltaEvent(BaseContentEvent):
    """A piece of generated text.

    Attributes:
        type: Always ``"response.output_text.delta"``.
        delta: The new text.
    """

    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    delta: str


class OutputTextDoneEvent(BaseContentEvent):
    """A text part is finished.

    Attributes:
        type: Always ``"response.output_text.done"``.
        text: The whole text of the part.
    """

    type: Literal["response.output_text.done"] = "response.output_text.done"
    text: str


class RefusalDeltaEvent(BaseContentEvent):
    """A piece of a refusal.

    Attributes:
        type: Always ``"response.refusal.delta"``.
        delta: The new text of the refusal.
    """

    type: Literal["response.refusal.delta"] = "response.refusal.delta"
    delta: str


class RefusalDoneEvent(BaseContentEvent):
    """A refusal is finished.

    Attributes:
        type: Always ``"response.refusal.done"``.
        refusal: The whole refusal.
    """

    type: Literal["response.refusal.done"] = "response.refusal.done"
    refusal: str


class UnknownEvent(BaseUnknown):
    """An event the library does not model yet, kept whole.

    Tool calls, reasoning summaries, annotations and every event OpenAI
    adds later arrive as this, with all their fields as extra fields.

    Attributes:
        sequence_number: Position of the event in the stream, if sent.
    """

    sequence_number: int | None = None


_EVENT_TAGS: Final[Mapping[str, str]] = {
    "response.created": "state",
    "response.queued": "state",
    "response.in_progress": "state",
    "response.completed": "state",
    "response.failed": "state",
    "response.incomplete": "state",
    "response.output_item.added": "item",
    "response.output_item.done": "item",
    "response.content_part.added": "part",
    "response.content_part.done": "part",
    "response.output_text.delta": "text_delta",
    "response.output_text.done": "text_done",
    "response.refusal.delta": "refusal_delta",
    "response.refusal.done": "refusal_done",
}
"""Tag of each modeled event type; the rest become ``UnknownEvent``."""


type ResponseStreamEvent = Annotated[
    Annotated[ResponseStateEvent, Tag("state")]
    | Annotated[OutputItemEvent, Tag("item")]
    | Annotated[ContentPartEvent, Tag("part")]
    | Annotated[OutputTextDeltaEvent, Tag("text_delta")]
    | Annotated[OutputTextDoneEvent, Tag("text_done")]
    | Annotated[RefusalDeltaEvent, Tag("refusal_delta")]
    | Annotated[RefusalDoneEvent, Tag("refusal_done")]
    | Annotated[UnknownEvent, Tag(UNKNOWN)],
    Discriminator(by_type(_EVENT_TAGS)),
]
"""Any event of a response stream, told apart by ``type``.

The ``error`` event never arrives as an item: the stream parser raises it
as an ``APIStatusError``.
"""
