"""Stream events of Anthropic's Messages API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal

from pydantic import Discriminator, Tag

from ai_api.providers.anthropic.types.messages import (
    ContentBlock,
    Message,
    StopDetails,
)
from ai_api.types.base import BaseAPIType
from ai_api.types.unknown import UNKNOWN, BaseUnknown, by_type


class MessageStartEvent(BaseAPIType):
    """The stream started: a message without content yet.

    Attributes:
        type: Always ``"message_start"``.
        message: The message, with empty ``content``; its ``usage``
            carries the input tokens.
    """

    type: Literal["message_start"] = "message_start"
    message: Message


class ContentBlockStartEvent(BaseAPIType):
    """A content block started.

    Attributes:
        type: Always ``"content_block_start"``.
        index: Position of the block in the message ``content``.
        content_block: The block, without its streamed part yet.
    """

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class TextDelta(BaseAPIType):
    """New text of a ``text`` block.

    Attributes:
        type: Always ``"text_delta"``.
        text: The new text.
    """

    type: Literal["text_delta"] = "text_delta"
    text: str


class UnknownDelta(BaseUnknown):
    """A delta the library does not model yet, kept whole.

    Tool input JSON, thinking, signatures and citations arrive as this.
    """


type ContentDelta = Annotated[
    Annotated[TextDelta, Tag("text_delta")]
    | Annotated[UnknownDelta, Tag(UNKNOWN)],
    Discriminator(by_type({"text_delta": "text_delta"})),
]
"""A delta of a content block, told apart by ``type``."""


class ContentBlockDeltaEvent(BaseAPIType):
    """A part of a content block.

    Attributes:
        type: Always ``"content_block_delta"``.
        index: Position of the block in the message ``content``.
        delta: The part.
    """

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ContentDelta


class ContentBlockStopEvent(BaseAPIType):
    """A content block is complete.

    Attributes:
        type: Always ``"content_block_stop"``.
        index: Position of the block in the message ``content``.
    """

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDelta(BaseAPIType):
    """Top-level changes to the message.

    Attributes:
        stop_reason: Why generation stopped, such as ``"max_tokens"``.
        stop_sequence: The custom stop sequence generated, if any.
        stop_details: More about the stop, such as a refusal's category.
    """

    stop_reason: str | None = None
    stop_sequence: str | None = None
    stop_details: StopDetails | None = None


class MessageDeltaUsage(BaseAPIType):
    """Token counts so far; cumulative, not an increment.

    Attributes:
        input_tokens: Input tokens, if sent again.
        output_tokens: Generated tokens so far.
        cache_creation_input_tokens: Input tokens written to the cache.
        cache_read_input_tokens: Input tokens read from the cache.
    """

    input_tokens: int | None = None
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class MessageDeltaEvent(BaseAPIType):
    """Top-level changes to the message, such as the stop reason.

    Attributes:
        type: Always ``"message_delta"``.
        delta: The changes.
        usage: Cumulative token counts, if sent.
    """

    type: Literal["message_delta"] = "message_delta"
    delta: MessageDelta
    usage: MessageDeltaUsage | None = None


class MessageStopEvent(BaseAPIType):
    """The stream is complete.

    Attributes:
        type: Always ``"message_stop"``.
    """

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseAPIType):
    """Keeps the connection alive; carries nothing.

    Attributes:
        type: Always ``"ping"``.
    """

    type: Literal["ping"] = "ping"


class UnknownEvent(BaseUnknown):
    """An event the library does not model yet, kept whole."""


_EVENT_TAGS: Final[Mapping[str, str]] = {
    "message_start": "message_start",
    "content_block_start": "content_block_start",
    "content_block_delta": "content_block_delta",
    "content_block_stop": "content_block_stop",
    "message_delta": "message_delta",
    "message_stop": "message_stop",
    "ping": "ping",
}
"""Tag of each modeled event type; the rest become ``UnknownEvent``."""


type MessageStreamEvent = Annotated[
    Annotated[MessageStartEvent, Tag("message_start")]
    | Annotated[ContentBlockStartEvent, Tag("content_block_start")]
    | Annotated[ContentBlockDeltaEvent, Tag("content_block_delta")]
    | Annotated[ContentBlockStopEvent, Tag("content_block_stop")]
    | Annotated[MessageDeltaEvent, Tag("message_delta")]
    | Annotated[MessageStopEvent, Tag("message_stop")]
    | Annotated[PingEvent, Tag("ping")]
    | Annotated[UnknownEvent, Tag(UNKNOWN)],
    Discriminator(by_type(_EVENT_TAGS)),
]
"""Any event of a message stream, told apart by ``type``.

The ``error`` event never arrives as an item: the stream parser raises it
as an ``APIStatusError``.
"""
