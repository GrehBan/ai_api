"""Stream events of Anthropic's Messages API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal

from pydantic import Discriminator, Tag

from aiollm.providers.anthropic.types.messages import (
    ContentBlock,
    Message,
    OutputTokensDetails,
    StopDetails,
)
from aiollm.types.base import BaseAPITypeExtra
from aiollm.types.unknown import UNKNOWN, BaseUnknown, by_type


class MessageStartEvent(BaseAPITypeExtra):
    """The stream started: a message without content yet.

    Attributes:
        type: Always ``"message_start"``.
        message: The message, with empty ``content``; its ``usage``
            carries the input tokens.
    """

    __upstream_schema__ = "RawMessageStartEvent"

    type: Literal["message_start"] = "message_start"
    message: Message


class ContentBlockStartEvent(BaseAPITypeExtra):
    """A content block started.

    Attributes:
        type: Always ``"content_block_start"``.
        index: Position of the block in the message ``content``.
        content_block: The block, without its streamed part yet.
    """

    __upstream_schema__ = "RawContentBlockStartEvent"

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class TextDelta(BaseAPITypeExtra):
    """New text of a ``text`` block.

    Attributes:
        type: Always ``"text_delta"``.
        text: The new text.
    """

    __upstream_schema__ = "TextDelta"

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseAPITypeExtra):
    """A piece of thinking in a ``content_block_delta``.

    Attributes:
        type: Always ``"thinking_delta"``.
        thinking: The new thinking.
    """

    __upstream_schema__ = "ThinkingDelta"

    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class UnknownDelta(BaseUnknown):
    """A delta the library does not model yet, kept whole.

    Tool input JSON, thinking, signatures and citations arrive as this.
    """


type ContentDelta = Annotated[
    Annotated[TextDelta, Tag("text_delta")]
    | Annotated[ThinkingDelta, Tag("thinking_delta")]
    | Annotated[UnknownDelta, Tag(UNKNOWN)],
    Discriminator(
        by_type(
            {"text_delta": "text_delta", "thinking_delta": "thinking_delta"}
        )
    ),
]
"""A delta of a content block, told apart by ``type``."""


class ContentBlockDeltaEvent(BaseAPITypeExtra):
    """A part of a content block.

    Attributes:
        type: Always ``"content_block_delta"``.
        index: Position of the block in the message ``content``.
        delta: The part.
    """

    __upstream_schema__ = "RawContentBlockDeltaEvent"

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: ContentDelta


class ContentBlockStopEvent(BaseAPITypeExtra):
    """A content block is complete.

    Attributes:
        type: Always ``"content_block_stop"``.
        index: Position of the block in the message ``content``.
    """

    __upstream_schema__ = "RawContentBlockStopEvent"

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDelta(BaseAPITypeExtra):
    """Top-level changes to the message.

    Attributes:
        stop_reason: Why generation stopped, such as ``"max_tokens"``.
        stop_sequence: The custom stop sequence generated, if any.
        stop_details: More about the stop, such as a refusal's category.
    """

    __upstream_schema__ = "Delta"

    stop_reason: str | None = None
    stop_sequence: str | None = None
    stop_details: StopDetails | None = None


class MessageDeltaUsage(BaseAPITypeExtra):
    """Token counts so far; cumulative, not an increment.

    Attributes:
        input_tokens: Input tokens, if sent again.
        output_tokens: Generated tokens so far, thinking included.
        cache_creation_input_tokens: Input tokens written to the cache.
        cache_read_input_tokens: Input tokens read from the cache.
        output_tokens_details: Breakdown of the output tokens, if sent.
    """

    __upstream_schema__ = "MessageDeltaUsage"

    input_tokens: int | None = None
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens_details: OutputTokensDetails | None = None


class MessageDeltaEvent(BaseAPITypeExtra):
    """Top-level changes to the message, such as the stop reason.

    Attributes:
        type: Always ``"message_delta"``.
        delta: The changes.
        usage: Cumulative token counts, if sent.
    """

    __upstream_schema__ = "RawMessageDeltaEvent"

    type: Literal["message_delta"] = "message_delta"
    delta: MessageDelta
    usage: MessageDeltaUsage | None = None


class MessageStopEvent(BaseAPITypeExtra):
    """The stream is complete.

    Attributes:
        type: Always ``"message_stop"``.
    """

    __upstream_schema__ = "RawMessageStopEvent"

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseAPITypeExtra):
    """Keeps the connection alive; carries nothing.

    Attributes:
        type: Always ``"ping"``.
    """

    # Not in the upstream stream schema.
    __upstream_schema__ = ()

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
