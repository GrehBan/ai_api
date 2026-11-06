"""Methods of Anthropic's Messages API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final

from pydantic import MISSING, BaseModel, Field, field_validator

from aiollm.enums import BodyFormat, ResponseFormat
from aiollm.http.headers import ANTHROPIC_VERSION
from aiollm.methods.base import BaseAPIMethod, BaseStreamMethod
from aiollm.providers.anthropic.api import API_VERSION
from aiollm.providers.anthropic.methods.params.messages import (
    AssistantMessageParam,
    CreateMessageParams,
    MetadataParam,
    StreamMessageParams,
    SystemMessageParam,
    UserMessageParam,
)
from aiollm.providers.anthropic.types.events import MessageStreamEvent
from aiollm.providers.anthropic.types.messages import Message
from aiollm.types.matchers import EventTypes

VERSION_HEADERS: Final[Mapping[str, str]] = {ANTHROPIC_VERSION: API_VERSION}
"""``__request_headers__`` of every Anthropic method.

Anthropic requires ``anthropic-version`` on every request, and the
version fixes the request and response format, so a method written
against ``API_VERSION`` sends it itself. A method of your own sets
``__request_headers__ = VERSION_HEADERS`` too.
"""

TERMINAL_EVENTS: Final = EventTypes("message_stop")
"""The event that ends a complete Messages stream."""


USER_ID_LENGTH: Final = 512
"""The longest ``metadata.user_id`` Anthropic takes."""


class MessageLimits(BaseModel):
    """Limits of a message request that Anthropic's schema leaves out.

    The generated ``methods.params`` follow the schema, which names no
    limit here, while the API answers 400 past them; this mixin, first in
    both methods, checks them before the request is sent.

    Attributes:
        messages: The conversation; at least one turn.
        max_tokens: Upper bound on generated tokens; at least 0 (0 only
            fills the prompt cache).
        metadata: Metadata of the request; ``user_id`` up to 512
            characters.
    """

    messages: Annotated[
        list[UserMessageParam | AssistantMessageParam | SystemMessageParam],
        Field(min_length=1),
    ]
    max_tokens: Annotated[int, Field(ge=0)]
    metadata: MetadataParam | MISSING = MISSING

    @field_validator("metadata")
    @classmethod
    def _user_id_fits(cls, value: MetadataParam | MISSING) -> object:
        """Refuse a ``user_id`` longer than Anthropic takes.

        Args:
            value: The metadata, if given.

        Returns:
            The metadata as it is.

        Raises:
            ValueError: ``user_id`` is longer than 512 characters.
        """
        if value is MISSING:
            return value
        user_id = value.user_id
        if isinstance(user_id, str) and len(user_id) > USER_ID_LENGTH:
            message = f"user_id is longer than {USER_ID_LENGTH} characters"
            raise ValueError(message)
        return value


class CreateMessage(
    MessageLimits, CreateMessageParams, BaseAPIMethod[Message]
):
    """Create a message and wait for the whole of it.

    A transient failure (a lost connection, a timeout, a 5xx, the 529
    ``overloaded_error``, a 429 with ``Retry-After``) is retried, as the
    official SDKs do. A timeout or a lost connection may hide a
    generation that ran on the server, a risk the SDKs accept too.

    Example:
        >>> message = await client.send_request(
        ...     CreateMessage(
        ...         model="claude-opus-5-5",
        ...         max_tokens=1024,
        ...         messages=[{"role": "user", "content": "Hi"}],
        ...     )
        ... )
        >>> message.content
    """

    __request_path__ = "/messages"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __request_headers__ = VERSION_HEADERS
    __retryable__ = True


class StreamMessage(
    MessageLimits,
    StreamMessageParams,
    BaseStreamMethod[MessageStreamEvent],
):
    """Create a message and stream its events.

    The stream is ``message_start``, content blocks (each a
    ``content_block_start``, deltas and a ``content_block_stop``), one or
    more ``message_delta`` and ``message_stop``, with ``ping`` events in
    between; an ``error`` event is raised as an ``APIStatusError``. Only
    opening the stream is retried, on a transient failure. A body
    that ends before ``message_stop`` raises
    ``APIStreamInterruptedError``.

    Its ``stream`` field is always ``True``: it asks for server-sent
    events.

    Example:
        >>> method = StreamMessage(
        ...     model="claude-opus-5-5",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hi"}],
        ... )
        >>> async with client.stream(method) as events:
        ...     async for event in events:
        ...         if isinstance(event, ContentBlockDeltaEvent):
        ...             ...
    """

    __request_path__ = "/messages"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __request_headers__ = VERSION_HEADERS
    __retryable__ = True
    __response_format__ = ResponseFormat.SSE
    __stream_terminal__ = TERMINAL_EVENTS
