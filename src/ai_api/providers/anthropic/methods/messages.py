"""Methods of Anthropic's Messages API."""

from __future__ import annotations

from typing import Literal

from ai_api.enums import BodyFormat, ResponseFormat
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.anthropic.methods.base import (
    VERSION_HEADERS,
    MessageParams,
)
from ai_api.providers.anthropic.types.events import MessageStreamEvent
from ai_api.providers.anthropic.types.messages import Message


class CreateMessage(MessageParams, BaseAPIMethod[Message]):
    """Create a message and wait for the whole of it.

    Example:
        >>> message = await client.send_request(
        ...     CreateMessage(
        ...         model="claude-opus-5-5",
        ...         max_tokens=1024,
        ...         messages=[{"role": "user", "content": "Hi"}],
        ...     )
        ... )
        >>> message.text
    """

    __request_path__ = "/messages"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __request_headers__ = VERSION_HEADERS


class StreamMessage(MessageParams, BaseStreamMethod[MessageStreamEvent]):
    """Create a message and stream its events.

    The stream is ``message_start``, content blocks (each a
    ``content_block_start``, deltas and a ``content_block_stop``), one or
    more ``message_delta`` and ``message_stop``, with ``ping`` events in
    between; an ``error`` event is raised as an ``APIStatusError``.

    Attributes:
        stream: Always ``True``; asks Anthropic for server-sent events.

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
    __response_format__ = ResponseFormat.SSE

    stream: Literal[True] = True
