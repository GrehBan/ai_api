"""Methods of OpenAI's Responses API."""

from __future__ import annotations

from typing import Final, Literal

from ai_api.enums import BodyFormat, ResponseFormat
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.openai.methods.base import ResponseParams
from ai_api.providers.openai.resolver import FAILED_RESPONSE
from ai_api.providers.openai.types.events import ResponseStreamEvent
from ai_api.providers.openai.types.responses import Response
from ai_api.response.parse.buffered import BufferedResponseParser
from ai_api.response.parse.stream import StreamResponseParser
from ai_api.types.matchers import EventTypes, TopLevelError

TERMINAL_EVENTS: Final = EventTypes(
    "response.completed", "response.incomplete", "response.failed"
)
"""The events that end a complete Responses stream."""

ERROR_EVENTS: Final = EventTypes("error", "response.failed") | TopLevelError()
"""The events of a Responses stream that are raised as errors."""


class CreateResponse(ResponseParams, BaseAPIMethod[Response]):
    """Create a response and wait for the whole of it.

    A response with status ``failed`` is raised as the ``APIStatusError``
    that ``OpenAIErrorResolver`` picks from its ``error.code`` (such as a
    transient ``InternalServerError`` for ``server_error``), with the
    ``Response`` as the error's ``raw``. This is stricter than the
    official SDK, which returns it as data, and matches its
    ``get_final_response()`` and Anthropic, where a failure is always an
    error. A transient failure (a lost connection, a timeout, a 5xx, a
    429 with ``Retry-After``) is retried, as the official SDKs do. A
    timeout or a lost connection may hide a generation that ran on the
    server, a risk the SDKs accept too.

    Example:
        >>> response = await client.send_request(
        ...     CreateResponse(model="gpt-5", input="Hi")
        ... )
        >>> response.output_text
    """

    __request_path__ = "/responses"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __retryable__ = True
    __response_parser__ = BufferedResponseParser(
        error_responses=FAILED_RESPONSE
    )


class StreamResponse(ResponseParams, BaseStreamMethod[ResponseStreamEvent]):
    """Create a response and stream its events.

    Only opening the stream is retried, on a transient failure; once
    events arrive, a failure reaches the caller. The stream ends with
    ``response.completed`` or ``response.incomplete``; a body that ends
    before one of them raises ``APIStreamInterruptedError``. An
    ``error`` event and ``response.failed`` are raised as the
    ``APIStatusError`` that ``OpenAIErrorResolver`` picks from their
    code; for ``response.failed`` the event is the error's ``raw``.

    Attributes:
        stream: Always ``True``; asks OpenAI for server-sent events.

    Example:
        >>> method = StreamResponse(model="gpt-5", input="Hi")
        >>> async with client.stream(method) as events:
        ...     async for event in events:
        ...         if isinstance(event, OutputTextDeltaEvent):
        ...             print(event.delta, end="")
    """

    __request_path__ = "/responses"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __retryable__ = True
    __response_format__ = ResponseFormat.SSE
    __stream_terminal__ = TERMINAL_EVENTS
    __stream_parser__ = StreamResponseParser(error_events=ERROR_EVENTS)

    stream: Literal[True] = True
