"""Methods of OpenAI's Responses API."""

from __future__ import annotations

from typing import Annotated, Final

from annotated_types import MaxLen
from pydantic import MISSING, StringConstraints

from aiollm.enums import BodyFormat, ResponseFormat
from aiollm.methods.base import BaseAPIMethod, BaseStreamMethod
from aiollm.providers.openai.methods.params.responses import (
    CreateResponseParams,
    StreamResponseParams,
)
from aiollm.providers.openai.resolver import FAILED_RESPONSE
from aiollm.providers.openai.types.events import ResponseStreamEvent
from aiollm.providers.openai.types.responses import Response
from aiollm.response.parse.buffered import BufferedResponseParser
from aiollm.response.parse.stream import StreamResponseParser
from aiollm.types.matchers import EventTypes, TopLevelError

type Metadata = Annotated[
    dict[
        Annotated[str, StringConstraints(max_length=64)],
        Annotated[str, StringConstraints(max_length=512)],
    ],
    MaxLen(16),
]
"""OpenAI metadata: up to 16 pairs, keys up to 64 and values up to 512
characters, checked before the request is sent.

The limits are only written in the schema's description, so the
generated ``methods.params`` type does not check them; both methods
narrow their ``metadata`` field to this type.
"""

TERMINAL_EVENTS: Final = EventTypes(
    "response.completed", "response.incomplete", "response.failed"
)
"""The events that end a complete Responses stream."""

RESPONSE_ERROR_EVENTS: Final = (
    EventTypes("error", "response.failed") | TopLevelError()
)
"""The events of a Responses stream that are raised as errors.

The parser's default ``ERROR_EVENTS`` plus ``response.failed``; named
apart so the two are not mistaken for each other.
"""


class CreateResponse(CreateResponseParams, BaseAPIMethod[Response]):
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
        >>> response.output
    """

    __request_path__ = "/responses"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __retryable__ = True
    __response_parser__ = BufferedResponseParser(
        error_responses=FAILED_RESPONSE
    )

    metadata: Metadata | None | MISSING = MISSING


class StreamResponse(
    StreamResponseParams, BaseStreamMethod[ResponseStreamEvent]
):
    """Create a response and stream its events.

    Only opening the stream is retried, on a transient failure; once
    events arrive, a failure reaches the caller. The stream ends with
    ``response.completed`` or ``response.incomplete``; a body that ends
    before one of them raises ``APIStreamInterruptedError``. An
    ``error`` event and ``response.failed`` are raised as the
    ``APIStatusError`` that ``OpenAIErrorResolver`` picks from their
    code; for ``response.failed`` the event is the error's ``raw``.

    Its ``stream`` field is always ``True``: it asks for server-sent
    events.

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
    __stream_parser__ = StreamResponseParser(
        error_events=RESPONSE_ERROR_EVENTS
    )

    metadata: Metadata | None | MISSING = MISSING
