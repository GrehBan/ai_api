"""Methods of OpenAI's Responses API."""

from __future__ import annotations

from typing import Literal

from ai_api.enums import BodyFormat, ResponseFormat
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.openai.methods.base import ResponseParams
from ai_api.providers.openai.types.events import ResponseStreamEvent
from ai_api.providers.openai.types.responses import Response


class CreateResponse(ResponseParams, BaseAPIMethod[Response]):
    """Create a response and wait for the whole of it.

    Example:
        >>> response = await client.send_request(
        ...     CreateResponse(model="gpt-5", input="Hi")
        ... )
        >>> response.output_text
    """

    __request_path__ = "/responses"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON


class StreamResponse(ResponseParams, BaseStreamMethod[ResponseStreamEvent]):
    """Create a response and stream its events.

    The stream ends with ``response.completed``, ``response.failed`` or
    ``response.incomplete``; an ``error`` event is raised as an
    ``APIStatusError``.

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
    __response_format__ = ResponseFormat.SSE

    stream: Literal[True] = True
