"""Conversions of OpenAI's Responses types into shared types."""

from __future__ import annotations

from typing import Final

from ai_api.http.exceptions.response import APIResponseError
from ai_api.providers.openai.types.events import (
    OutputTextDeltaEvent,
    ResponseStateEvent,
    ResponseStreamEvent,
)
from ai_api.providers.openai.types.responses import Response, ResponseUsage
from ai_api.types.responses.events import (
    CompletedEvent,
    FailedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)
from ai_api.types.responses.generation import TokenUsage

TERMINAL_EVENTS: Final = frozenset(
    {"response.completed", "response.failed", "response.incomplete"}
)
"""Types of the events that end a Responses stream."""

_UNKNOWN_REASON: Final = "unknown"
"""Reason of an incomplete response that does not say why it stopped."""


def to_usage(usage: ResponseUsage | None) -> TokenUsage | None:
    """Convert OpenAI's token counts.

    Args:
        usage: Token counts of a response, if reported.

    Returns:
        The shared token counts, or ``None`` if none were reported.
    """
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def incomplete_reason(response: Response) -> str | None:
    """Tell why a response stopped before the end.

    Args:
        response: A finished response.

    Returns:
        OpenAI's reason, such as ``"max_output_tokens"``, if the
        response is incomplete (``"unknown"`` if OpenAI gives none), and
        ``None`` otherwise.
    """
    if response.status != "incomplete":
        return None
    details = response.incomplete_details
    if details is None or details.reason is None:
        return _UNKNOWN_REASON
    return details.reason


def failure_message(response: Response) -> str:
    """Describe why a response failed.

    Args:
        response: A response with status ``failed``.

    Returns:
        OpenAI's error message, or a note that it sent none.
    """
    if response.error is None:
        return f"response {response.id} failed without an error"
    return response.error.message


def check_status(response: Response) -> None:
    """Reject a response that did not finish.

    A response is created without ``background``, so OpenAI answers
    only when it has finished: ``completed`` or ``incomplete`` (stopped
    early, with the text generated so far). Any other status, ``failed``
    first of all, leaves no generated text to return.

    Args:
        response: The parsed response.

    Raises:
        APIResponseError: The status is neither ``completed`` nor
            ``incomplete``.
    """
    if response.status in {"completed", "incomplete"}:
        return
    if response.status == "failed":
        code = "" if response.error is None else f" ({response.error.code})"
        message = f"OpenAI response failed{code}: {failure_message(response)}"
        raise APIResponseError(message)
    message = (
        f"OpenAI response {response.id} has status {response.status!r}, "
        "expected 'completed' or 'incomplete'"
    )
    raise APIResponseError(message)


def to_event(
    event: ResponseStreamEvent,
) -> GenerationEvent[ResponseStreamEvent]:
    """Convert one event of a Responses stream.

    Args:
        event: The event OpenAI sent.

    Returns:
        A text delta, the completion (also of an incomplete response),
        the failure, or ``OtherEvent`` for every other event.
    """
    match event:
        case OutputTextDeltaEvent():
            return TextDeltaEvent[ResponseStreamEvent](
                text=event.delta, raw=event
            )
        case ResponseStateEvent(
            type="response.completed" | "response.incomplete"
        ):
            return CompletedEvent[ResponseStreamEvent](
                usage=to_usage(event.response.usage),
                incomplete_reason=incomplete_reason(event.response),
                raw=event,
            )
        case ResponseStateEvent(type="response.failed"):
            return FailedEvent[ResponseStreamEvent](
                message=failure_message(event.response), raw=event
            )
        case _:
            return OtherEvent[ResponseStreamEvent](raw=event)
