"""Conversions of OpenAI's Responses types into shared types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from ai_api.enums import IncompleteReason
from ai_api.http.exceptions.response import APIResponseError
from ai_api.providers.openai.types.events import (
    OutputTextDeltaEvent,
    ResponseStateEvent,
    ResponseStreamEvent,
)
from ai_api.providers.openai.types.responses import (
    OutputMessage,
    OutputRefusal,
    Response,
    ResponseUsage,
)
from ai_api.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)
from ai_api.types.responses.generation import TokenUsage

INCOMPLETE_REASONS: Final[Mapping[str, IncompleteReason]] = {
    "max_output_tokens": IncompleteReason.MAX_TOKENS,
}
"""Shared reason of each ``incomplete_details.reason`` that has one.

Every other reason (``content_filter``, ``max_messages``, ``steered``,
none at all, a reason OpenAI adds later) is ``IncompleteReason.OTHER``.
"""


def to_usage(usage: ResponseUsage | None) -> TokenUsage | None:
    """Convert OpenAI's token counts.

    Args:
        usage: Token counts of a response, if reported.

    Returns:
        The shared token counts, or ``None`` if none were reported.
    """
    if usage is None:
        return None
    inputs = usage.input_tokens_details
    outputs = usage.output_tokens_details
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_tokens=None if inputs is None else inputs.cached_tokens,
        cache_write_tokens=(
            None if inputs is None else inputs.cache_write_tokens
        ),
        reasoning_tokens=(
            None if outputs is None else outputs.reasoning_tokens
        ),
    )


def refusal(response: Response) -> str | None:
    """Collect the refusal of a response.

    Args:
        response: A finished response.

    Returns:
        The text of all ``refusal`` parts of the output messages joined
        together, or ``None`` if there are none.
    """
    parts = [
        part.refusal
        for item in response.output
        if isinstance(item, OutputMessage)
        for part in item.content
        if isinstance(part, OutputRefusal)
    ]
    return "".join(parts) if parts else None


def raw_incomplete_reason(response: Response) -> str | None:
    """Take OpenAI's own reason of an early stop.

    Args:
        response: A finished response.

    Returns:
        ``incomplete_details.reason`` of an incomplete response, such as
        ``"max_output_tokens"``, or ``None`` if the response is not
        incomplete or OpenAI gives no reason.
    """
    if response.status != "incomplete":
        return None
    details = response.incomplete_details
    return None if details is None else details.reason


def incomplete_reason(response: Response) -> IncompleteReason | None:
    """Tell why a response stopped before the end.

    A refusal part makes it a refusal, whatever the status, as OpenAI
    marks a refusal by a content part, not by a status.

    Args:
        response: A finished response.

    Returns:
        ``REFUSAL`` if the output has a refusal part; else, for an
        incomplete response, ``MAX_TOKENS`` or ``OTHER`` from its
        details; ``None`` for a response that finished.
    """
    if refusal(response) is not None:
        return IncompleteReason.REFUSAL
    if response.status != "incomplete":
        return None
    reason = raw_incomplete_reason(response)
    return INCOMPLETE_REASONS.get(reason or "", IncompleteReason.OTHER)


def check_status(response: Response) -> None:
    """Reject a response that did not finish.

    A response is created without ``background``, so OpenAI answers
    only when it has finished: ``completed`` or ``incomplete`` (stopped
    early, with the text generated so far). A ``failed`` one never gets
    here: ``CreateResponse`` raises it as an ``APIStatusError``. Any
    other status leaves no generated text to return.

    Args:
        response: The parsed response.

    Raises:
        APIResponseError: The status is neither ``completed`` nor
            ``incomplete``.
    """
    if response.status in {"completed", "incomplete"}:
        return
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
        or ``OtherEvent`` for every other event. ``response.failed``
        never gets here: ``StreamResponse`` raises it as an
        ``APIStatusError``.
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
                raw_incomplete_reason=raw_incomplete_reason(event.response),
                refusal=refusal(event.response),
                raw=event,
            )
        case _:
            return OtherEvent[ResponseStreamEvent](raw=event)
