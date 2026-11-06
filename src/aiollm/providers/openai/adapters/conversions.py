"""Conversions of OpenAI's Responses types into shared types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from aiollm.adapters.base import BaseConversion
from aiollm.adapters.stop import Stop
from aiollm.enums import IncompleteReason
from aiollm.http.exceptions.response import APIResponseError
from aiollm.providers.openai.types import (
    OutputMessage,
    OutputTextContent,
    ReasoningItem,
    RefusalContent,
    Response,
    ResponseUsage,
)
from aiollm.types.requests.generation import Turn
from aiollm.types.responses.generation import TokenUsage

INCOMPLETE_REASONS: Final[Mapping[str, IncompleteReason]] = {
    "max_output_tokens": IncompleteReason.MAX_TOKENS,
}
"""Shared reason of each ``incomplete_details.reason`` that has one.

Every other reason (``content_filter``, ``max_messages``, ``steered``,
none at all, a reason OpenAI adds later) is ``IncompleteReason.OTHER``.
"""


class InputConversion(
    BaseConversion[str | list[Turn], str | list[dict[str, Any]]],
):
    """Converts the shared ``input`` into OpenAI's ``input``.

    A text is sent as it is; a list of turns as ``EasyInputMessage``
    items (``role``, ``content``).
    """

    def __call__(
        self, source: str | list[Turn], /
    ) -> str | list[dict[str, Any]]:
        """Convert the input.

        Args:
            source: The text or the turns of the shared request.

        Returns:
            The text, or one input item per turn.
        """
        if isinstance(source, str):
            return source
        return [
            {"role": turn.role.value, "content": turn.content}
            for turn in source
        ]


class UsageConversion(BaseConversion[ResponseUsage, TokenUsage]):
    """Converts OpenAI's token counts; they map one to one."""

    def __call__(self, source: ResponseUsage, /) -> TokenUsage:
        """Convert the counts of a response.

        Args:
            source: Token counts of a response.

        Returns:
            The shared token counts.
        """
        inputs = source.input_tokens_details
        outputs = source.output_tokens_details
        return TokenUsage(
            input_tokens=source.input_tokens,
            output_tokens=source.output_tokens,
            total_tokens=source.total_tokens,
            cache_read_tokens=inputs.cached_tokens,
            cache_write_tokens=inputs.cache_write_tokens,
            reasoning_tokens=outputs.reasoning_tokens,
        )


class StopConversion(BaseConversion[Response, Stop]):
    """Tells how a finished response stopped.

    A refusal part makes it a refusal, whatever the status, as OpenAI
    marks a refusal by a content part, not by a status.
    """

    def __call__(self, source: Response, /) -> Stop:
        """Convert the stop of a finished response.

        Args:
            source: A ``completed`` or ``incomplete`` response, also the
                one inside a stream's terminal event.

        Returns:
            ``REFUSAL`` and the refusal text if the output has a refusal
            part; else, for an incomplete response, ``MAX_TOKENS`` or
            ``OTHER`` with OpenAI's reason; nothing for a response that
            finished.
        """
        return Stop(
            _incomplete_reason(source),
            _raw_incomplete_reason(source),
            _refusal(source),
        )


def _refusal(response: Response) -> str | None:
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
        if isinstance(part, RefusalContent)
    ]
    return "".join(parts) if parts else None


def _raw_incomplete_reason(response: Response) -> str | None:
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


def _incomplete_reason(response: Response) -> IncompleteReason | None:
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
    if _refusal(response) is not None:
        return IncompleteReason.REFUSAL
    if response.status != "incomplete":
        return None
    reason = _raw_incomplete_reason(response)
    return INCOMPLETE_REASONS.get(reason or "", IncompleteReason.OTHER)


def output_text(response: Response) -> str | None:
    """Join the text of all output messages.

    The Responses API sends no such field: its schema names
    ``output_text`` as a property the SDKs compute, so ``Response``
    reads it as ``None``. It is collected here the way the SDKs do,
    from the ``output_text`` parts of the ``message`` items.

    Args:
        response: A response.

    Returns:
        The joined text, or ``None`` if the output has no text part.
    """
    parts = [
        part.text
        for item in response.output
        if isinstance(item, OutputMessage)
        for part in item.content
        if isinstance(part, OutputTextContent)
    ]
    return "".join(parts) if parts else None


def reasoning_summary(response: Response) -> str | None:
    """Join the summaries of all reasoning items.

    Args:
        response: A response.

    Returns:
        The joined summaries, or ``None`` if there are none (the
        request did not ask for them).
    """
    summary = "".join(
        part.text
        for item in response.output
        if isinstance(item, ReasoningItem)
        for part in item.summary
    )
    return summary or None


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


INPUT_CONVERSION: Final = InputConversion()
"""The default conversion of the shared ``input``."""

USAGE_CONVERSION: Final = UsageConversion()
"""The default conversion of token counts."""

STOP_CONVERSION: Final = StopConversion()
"""The default conversion of a response's stop."""
