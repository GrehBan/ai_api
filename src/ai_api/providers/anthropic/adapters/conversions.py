"""Conversions of Anthropic's Messages types into shared types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from ai_api.enums import IncompleteReason
from ai_api.http.exceptions.response import APIResponseError
from ai_api.providers.anthropic.types.events import (
    MessageDeltaEvent,
    MessageStartEvent,
)
from ai_api.providers.anthropic.types.messages import (
    OutputTokensDetails,
    StopDetails,
    Usage,
)
from ai_api.types.responses.generation import TokenUsage

FINISHED: Final = frozenset({"end_turn", "stop_sequence"})
"""Stop reasons of a generation that reached its end.

Any other reason (``max_tokens``, ``refusal``,
``model_context_window_exceeded``, ``pause_turn``, ``tool_use``, a
reason Anthropic adds later) stopped it early.
"""

INCOMPLETE_REASONS: Final[Mapping[str, IncompleteReason]] = {
    "max_tokens": IncompleteReason.MAX_TOKENS,
    "refusal": IncompleteReason.REFUSAL,
}
"""Shared reason of each early stop reason that has one.

Every other early stop is ``IncompleteReason.OTHER``.
"""

REFUSAL: Final = "refusal"
"""The stop reason of a refusal."""


@dataclass(frozen=True, slots=True)
class Stop:
    """How a message stopped, in shared terms.

    Attributes:
        incomplete_reason: Shared reason of an early stop; ``None`` if
            the message reached its end.
        raw_incomplete_reason: Anthropic's stop reason of an early stop.
        refusal: The refusal's explanation, else its category, if the
            model refused and Anthropic says why.
    """

    incomplete_reason: IncompleteReason | None
    raw_incomplete_reason: str | None
    refusal: str | None


def to_stop(
    stop_reason: str | None,
    stop_details: StopDetails | None,
) -> Stop:
    """Tell how a message stopped.

    Args:
        stop_reason: The stop reason of a finished message.
        stop_details: More about the stop, if sent.

    Returns:
        Nothing set if the message reached its end; else the shared and
        Anthropic's reason, and for a refusal its explanation.

    Raises:
        APIResponseError: The message has no stop reason, which Anthropic
            sends on every finished message.
    """
    if stop_reason is None:
        message = "Anthropic message finished without a stop_reason"
        raise APIResponseError(message)
    if stop_reason in FINISHED:
        return Stop(None, None, None)
    refusal = None
    if stop_reason == REFUSAL and stop_details is not None:
        refusal = stop_details.explanation or stop_details.category
    return Stop(
        INCOMPLETE_REASONS.get(stop_reason, IncompleteReason.OTHER),
        stop_reason,
        refusal,
    )


def to_usage(
    *,
    input_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    output_tokens: int | None,
    output_tokens_details: OutputTokensDetails | None,
) -> TokenUsage:
    """Convert Anthropic's token counts.

    Anthropic counts input read from and written to the cache apart from
    the rest; the shared ``input_tokens`` is their sum, and
    ``total_tokens`` adds the output.

    Args:
        input_tokens: Input tokens outside the cache, if known.
        cache_creation_input_tokens: Input tokens written to the cache.
        cache_read_input_tokens: Input tokens read from the cache.
        output_tokens: Generated tokens, if known.
        output_tokens_details: Breakdown of the output tokens, if sent.

    Returns:
        The shared token counts; a sum is ``None`` if a part of it that
        is always sent is unknown.
    """
    total_input = (
        None
        if input_tokens is None
        else input_tokens
        + (cache_creation_input_tokens or 0)
        + (cache_read_input_tokens or 0)
    )
    total = (
        None
        if total_input is None or output_tokens is None
        else total_input + output_tokens
    )
    return TokenUsage(
        input_tokens=total_input,
        output_tokens=output_tokens,
        total_tokens=total,
        cache_read_tokens=cache_read_input_tokens,
        cache_write_tokens=cache_creation_input_tokens,
        reasoning_tokens=(
            None
            if output_tokens_details is None
            else output_tokens_details.thinking_tokens
        ),
    )


def message_usage(usage: Usage) -> TokenUsage:
    """Convert the token counts of a whole message.

    Args:
        usage: Token counts of a message.

    Returns:
        The shared token counts.
    """
    return to_usage(
        input_tokens=usage.input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        output_tokens=usage.output_tokens,
        output_tokens_details=usage.output_tokens_details,
    )


@dataclass
class StreamState:
    """What a message stream has told so far about the whole message.

    ``message_start`` carries the input tokens, and every
    ``message_delta`` the stop reason and the cumulative counts, so the
    counts are replaced, never added up.

    Attributes:
        input_tokens: Input tokens outside the cache.
        cache_creation_input_tokens: Input tokens written to the cache.
        cache_read_input_tokens: Input tokens read from the cache.
        output_tokens: Generated tokens so far.
        output_tokens_details: Breakdown of the output tokens, if sent.
        stop_reason: Why generation stopped, once known.
        stop_details: More about the stop, once known.
    """

    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens: int | None = None
    output_tokens_details: OutputTokensDetails | None = None
    stop_reason: str | None = None
    stop_details: StopDetails | None = None

    def start(self, event: MessageStartEvent) -> None:
        """Take the counts of ``message_start``.

        Args:
            event: The first event of the stream.
        """
        usage = event.message.usage
        self.input_tokens = usage.input_tokens
        self.cache_creation_input_tokens = usage.cache_creation_input_tokens
        self.cache_read_input_tokens = usage.cache_read_input_tokens
        self.output_tokens = usage.output_tokens
        self.output_tokens_details = usage.output_tokens_details

    def update(self, event: MessageDeltaEvent) -> None:
        """Take the stop reason and counts of a ``message_delta``.

        Args:
            event: A ``message_delta`` event.
        """
        if event.delta.stop_reason is not None:
            self.stop_reason = event.delta.stop_reason
        if event.delta.stop_details is not None:
            self.stop_details = event.delta.stop_details
        usage = event.usage
        if usage is None:
            return
        self.output_tokens = usage.output_tokens
        if usage.output_tokens_details is not None:
            self.output_tokens_details = usage.output_tokens_details
        if usage.input_tokens is not None:
            self.input_tokens = usage.input_tokens
        if usage.cache_creation_input_tokens is not None:
            self.cache_creation_input_tokens = (
                usage.cache_creation_input_tokens
            )
        if usage.cache_read_input_tokens is not None:
            self.cache_read_input_tokens = usage.cache_read_input_tokens

    def usage(self) -> TokenUsage:
        """Convert the counts known so far.

        Returns:
            The shared token counts.
        """
        return to_usage(
            input_tokens=self.input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            output_tokens=self.output_tokens,
            output_tokens_details=self.output_tokens_details,
        )

    def stop(self) -> Stop:
        """Tell how the message stopped, from what was sent so far.

        Returns:
            The shared and Anthropic's reason, and a refusal's
            explanation.

        Raises:
            APIResponseError: No ``message_delta`` gave a stop reason.
        """
        return to_stop(self.stop_reason, self.stop_details)
