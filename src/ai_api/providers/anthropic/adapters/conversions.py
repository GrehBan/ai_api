"""Conversions of Anthropic's Messages types into shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ai_api.http.exceptions.response import APIResponseError
from ai_api.providers.anthropic.types.events import (
    MessageDeltaEvent,
    MessageStartEvent,
)
from ai_api.providers.anthropic.types.messages import Usage
from ai_api.types.responses.generation import TokenUsage

FINISHED: Final = frozenset({"end_turn", "stop_sequence"})
"""Stop reasons of a generation that reached its end.

Any other reason (``max_tokens``, ``refusal``,
``model_context_window_exceeded``, ``pause_turn``, a reason Anthropic
adds later) stopped it early and becomes ``incomplete_reason``.
"""


def incomplete_reason(stop_reason: str | None) -> str | None:
    """Tell why a message stopped before its end.

    Args:
        stop_reason: The stop reason of a finished message.

    Returns:
        ``None`` if the message reached its end, else the stop reason.

    Raises:
        APIResponseError: The message has no stop reason, which Anthropic
            sends on every finished message.
    """
    if stop_reason is None:
        message = "Anthropic message finished without a stop_reason"
        raise APIResponseError(message)
    return None if stop_reason in FINISHED else stop_reason


def to_usage(
    *,
    input_tokens: int | None,
    cache_creation_input_tokens: int | None,
    cache_read_input_tokens: int | None,
    output_tokens: int | None,
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
        stop_reason: Why generation stopped, once known.
    """

    input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None

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

    def update(self, event: MessageDeltaEvent) -> None:
        """Take the stop reason and counts of a ``message_delta``.

        Args:
            event: A ``message_delta`` event.
        """
        if event.delta.stop_reason is not None:
            self.stop_reason = event.delta.stop_reason
        usage = event.usage
        if usage is None:
            return
        self.output_tokens = usage.output_tokens
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
        )
