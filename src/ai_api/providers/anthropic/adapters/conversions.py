"""Conversions of Anthropic's Messages types into shared types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel

from ai_api.adapters.base import BaseConversion
from ai_api.adapters.stop import Stop
from ai_api.enums import IncompleteReason, Role
from ai_api.http.exceptions.response import APIResponseError
from ai_api.providers.anthropic.methods.base import MessageParam
from ai_api.providers.anthropic.types.events import (
    MessageDelta,
    MessageDeltaEvent,
    MessageStartEvent,
)
from ai_api.providers.anthropic.types.messages import Message, Usage
from ai_api.types.requests.generation import Turn
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


class InputConversion(
    BaseConversion[str | list[Turn], list[MessageParam]],
):
    """Converts the shared ``input`` into ``messages``.

    A text becomes one ``user`` turn; a list of turns is sent as it is.
    """

    def __call__(self, source: str | list[Turn], /) -> list[MessageParam]:
        """Convert the input.

        Args:
            source: The text or the turns of the shared request.

        Returns:
            One message per turn.
        """
        if isinstance(source, str):
            return [{"role": Role.USER.value, "content": source}]
        return [
            {"role": turn.role.value, "content": turn.content}
            for turn in source
        ]


class UsageConversion(BaseConversion[Usage, TokenUsage]):
    """Converts Anthropic's token counts.

    Anthropic counts input read from and written to the cache apart from
    the rest; the shared ``input_tokens`` is their sum, and
    ``total_tokens`` adds the output.
    """

    def __call__(self, source: Usage, /) -> TokenUsage:
        """Convert the counts of a message.

        Args:
            source: Token counts of a message.

        Returns:
            The shared token counts.
        """
        cache_write = source.cache_creation_input_tokens
        cache_read = source.cache_read_input_tokens
        input_tokens = (
            source.input_tokens + (cache_write or 0) + (cache_read or 0)
        )
        details = source.output_tokens_details
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=source.output_tokens,
            total_tokens=input_tokens + source.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=(
                None if details is None else details.thinking_tokens
            ),
        )


class StopConversion(BaseConversion[Message | MessageDelta, Stop]):
    """Tells how a message stopped.

    It takes a ``Message``, or for a stream the ``MessageDelta`` that
    ``StreamState`` collects: both carry ``stop_reason`` and
    ``stop_details``.
    """

    def __call__(self, source: Message | MessageDelta, /) -> Stop:
        """Convert the stop of a finished message.

        Args:
            source: The message, or the collected changes of a stream.

        Returns:
            Nothing set if the message reached its end; else the shared and
            Anthropic's reason, and for a refusal its explanation, else its
            category.

        Raises:
            APIResponseError: There is no stop reason, which Anthropic
                sends on every finished message.
        """
        stop_reason = source.stop_reason
        if stop_reason is None:
            message = "Anthropic message finished without a stop_reason"
            raise APIResponseError(message)
        if stop_reason in FINISHED:
            return Stop()
        details = source.stop_details
        refusal = None
        if stop_reason == REFUSAL and details is not None:
            refusal = details.explanation or details.category
        return Stop(
            INCOMPLETE_REASONS.get(stop_reason, IncompleteReason.OTHER),
            stop_reason,
            refusal,
        )


INPUT_CONVERSION: Final = InputConversion()
"""The default conversion of the shared ``input``."""

USAGE_CONVERSION: Final = UsageConversion()
"""The default conversion of token counts."""

STOP_CONVERSION: Final = StopConversion()
"""The default conversion of a message's stop."""


@dataclass
class StreamState:
    """What a message stream has told so far about the whole message.

    ``message_start`` carries the message's ``Usage``, and every
    ``message_delta`` the stop reason and the cumulative counts, so the
    counts are replaced, never added up. At the end the state gives a
    real ``Usage`` and ``MessageDelta``, so a stream goes through the
    same conversions as a message.

    Attributes:
        delta: The stop reason and details sent so far.
    """

    delta: MessageDelta = field(default_factory=MessageDelta)
    _usage: Usage | None = field(default=None, init=False, repr=False)

    def start(self, event: MessageStartEvent) -> None:
        """Take the counts of ``message_start``.

        Args:
            event: The first event of the stream.
        """
        self._usage = event.message.usage

    def update(self, event: MessageDeltaEvent) -> None:
        """Take the stop reason and counts of a ``message_delta``.

        Args:
            event: A ``message_delta`` event.

        Raises:
            APIResponseError: The event has counts, but no
                ``message_start`` came before it to take them into.
        """
        changes = _sent(event.delta)
        if changes:
            self.delta = self.delta.model_copy(update=changes)
        usage = event.usage
        if usage is None:
            return
        if self._usage is None:
            message = (
                "Anthropic stream sent message_delta before message_start"
            )
            raise APIResponseError(message)
        self._usage = self._usage.model_copy(update=_sent(usage))

    def usage(self) -> Usage:
        """Give the counts of the whole message.

        Returns:
            The counts of ``message_start`` with the last cumulative
            counts of ``message_delta``.

        Raises:
            APIResponseError: The stream had no ``message_start``.
        """
        if self._usage is None:
            message = "Anthropic stream finished without message_start"
            raise APIResponseError(message)
        return self._usage


def _sent(model: BaseModel) -> dict[str, object]:
    """Take the fields of a model that have a value.

    The values stay as they are, nested models included, so they can
    update another model with ``model_copy``.

    Args:
        model: A partial update, such as a ``message_delta``'s counts.

    Returns:
        Each field that is not ``None``, with its value.
    """
    return {
        name: value
        for name in type(model).model_fields
        if (value := getattr(model, name)) is not None
    }
