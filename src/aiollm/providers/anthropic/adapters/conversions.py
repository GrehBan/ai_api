"""Conversions of Anthropic's Messages types into shared types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import BaseModel

from aiollm.adapters.base import BaseConversion
from aiollm.adapters.stop import Stop
from aiollm.enums import IncompleteReason, Role
from aiollm.http.exceptions.response import APIResponseError
from aiollm.providers.anthropic.types import (
    Delta,
    Message,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    TextBlock,
    ThinkingBlock,
    Usage,
)
from aiollm.types.requests.generation import Turn
from aiollm.types.responses.generation import TokenUsage

FINISHED: Final = frozenset({"end_turn", "stop_sequence", "tool_use"})
"""Stop reasons of a generation that reached its end.

``tool_use`` is one: the model waits for the tool results, as an
OpenAI response with function calls is ``completed``. Any other reason
(``max_tokens``, ``refusal``, ``model_context_window_exceeded``,
``pause_turn``, a reason Anthropic adds later) stopped it early.
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
    BaseConversion[str | list[Turn], list[dict[str, Any]]],
):
    """Converts the shared ``input`` into ``messages``.

    A text becomes one ``user`` turn; a list of turns is sent as it is.
    """

    def __call__(self, source: str | list[Turn], /) -> list[dict[str, Any]]:
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


class StopConversion(BaseConversion[Message | Delta, Stop]):
    """Tells how a message stopped.

    It takes a ``Message``, or for a stream the ``Delta`` that
    ``StreamState`` collects: both carry ``stop_reason`` and
    ``stop_details``.
    """

    def __call__(self, source: Message | Delta, /) -> Stop:
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


def message_text(message: Message) -> str | None:
    """Join the text of a message.

    Args:
        message: A message.

    Returns:
        The text of all ``text`` blocks, or ``None`` if there are none.
    """
    texts = [
        block.text for block in message.content if isinstance(block, TextBlock)
    ]
    return "".join(texts) if texts else None


def message_thinking(message: Message) -> str | None:
    """Join the thinking of a message.

    Args:
        message: A message.

    Returns:
        The text of all ``thinking`` blocks, or ``None`` if there are
        none or all are empty (thinking whose display is ``omitted``).
    """
    thinking = "".join(
        block.thinking
        for block in message.content
        if isinstance(block, ThinkingBlock)
    )
    return thinking or None


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
    real ``Usage`` and ``Delta``, so a stream goes through the
    same conversions as a message, and the message's id, model and
    whole text.

    Attributes:
        delta: The stop reason and details sent so far.
    """

    delta: Delta = field(default_factory=Delta)
    _message: Message | None = field(default=None, init=False, repr=False)
    _texts: list[str] = field(default_factory=list, init=False, repr=False)
    _thinking: list[str] = field(default_factory=list, init=False, repr=False)

    def start(self, event: RawMessageStartEvent) -> None:
        """Take the message and the counts of ``message_start``.

        Args:
            event: The first event of the stream.
        """
        self._message = event.message

    def add_text(self, text: str) -> None:
        """Keep a text delta for the whole text.

        Args:
            text: The new text of a ``text_delta``.
        """
        self._texts.append(text)

    def add_thinking(self, thinking: str) -> None:
        """Keep a thinking delta for the whole reasoning.

        Args:
            thinking: The new text of a ``thinking_delta``.
        """
        self._thinking.append(thinking)

    def thinking(self) -> str | None:
        """Join the thinking deltas.

        Returns:
            The thinking received, or ``None`` if there was none.
        """
        return "".join(self._thinking) or None

    def update(self, event: RawMessageDeltaEvent) -> None:
        """Take the stop reason and counts of a ``message_delta``.

        Args:
            event: A ``message_delta`` event.

        Raises:
            APIResponseError: No ``message_start`` came before it to
                take its counts into.
        """
        changes = _sent(event.delta)
        if changes:
            self.delta = self.delta.model_copy(update=changes)
        if self._message is None:
            message = (
                "Anthropic stream sent message_delta before message_start"
            )
            raise APIResponseError(message)
        counts = self._message.usage.model_copy(update=_sent(event.usage))
        self._message = self._message.model_copy(update={"usage": counts})

    def usage(self) -> Usage:
        """Give the counts of the whole message.

        Returns:
            The counts of ``message_start`` with the last cumulative
            counts of ``message_delta``.

        Raises:
            APIResponseError: The stream had no ``message_start``.
        """
        return self.message().usage

    def message(self) -> Message:
        """Give the message of ``message_start``.

        Returns:
            The message as the stream started it, for its id and model,
            with the counts of the last ``message_delta``.

        Raises:
            APIResponseError: The stream had no ``message_start``.
        """
        if self._message is None:
            message = "Anthropic stream finished without message_start"
            raise APIResponseError(message)
        return self._message

    def text(self) -> str | None:
        """Give the whole text streamed so far.

        Returns:
            The text deltas joined, or ``None`` if none came, as
            ``message_text`` gives for a message without text.
        """
        return "".join(self._texts) if self._texts else None


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
