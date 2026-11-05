"""Anthropic's own options of a generation call."""

from __future__ import annotations

from typing import TypedDict

from aiollm.providers.anthropic.methods.base import ThinkingConfig


class AnthropicGenerationOptions(TypedDict, total=False):
    """Fields of ``POST /v1/messages`` that the shared request does not hold.

    Given to ``client.generation.create()`` or ``stream()`` as
    ``provider_options``; a key left out is not sent. Each key is a field
    of ``CreateMessage`` with its type, and the method checks the value.

    Attributes:
        stop_sequences: Texts that stop generation.
        thinking: How the model thinks; ``{"type": "adaptive",
            "display": "summarized"}`` brings a summary back as
            ``GenerationResponse.reasoning`` and ``ReasoningDeltaEvent``.

    Example:
        >>> await client.generation.create(
        ...     model="claude-opus-5-5",
        ...     input="Count to ten",
        ...     provider_options={"stop_sequences": ["5"]},
        ... )
    """

    stop_sequences: list[str]
    thinking: ThinkingConfig
