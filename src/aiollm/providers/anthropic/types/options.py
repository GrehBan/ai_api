"""Anthropic's own options of a generation call."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class ThinkingAdaptive(TypedDict):
    """The model decides how much it thinks.

    Mirrors ``ThinkingConfigAdaptiveParam`` as a ``TypedDict``, so a dict
    literal type-checks.

    Attributes:
        type: Always ``adaptive``.
        display: Whether a summary of the thinking comes back.
    """

    type: Literal["adaptive"]
    display: NotRequired[Literal["summarized", "omitted"] | None]


class ThinkingEnabled(TypedDict):
    """Thinking with a fixed budget, on the models that take one.

    Mirrors ``ThinkingConfigEnabledParam``.

    Attributes:
        type: Always ``enabled``.
        budget_tokens: Tokens the model may spend on thinking.
        display: Whether a summary of the thinking comes back.
    """

    type: Literal["enabled"]
    budget_tokens: int
    display: NotRequired[Literal["summarized", "omitted"] | None]


class ThinkingDisabled(TypedDict):
    """No thinking, on the models that allow it.

    Mirrors ``ThinkingConfigDisabledParam``.

    Attributes:
        type: Always ``disabled``.
    """

    type: Literal["disabled"]


class ThinkingBetweenTools(TypedDict):
    """Thinking only between tool calls.

    Mirrors ``ThinkingConfigBetweenToolsParam``.

    Attributes:
        type: Always ``between_tools``.
    """

    type: Literal["between_tools"]


class AnthropicGenerationOptions(TypedDict, total=False):
    """Fields of ``POST /v1/messages`` that the shared request does not hold.

    Given to ``client.generation.create()`` or ``stream()`` as
    ``provider_options``; a key left out is not sent. Each key is a field
    of ``CreateMessage``, its value of the field's type or a ``TypedDict``
    with the same keys, so a dict literal type-checks; the method checks
    the value.

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
    thinking: (
        ThinkingEnabled
        | ThinkingDisabled
        | ThinkingBetweenTools
        | ThinkingAdaptive
    )
