"""Anthropic's adapters between shared types and its methods."""

from __future__ import annotations

from aiollm.providers.anthropic.adapters.conversions import (
    message_text,
    message_thinking,
    message_tool_calls,
)
from aiollm.providers.anthropic.adapters.generation import (
    AnthropicGenerationAdapter,
)

__all__ = [
    "AnthropicGenerationAdapter",
    "message_text",
    "message_thinking",
    "message_tool_calls",
]
