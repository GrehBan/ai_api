"""OpenAI's adapters between shared types and its methods."""

from __future__ import annotations

from aiollm.providers.openai.adapters.conversions import (
    output_text,
    reasoning_summary,
)
from aiollm.providers.openai.adapters.generation import (
    OpenAIGenerationAdapter,
)

__all__ = [
    "OpenAIGenerationAdapter",
    "output_text",
    "reasoning_summary",
]
