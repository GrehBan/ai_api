"""Requests in types shared by all providers."""

from __future__ import annotations

from aiollm.types.requests.base import (
    BaseRequest,
)
from aiollm.types.requests.generation import (
    GenerationRequest,
    Turn,
)

__all__ = [
    "BaseRequest",
    "GenerationRequest",
    "Turn",
]
