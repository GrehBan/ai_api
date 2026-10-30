"""Requests in types shared by all providers."""

from __future__ import annotations

from ai_api.types.requests.base import (
    BaseRequest,
)
from ai_api.types.requests.generation import (
    GenerationRequest,
)

__all__ = [
    "BaseRequest",
    "GenerationRequest",
]
