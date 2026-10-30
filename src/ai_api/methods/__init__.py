"""Declarative API methods."""

from __future__ import annotations

from ai_api.methods.base import (
    BaseAPIMethod,
    BaseMethod,
    BaseStreamMethod,
)
from ai_api.methods.protocols import (
    IAPIMethod,
    IBaseMethod,
    IStreamMethod,
)

__all__ = [
    "BaseAPIMethod",
    "BaseMethod",
    "BaseStreamMethod",
    "IAPIMethod",
    "IBaseMethod",
    "IStreamMethod",
]
