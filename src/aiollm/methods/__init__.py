"""Declarative API methods."""

from __future__ import annotations

from aiollm.methods.base import (
    BaseAPIMethod,
    BaseMethod,
    BaseStreamMethod,
)
from aiollm.methods.protocols import (
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
