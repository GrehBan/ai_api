"""Adapters between shared request and response types and provider methods."""

from __future__ import annotations

from ai_api.adapters.base import (
    BaseAdapter,
    BaseCompositeAdapter,
    BaseStreamAdapter,
)
from ai_api.adapters.protocols import (
    IAdapter,
    ICompositeAdapter,
    IStreamAdapter,
)

__all__ = [
    "BaseAdapter",
    "BaseCompositeAdapter",
    "BaseStreamAdapter",
    "IAdapter",
    "ICompositeAdapter",
    "IStreamAdapter",
]
