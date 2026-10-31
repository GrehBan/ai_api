"""Adapters between shared request and response types and provider methods."""

from __future__ import annotations

from ai_api.adapters.base import (
    BaseAdapter,
    BaseCompositeAdapter,
    BaseConversion,
    BaseStreamAdapter,
)
from ai_api.adapters.protocols import (
    IAdapter,
    ICompositeAdapter,
    IConversion,
    IStreamAdapter,
)
from ai_api.adapters.stop import Stop

__all__ = [
    "BaseAdapter",
    "BaseCompositeAdapter",
    "BaseConversion",
    "BaseStreamAdapter",
    "IAdapter",
    "ICompositeAdapter",
    "IConversion",
    "IStreamAdapter",
    "Stop",
]
