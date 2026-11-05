"""Adapters between shared request and response types and provider methods."""

from __future__ import annotations

from aiollm.adapters.base import (
    BaseAdapter,
    BaseCompositeAdapter,
    BaseConversion,
    BaseStreamAdapter,
)
from aiollm.adapters.protocols import (
    IAdapter,
    ICompositeAdapter,
    IConversion,
    IStreamAdapter,
)
from aiollm.adapters.stop import Stop

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
