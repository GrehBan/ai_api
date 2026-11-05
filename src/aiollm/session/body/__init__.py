"""Request body builders, one module per transport."""

from __future__ import annotations

from aiollm.session.body.protocols import (
    IBodyBuilder,
)

__all__ = [
    "IBodyBuilder",
]
