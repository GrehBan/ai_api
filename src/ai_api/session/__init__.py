"""Sessions that send methods to an API."""

from __future__ import annotations

from ai_api.session.base import (
    BaseSession,
)
from ai_api.session.protocols import (
    ISession,
    Timeout,
)
from ai_api.session.retry import (
    DEFAULT_MAX_RETRY_AFTER,
    DEFAULT_RETRY_DEADLINE,
)

__all__ = [
    "BaseSession",
    "DEFAULT_MAX_RETRY_AFTER",
    "DEFAULT_RETRY_DEADLINE",
    "ISession",
    "Timeout",
]
