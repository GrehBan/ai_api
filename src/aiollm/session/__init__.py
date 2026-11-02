"""Sessions that send methods to an API."""

from __future__ import annotations

from aiollm.session.base import (
    BaseSession,
)
from aiollm.session.factory import (
    SessionFactory,
)
from aiollm.session.protocols import (
    ISession,
    ISessionFactory,
    Timeout,
)
from aiollm.session.retry import (
    DEFAULT_MAX_RETRY_AFTER,
    DEFAULT_RETRY_DEADLINE,
)

__all__ = [
    "BaseSession",
    "DEFAULT_MAX_RETRY_AFTER",
    "DEFAULT_RETRY_DEADLINE",
    "ISession",
    "ISessionFactory",
    "SessionFactory",
    "Timeout",
]
