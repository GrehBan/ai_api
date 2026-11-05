"""Sessions that send methods to an API."""

from __future__ import annotations

from aiollm.session.base import (
    BaseSession,
)
from aiollm.session.factory import (
    SessionFactory,
)
from aiollm.session.protocols import (
    CallOptions,
    ISession,
    ISessionFactory,
    Timeout,
)
from aiollm.session.retry import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_RETRY_AFTER,
    DEFAULT_RETRY_DEADLINE,
    DEFAULT_RETRY_POLICY,
    FullJitterBackoff,
    IBackoff,
    IRetryPredicate,
    RetryPolicy,
    RetryState,
)

__all__ = [
    "BaseSession",
    "CallOptions",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_RETRY_AFTER",
    "DEFAULT_RETRY_DEADLINE",
    "DEFAULT_RETRY_POLICY",
    "FullJitterBackoff",
    "IBackoff",
    "IRetryPredicate",
    "ISession",
    "ISessionFactory",
    "RetryPolicy",
    "RetryState",
    "SessionFactory",
    "Timeout",
]
