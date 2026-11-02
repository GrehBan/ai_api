"""The failed attempt a retry decision is made from."""

from __future__ import annotations

from dataclasses import dataclass

from aiollm.http.exceptions.status import APIStatusError
from aiollm.http.exceptions.transport import APITransportError


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryState:
    """A failed attempt whose error the library would retry.

    The session builds it for a policy's ``retry_if`` and ``backoff``
    only when the error is transient and its ``Retry-After`` is within
    ``max_retry_after``, so ``error`` is never anything else.

    Attributes:
        attempt: Number of the attempt that failed, counting from 1.
        elapsed: Seconds since the first attempt started.
        error: The transient error the attempt ended with.
    """

    attempt: int
    elapsed: float
    error: APIStatusError | APITransportError
