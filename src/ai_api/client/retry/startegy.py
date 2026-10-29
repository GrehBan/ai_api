"""When and how long to wait before retrying a request."""

from tenacity import (
    RetryCallState,
    retry_if_exception,
    wait_random_exponential,
)

from ai_api.client.exceptions.http import APIError
from ai_api.client.exceptions.transport import APITransportError

_jitter = wait_random_exponential(
    multiplier=0.5,
    max=30,
)


def should_retry(exc: BaseException) -> bool:
    """Decide whether a failed attempt should be retried.

    Args:
        exc: The exception raised by the attempt.

    Returns:
        The ``transient`` flag for API and transport errors, ``False`` for
        anything else.
    """
    if isinstance(exc, (APIError, APITransportError)):
        return exc.transient
    return False


def calculate_wait(state: RetryCallState) -> float:
    """Compute the delay before the next attempt.

    Uses random exponential backoff (up to 30 seconds), but never waits
    less than the ``Retry-After`` value the API asked for.

    Args:
        state: Tenacity's state of the current retry loop.

    Returns:
        Seconds to wait.
    """
    delay = _jitter(state)

    if state.outcome is None:
        return delay

    exc = state.outcome.exception()

    if isinstance(exc, APIError) and exc.retry_after is not None:
        return max(delay, exc.retry_after)

    return delay


RETRY_CONDITION = retry_if_exception(should_retry)
"""Tenacity retry condition built from ``should_retry``."""
