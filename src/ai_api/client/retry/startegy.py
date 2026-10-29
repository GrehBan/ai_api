"""When and how long to wait before retrying a request."""

from functools import partial
from typing import Final

from tenacity import (
    RetryCallState,
    retry_if_exception,
    wait_random_exponential,
)

from ai_api.client.exceptions.http import APIError
from ai_api.client.exceptions.transport import APITransportError

DEFAULT_MAX_RETRY_AFTER: Final[float] = 60.0
"""Longest ``Retry-After`` the session waits for, in seconds."""

DEFAULT_RETRY_DEADLINE: Final[float] = 900.0
"""Time after which no new retry starts, in seconds (15 minutes)."""

_jitter = wait_random_exponential(
    multiplier=0.5,
    max=30,
)


def should_retry(
    exc: BaseException,
    max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
) -> bool:
    """Decide whether a failed attempt should be retried.

    Args:
        exc: The exception raised by the attempt.
        max_retry_after: Longest ``Retry-After`` worth waiting for; an
            API that asks for a longer wait gets its error raised at once
            instead of blocking the caller.

    Returns:
        The ``transient`` flag for API and transport errors, ``False`` for
        anything else and for a ``Retry-After`` above the limit.
    """
    if isinstance(exc, APIError):
        return exc.transient and (
            exc.retry_after is None or exc.retry_after <= max_retry_after
        )

    if isinstance(exc, APITransportError):
        return exc.transient

    return False


def retry_condition(
    max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
) -> retry_if_exception:
    """Build the tenacity retry condition.

    Args:
        max_retry_after: Longest ``Retry-After`` worth waiting for.

    Returns:
        A condition that applies ``should_retry`` with this limit.
    """
    return retry_if_exception(
        partial(should_retry, max_retry_after=max_retry_after)
    )


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


RETRY_CONDITION = retry_condition()
"""Tenacity retry condition with the default ``Retry-After`` limit."""
