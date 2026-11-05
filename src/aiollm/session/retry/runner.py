"""Run a retry policy with tenacity."""

from __future__ import annotations

import logging
from collections.abc import Callable
from math import isfinite

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    stop_after_attempt,
    stop_before_delay,
)
from tenacity.stop import stop_base

from aiollm.http.exceptions.status import APIStatusError
from aiollm.http.exceptions.transport import APITransportError
from aiollm.session.retry.policy import DEFAULT_MAX_RETRY_AFTER, RetryPolicy
from aiollm.session.retry.state import RetryState

logger = logging.getLogger(__name__)


def should_retry(
    exc: BaseException,
    max_retry_after: float = DEFAULT_MAX_RETRY_AFTER,
) -> bool:
    """Decide whether the library allows a failed attempt to be retried.

    A policy's ``retry_if`` is asked only when this returns ``True``.

    Args:
        exc: The exception raised by the attempt.
        max_retry_after: Longest ``Retry-After`` worth waiting for; an
            API that asks for a longer wait gets its error raised at once
            instead of blocking the caller.

    Returns:
        The ``transient`` flag for API and transport errors, ``False`` for
        anything else and for a ``Retry-After`` above the limit.
    """
    if isinstance(exc, APIStatusError):
        if (
            exc.transient
            and exc.retry_after is not None
            and exc.retry_after > max_retry_after
        ):
            logger.debug(
                "Not retrying %s: the API asks to wait %s s, more than "
                "max_retry_after (%s s)",
                type(exc).__name__,
                exc.retry_after,
                max_retry_after,
            )
            return False

        return exc.transient

    if isinstance(exc, APITransportError):
        return exc.transient

    return False


def _retry_state(state: RetryCallState) -> RetryState:
    """Describe tenacity's failed attempt for a policy's callables.

    Args:
        state: Tenacity's state of the current retry loop.

    Returns:
        The attempt's number, the time since the first one and its error.

    Raises:
        RuntimeError: If the attempt did not fail with an API or a
            transport error; tenacity asks for a wait only after
            ``is_retried`` allowed the retry, so this is a bug.
    """
    error = state.outcome.exception() if state.outcome else None

    if not isinstance(error, APIStatusError | APITransportError):
        msg = "a retry needs an attempt that failed with an API error"
        raise RuntimeError(msg)

    return RetryState(
        attempt=state.attempt_number,
        elapsed=state.seconds_since_start or 0.0,
        error=error,
    )


def _call_hook[T](hook: Callable[[RetryState], T], state: RetryState) -> T:
    """Call a callable of the policy, keeping the error it was given.

    Tenacity would raise the callable's own failure in place of the
    request's error, with nothing to link them, so that error becomes the
    failure's ``__cause__`` (unless it has one, or is that error).

    Args:
        hook: The policy's ``retry_if`` or ``backoff``.
        state: The failed attempt.

    Returns:
        What ``hook`` returned.
    """
    try:
        return hook(state)
    except Exception as error:
        if error is not state.error and error.__cause__ is None:
            raise error from state.error
        raise


def is_retried(state: RetryCallState, policy: RetryPolicy) -> bool:
    """Decide whether tenacity retries a finished attempt.

    Args:
        state: Tenacity's state of the current retry loop.
        policy: The call's retry policy.

    Returns:
        ``False`` for a success, for an error ``should_retry`` refuses
        and after the last attempt ``max_attempts`` allows; otherwise
        what the policy's ``retry_if`` answers, or ``True`` without one.
    """
    error = state.outcome.exception() if state.outcome else None

    if error is None or not should_retry(error, policy.max_retry_after):
        return False

    # Tenacity asks for the wait before it checks the stop, so without
    # this the last attempt would reach retry_if and backoff too, though
    # no retry can follow it.
    if state.attempt_number >= policy.max_attempts:
        return False

    if policy.retry_if is None:
        return True

    return _call_hook(policy.retry_if, _retry_state(state))


def calculate_wait(state: RetryCallState, policy: RetryPolicy) -> float:
    """Compute the delay before the next attempt.

    The policy's ``backoff`` decides it, but the session never waits less
    than the ``Retry-After`` the API asked for.

    Args:
        state: Tenacity's state of the current retry loop.
        policy: The call's retry policy.

    Returns:
        Seconds to wait.

    Raises:
        TypeError: If ``backoff`` returned something other than a number.
        ValueError: If it returned an infinite, NaN or negative number.
    """
    retry_state = _retry_state(state)
    error = retry_state.error
    delay = _checked_delay(_call_hook(policy.backoff, retry_state), error)

    if isinstance(error, APIStatusError) and error.retry_after is not None:
        return max(delay, error.retry_after)

    return delay


def _checked_delay(delay: object, error: BaseException) -> float:
    """Check what a backoff returned, which a type checker cannot ensure.

    Args:
        delay: The backoff's result.
        error: The retried error, the cause of a rejection.

    Returns:
        ``delay`` as a ``float``.

    Raises:
        TypeError: If ``delay`` is not an ``int`` or a ``float``.
        ValueError: If it is infinite, NaN or negative.
    """
    if isinstance(delay, bool) or not isinstance(delay, int | float):
        msg = f"backoff must return a number, not {type(delay).__name__}"
        raise TypeError(msg) from error

    if not isfinite(delay) or delay < 0:
        msg = "backoff must return a finite, non-negative number"
        raise ValueError(msg) from error

    return float(delay)


def _stop(policy: RetryPolicy) -> stop_base:
    """Build the condition that ends the retries.

    Args:
        policy: The call's retry policy.

    Returns:
        A stop after ``max_attempts``, or before a wait that would end
        after ``deadline``, whichever comes first.
    """
    stop: stop_base = stop_after_attempt(policy.max_attempts)

    if policy.deadline is not None:
        stop |= stop_before_delay(policy.deadline)

    return stop


def build_retrying(
    policy: RetryPolicy,
    *,
    before_sleep: Callable[[RetryState, float], None],
) -> AsyncRetrying:
    """Build the tenacity loop of one call.

    With ``reraise=True`` it returns the first result or raises the last
    error, so no state is left to guard against.

    Args:
        policy: The call's retry policy.
        before_sleep: Called before each wait with the failed attempt
            and the wait in seconds, to log the retry; it never sees
            tenacity's own state.

    Returns:
        A retry loop to call with the attempt's coroutine function.
    """
    return AsyncRetrying(
        retry=lambda state: is_retried(state, policy),
        wait=lambda state: calculate_wait(state, policy),
        stop=_stop(policy),
        before_sleep=lambda state: before_sleep(
            _retry_state(state), state.upcoming_sleep
        ),
        reraise=True,
    )
