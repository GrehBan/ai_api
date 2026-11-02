"""How often and how long a retryable request is retried."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from random import uniform
from typing import Final

from aiollm.session.retry.protocols import IBackoff, IRetryPredicate
from aiollm.session.retry.state import RetryState

DEFAULT_MAX_ATTEMPTS: Final[int] = 4
"""Attempts of a retryable request, including the first one."""

DEFAULT_RETRY_DEADLINE: Final[float] = 900.0
"""Time after which no new retry starts, in seconds (15 minutes)."""

DEFAULT_MAX_RETRY_AFTER: Final[float] = 60.0
"""Longest ``Retry-After`` the session waits for, in seconds."""


def _check_number(name: str, value: object, *, positive: bool) -> None:
    """Reject a setting that is not a finite number in range.

    Args:
        name: The setting's name, for the message.
        value: Its value.
        positive: Whether zero is out of range too.

    Raises:
        TypeError: If ``value`` is not an ``int`` or a ``float`` (a
            ``bool`` is not a number here).
        ValueError: If it is infinite, NaN, negative, or zero when
            ``positive`` is set.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{name} must be a number, not {type(value).__name__}"
        raise TypeError(msg)

    if not isfinite(value) or value < 0 or (positive and value == 0):
        bound = "positive" if positive else "non-negative"
        msg = f"{name} must be finite and {bound}"
        raise ValueError(msg)


def _check_count(name: str, value: object) -> None:
    """Reject a setting that is not a whole number of at least 1.

    Args:
        name: The setting's name, for the message.
        value: Its value.

    Raises:
        TypeError: If ``value`` is not an ``int`` (a ``bool`` is not a
            number here).
        ValueError: If it is below 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an int, not {type(value).__name__}"
        raise TypeError(msg)

    if value < 1:
        msg = f"{name} must be at least 1"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class FullJitterBackoff:
    """Random exponential backoff, the default of ``RetryPolicy``.

    The wait after attempt ``n`` is drawn uniformly from zero to
    ``multiplier * 2 ** (n - 1)``, capped at ``max_delay``, so clients
    that failed together do not come back together.

    Attributes:
        multiplier: Upper end of the first wait, in seconds.
        max_delay: Largest wait, in seconds.

    Example:
        >>> RetryPolicy(backoff=FullJitterBackoff(max_delay=5))
    """

    multiplier: float = 0.5
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            TypeError: If a setting is not a number.
            ValueError: If a setting is infinite, NaN or negative.
        """
        _check_number("multiplier", self.multiplier, positive=False)
        _check_number("max_delay", self.max_delay, positive=False)

    def __call__(self, state: RetryState, /) -> float:
        """Draw the wait after a failed attempt.

        Args:
            state: The failed attempt.

        Returns:
            Seconds to wait, from 0 to the attempt's cap.
        """
        try:
            upper = self.multiplier * 2.0 ** (state.attempt - 1)
        except OverflowError:
            upper = self.max_delay

        return uniform(0.0, min(upper, self.max_delay))


@dataclass(frozen=True, slots=True, kw_only=True)
class RetryPolicy:
    """When and how long a retryable request is retried.

    A policy applies only to methods with ``__retryable__ = True`` and
    only to errors whose ``transient`` flag is true; it cannot make any
    other error retried. Give one to a session for all its calls, or to
    a single ``send_request`` or ``stream`` call.

    Attributes:
        max_attempts: Attempts in total, including the first one; ``1``
            turns retries off.
        deadline: Seconds since the first attempt after which no retry
            starts; a wait that would end later is not started. An
            attempt in progress is limited by its own timeout. ``None``
            for no limit.
        max_retry_after: Longest ``Retry-After`` worth waiting for, in
            seconds; an API that asks for longer gets its error raised
            at once.
        backoff: Computes the wait before each retry; the session waits
            at least the error's ``Retry-After``.
        retry_if: Called for each transient error, the last attempt's
            too; ``False`` raises the error at once. ``None`` retries
            every transient error.

    Example:
        >>> patient = RetryPolicy(max_attempts=6, deadline=None)
        >>> await session.send_request(Ping(), retry_policy=patient)
    """

    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    deadline: float | None = DEFAULT_RETRY_DEADLINE
    max_retry_after: float = DEFAULT_MAX_RETRY_AFTER
    backoff: IBackoff = field(default_factory=FullJitterBackoff)
    retry_if: IRetryPredicate | None = None

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            TypeError: If a setting has the wrong type, or ``backoff`` or
                ``retry_if`` is not callable.
            ValueError: If ``max_attempts`` is below 1, ``deadline`` is
                not positive, or a number is infinite, NaN or negative.
        """
        _check_count("max_attempts", self.max_attempts)

        if self.deadline is not None:
            _check_number("deadline", self.deadline, positive=True)

        _check_number("max_retry_after", self.max_retry_after, positive=False)

        if not callable(self.backoff):
            msg = "backoff must be callable"
            raise TypeError(msg)

        if self.retry_if is not None and not callable(self.retry_if):
            msg = "retry_if must be callable or None"
            raise TypeError(msg)


DEFAULT_RETRY_POLICY: Final[RetryPolicy] = RetryPolicy()
"""The policy of a session that is given none."""
