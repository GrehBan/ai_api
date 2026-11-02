"""Interfaces of the callables a retry policy takes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from aiollm.session.retry.state import RetryState


class IBackoff(Protocol):
    """Computes the wait before the next attempt."""

    def __call__(self, state: RetryState, /) -> float:
        """Return the wait after the failed attempt.

        Args:
            state: The failed attempt.

        Returns:
            Seconds to wait: a finite, non-negative number. The session
            waits at least the error's ``Retry-After``.
        """
        ...


class IRetryPredicate(Protocol):
    """Decides whether a transient error is retried."""

    def __call__(self, state: RetryState, /) -> bool:
        """Return whether to retry the failed attempt.

        Args:
            state: The failed attempt; its error is always transient.

        Returns:
            ``False`` to raise the error now, ``True`` to retry it.
        """
        ...
