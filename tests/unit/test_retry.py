"""Block C: the retry strategy."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tenacity import AsyncRetrying, RetryCallState

from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import (
    APIStatusError,
    NotFoundError,
    ServiceUnavailableError,
)
from ai_api.http.exceptions.transport import (
    APIConnectionError,
    APITLSError,
)
from ai_api.session.retry import (
    calculate_wait,
    retry_condition,
    should_retry,
)


def make_state(
    attempt: int = 1,
    error: BaseException | None = None,
    *,
    result: object = None,
) -> RetryCallState:
    state = RetryCallState(
        retry_object=AsyncRetrying(),
        fn=None,
        args=(),
        kwargs={},
    )
    state.attempt_number = attempt

    if error is not None:
        state.set_exception((type(error), error, None))
    elif result is not None:
        state.set_result(result)

    return state


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ServiceUnavailableError(status=503, transient=True), True),
        (NotFoundError(status=404), False),
        (APIConnectionError(), True),
        (APITLSError(), False),
        (APIResponseError("bad body"), False),
        (ValueError("other"), False),
    ],
    ids=[
        "api-transient",
        "api-permanent",
        "transport-transient",
        "transport-permanent",
        "response-error",
        "foreign",
    ],
)
def test_should_retry(exc: BaseException, expected: bool) -> None:
    assert should_retry(exc) is expected


def test_retry_condition_uses_should_retry() -> None:
    condition = retry_condition()

    assert condition(make_state(error=APIConnectionError())) is True
    assert condition(make_state(error=APITLSError())) is False


def jitter_cap(attempt: int) -> float:
    """The upper end of the random exponential backoff of an attempt."""
    return min(30.0, 0.5 * 2 ** (attempt - 1))


OUTCOMES = st.sampled_from(
    [None, APIConnectionError(), APIStatusError(status=503), "ok"]
)
"""What an attempt ended with, besides a Retry-After."""


@settings(max_examples=300, deadline=None)
@given(attempt=st.integers(min_value=1, max_value=60), outcome=OUTCOMES)
def test_wait_without_retry_after_is_the_backoff(
    attempt: int,
    outcome: object,
) -> None:
    # The real jitter: random, but within the exponential cap, never more
    # than 30 s.
    if isinstance(outcome, BaseException):
        state = make_state(attempt, outcome)
    else:
        state = make_state(attempt, result=outcome)

    assert 0 <= calculate_wait(state) <= jitter_cap(attempt)


@settings(max_examples=300, deadline=None)
@given(
    attempt=st.integers(min_value=1, max_value=60),
    retry_after=st.floats(min_value=0, max_value=3600, allow_nan=False),
)
def test_wait_is_never_below_retry_after(
    attempt: int,
    retry_after: float,
) -> None:
    # The API's own wait is honored, and the backoff never adds more than
    # its cap on top: Retry-After <= wait <= max(cap, Retry-After).
    error = APIStatusError(status=503, retry_after=retry_after, transient=True)

    wait = calculate_wait(make_state(attempt, error))

    assert retry_after <= wait <= max(jitter_cap(attempt), retry_after)


def test_retry_after_above_the_cap_is_waited_exactly() -> None:
    error = APIStatusError(status=503, retry_after=45.0, transient=True)

    assert calculate_wait(make_state(50, error)) == 45.0
