"""Block C: the retry policy and the strategy that runs it."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tenacity import AsyncRetrying, RetryCallState

from aiollm.http.exceptions.response import APIResponseError
from aiollm.http.exceptions.status import (
    APIStatusError,
    NotFoundError,
    ServiceUnavailableError,
)
from aiollm.http.exceptions.transport import (
    APIConnectionError,
    APITLSError,
)
from aiollm.session.retry import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_RETRY_AFTER,
    DEFAULT_RETRY_DEADLINE,
    DEFAULT_RETRY_POLICY,
    FullJitterBackoff,
    RetryPolicy,
    RetryState,
)
from aiollm.session.retry.runner import (
    calculate_wait,
    is_retried,
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


def transient() -> ServiceUnavailableError:
    return ServiceUnavailableError(status=503, transient=True)


def snapshot(attempt: int = 1) -> RetryState:
    return RetryState(attempt=attempt, elapsed=0.0, error=transient())


def always(_: RetryState) -> bool:
    return True


def one_second(_: RetryState) -> int:
    return 1


# --- should_retry: the library's own rule ---------------------------------


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


def test_should_retry_refuses_a_retry_after_above_the_limit() -> None:
    error = APIStatusError(status=503, retry_after=61.0, transient=True)

    assert should_retry(error) is False
    assert should_retry(error, max_retry_after=61) is True


# --- RetryPolicy ------------------------------------------------------------


def test_the_default_policy_uses_the_constants() -> None:
    assert RetryPolicy() == DEFAULT_RETRY_POLICY
    assert DEFAULT_RETRY_POLICY.max_attempts == DEFAULT_MAX_ATTEMPTS == 4
    assert DEFAULT_RETRY_POLICY.deadline == DEFAULT_RETRY_DEADLINE == 900
    assert (
        DEFAULT_RETRY_POLICY.max_retry_after == DEFAULT_MAX_RETRY_AFTER == 60
    )
    assert DEFAULT_RETRY_POLICY.backoff == FullJitterBackoff()
    assert DEFAULT_RETRY_POLICY.retry_if is None


def test_a_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_RETRY_POLICY.max_attempts = 1  # type: ignore[misc]


def test_a_policy_takes_its_arguments_by_keyword() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(2)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "settings_",
    [
        {"max_attempts": 1},
        {"deadline": None},
        {"deadline": 0.5},
        {"max_retry_after": 0},
        {"retry_if": always},
        {"backoff": one_second},
    ],
    ids=["one-attempt", "no-deadline", "short", "no-wait", "retry-if", "fn"],
)
def test_a_valid_policy_is_accepted(settings_: dict[str, Any]) -> None:
    RetryPolicy(**settings_)


@pytest.mark.parametrize(
    ("settings_", "error", "message"),
    [
        ({"max_attempts": 0}, ValueError, "max_attempts must be at least 1"),
        ({"max_attempts": -1}, ValueError, "at least 1"),
        ({"max_attempts": True}, TypeError, "an int, not bool"),
        ({"max_attempts": 2.0}, TypeError, "an int, not float"),
        ({"deadline": 0}, ValueError, "deadline must be finite and pos"),
        ({"deadline": -1.0}, ValueError, "positive"),
        ({"deadline": math.inf}, ValueError, "positive"),
        ({"deadline": math.nan}, ValueError, "positive"),
        ({"deadline": "10"}, TypeError, "deadline must be a number"),
        ({"deadline": True}, TypeError, "not bool"),
        ({"max_retry_after": -1}, ValueError, "non-negative"),
        ({"max_retry_after": math.inf}, ValueError, "non-negative"),
        ({"backoff": 1.0}, TypeError, "backoff must be callable"),
        ({"retry_if": True}, TypeError, "retry_if must be callable"),
    ],
)
def test_an_invalid_policy_is_rejected(
    settings_: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        RetryPolicy(**settings_)


# --- FullJitterBackoff ------------------------------------------------------


def jitter_cap(attempt: int) -> float:
    """The upper end of the random exponential backoff of an attempt."""
    return min(30.0, 0.5 * 2 ** (attempt - 1))


@settings(max_examples=300, deadline=None)
@given(attempt=st.integers(min_value=1, max_value=60))
def test_the_default_backoff_stays_within_its_cap(attempt: int) -> None:
    assert 0 <= FullJitterBackoff()(snapshot(attempt)) <= jitter_cap(attempt)


def test_the_backoff_survives_a_huge_attempt_number() -> None:
    backoff = FullJitterBackoff(max_delay=2.0)

    assert 0 <= backoff(snapshot(5000)) <= 2.0


def test_a_zero_multiplier_never_waits() -> None:
    assert FullJitterBackoff(multiplier=0)(snapshot(10)) == 0


@pytest.mark.parametrize(
    ("settings_", "error"),
    [
        ({"multiplier": -0.1}, ValueError),
        ({"multiplier": math.nan}, ValueError),
        ({"max_delay": math.inf}, ValueError),
        ({"max_delay": "30"}, TypeError),
        ({"multiplier": False}, TypeError),
    ],
)
def test_an_invalid_backoff_is_rejected(
    settings_: dict[str, Any],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        FullJitterBackoff(**settings_)


# --- is_retried -------------------------------------------------------------


def test_a_success_is_not_retried() -> None:
    assert is_retried(make_state(result="ok"), DEFAULT_RETRY_POLICY) is False


@pytest.mark.parametrize(
    ("error", "expected"),
    [(transient(), True), (NotFoundError(status=404), False)],
    ids=["transient", "permanent"],
)
def test_without_retry_if_every_transient_error_is_retried(
    error: BaseException,
    expected: bool,
) -> None:
    assert is_retried(make_state(error=error), DEFAULT_RETRY_POLICY) is (
        expected
    )


def test_retry_if_receives_the_failed_attempt() -> None:
    seen: list[RetryState] = []
    error = transient()

    def refuse(state: RetryState) -> bool:
        seen.append(state)
        return False

    policy = RetryPolicy(retry_if=refuse)

    assert is_retried(make_state(3, error), policy) is False
    (state,) = seen
    assert state.attempt == 3
    assert state.error is error
    assert state.elapsed >= 0


def test_retry_if_is_not_asked_above_max_retry_after() -> None:
    calls: list[RetryState] = []
    policy = RetryPolicy(
        max_retry_after=5,
        retry_if=lambda state: bool(calls.append(state)) or True,
    )
    error = APIStatusError(status=503, retry_after=6.0, transient=True)

    assert is_retried(make_state(error=error), policy) is False
    assert calls == []


def test_a_failing_retry_if_keeps_the_request_error_as_cause() -> None:
    error = transient()

    def broken(_: RetryState) -> bool:
        raise KeyError("x")

    with pytest.raises(KeyError) as caught:
        is_retried(make_state(error=error), RetryPolicy(retry_if=broken))

    assert caught.value.__cause__ is error


def test_a_failing_retry_if_keeps_its_own_cause() -> None:
    own = OSError("own")

    def broken(_: RetryState) -> bool:
        raise KeyError("x") from own

    with pytest.raises(KeyError) as caught:
        is_retried(make_state(error=transient()), RetryPolicy(retry_if=broken))

    assert caught.value.__cause__ is own


def test_retry_if_may_raise_the_request_error_itself() -> None:
    error = transient()

    def give_up(state: RetryState) -> bool:
        raise state.error

    with pytest.raises(ServiceUnavailableError) as caught:
        is_retried(make_state(error=error), RetryPolicy(retry_if=give_up))

    assert caught.value is error
    assert caught.value.__cause__ is None


# --- calculate_wait ---------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(
    attempt=st.integers(min_value=1, max_value=60),
    error=st.sampled_from([APIConnectionError(), transient()]),
)
def test_wait_without_retry_after_is_the_backoff(
    attempt: int,
    error: BaseException,
) -> None:
    # The real jitter: random, but within the exponential cap, never more
    # than 30 s.
    wait = calculate_wait(make_state(attempt, error), DEFAULT_RETRY_POLICY)

    assert 0 <= wait <= jitter_cap(attempt)


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

    wait = calculate_wait(make_state(attempt, error), DEFAULT_RETRY_POLICY)

    assert retry_after <= wait <= max(jitter_cap(attempt), retry_after)


def test_retry_after_above_the_cap_is_waited_exactly() -> None:
    error = APIStatusError(status=503, retry_after=45.0, transient=True)

    assert calculate_wait(make_state(50, error), DEFAULT_RETRY_POLICY) == 45.0


def test_a_custom_backoff_gives_the_wait() -> None:
    policy = RetryPolicy(backoff=lambda state: state.attempt)

    wait = calculate_wait(make_state(3, transient()), policy)

    assert wait == 3.0
    assert type(wait) is float


def test_retry_after_still_floors_a_custom_backoff() -> None:
    policy = RetryPolicy(backoff=lambda state: 0.0)
    error = APIStatusError(status=503, retry_after=7.0, transient=True)

    assert calculate_wait(make_state(error=error), policy) == 7.0


@pytest.mark.parametrize(
    ("delay", "error"),
    [
        ("1", TypeError),
        (None, TypeError),
        (True, TypeError),
        (-0.5, ValueError),
        (math.inf, ValueError),
        (math.nan, ValueError),
    ],
)
def test_an_invalid_wait_keeps_the_request_error_as_cause(
    delay: object,
    error: type[Exception],
) -> None:
    request_error = transient()
    policy = RetryPolicy(backoff=lambda state: delay)  # type: ignore[arg-type,return-value]

    with pytest.raises(error, match="backoff must return") as caught:
        calculate_wait(make_state(error=request_error), policy)

    assert caught.value.__cause__ is request_error


def test_a_failing_backoff_keeps_the_request_error_as_cause() -> None:
    request_error = APIConnectionError()

    def broken(_: RetryState) -> float:
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError) as caught:
        calculate_wait(
            make_state(error=request_error), RetryPolicy(backoff=broken)
        )

    assert caught.value.__cause__ is request_error


@pytest.mark.parametrize(
    "state",
    [make_state(result="ok"), make_state(error=ValueError("foreign"))],
    ids=["success", "foreign-error"],
)
def test_a_wait_needs_an_attempt_that_failed_with_an_api_error(
    state: RetryCallState,
) -> None:
    with pytest.raises(RuntimeError, match="failed with an API error"):
        calculate_wait(state, DEFAULT_RETRY_POLICY)
