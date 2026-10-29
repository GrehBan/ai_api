"""Block C: the retry strategy."""

import pytest
from pytest_mock import MockerFixture
from tenacity import AsyncRetrying, RetryCallState

from ai_api.client.exceptions.http import (
    APIError,
    NotFoundError,
    ServiceUnavailableError,
)
from ai_api.client.exceptions.transport import (
    APIConnectionError,
    APIResponseError,
    APITLSError,
)
from ai_api.client.retry.startegy import (
    RETRY_CONDITION,
    calculate_wait,
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
    assert RETRY_CONDITION(make_state(error=APIConnectionError())) is True
    assert RETRY_CONDITION(make_state(error=APITLSError())) is False


@pytest.mark.parametrize("attempt", [1, 2, 5, 10, 50])
def test_jitter_stays_within_bounds(attempt: int) -> None:
    for _ in range(50):
        assert 0 <= calculate_wait(make_state(attempt)) <= 30


@pytest.mark.parametrize(
    "state",
    [
        make_state(),
        make_state(error=APIConnectionError()),
        make_state(error=APIError(status=503)),
        make_state(result="ok"),
    ],
    ids=["no-outcome", "transport-error", "no-retry-after", "result"],
)
def test_wait_is_jitter_without_retry_after(
    mocker: MockerFixture,
    state: RetryCallState,
) -> None:
    jitter = mocker.patch(
        "ai_api.client.retry.startegy._jitter",
        return_value=4.0,
    )

    assert calculate_wait(state) == 4.0
    jitter.assert_called_once_with(state)


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [(10.0, 10.0), (100.0, 100.0), (1.0, 4.0), (0.0, 4.0)],
    ids=["above-jitter", "above-cap", "below-jitter", "zero"],
)
def test_wait_honors_retry_after(
    mocker: MockerFixture,
    retry_after: float,
    expected: float,
) -> None:
    mocker.patch("ai_api.client.retry.startegy._jitter", return_value=4.0)
    error = APIError(status=503, retry_after=retry_after, transient=True)

    assert calculate_wait(make_state(error=error)) == expected


def test_retry_after_above_jitter_with_real_jitter() -> None:
    error = APIError(status=503, retry_after=45.0, transient=True)

    assert calculate_wait(make_state(error=error)) == 45.0
