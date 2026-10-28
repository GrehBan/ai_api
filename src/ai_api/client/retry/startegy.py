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
    if isinstance(exc, (APIError, APITransportError)):
        return exc.transient
    return False


def calculate_wait(state: RetryCallState) -> float:
    delay = _jitter(state)

    if state.outcome is None:
        return delay

    exc = state.outcome.exception()

    if isinstance(exc, APIError) and exc.retry_after is not None:
        return max(delay, exc.retry_after)

    return delay


RETRY_CONDITION = retry_if_exception(should_retry)