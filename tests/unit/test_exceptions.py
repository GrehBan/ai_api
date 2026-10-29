"""Block A: the exception hierarchy."""

import copy
import pickle
import subprocess
import sys
from typing import Any, ClassVar

import pytest

from ai_api.client.exceptions import http
from ai_api.client.exceptions.http import HTTP_ERRORS, APIError
from ai_api.client.exceptions.transport import (
    APIConnectionError,
    APIRedirectError,
    APIRequestError,
    APIResponseError,
    APITimeoutError,
    APITLSError,
    APITransportError,
)
from ai_api.exceptions import AIAPIError


class TemplatedError(AIAPIError):
    default_message: ClassVar[str | None] = "Failed {what} after {n} tries."


class PrefixedError(AIAPIError):
    default_message: ClassVar[str | None] = "Prefix."


@pytest.mark.parametrize(
    ("error_cls", "message", "expected"),
    [
        (PrefixedError, "boom", "Prefix. detail: boom"),
        (PrefixedError, None, "Prefix."),
        (AIAPIError, "boom", "detail: boom"),
        (AIAPIError, None, ""),
        (PrefixedError, "", "Prefix."),
    ],
    ids=["both", "default-only", "message-only", "neither", "empty-message"],
)
def test_message_composition(
    error_cls: type[AIAPIError],
    message: str | None,
    expected: str,
) -> None:
    error = error_cls(message)

    assert error.message == expected
    assert str(error) == expected


def test_kwargs_fill_default_message_template() -> None:
    error = TemplatedError("late", what="upload", n=3)

    assert error.message == "Failed upload after 3 tries. detail: late"


def test_extra_args_are_forwarded_to_exception() -> None:
    error = AIAPIError("boom", 1, "x")

    assert error.args == ("detail: boom", 1, "x")


def test_api_error_keeps_all_attributes() -> None:
    error = APIError(
        status=418,
        message="teapot",
        code="tea",
        error_type="kitchen",
        provider="openai",
        request_id="req-1",
        retry_after=1.5,
        transient=True,
    )

    assert error.status == 418
    assert error.code == "tea"
    assert error.error_type == "kitchen"
    assert error.provider == "openai"
    assert error.request_id == "req-1"
    assert error.retry_after == 1.5
    assert error.transient is True
    assert error.message == "detail: teapot"


def test_api_error_defaults() -> None:
    error = APIError(status=400)

    assert error.transient is False
    assert error.code is None
    assert error.error_type is None
    assert error.provider is None
    assert error.request_id is None
    assert error.retry_after is None
    assert error.message == ""


EXPECTED_HTTP_ERRORS = {
    400: http.BadRequestError,
    401: http.AuthenticationError,
    402: http.BillingError,
    403: http.PermissionDeniedError,
    404: http.NotFoundError,
    408: http.RequestTimeoutError,
    409: http.ConflictError,
    422: http.UnprocessableEntityError,
    429: http.RateLimitError,
    500: http.InternalServerError,
    502: http.BadGatewayError,
    503: http.ServiceUnavailableError,
    504: http.GatewayTimeoutError,
}


def test_http_errors_cover_exactly_the_known_statuses() -> None:
    assert HTTP_ERRORS == EXPECTED_HTTP_ERRORS


def test_http_errors_map_to_distinct_classes() -> None:
    assert len(set(HTTP_ERRORS.values())) == len(HTTP_ERRORS)


@pytest.mark.parametrize(
    ("status", "error_cls"),
    list(EXPECTED_HTTP_ERRORS.items()),
    ids=[str(status) for status in EXPECTED_HTTP_ERRORS],
)
def test_http_error_class(status: int, error_cls: type[APIError]) -> None:
    error = error_cls(status=status, message="boom")

    assert issubclass(error_cls, APIError)
    assert error_cls.default_message
    assert error.message == f"{error_cls.default_message} detail: boom"


def test_quota_exceeded_is_not_mapped_by_status() -> None:
    assert issubclass(http.QuotaExceededError, APIError)
    assert http.QuotaExceededError.default_message
    assert http.QuotaExceededError not in HTTP_ERRORS.values()


@pytest.mark.parametrize(
    ("error_cls", "transient"),
    [
        (APIConnectionError, True),
        (APITimeoutError, True),
        (APITLSError, False),
    ],
    ids=["connection", "timeout", "tls"],
)
def test_transport_error_transience(
    error_cls: type[APIConnectionError | APITimeoutError | APITLSError],
    transient: bool,
) -> None:
    error = error_cls()

    assert isinstance(error, APITransportError)
    assert error.transient is transient
    assert error.message.startswith("detail: ")


def test_transport_error_default_and_override() -> None:
    assert APITransportError("boom").transient is True
    assert APITransportError("boom", transient=False).transient is False


def test_transport_error_custom_message() -> None:
    assert APITimeoutError("slow").message == "detail: slow"


def test_response_error_keeps_status() -> None:
    error = APIResponseError("bad body", status=502)

    assert error.status == 502
    assert error.message == "detail: bad body"
    assert APIResponseError("bad body").status is None


def test_response_error_is_not_a_transport_error() -> None:
    assert not issubclass(APIResponseError, APITransportError)
    assert issubclass(APIResponseError, AIAPIError)


@pytest.mark.parametrize(
    "error_cls",
    [APIError, APITransportError, APIResponseError],
    ids=lambda cls: cls.__name__,
)
def test_everything_derives_from_root(error_cls: type[Exception]) -> None:
    assert issubclass(error_cls, AIAPIError)


def test_error_resolver_protocols_import_in_fresh_interpreter() -> None:
    # The module once used APIError without importing it; a fresh
    # interpreter makes sure nothing else imported it first.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ai_api.client.exceptions.errror_resolver.protocols",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def pickled(error: AIAPIError) -> AIAPIError:
    restored: AIAPIError = pickle.loads(pickle.dumps(error))  # noqa: S301
    return restored


@pytest.mark.parametrize(
    "error",
    [
        HTTP_ERRORS[429](
            status=429,
            message="slow down",
            code="rate_limit",
            request_id="req-1",
            retry_after=2.0,
            transient=True,
        ),
        APIError(status=599),
        APITimeoutError(),
        APIRequestError("bad header"),
        APIResponseError("broken", status=200),
        TemplatedError("late", what="upload", n=3),
        AIAPIError("boom", 1, "x"),
    ],
    ids=lambda error: type(error).__name__,
)
@pytest.mark.parametrize(
    "clone",
    [
        pickled,
        copy.copy,
        copy.deepcopy,
    ],
    ids=["pickle", "copy", "deepcopy"],
)
def test_errors_survive_pickle_and_copy(error: Any, clone: Any) -> None:
    restored = clone(error)

    assert type(restored) is type(error)
    assert restored.args == error.args
    assert str(restored) == str(error)
    assert vars(restored) == vars(error)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (APIRedirectError(), "Too many redirects."),
        (APIRequestError(), "Failed to send the request."),
    ],
    ids=["redirect", "request"],
)
def test_request_side_errors_are_not_transient(
    error: APITransportError,
    message: str,
) -> None:
    assert error.transient is False
    assert error.message == f"detail: {message}"
