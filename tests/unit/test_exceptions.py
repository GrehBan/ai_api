"""Block A: the exception hierarchy."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from ai_api.adapters.validation import request_errors
from ai_api.enums import Provider
from ai_api.exceptions import AIAPIError, RequestValidationError
from ai_api.http.exceptions import status as http_errors
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import HTTP_ERRORS, APIStatusError
from ai_api.http.exceptions.transport import (
    APIConnectionError,
    APIRedirectError,
    APIRequestError,
    APIStreamInterruptedError,
    APITimeoutError,
    APITLSError,
    APITransportError,
)


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
    error = APIStatusError(
        status=418,
        message="teapot",
        code="tea",
        error_type="kitchen",
        provider=Provider.OPENAI,
        request_id="req-1",
        retry_after=1.5,
        transient=True,
    )

    assert error.status == 418
    assert error.code == "tea"
    assert error.error_type == "kitchen"
    assert error.provider is Provider.OPENAI
    assert error.request_id == "req-1"
    assert error.retry_after == 1.5
    assert error.transient is True
    assert error.message == "detail: teapot"


def test_api_error_defaults() -> None:
    error = APIStatusError(status=400)

    assert error.transient is False
    assert error.code is None
    assert error.error_type is None
    assert error.provider is Provider.GENERIC
    assert error.request_id is None
    assert error.retry_after is None
    assert error.message == ""


EXPECTED_HTTP_ERRORS = {
    400: http_errors.BadRequestError,
    401: http_errors.AuthenticationError,
    402: http_errors.BillingError,
    403: http_errors.PermissionDeniedError,
    404: http_errors.NotFoundError,
    408: http_errors.RequestTimeoutError,
    409: http_errors.ConflictError,
    422: http_errors.UnprocessableEntityError,
    429: http_errors.RateLimitError,
    500: http_errors.InternalServerError,
    502: http_errors.BadGatewayError,
    503: http_errors.ServiceUnavailableError,
    504: http_errors.GatewayTimeoutError,
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
def test_http_error_class(
    status: int, error_cls: type[APIStatusError]
) -> None:
    error = error_cls(status=status, message="boom")

    assert issubclass(error_cls, APIStatusError)
    assert error_cls.default_message
    assert error.message == f"{error_cls.default_message} detail: boom"


def test_quota_exceeded_is_not_mapped_by_status() -> None:
    assert issubclass(http_errors.QuotaExceededError, APIStatusError)
    assert http_errors.QuotaExceededError.default_message
    assert http_errors.QuotaExceededError not in HTTP_ERRORS.values()


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


def test_stream_interruption_is_never_transient() -> None:
    error = APIStreamInterruptedError(items_received=2)

    assert isinstance(error, APITransportError)
    assert not isinstance(error, APIConnectionError)
    assert error.transient is False
    assert error.items_received == 2
    assert error.message == "detail: The stream was interrupted."


def test_status_error_has_no_raw_by_default() -> None:
    assert APIStatusError(status=500).raw is None


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


class Limited(BaseModel):
    max_tokens: int = Field(gt=0)
    note: str = ""

    @model_validator(mode="after")
    def check(self) -> Limited:
        if self.note == "whole":
            raise ValueError("the method as a whole is wrong")
        return self


def test_request_errors_use_the_shared_names() -> None:
    with (
        pytest.raises(RequestValidationError) as caught,
        request_errors(Provider.OPENAI, {"max_tokens": "max_output_tokens"}),
    ):
        Limited(max_tokens=0, note=5)  # type: ignore[arg-type]

    message = str(caught.value)
    assert caught.value.field == "max_output_tokens"
    assert "max_output_tokens: Input should be greater than 0" in message
    assert "note: Input should be a valid string" in message
    assert isinstance(caught.value.__cause__, ValidationError)


def test_request_error_of_the_whole_method_has_no_location() -> None:
    with (
        pytest.raises(RequestValidationError) as caught,
        request_errors(Provider.ANTHROPIC),
    ):
        Limited(max_tokens=1, note="whole")

    assert caught.value.field == ""
    assert str(caught.value).endswith(
        "detail: Value error, the method as a whole is wrong"
    )


def test_request_validation_error_names_provider_and_field() -> None:
    error = RequestValidationError(
        "must be at least 16", provider=Provider.OPENAI, field="max_tokens"
    )

    assert isinstance(error, ValueError)
    assert error.provider is Provider.OPENAI
    assert error.field == "max_tokens"
    assert str(error) == (
        "openai cannot take the request field 'max_tokens'. "
        "detail: must be at least 16"
    )


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, "pickle"])
def test_request_validation_error_copies(clone: Any) -> None:
    error = RequestValidationError(
        "bad", provider=Provider.ANTHROPIC, field="metadata"
    )

    copied = (
        pickle.loads(pickle.dumps(error))
        if clone == "pickle"
        else clone(error)
    )

    assert type(copied) is RequestValidationError
    assert str(copied) == str(error)
    assert copied.provider is Provider.ANTHROPIC
    assert copied.field == "metadata"


@pytest.mark.parametrize(
    "error_cls",
    [
        APIStatusError,
        APITransportError,
        APIResponseError,
        RequestValidationError,
    ],
    ids=lambda cls: cls.__name__,
)
def test_everything_derives_from_root(error_cls: type[Exception]) -> None:
    assert issubclass(error_cls, AIAPIError)


def test_error_resolver_protocols_import_in_fresh_interpreter() -> None:
    # The module once used APIStatusError without importing it; a fresh
    # interpreter makes sure nothing else imported it first.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ai_api.http.exceptions.protocols",
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
        APIStatusError(status=599),
        APIStatusError(status=200, raw={"status": "failed"}),
        APITimeoutError(),
        APIStreamInterruptedError("cut", items_received=3),
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
        (APIRedirectError(), "Redirect was not followed."),
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
