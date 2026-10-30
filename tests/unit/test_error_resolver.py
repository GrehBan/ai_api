"""Block B: the default error resolver."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any

import pytest
from multidict import CIMultiDict

from ai_api.enums import Provider
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import (
    HTTP_ERRORS,
    APIStatusError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
)
from ai_api.http.headers import get_header, parse_retry_after
from tests.conftest import NOW


@pytest.mark.parametrize(
    ("value", "expected"),
    [("abc", "abc"), ("", ""), (1, None), (None, None), ({"a": 1}, None)],
    ids=["str", "empty-str", "int", "none", "dict"],
)
def test_error_fields_keep_only_strings(
    value: Any,
    expected: str | None,
) -> None:
    error = BaseErrorResolver().resolve(
        status=400,
        body={"error": {"type": value}},
        headers={},
    )

    assert error.error_type == expected


@pytest.mark.parametrize(
    "headers",
    [{"X-Request-Id": "abc"}, CIMultiDict({"x-request-id": "abc"})],
    ids=["dict", "cimultidict"],
)
def test_get_header_is_case_insensitive(headers: Mapping[str, str]) -> None:
    assert get_header(headers, "X-REQUEST-ID") == "abc"


def test_get_header_absent() -> None:
    assert get_header({"Other": "x"}, "x-request-id") is None


def _http_date(moment: datetime.datetime) -> str:
    return moment.strftime("%a, %d %b %Y %H:%M:%S GMT")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("120", 120.0),
        ("1.5", 1.5),
        ("0", 0.0),
        ("-5", 0.0),
        ("inf", None),
        ("nan", None),
        ("soon", None),
        ("", None),
    ],
    ids=[
        "seconds",
        "fraction",
        "zero",
        "negative",
        "inf",
        "nan",
        "garbage",
        "empty",
    ],
)
def test_parse_retry_after_values(
    frozen_now: datetime.datetime,
    value: str,
    expected: float | None,
) -> None:
    assert parse_retry_after({"Retry-After": value}) == expected


def test_parse_retry_after_absent() -> None:
    assert parse_retry_after({}) is None


def test_parse_retry_after_header_name_is_case_insensitive() -> None:
    assert parse_retry_after({"retry-after": "3"}) == 3.0


def test_parse_retry_after_future_date(frozen_now: datetime.datetime) -> None:
    value = _http_date(frozen_now + datetime.timedelta(seconds=90))

    assert parse_retry_after({"Retry-After": value}) == 90.0


def test_parse_retry_after_past_date(frozen_now: datetime.datetime) -> None:
    value = _http_date(frozen_now - datetime.timedelta(hours=1))

    assert parse_retry_after({"Retry-After": value}) == 0.0


def test_parse_retry_after_date_without_timezone_is_utc(
    frozen_now: datetime.datetime,
) -> None:
    # "-0000" makes email.utils return a naive datetime.
    future = frozen_now + datetime.timedelta(seconds=30)
    value = future.strftime("%a, %d %b %Y %H:%M:%S -0000")

    assert parse_retry_after({"Retry-After": value}) == 30.0


def resolve(
    status: int = 400,
    body: Any = None,
    headers: Mapping[str, str] | None = None,
    resolver: BaseErrorResolver | None = None,
) -> APIStatusError:
    return (resolver or BaseErrorResolver()).resolve(
        status=status,
        body=body,
        headers=headers or {},
    )


def test_resolve_reads_error_object() -> None:
    error = resolve(
        404,
        {
            "error": {
                "code": "model_not_found",
                "type": "invalid_request_error",
                "message": "No such model",
            }
        },
        {"x-request-id": "req-1", "Retry-After": "2"},
    )

    assert type(error) is NotFoundError
    assert error.status == 404
    assert error.code == "model_not_found"
    assert error.error_type == "invalid_request_error"
    assert error.message.endswith("detail: No such model")
    assert error.request_id == "req-1"
    assert error.retry_after == 2.0
    assert error.provider == Provider.GENERIC
    assert error.transient is False


@pytest.mark.parametrize(
    "body",
    [None, "plain text", ["error"], {"detail": "x"}, {"error": 1}],
    ids=["none", "str", "list", "no-error-key", "error-not-object"],
)
def test_resolve_tolerates_unexpected_bodies(body: Any) -> None:
    error = resolve(400, body)

    assert error.code is None
    assert error.error_type is None
    assert error.message == HTTP_ERRORS[400].default_message
    assert error.request_id is None


def test_string_error_is_the_message() -> None:
    error = resolve(500, {"error": "model not found"})

    assert error.code is None
    assert error.message.endswith("detail: model not found")


def test_resolve_ignores_non_string_fields() -> None:
    error = resolve(400, {"error": {"code": 42, "type": [], "message": 1}})

    assert error.code is None
    assert error.error_type is None
    assert "detail" not in error.message


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        (
            {"x-request-id": "x", "request-id": "r"},
            {"request_id": "b"},
            "x",
        ),
        ({"request-id": "r"}, {"request_id": "b"}, "r"),
        ({}, {"request_id": "b"}, "b"),
        ({}, {"request_id": 5}, None),
    ],
    ids=["x-request-id", "request-id", "body", "body-not-str"],
)
def test_resolve_request_id_order(
    headers: Mapping[str, str],
    body: Any,
    expected: str | None,
) -> None:
    assert resolve(500, body, headers).request_id == expected


@pytest.mark.parametrize(
    ("status", "error_cls", "transient"),
    [
        (400, HTTP_ERRORS[400], False),
        (404, HTTP_ERRORS[404], False),
        (408, HTTP_ERRORS[408], True),
        (429, RateLimitError, False),
        (500, HTTP_ERRORS[500], True),
        (502, HTTP_ERRORS[502], True),
        (503, HTTP_ERRORS[503], True),
        (504, HTTP_ERRORS[504], True),
        (529, APIStatusError, True),
        (418, APIStatusError, False),
        (599, APIStatusError, False),
    ],
    ids=lambda value: str(value) if isinstance(value, int) else None,
)
def test_classify(
    status: int,
    error_cls: type[APIStatusError],
    transient: bool,
) -> None:
    result = BaseErrorResolver().classify(
        status=status,
        code=None,
        error_type=None,
        message=None,
        retry_after=None,
    )

    assert result == (error_cls, transient)


@pytest.mark.parametrize(
    ("retry_after", "transient"),
    [(None, False), (0.0, True), (30.0, True)],
    ids=["no-retry-after", "zero", "seconds"],
)
def test_rate_limit_is_transient_with_retry_after(
    retry_after: float | None,
    transient: bool,
) -> None:
    result = BaseErrorResolver().classify(
        status=429,
        code=None,
        error_type=None,
        message=None,
        retry_after=retry_after,
    )

    assert result == (RateLimitError, transient)


def test_resolve_uses_classification() -> None:
    error = resolve(529)

    assert type(error) is APIStatusError
    assert error.transient is True


class ExampleErrorResolver(BaseErrorResolver):
    """The subclass from the README's error-handling section."""

    provider = Provider.OPENAI

    def classify(
        self,
        *,
        status: int,
        code: str | None,
        error_type: str | None,
        message: str | None,
        retry_after: float | None,
    ) -> tuple[type[APIStatusError], bool]:
        if status == 429 and code == "insufficient_quota":
            return QuotaExceededError, False

        return super().classify(
            status=status,
            code=code,
            error_type=error_type,
            message=message,
            retry_after=retry_after,
        )


class NumericCodeResolver(ExampleErrorResolver):
    """Provider that sends error codes as numbers."""

    def get_error_code(self, error: Mapping[str, Any]) -> str | None:
        code = error.get("code")
        return None if code is None else str(code)


def test_subclass_classify_override() -> None:
    error = resolve(
        429,
        {"error": {"code": "insufficient_quota"}},
        resolver=ExampleErrorResolver(),
    )

    assert type(error) is QuotaExceededError
    assert error.provider == Provider.OPENAI
    assert error.transient is False


def test_subclass_falls_back_to_default_classify() -> None:
    error = resolve(
        429,
        {"error": {"code": "rate_limit"}},
        resolver=ExampleErrorResolver(),
    )

    assert type(error) is RateLimitError
    assert error.provider == Provider.OPENAI


def test_subclass_get_error_code_override() -> None:
    error = resolve(
        400,
        {"error": {"code": 1001}},
        resolver=NumericCodeResolver(),
    )

    assert error.code == "1001"


def test_resolve_retry_after_uses_frozen_clock(
    frozen_now: datetime.datetime,
) -> None:
    value = _http_date(NOW + datetime.timedelta(seconds=10))

    assert resolve(503, headers={"Retry-After": value}).retry_after == 10.0
