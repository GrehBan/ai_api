"""OpenAI: the error resolver.

What every provider client shares is in ``test_provider_clients.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_api.enums import Provider
from ai_api.http.exceptions.status import (
    APIStatusError,
    BadRequestError,
    InternalServerError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)
from ai_api.providers.openai.resolver import (
    OpenAIErrorResolver,
)


def test_openai_quota_error_is_not_retried() -> None:
    error = OpenAIErrorResolver().resolve(
        status=429,
        body={"error": {"code": "insufficient_quota", "message": "No"}},
        headers={"Retry-After": "1"},
    )

    assert type(error) is QuotaExceededError
    assert error.transient is False
    assert error.provider is Provider.OPENAI


def test_openai_failed_response_is_read_without_its_object_field() -> None:
    # The same rule as the parser's FAILED_RESPONSE: status alone.
    error = OpenAIErrorResolver().resolve(
        status=200,
        body={
            "id": "resp_1",
            "status": "failed",
            "error": {"code": "server_error", "message": "Boom"},
        },
        headers={},
    )

    assert type(error) is InternalServerError
    assert error.code == "server_error"
    assert "Boom" in str(error)
    assert error.transient is False


def test_openai_rate_limit_falls_back_to_the_default() -> None:
    error = OpenAIErrorResolver().resolve(
        status=429,
        body={"error": {"code": "rate_limit_exceeded"}},
        headers={"Retry-After": "1"},
    )

    assert type(error) is RateLimitError
    assert error.transient is True


@pytest.mark.parametrize(
    "error",
    [
        {"code": "credit_balance_exhausted"},
        {"code": "organization_spend_limit_exceeded"},
        {"code": "project_spend_limit_exceeded"},
        {"code": "organization_usage_limit_exceeded"},
        {"code": "other", "type": "insufficient_quota"},
    ],
    ids=["credit", "org-spend", "project-spend", "usage", "type"],
)
def test_openai_exhausted_limits_are_quota_errors(
    error: dict[str, str],
) -> None:
    resolved = OpenAIErrorResolver().resolve(
        status=429,
        body={"error": error | {"message": "No"}},
        headers={"Retry-After": "1"},
    )

    assert type(resolved) is QuotaExceededError
    assert resolved.transient is False


@pytest.mark.parametrize(
    ("body", "error_class"),
    [
        (
            {"type": "error", "code": "server_is_overloaded"},
            ServiceUnavailableError,
        ),
        ({"type": "error", "code": "server_error"}, InternalServerError),
        (
            {"error": {"type": "invalid_request_error", "code": None}},
            BadRequestError,
        ),
        ({"type": "error", "code": "ERR_SOMETHING"}, APIStatusError),
    ],
    ids=["overloaded", "server", "type", "unknown"],
)
def test_openai_stream_error_is_classified_by_code_or_type(
    body: dict[str, Any],
    error_class: type[APIStatusError],
) -> None:
    resolved = OpenAIErrorResolver().resolve(status=200, body=body, headers={})

    assert type(resolved) is error_class
    assert resolved.status == 200
    # The generation ran and failed: running it again is the caller's
    # choice, so an in-band failure is never transient.
    assert resolved.transient is False
