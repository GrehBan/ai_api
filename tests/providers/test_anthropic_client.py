"""Anthropic: the error resolver and the requests of its client.

What every provider client shares is in ``test_provider_clients.py``.

The error bodies follow the examples of Anthropic's API reference
(captured 2026-10-30).
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_api.enums import Provider
from ai_api.http.exceptions.status import (
    APIStatusError,
    BadRequestError,
    ContentTooLargeError,
    InternalServerError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)
from ai_api.methods.base import BaseAPIMethod
from ai_api.providers.anthropic import (
    AnthropicClient,
    AnthropicErrorResolver,
)
from ai_api.providers.anthropic.client import session_factory
from tests.conftest import (
    TEST_TOKEN,
    LocalAPI,
    Recorder,
    Reply,
    make_method,
)

SPEND_CAP_MESSAGE = (
    "You have reached your API usage limits: your organization has "
    "crossed its monthly API usage threshold, set based on your "
    "organization's API tier. You will regain access on 2026-09-01 at "
    "00:00 UTC."
)

SPEND_LIMIT_MESSAGE = (
    "You have reached your specified API usage limits. You will regain "
    "access on 2026-11-01 at 00:00 UTC."
)


def body(error_type: str, message: str = "Error", **error: Any) -> Any:
    """An error body in Anthropic's shape."""
    return {
        "type": "error",
        "error": {"type": error_type, "message": message, **error},
        "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy",
    }


def resolve(
    status: int,
    payload: Any,
    headers: dict[str, str] | None = None,
) -> APIStatusError:
    return AnthropicErrorResolver().resolve(
        status=status, body=payload, headers=headers or {}
    )


def test_error_reads_the_type_message_and_request_id() -> None:
    error = resolve(404, body("not_found_error", "Not found"))

    assert type(error) is NotFoundError
    assert error.error_type == "not_found_error"
    assert error.code is None
    assert error.request_id == "req_011CSHoEeqs5C35K2UUqR7Fy"
    assert error.provider is Provider.ANTHROPIC
    assert "Not found" in str(error)


def test_overloaded_is_a_transient_unavailable_error() -> None:
    error = resolve(529, body("overloaded_error", "Overloaded"))

    assert type(error) is ServiceUnavailableError
    assert error.status == 529
    assert error.transient is True


def test_rate_limit_with_retry_after_is_transient() -> None:
    error = resolve(429, body("rate_limit_error"), {"retry-after": "3"})

    assert type(error) is RateLimitError
    assert error.transient is True


def test_spend_cap_is_a_quota_error() -> None:
    details = {"error_code": "enforced_spend_limit_reached"}
    error = resolve(
        429, body("rate_limit_error", SPEND_CAP_MESSAGE, details=details)
    )

    assert type(error) is QuotaExceededError
    assert error.code == "enforced_spend_limit_reached"
    assert error.transient is False


@pytest.mark.parametrize(
    "message",
    [
        SPEND_LIMIT_MESSAGE,
        "You have reached your specified workspace API usage limits. You "
        "will regain access on 2026-11-01 at 00:00 UTC.",
    ],
    ids=["organization", "workspace"],
)
def test_own_spend_limit_is_a_quota_error(message: str) -> None:
    # Both starts that api/rate-limits documents: this 400 has no code of
    # its own, so the prefix is all there is to recognize it by.
    error = resolve(400, body("invalid_request_error", message))

    assert type(error) is QuotaExceededError
    assert error.transient is False


def test_too_large_request_is_a_content_too_large_error() -> None:
    error = resolve(413, body("request_too_large", "Request too large"))

    assert type(error) is ContentTooLargeError
    assert error.transient is False


def test_too_large_request_inside_a_stream_keeps_its_class() -> None:
    error = resolve(200, body("request_too_large", "Request too large"))

    assert type(error) is ContentTooLargeError
    assert error.status == 200


@pytest.mark.parametrize(
    "error",
    [{"type": "invalid_request_error"}, {"details": "x"}],
    ids=["no-message", "details-not-an-object"],
)
def test_other_bad_requests_stay_bad_requests(error: dict[str, Any]) -> None:
    resolved = resolve(400, {"type": "error", "error": error})

    assert type(resolved) is BadRequestError
    assert resolved.code is None


def test_non_string_error_code_is_ignored_as_a_code() -> None:
    error = resolve(429, body("rate_limit_error", details={"error_code": 1}))

    assert type(error) is RateLimitError
    assert error.code is None


@pytest.mark.parametrize(
    ("error_type", "error_class"),
    [
        ("overloaded_error", ServiceUnavailableError),
        ("api_error", InternalServerError),
        ("invalid_request_error", BadRequestError),
        ("brand_new_error", APIStatusError),
    ],
)
def test_stream_error_event_is_classified_by_its_type(
    error_type: str,
    error_class: type[APIStatusError],
) -> None:
    error = resolve(200, body(error_type))

    assert type(error) is error_class
    assert error.status == 200
    # The generation ran and failed: running it again is the caller's
    # choice, so an in-band failure is never transient.
    assert error.transient is False


async def test_client_sends_bearer_auth_and_maps_errors(
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(
        Reply(
            status=404,
            json=body("not_found_error", "Not found"),
            headers={"request-id": "req_header"},
        )
    )
    method = make_method(BaseAPIMethod[bytes], http_method="GET")()

    async with AnthropicClient(TEST_TOKEN) as client:
        with pytest.raises(NotFoundError) as caught:
            await client.send_request(method)

    assert caught.value.request_id == "req_header"
    assert recorder.last.headers["Authorization"] == f"Bearer {TEST_TOKEN}"
