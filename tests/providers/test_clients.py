"""Block M: ``BaseClient``, ``OpenAIClient`` and ``SessionFactory``."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from ai_api.client.base import BaseClient
from ai_api.client.factory import SessionFactory
from ai_api.enums import Provider
from ai_api.http.auth import APIKeyAuth, BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.status import (
    APIStatusError,
    BadRequestError,
    InternalServerError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
)
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.openai.api import OPENAI_API
from ai_api.providers.openai.client import OpenAIClient
from ai_api.providers.openai.resolver import (
    OpenAIErrorResolver,
)
from ai_api.response.buffered import BufferedResponse
from ai_api.session.aiohttp import AiohttpSession
from tests.conftest import (
    TEST_TOKEN,
    FakeStreamResponse,
    FakeTransport,
    make_method,
)


def buffered() -> BaseAPIMethod[bytes]:
    return make_method(BaseAPIMethod[bytes])()


def streamed() -> BaseStreamMethod[bytes]:
    return make_method(BaseStreamMethod[bytes], http_method="GET")()


async def test_client_delegates_to_its_session() -> None:
    transport = FakeTransport(
        [BufferedResponse(200, {}, b"done")],
        [FakeStreamResponse([b"a", b"b"])],
    )
    client = BaseClient(transport)

    assert await client.send_request(buffered()) == b"done"

    async with client.stream(streamed()) as chunks:
        assert [chunk async for chunk in chunks] == [b"a", b"b"]


async def test_client_forwards_call_options() -> None:
    transport = FakeTransport(
        [BufferedResponse(200, {}, b"done")],
        [FakeStreamResponse([b"a"])],
    )
    client = BaseClient(transport)

    await client.send_request(buffered(), headers={"X-A": "1"}, timeout=3)

    async with client.stream(streamed(), headers={"X-B": "2"}, timeout=4):
        pass

    assert transport.call_options == [({"X-A": "1"}, 3), ({"X-B": "2"}, 4)]


async def test_client_context_closes_its_session() -> None:
    transport = FakeTransport()

    async with BaseClient(transport) as client:
        assert client.session is transport

    assert transport.close_calls == 1


async def test_client_close_closes_its_session() -> None:
    transport = FakeTransport()

    await BaseClient(transport).close()

    assert transport.close_calls == 1


def test_openai_client_from_a_token_outside_a_coroutine() -> None:
    session = OpenAIClient(TEST_TOKEN).session

    assert isinstance(session, AiohttpSession)
    assert session.api is OPENAI_API
    assert session.auth == BearerAuth(TEST_TOKEN)
    assert isinstance(session.error_resolver, OpenAIErrorResolver)


def test_openai_api_url() -> None:
    assert (
        OPENAI_API.build_url("/models") == "https://api.openai.com/v1/models"
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"token": "t", "session": FakeTransport()}],
    ids=["neither", "both"],
)
def test_openai_client_needs_a_token_or_a_session(kwargs: Any) -> None:
    with pytest.raises(ValueError, match="token or a session"):
        OpenAIClient(**kwargs)


def test_openai_client_reads_the_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    session = OpenAIClient().session

    assert isinstance(session, AiohttpSession)
    assert session.auth.encode() == "Bearer sk-env"


def test_explicit_token_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    session = OpenAIClient("sk-explicit").session

    assert isinstance(session, AiohttpSession)
    assert session.auth.encode() == "Bearer sk-explicit"


def test_session_ignores_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    session = FakeTransport(error_resolver=OpenAIErrorResolver())

    assert OpenAIClient(session=session).session is session


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_missing_key_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    if value is not None:
        monkeypatch.setenv("OPENAI_API_KEY", value)

    with pytest.raises(ValueError, match="or set OPENAI_API_KEY"):
        OpenAIClient()


def test_openai_client_accepts_a_fitting_session() -> None:
    session = FakeTransport(error_resolver=OpenAIErrorResolver())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client = OpenAIClient(session=session)

    assert client.session is session


def test_generic_resolver_is_warned_about_at_the_call_site() -> None:
    with pytest.warns(UserWarning, match="generic BaseErrorResolver") as info:
        OpenAIClient(session=FakeTransport(error_resolver=BaseErrorResolver()))

    assert info[0].filename == __file__


def test_non_bearer_auth_is_warned_about() -> None:
    session = FakeTransport(
        auth=APIKeyAuth("k"),
        error_resolver=OpenAIErrorResolver(),
    )

    with pytest.warns(UserWarning, match="Authorization"):
        OpenAIClient(session=session)


def test_other_sessions_are_not_checked() -> None:
    # Not a BaseSession, but it has the ISession methods.
    session = BaseClient(FakeTransport(error_resolver=BaseErrorResolver()))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client = OpenAIClient(session=session)

    assert client.session is session


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


def test_factory_turns_a_token_into_auth_and_forwards_arguments() -> None:
    calls: list[tuple[Any, ...]] = []

    def build(
        api: Any,
        auth: BearerAuth,
        resolver: Any,
        *,
        limit: int = 1,
    ) -> FakeTransport:
        calls.append((api, auth, resolver, limit))
        return FakeTransport()

    resolver = OpenAIErrorResolver()
    factory = SessionFactory(
        build,
        auth_factory=BearerAuth,
        api=OPENAI_API,
        error_resolver=resolver,
    )

    factory("token", limit=5)
    factory(BearerAuth("ready"))

    assert calls == [
        (OPENAI_API, BearerAuth("token"), resolver, 5),
        (OPENAI_API, BearerAuth("ready"), resolver, 1),
    ]
