"""Block M: ``BaseClient``, ``OpenAIClient`` and ``SessionFactory``."""

import warnings
from typing import Any

import pytest

from ai_api.client.auth.api_key import APIKeyAuth
from ai_api.client.auth.bearer import BearerAuth
from ai_api.client.base import BaseClient
from ai_api.client.exceptions.error_resolver.base import BaseErrorResolver
from ai_api.client.exceptions.http import (
    QuotaExceededError,
    RateLimitError,
)
from ai_api.client.factory.factory import SessionFactory
from ai_api.client.response.base import BufferedResponse
from ai_api.client.session.aiohttp import AiohttpSession
from ai_api.enums import Provider
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.openai.client.api.openai import OPENAI_API
from ai_api.providers.openai.client.exceptions.error_resolver.openai import (
    OpenAIErrorResolver,
)
from ai_api.providers.openai.client.openai import OpenAIClient
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


def test_openai_rate_limit_falls_back_to_the_default() -> None:
    error = OpenAIErrorResolver().resolve(
        status=429,
        body={"error": {"code": "rate_limit_exceeded"}},
        headers={"Retry-After": "1"},
    )

    assert type(error) is RateLimitError
    assert error.transient is True


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
