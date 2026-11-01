"""What every provider client does the same way, run for each provider.

A new provider adds a ``ClientCase``; what differs between providers (the
error resolvers, Anthropic's ``default_max_tokens``) stays in the
provider's own module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import pytest

from ai_api import SessionMismatchWarning
from ai_api.client.base import BaseClient, BaseProviderClient
from ai_api.enums import Provider
from ai_api.http.auth import APIKeyAuth, BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.protocols import IAPI, IAuth
from ai_api.providers.anthropic import (
    ANTHROPIC_API,
    AnthropicClient,
    AnthropicErrorResolver,
)
from ai_api.providers.openai.api import OPENAI_API
from ai_api.providers.openai.client import OpenAIClient
from ai_api.providers.openai.resolver import OpenAIErrorResolver
from ai_api.session.aiohttp import AiohttpSession
from tests.conftest import TEST_TOKEN, FakeTransport


@dataclass(frozen=True)
class ClientCase:
    """One provider client in the scenarios.

    Attributes:
        client: The client class.
        provider: Its provider.
        api: The API its sessions are built for.
        url: A path of the API and the URL it builds.
        resolver: The provider's error resolver class.
        env: The environment variable with the key.
        auths: Authentications the provider takes.
        foreign_resolver: Another provider's resolver, warned about.
    """

    client: type[BaseProviderClient]
    provider: Provider
    api: IAPI
    url: tuple[str, str]
    resolver: type[BaseErrorResolver]
    env: str
    auths: tuple[IAuth, ...]
    foreign_resolver: type[BaseErrorResolver]


OPENAI = ClientCase(
    client=OpenAIClient,
    provider=Provider.OPENAI,
    api=OPENAI_API,
    url=("/models", "https://api.openai.com/v1/models"),
    resolver=OpenAIErrorResolver,
    env="OPENAI_API_KEY",
    auths=(BearerAuth("k"),),
    foreign_resolver=AnthropicErrorResolver,
)

ANTHROPIC = ClientCase(
    client=AnthropicClient,
    provider=Provider.ANTHROPIC,
    api=ANTHROPIC_API,
    url=("/messages", "https://api.anthropic.com/v1/messages"),
    resolver=AnthropicErrorResolver,
    env="ANTHROPIC_API_KEY",
    auths=(BearerAuth("k"), APIKeyAuth("k")),
    foreign_resolver=OpenAIErrorResolver,
)


@pytest.fixture(params=[OPENAI, ANTHROPIC], ids=["openai", "anthropic"])
def case(request: pytest.FixtureRequest) -> ClientCase:
    """Each provider client in turn."""
    return request.param


def test_client_from_a_token_outside_a_coroutine(case: ClientCase) -> None:
    session = case.client(TEST_TOKEN).session

    assert isinstance(session, AiohttpSession)
    assert session.api is case.api
    assert session.auth == BearerAuth(TEST_TOKEN)
    assert type(session.error_resolver) is case.resolver
    assert case.client.provider is case.provider


def test_api_url(case: ClientCase) -> None:
    path, url = case.url

    assert case.api.build_url(path) == url


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"token": "t", "session": FakeTransport()}],
    ids=["neither", "both"],
)
def test_client_needs_a_token_or_a_session(
    case: ClientCase,
    kwargs: Any,
) -> None:
    with pytest.raises(ValueError, match="token or a session"):
        case.client(**kwargs)


def test_client_reads_the_key_from_the_environment(
    case: ClientCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(case.env, "sk-env")

    session = case.client().session

    assert isinstance(session, AiohttpSession)
    assert session.auth.encode() == "Bearer sk-env"


def test_explicit_token_wins_over_the_environment(
    case: ClientCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(case.env, "sk-env")

    session = case.client("sk-explicit").session

    assert isinstance(session, AiohttpSession)
    assert session.auth.encode() == "Bearer sk-explicit"


def test_session_ignores_the_environment(
    case: ClientCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(case.env, "sk-env")
    session = FakeTransport(error_resolver=case.resolver())

    assert case.client(session=session).session is session


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_missing_key_names_the_variable(
    case: ClientCase,
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    # An empty variable counts as unset.
    if value is not None:
        monkeypatch.setenv(case.env, value)

    with pytest.raises(ValueError, match=f"or set {case.env}"):
        case.client()


def test_a_fitting_session_gets_no_warning(case: ClientCase) -> None:
    for auth in case.auths:
        session = FakeTransport(auth=auth, error_resolver=case.resolver())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            client = case.client(session=session)

        assert client.session is session


@pytest.mark.parametrize(
    "resolver",
    ["generic", "foreign"],
)
def test_a_resolver_of_no_or_another_provider_is_warned_about(
    case: ClientCase,
    resolver: str,
) -> None:
    given = (
        BaseErrorResolver()
        if resolver == "generic"
        else case.foreign_resolver()
    )
    session = FakeTransport(error_resolver=given)

    with pytest.warns(
        SessionMismatchWarning,
        match=f"is using {type(given).__name__}. Consider using "
        f"{case.resolver.__name__}",
    ) as info:
        case.client(session=session)

    # The warning points at the code that built the client.
    assert info[0].filename == __file__


def test_another_auth_header_is_warned_about(case: ClientCase) -> None:
    session = FakeTransport(
        auth=APIKeyAuth("k", header_name="X-Key"),
        error_resolver=case.resolver(),
    )

    with pytest.warns(
        SessionMismatchWarning,
        match=f"sends the key in X-Key, but {case.provider.value} takes it",
    ):
        case.client(session=session)


def test_other_sessions_are_not_checked(case: ClientCase) -> None:
    # Not a BaseSession, but it has the ISession methods.
    session = BaseClient(FakeTransport(error_resolver=BaseErrorResolver()))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client = case.client(session=session)

    assert client.session is session


def test_session_arguments_are_refused_with_a_session(
    case: ClientCase,
) -> None:
    session = FakeTransport(error_resolver=case.resolver())

    with pytest.raises(ValueError, match="pass them to the given session"):
        case.client(session=session, max_body_size=1)
