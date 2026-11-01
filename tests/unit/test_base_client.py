"""``BaseProviderClient``: one provider for every part, checked early."""

from __future__ import annotations

import types
from typing import Any

import pytest

from ai_api.adapters.base import BaseAdapter, BaseCompositeAdapter
from ai_api.client.base import BaseProviderClient
from ai_api.client.factory import SessionFactory
from ai_api.client.validators import check_same_provider, provider_of
from ai_api.enums import Provider
from ai_api.http.auth import APIKeyAuth, BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.headers import AUTHORIZATION
from ai_api.providers.anthropic.adapters import AnthropicGenerationAdapter
from ai_api.providers.anthropic.resolver import AnthropicErrorResolver
from ai_api.providers.openai.adapters import OpenAIGenerationAdapter
from ai_api.providers.openai.api import OPENAI_API
from ai_api.providers.openai.client import OpenAIClient
from ai_api.providers.openai.resolver import OpenAIErrorResolver
from ai_api.session.aiohttp import AiohttpSession
from ai_api.types.requests import GenerationRequest
from ai_api.types.responses import GenerationResponse
from tests.conftest import TEST_TOKEN, LocalAPI


def factory(resolver: BaseErrorResolver) -> SessionFactory[Any, Any, ...]:
    return SessionFactory(
        AiohttpSession,
        auth_factory=BearerAuth,
        api=OPENAI_API,
        error_resolver=resolver,
    )


OPENAI_FACTORY = factory(OpenAIErrorResolver())


def declare(**parts: Any) -> type[BaseProviderClient[Any]]:
    """Declare a provider client with the given parts."""
    namespace: dict[str, Any] = {
        "provider": Provider.OPENAI,
        "session_factory": OPENAI_FACTORY,
        "api_key_env": "EXAMPLE_API_KEY",
        "auth_headers": frozenset({AUTHORIZATION}),
        "adapter_type": OpenAIGenerationAdapter,
    }
    namespace.update(parts)
    namespace = {k: v for k, v in namespace.items() if v is not None}
    return types.new_class(
        "ExampleClient",
        (BaseProviderClient[Any],),
        exec_body=lambda body: body.update(namespace),
    )


def test_parts_of_one_provider_are_accepted() -> None:
    client_type = declare()

    client = client_type(TEST_TOKEN)

    assert client.provider is Provider.OPENAI
    assert isinstance(client.session, AiohttpSession)


@pytest.mark.parametrize(
    ("parts", "listed"),
    [
        (
            {"provider": Provider.ANTHROPIC},
            "client=anthropic, error_resolver=openai, adapter=openai",
        ),
        (
            {"session_factory": factory(AnthropicErrorResolver())},
            "client=openai, error_resolver=anthropic, adapter=openai",
        ),
        (
            {"adapter_type": AnthropicGenerationAdapter},
            "client=openai, error_resolver=openai, adapter=anthropic",
        ),
    ],
    ids=["client", "resolver", "adapter"],
)
def test_a_part_of_another_provider_fails_at_definition(
    parts: dict[str, Any],
    listed: str,
) -> None:
    # Every source is named, so the odd one out is plain to see.
    with pytest.raises(TypeError, match=f"disagree: {listed}"):
        declare(**parts)


@pytest.mark.parametrize(
    "missing",
    ["provider", "session_factory", "api_key_env", "auth_headers"],
)
def test_a_missing_part_fails_at_definition(missing: str) -> None:
    with pytest.raises(TypeError, match=f"must set {missing}"):
        declare(**{missing: None})


def test_a_resolver_without_provider_fails_at_definition() -> None:
    class Bare:
        """A resolver that names no provider."""

    session_factory: Any = factory(OpenAIErrorResolver())
    session_factory.error_resolver = Bare()

    with pytest.raises(TypeError, match="error resolver .* no provider"):
        declare(session_factory=session_factory)


def test_check_same_provider_accepts_one_provider() -> None:
    check_same_provider(a=Provider.OPENAI, b=Provider.OPENAI)


def test_provider_of_names_the_source() -> None:
    assert provider_of(OpenAIGenerationAdapter, "adapter") is Provider.OPENAI

    with pytest.raises(TypeError, match="The adapter 1 declares no provider"):
        provider_of(1, "adapter")


def test_the_environment_variable_of_the_client_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "from-env")
    client = declare()()

    assert isinstance(client.session, AiohttpSession)
    assert client.session.auth.encode() == "Bearer from-env"


def test_warnings_name_the_client_and_what_it_takes(api: LocalAPI) -> None:
    session = AiohttpSession(
        api, APIKeyAuth(TEST_TOKEN, header_name="X-Key"), BaseErrorResolver()
    )

    with pytest.warns(UserWarning) as caught:
        OpenAIClient(session=session)

    messages = [str(warning.message) for warning in caught]
    assert messages == [
        "OpenAIClient is using the generic BaseErrorResolver. Consider "
        "using OpenAIErrorResolver for provider-specific error handling.",
        "OpenAIClient sends the key in X-Key, but openai takes it in "
        "Authorization.",
    ]
    assert all(warning.filename == __file__ for warning in caught)


class NoProvider(BaseAdapter[GenerationRequest, str, GenerationResponse[str]]):
    """A concrete adapter is checked; this one is not defined."""


def test_a_concrete_adapter_without_provider_fails() -> None:
    with pytest.raises(TypeError, match="Late must set provider"):

        class Late(NoProvider):
            def to_method(self, request: GenerationRequest) -> Any:
                return None

            def to_response(
                self,
                raw: str,
                request: GenerationRequest,
            ) -> GenerationResponse[str]:
                raise NotImplementedError

        _ = Late


def test_an_abstract_adapter_needs_no_provider() -> None:
    # A base of a provider's adapters sets no provider of its own.
    class Partial(
        BaseCompositeAdapter[
            GenerationRequest,
            str,
            GenerationResponse[str],
            str,
            Any,
        ]
    ):
        def to_method(self, request: GenerationRequest) -> Any:
            return None

    assert not hasattr(Partial, "provider")
