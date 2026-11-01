"""``BaseClient`` and ``BaseProviderClient``.

``BaseClient`` delegates to its session and owns it; a provider client
names one provider in every part, checked early.
"""

from __future__ import annotations

import types
import warnings
from types import TracebackType
from typing import Any

import pytest

from ai_api import SessionMismatchWarning
from ai_api.adapters.base import BaseAdapter, BaseCompositeAdapter
from ai_api.client.base import BaseClient, BaseProviderClient
from ai_api.client.validators import (
    check_same_provider,
    provider_of,
    providers_of,
)
from ai_api.enums import Provider
from ai_api.http.auth import APIKeyAuth, BearerAuth
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.headers import AUTHORIZATION
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.anthropic.adapters import AnthropicGenerationAdapter
from ai_api.providers.anthropic.resolver import AnthropicErrorResolver
from ai_api.providers.openai.adapters import OpenAIGenerationAdapter
from ai_api.providers.openai.api import OPENAI_API
from ai_api.providers.openai.client import OpenAIClient
from ai_api.providers.openai.resolver import OpenAIErrorResolver
from ai_api.response.buffered import BufferedResponse
from ai_api.session.aiohttp import AiohttpSession
from ai_api.session.factory import SessionFactory
from ai_api.types.requests import GenerationRequest
from ai_api.types.responses import GenerationResponse
from tests.conftest import (
    TEST_TOKEN,
    FakeStreamResponse,
    FakeTransport,
    LocalAPI,
    make_method,
)


def factory(resolver: BaseErrorResolver) -> SessionFactory[Any, Any, ...]:
    return SessionFactory(
        AiohttpSession,
        auth_factory=BearerAuth,
        api=OPENAI_API,
        error_resolver=resolver,
    )


OPENAI_FACTORY = factory(OpenAIErrorResolver())


class ModelsAdapter(
    BaseAdapter[GenerationRequest, str, GenerationResponse[str]]
):
    """A second adapter of the client, as a ``models`` resource would add."""

    provider = Provider.OPENAI

    def __init__(self, *, page_size: int = 20) -> None:
        self.page_size = page_size

    def to_method(self, request: GenerationRequest) -> Any:
        raise NotImplementedError

    def to_response(
        self,
        raw: str,
        request: GenerationRequest,
    ) -> GenerationResponse[str]:
        raise NotImplementedError


class AnthropicModelsAdapter(ModelsAdapter):
    """The same adapter, of another provider."""

    provider = Provider.ANTHROPIC


def declare(**parts: Any) -> type[BaseProviderClient]:
    """Declare a provider client with the given parts; ``None`` drops one."""
    namespace: dict[str, Any] = {
        "provider": Provider.OPENAI,
        "session_factory": OPENAI_FACTORY,
        "api_key_env": "EXAMPLE_API_KEY",
        "auth_headers": frozenset({AUTHORIZATION}),
        "adapter_types": (OpenAIGenerationAdapter, ModelsAdapter),
    }
    namespace.update(parts)
    namespace = {k: v for k, v in namespace.items() if v is not None}
    return types.new_class(
        "ExampleClient",
        (BaseProviderClient,),
        exec_body=lambda body: body.update(namespace),
    )


def test_parts_of_one_provider_are_accepted() -> None:
    client = declare()(TEST_TOKEN)

    assert client.provider is Provider.OPENAI
    assert isinstance(client.session, AiohttpSession)


@pytest.mark.parametrize(
    ("parts", "listed"),
    [
        (
            {"provider": Provider.ANTHROPIC},
            "client=anthropic, error_resolver=openai, "
            "adapter OpenAIGenerationAdapter=openai, "
            "adapter ModelsAdapter=openai",
        ),
        (
            {"session_factory": factory(AnthropicErrorResolver())},
            "client=openai, error_resolver=anthropic, "
            "adapter OpenAIGenerationAdapter=openai, "
            "adapter ModelsAdapter=openai",
        ),
        (
            {"adapter_types": (AnthropicGenerationAdapter, ModelsAdapter)},
            "client=openai, error_resolver=openai, "
            "adapter AnthropicGenerationAdapter=anthropic, "
            "adapter ModelsAdapter=openai",
        ),
        (
            {
                "adapter_types": (
                    OpenAIGenerationAdapter,
                    AnthropicModelsAdapter,
                )
            },
            "client=openai, error_resolver=openai, "
            "adapter OpenAIGenerationAdapter=openai, "
            "adapter AnthropicModelsAdapter=anthropic",
        ),
    ],
    ids=["client", "resolver", "first-adapter", "second-adapter"],
)
def test_a_part_of_another_provider_fails_at_definition(
    parts: dict[str, Any],
    listed: str,
) -> None:
    # Every source is named, so the odd one out is plain to see; every
    # adapter is checked, not only the first.
    with pytest.raises(TypeError, match=f"disagree: {listed}"):
        declare(**parts)


@pytest.mark.parametrize(
    "missing",
    [
        "provider",
        "session_factory",
        "api_key_env",
        "auth_headers",
        "adapter_types",
    ],
)
def test_a_missing_part_fails_at_definition(missing: str) -> None:
    with pytest.raises(TypeError, match=f"must set {missing}"):
        declare(**{missing: None})


def test_a_client_without_shared_resources_declares_no_adapter() -> None:
    client = declare(adapter_types=())(TEST_TOKEN)

    with pytest.raises(TypeError, match="not in its adapter_types"):
        client.adapter(OpenAIGenerationAdapter)


def test_adapter_classes_with_one_name_fail_at_definition() -> None:
    twin = types.new_class("ModelsAdapter", (ModelsAdapter,))

    # Keyed by name, one of the two would drop out of the check.
    with pytest.raises(TypeError, match="Two adapter classes are named"):
        declare(adapter_types=(ModelsAdapter, twin))


def test_a_resolver_without_provider_fails_at_definition() -> None:
    class Bare:
        """A resolver that names no provider."""

    session_factory: Any = factory(OpenAIErrorResolver())
    session_factory.error_resolver = Bare()

    with pytest.raises(TypeError, match="error resolver .* no provider"):
        declare(session_factory=session_factory)


def test_adapter_builds_a_declared_adapter_with_its_arguments() -> None:
    client = declare()(TEST_TOKEN)

    adapter = client.adapter(ModelsAdapter, page_size=5)

    assert type(adapter) is ModelsAdapter
    assert adapter.page_size == 5


def test_adapter_refuses_a_class_the_client_does_not_declare() -> None:
    # Only declared classes had their provider checked.
    client = declare()(TEST_TOKEN)

    with pytest.raises(
        TypeError,
        match="ExampleClient builds AnthropicModelsAdapter, which is not",
    ):
        client.adapter(AnthropicModelsAdapter)


def test_a_subclass_of_a_declared_adapter_is_not_declared() -> None:
    # It may name another provider, so it must be listed itself.
    client = declare(adapter_types=(ModelsAdapter,))(TEST_TOKEN)

    with pytest.raises(TypeError, match="not in its adapter_types"):
        client.adapter(AnthropicModelsAdapter)


def test_check_same_provider_accepts_one_provider() -> None:
    check_same_provider(a=Provider.OPENAI, b=Provider.OPENAI)


def test_provider_of_names_the_source() -> None:
    assert provider_of(OpenAIGenerationAdapter, "adapter") is Provider.OPENAI

    with pytest.raises(TypeError, match="The adapter 1 declares no provider"):
        provider_of(1, "adapter")


def test_providers_of_keys_each_class_by_kind_and_name() -> None:
    assert providers_of(
        (ModelsAdapter, AnthropicModelsAdapter), "adapter"
    ) == {
        "adapter ModelsAdapter": Provider.OPENAI,
        "adapter AnthropicModelsAdapter": Provider.ANTHROPIC,
    }


def test_the_environment_variable_of_the_client_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXAMPLE_API_KEY", "from-env")
    client = declare()()

    assert isinstance(client.session, AiohttpSession)
    assert client.session.auth.encode() == "Bearer from-env"


def session_with(api: LocalAPI, resolver: BaseErrorResolver) -> AiohttpSession:
    return AiohttpSession(api, BearerAuth(TEST_TOKEN), resolver)


def test_warnings_name_the_client_and_what_it_takes(api: LocalAPI) -> None:
    session = AiohttpSession(
        api, APIKeyAuth(TEST_TOKEN, header_name="X-Key"), BaseErrorResolver()
    )

    with pytest.warns(SessionMismatchWarning) as caught:
        OpenAIClient(session=session)

    messages = [str(warning.message) for warning in caught]
    assert messages == [
        "OpenAIClient is using BaseErrorResolver. Consider using "
        "OpenAIErrorResolver for provider-specific error handling.",
        "OpenAIClient sends the key in X-Key, but openai takes it in "
        "Authorization.",
    ]
    assert all(warning.filename == __file__ for warning in caught)
    assert all(issubclass(warning.category, UserWarning) for warning in caught)


def test_another_providers_resolver_is_warned_about(api: LocalAPI) -> None:
    with pytest.warns(
        SessionMismatchWarning,
        match="is using AnthropicErrorResolver. Consider using "
        "OpenAIErrorResolver",
    ):
        OpenAIClient(session=session_with(api, AnthropicErrorResolver()))


def test_a_subclass_of_the_providers_resolver_is_accepted(
    api: LocalAPI,
) -> None:
    class QuotaAware(OpenAIErrorResolver):
        """A customized resolver, as README suggests."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        session = session_with(api, QuotaAware())
        client = OpenAIClient(session=session)

    assert client.session is session


def test_the_warning_is_turned_off_by_its_category(api: LocalAPI) -> None:
    # The standard filters, not a setting of the library.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=SessionMismatchWarning)
        OpenAIClient(session=session_with(api, BaseErrorResolver()))


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


class ContextSession(FakeTransport):
    """Records what its ``__aexit__`` was given."""

    exited: tuple[Any, ...] | None = None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exited = (exc_type, exc_value, traceback)
        await self.close()


async def test_leaving_the_client_leaves_the_session() -> None:
    session = ContextSession()
    client = BaseClient(session)
    error = ValueError("in the block")

    with pytest.raises(ValueError, match="in the block"):
        async with client:
            raise error

    assert session.exited is not None
    assert session.exited[1] is error
