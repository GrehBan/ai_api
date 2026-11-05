"""``BaseClient`` and ``BaseProviderClient``.

``BaseClient`` delegates to its session and owns it; a provider client
names one provider in every part, checked early.
"""

from __future__ import annotations

import types
import warnings
from types import TracebackType
from typing import Any
from unittest import mock

import pytest

from aiollm import SessionMismatchWarning
from aiollm.adapters.base import BaseAdapter, BaseCompositeAdapter
from aiollm.client.base import BaseClient, BaseProviderClient
from aiollm.client.binding import ResourceBinding, bind_resource
from aiollm.client.validators import (
    check_same_provider,
    provider_of,
)
from aiollm.enums import Provider
from aiollm.http.auth import APIKeyAuth, BearerAuth
from aiollm.http.exceptions.resolver import BaseErrorResolver
from aiollm.http.exceptions.status import ServiceUnavailableError
from aiollm.http.headers import AUTHORIZATION
from aiollm.methods.base import BaseAPIMethod, BaseStreamMethod
from aiollm.providers.anthropic.adapters import AnthropicGenerationAdapter
from aiollm.providers.anthropic.resolver import AnthropicErrorResolver
from aiollm.providers.openai.adapters import OpenAIGenerationAdapter
from aiollm.providers.openai.api import OPENAI_API
from aiollm.providers.openai.client import OpenAIClient
from aiollm.providers.openai.resolver import OpenAIErrorResolver
from aiollm.resources.generation import GenerationResource
from aiollm.response.buffered import BufferedResponse
from aiollm.session.aiohttp import AiohttpSession
from aiollm.session.factory import SessionFactory
from aiollm.session.retry import RetryPolicy
from aiollm.types.requests import GenerationRequest
from aiollm.types.responses import GenerationResponse
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


class ModelsResource:
    """A second shared resource, as a ``models`` resource would be."""

    def __init__(self, session: Any, adapter: ModelsAdapter) -> None:
        self.session = session
        self.adapter = adapter


def bind_models(adapter_type: type[ModelsAdapter] = ModelsAdapter) -> Any:
    """Bind ``ModelsResource`` to an adapter class."""
    return bind_resource(ModelsResource, adapter_type)


def bind_generation(adapter_type: Any = OpenAIGenerationAdapter) -> Any:
    """Bind the generation resource to an adapter class."""
    return bind_resource(GenerationResource, adapter_type)


def declare(**parts: Any) -> type[BaseProviderClient]:
    """Declare a provider client with the given parts; ``None`` drops one."""
    namespace: dict[str, Any] = {
        "provider": Provider.OPENAI,
        "session_factory": OPENAI_FACTORY,
        "api_key_env": "EXAMPLE_API_KEY",
        "auth_headers": frozenset({AUTHORIZATION}),
        "generation": bind_generation(),
        "models": bind_models(),
    }
    namespace.update(parts)
    namespace = {k: v for k, v in namespace.items() if v is not None}
    return types.new_class(
        "ExampleClient",
        (BaseProviderClient,),
        exec_body=lambda body: body.update(namespace),
    )


def test_parts_of_one_provider_are_accepted() -> None:
    client: Any = declare()(TEST_TOKEN)

    assert client.provider is Provider.OPENAI
    assert isinstance(client.session, AiohttpSession)
    assert isinstance(client.generation, GenerationResource)
    assert client.generation.session is client.session
    assert isinstance(client.models, ModelsResource)
    assert client.models.session is client.session
    assert type(client.models.adapter) is ModelsAdapter


@pytest.mark.parametrize(
    ("parts", "listed"),
    [
        (
            {"provider": Provider.ANTHROPIC},
            "client=anthropic, error_resolver=openai, "
            "generation adapter OpenAIGenerationAdapter=openai, "
            "models adapter ModelsAdapter=openai",
        ),
        (
            {"session_factory": factory(AnthropicErrorResolver())},
            "client=openai, error_resolver=anthropic, "
            "generation adapter OpenAIGenerationAdapter=openai, "
            "models adapter ModelsAdapter=openai",
        ),
        (
            {"generation": bind_generation(AnthropicGenerationAdapter)},
            "client=openai, error_resolver=openai, "
            "generation adapter AnthropicGenerationAdapter=anthropic, "
            "models adapter ModelsAdapter=openai",
        ),
        (
            {"models": bind_models(AnthropicModelsAdapter)},
            "client=openai, error_resolver=openai, "
            "generation adapter OpenAIGenerationAdapter=openai, "
            "models adapter AnthropicModelsAdapter=anthropic",
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
    ],
)
def test_a_missing_part_fails_at_definition(missing: str) -> None:
    with pytest.raises(TypeError, match=f"must set {missing}"):
        declare(**{missing: None})


def test_a_client_without_shared_resources_builds_none() -> None:
    client = declare(generation=None, models=None)(TEST_TOKEN)

    assert not hasattr(client, "generation")
    assert not hasattr(client, "models")


def test_adapters_of_one_class_on_two_resources_are_both_checked() -> None:
    # Keyed by attribute name, neither drops out of the check.
    with pytest.raises(
        TypeError,
        match="models adapter ModelsAdapter=openai, "
        "spare adapter AnthropicModelsAdapter=anthropic",
    ):
        declare(spare=bind_models(AnthropicModelsAdapter))


def test_an_adapter_without_provider_fails_at_definition() -> None:
    class Bare:
        """An adapter that names no provider."""

    bare: Any = Bare

    with pytest.raises(TypeError, match="The models adapter .* no provider"):
        declare(models=bind_resource(ModelsResource, bare))


def test_a_resolver_without_provider_fails_at_definition() -> None:
    class Bare:
        """A resolver that names no provider."""

    session_factory: Any = factory(OpenAIErrorResolver())
    session_factory.error_resolver = Bare()

    with pytest.raises(TypeError, match="error resolver .* no provider"):
        declare(session_factory=session_factory)


def test_an_adapter_factory_builds_the_adapter_from_the_client() -> None:
    def adapter(client: Any) -> ModelsAdapter:
        return ModelsAdapter(page_size=client.page_size)

    def init(self: Any, token: str) -> None:
        self.page_size = 5
        BaseProviderClient.__init__(self, token)

    models = bind_resource(
        ModelsResource, ModelsAdapter, adapter_factory=adapter
    )
    client: Any = declare(models=models, __init__=init)(TEST_TOKEN)

    assert client.models.adapter.page_size == 5


def test_adapters_are_built_before_the_session() -> None:
    # A bad adapter option fails before a session and its pool exist.
    def adapter(client: Any) -> ModelsAdapter:
        raise ValueError("bad option")

    models = bind_resource(
        ModelsResource, ModelsAdapter, adapter_factory=adapter
    )
    client_type = declare(models=models)

    with (
        mock.patch.object(client_type, "session_factory") as session_factory,
        pytest.raises(ValueError, match="bad option"),
    ):
        client_type(TEST_TOKEN)

    session_factory.assert_not_called()


@pytest.mark.parametrize(
    "built",
    [OpenAIGenerationAdapter, AnthropicModelsAdapter],
    ids=["another-class", "subclass-of-another-provider"],
)
def test_a_factory_of_another_adapter_is_refused(built: type[Any]) -> None:
    # Only the declared class had its provider checked; a subclass of it
    # may name another provider.
    models = bind_resource(
        ModelsResource,
        ModelsAdapter,
        adapter_factory=lambda client: built(),
    )
    client_type = declare(models=models)

    with (
        mock.patch.object(client_type, "session_factory") as session_factory,
        pytest.raises(
            TypeError,
            match=f"factory returned {built.__name__}, not ModelsAdapter "
            "of its provider",
        ),
    ):
        client_type(TEST_TOKEN)

    session_factory.assert_not_called()


def test_a_subclass_replaces_or_drops_an_inherited_resource() -> None:
    base: Any = declare()
    replaced_models = bind_resource(
        ModelsResource,
        ModelsAdapter,
        adapter_factory=lambda client: ModelsAdapter(page_size=1),
    )
    replaced_type = types.new_class(
        "Replaced",
        (base,),
        exec_body=lambda body: body.update(models=replaced_models),
    )
    dropped_type = types.new_class(
        "Dropped",
        (base,),
        exec_body=lambda body: body.update(models=None),
    )

    replaced = replaced_type(TEST_TOKEN)
    dropped = dropped_type(TEST_TOKEN)

    assert replaced.models.adapter.page_size == 1
    assert isinstance(replaced.generation, GenerationResource)
    assert dropped.models is None
    assert isinstance(dropped.generation, GenerationResource)


def test_on_the_class_a_resource_is_its_binding() -> None:
    binding = vars(declare())["models"]

    assert isinstance(binding, ResourceBinding)
    assert repr(binding) == "ResourceBinding(ModelsAdapter)"


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


async def test_client_forwards_the_retry_policy() -> None:
    once = RetryPolicy(max_attempts=1)
    transport = FakeTransport(
        [BufferedResponse(503, {}, b""), BufferedResponse(200, {}, b"ok")],
        [FakeStreamResponse([], status=503), FakeStreamResponse([b"a"])],
    )
    client = BaseClient(transport)
    retryable = make_method(BaseAPIMethod[bytes], __retryable__=True)
    retryable_stream = make_method(
        BaseStreamMethod[bytes], http_method="GET", __retryable__=True
    )

    with pytest.raises(ServiceUnavailableError):
        await client.send_request(retryable(), retry_policy=once)

    with pytest.raises(ServiceUnavailableError):
        async with client.stream(retryable_stream(), retry_policy=once):
            pass

    assert transport.attempts == 2


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
