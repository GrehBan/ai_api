# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static checks of the public API.

Never run: mypy (through ``pytest --mypy``) and pyright check it. Every
``# type: ignore[...]`` marks a line that must stay a type error; mypy's
``warn_unused_ignores`` and pyright's ``reportUnnecessaryTypeIgnoreComment``
fail the check if that error disappears.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Literal, Self, TypedDict, Unpack, assert_type

from pydantic import Field

from aiollm import (
    BaseAPI,
    BaseAPIMethod,
    BaseAPIType,
    BaseClient,
    BaseStreamMethod,
    CompletedEvent,
    Effort,
    GenerationStream,
    Headers,
    IGenerationClient,
    IncompleteReason,
    Provider,
    ResponseFormat,
    RetryPolicy,
    RetryState,
    Role,
    ServerSentEvent,
    Stream,
    bind_resource,
)
from aiollm.adapters import (
    BaseAdapter,
    BaseCompositeAdapter,
    IConversion,
)
from aiollm.adapters import Stop as GenerationStop
from aiollm.adapters.validation import checked_options
from aiollm.client import IClient
from aiollm.http import IAPI, BearerAuth
from aiollm.methods import IAPIMethod, IStreamMethod
from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.anthropic.adapters import AnthropicGenerationAdapter
from aiollm.providers.anthropic.methods import (
    CreateMessage,
    StreamMessage,
)
from aiollm.providers.anthropic.methods.params import UserMessageParam
from aiollm.providers.anthropic.types import (
    Message,
    MessageStreamEvent,
    Usage,
)
from aiollm.providers.anthropic.types.options import (
    AnthropicGenerationOptions,
)
from aiollm.providers.openai import OpenAIClient
from aiollm.providers.openai.adapters import OpenAIGenerationAdapter
from aiollm.providers.openai.client import session_factory
from aiollm.providers.openai.methods import CreateResponse, StreamResponse
from aiollm.providers.openai.types import (
    Response,
    ResponseStreamEvent,
    ResponseUsage,
)
from aiollm.providers.openai.types.options import OpenAIGenerationOptions
from aiollm.resources import (
    AdapterStreamUseCase,
    AdapterUseCase,
    BaseResource,
    BaseUseCase,
    GenerationResource,
    IGenerationResource,
    ISessionAccess,
    IStreamUseCase,
    IUseCase,
)
from aiollm.response import map_stream
from aiollm.session import CallOptions, ISession, ISessionFactory
from aiollm.session.aiohttp import AiohttpSession
from aiollm.types.requests import GenerationRequest, NoProviderOptions, Turn
from aiollm.types.responses import (
    GenerationEvent,
    GenerationResponse,
    OtherEvent,
    TokenUsage,
)


class Model(BaseAPIType):
    id: str


class GetModel(BaseAPIMethod[Model]):
    __request_path__ = "/models/{model_id}"
    __request_method__ = "GET"

    model_id: str


class GetText(BaseAPIMethod[str]):
    __request_path__ = "/text"
    __request_method__ = "GET"


class GetBytes(BaseAPIMethod[bytes]):
    __request_path__ = "/bytes"
    __request_method__ = "GET"


class Delete(BaseAPIMethod[None]):
    __request_path__ = "/items/1"
    __request_method__ = "DELETE"


class Download(BaseStreamMethod[bytes]):
    __request_path__ = "/files/content"
    __request_method__ = "GET"


class Generate(BaseStreamMethod[str]):
    __request_path__ = "/generate"
    __request_method__ = "POST"


class Start(BaseAPIType):
    type: Literal["start"]


class Stop(BaseAPIType):
    type: Literal["stop"]


type Event = Annotated[Start | Stop, Field(discriminator="type")]


class Deltas(BaseStreamMethod[Model]):
    __request_path__ = "/deltas"
    __request_method__ = "POST"
    __response_format__ = ResponseFormat.NDJSON


class Events(BaseStreamMethod[Event]):
    __request_path__ = "/events"
    __request_method__ = "POST"
    __response_format__ = ResponseFormat.SSE


class RawEvents(BaseStreamMethod[ServerSentEvent]):
    __request_path__ = "/events"
    __request_method__ = "GET"


class ListPage(BaseAPIMethod[bytes]):
    __request_path__ = "/items"
    __request_method__ = "GET"
    __request_query__ = {"page_size"}

    page_size: int = Field(
        validation_alias="pageSize",
        serialization_alias="pageSize",
    )


async def send_request_infers_result(session: AiohttpSession) -> None:
    assert_type(await session.send_request(GetModel(model_id="m")), Model)
    assert_type(await session.send_request(GetText()), str)
    assert_type(await session.send_request(GetBytes()), bytes)
    # mypy rejects using the result of a call that returns None, so the
    # call and the await are split.
    deleting = session.send_request(Delete())
    assert_type(await deleting, None)


async def stream_infers_item(session: AiohttpSession) -> None:
    async with session.stream(Download()) as chunks:
        assert_type(chunks, Stream[bytes])
        assert_type(await chunks.save("out.bin"), int)

    async with session.stream(Generate()) as pieces:
        assert_type(pieces, Stream[str])


async def stream_infers_events(session: AiohttpSession) -> None:
    async with session.stream(Deltas()) as deltas:
        assert_type(deltas, Stream[Model])

    async with session.stream(Events()) as events:
        async for event in events:
            assert_type(event, Start | Stop)

    async with session.stream(RawEvents()) as raw:
        assert_type(raw, Stream[ServerSentEvent])


async def wrong_kind_of_method_is_rejected(session: AiohttpSession) -> None:
    await session.send_request(Download())  # type: ignore[arg-type]
    session.stream(GetText())  # type: ignore[arg-type]

    async with session.stream(Generate()) as pieces:
        await pieces.save("out.txt")  # type: ignore[misc]


async def call_options_are_keyword_only(session: AiohttpSession) -> None:
    await session.send_request(GetText(), headers={"X-A": "1"}, timeout=2)
    await session.send_request(GetText(), headers=Headers(), timeout=0.5)
    await session.send_request(GetText(), {"X-A": "1"})  # type: ignore[call-arg]
    await session.send_request(GetText(), timeout="2")  # type: ignore[arg-type]
    await session.send_request(GetText(), retry_policy=RetryPolicy())
    await session.send_request(GetText(), retry_policy=3)  # type: ignore[arg-type]

    async with session.stream(Download(), timeout=30) as chunks:
        assert_type(chunks, Stream[bytes])


def retry_policy_takes_typed_callables() -> None:
    def only_rate_limits(state: RetryState) -> bool:
        return state.attempt < 3

    RetryPolicy(retry_if=only_rate_limits, backoff=lambda state: 1.0)
    RetryPolicy(retry_if=lambda state: "yes")  # type: ignore[arg-type]
    RetryPolicy(backoff=lambda: 1.0)  # type: ignore[arg-type]
    RetryPolicy(5)  # type: ignore[call-arg]


def aliased_fields_are_set_by_name() -> None:
    # With validation_alias/serialization_alias, mypy and pyright both
    # take the field name (with alias=, they disagree).
    ListPage(page_size=1)
    ListPage(pageSize=1)  # type: ignore[call-arg]


def session_implements_protocol(session: AiohttpSession) -> ISession:
    return session


def base_api_implements_protocol() -> IAPI:
    # Regression: a base_url typed str | URL did not match IAPI.base_url.
    return BaseAPI("https://api.example.com", "v1")


def methods_implement_protocols() -> None:
    # Regression: pyright once rejected method classes whose dunder
    # attributes are not annotated as ClassVar.
    buffered: IAPIMethod[Model] = GetModel(model_id="m")
    streamed: IStreamMethod[bytes] = Download()
    assert_type(buffered.response_type, type[Model] | None)
    assert_type(streamed.response_type, type[bytes] | None)


async def client_infers_result(client: OpenAIClient) -> IClient:
    assert_type(await client.send_request(GetModel(model_id="m")), Model)
    request = GenerationRequest(model="m", input="x")
    assert_type(
        await client.generation.create(request), GenerationResponse[Response]
    )

    async with client.stream(Download()) as chunks:
        assert_type(chunks, Stream[bytes])

    return client


def client_keeps_the_session_type(session: AiohttpSession) -> IClient:
    client = BaseClient(session)
    assert_type(client.session, AiohttpSession)
    assert_type(client.session.loop, asyncio.AbstractEventLoop | None)
    BaseClient(object())  # type: ignore[type-var]

    return client


def session_factory_checks_session_arguments() -> None:
    assert_type(session_factory("sk", max_body_size=1), AiohttpSession)
    session_factory("sk", max_body_sizee=1)  # type: ignore[call-arg]


async def map_stream_infers_the_item(session: AiohttpSession) -> None:
    async with map_stream(session.stream(Download()), len) as sizes:
        assert_type(sizes, Stream[int])


class Echo(BaseUseCase[GenerationRequest, GenerationResponse[str]]):
    """Use case without a session type: it defaults to ``ISession``."""

    async def __call__(
        self,
        request: GenerationRequest,
        **options: Unpack[CallOptions],
    ) -> GenerationResponse[str]:
        assert_type(self.session, ISession)
        return GenerationResponse[str].model_validate(
            {"provider": "openai", "model": request.model, "raw": "x"}
        )


class NoOptions(BaseUseCase[GenerationRequest, GenerationResponse[str]]):
    """A use case must take the options of the call."""

    async def __call__(  # type: ignore[override]
        self,
        request: GenerationRequest,
    ) -> GenerationResponse[str]:
        raise NotImplementedError


def use_case_implements_protocol(
    session: AiohttpSession,
) -> IUseCase[GenerationRequest, GenerationResponse[str]]:
    return Echo(session)


def stream_use_case_implements_protocol(
    session: AiohttpSession,
) -> IStreamUseCase[GenerationRequest, GenerationEvent[ResponseStreamEvent]]:
    return AdapterStreamUseCase(session, OpenAIGenerationAdapter())


def stream_use_case_checks_its_event_type(session: AiohttpSession) -> None:
    wrong: IStreamUseCase[GenerationRequest, GenerationEvent[str]]
    wrong = AdapterStreamUseCase(session, OpenAIGenerationAdapter())  # type: ignore[arg-type]
    assert wrong


def everything_with_a_session_is_a_session_access(
    session: AiohttpSession,
) -> list[ISessionAccess]:
    return [
        BaseResource(session),
        Echo(session),
        AdapterStreamUseCase(session, OpenAIGenerationAdapter()),
        GenerationResource(session, OpenAIGenerationAdapter()),
    ]


def a_session_access_needs_a_session() -> ISessionAccess:
    return object()  # type: ignore[return-value]


def resource_keeps_the_session_type(session: AiohttpSession) -> None:
    resource = BaseResource(session)
    assert_type(resource.session, AiohttpSession)
    BaseResource(object())  # type: ignore[type-var]


class WrongRequest(BaseUseCase[Model, GenerationResponse[str]]):  # type: ignore[type-var]
    """A use case request must be a ``BaseRequest``."""

    async def __call__(
        self, request: Model, **options: Unpack[CallOptions]
    ) -> GenerationResponse[str]:
        raise NotImplementedError


class TextAdapter(
    BaseAdapter[GenerationRequest, str, GenerationResponse[str]],
):
    """Builds a method returning ``str`` and converts that ``str``."""

    provider = Provider.GENERIC

    def to_method(self, request: GenerationRequest) -> GetText:
        return GetText()

    def to_response(
        self,
        raw: str,
        request: GenerationRequest,
    ) -> GenerationResponse[str]:
        return GenerationResponse[str].model_validate(
            {"provider": "openai", "model": request.model, "raw": raw}
        )


class WrongMethodAdapter(
    BaseAdapter[GenerationRequest, str, GenerationResponse[str]],
):
    """The method's result must be the raw type of the response half."""

    provider = Provider.GENERIC

    def to_method(self, request: GenerationRequest) -> GetModel:  # type: ignore[override]
        return GetModel(model_id="m")

    def to_response(
        self,
        raw: str,
        request: GenerationRequest,
    ) -> GenerationResponse[str]:
        raise NotImplementedError


async def adapter_use_case_infers_the_response(
    session: AiohttpSession,
) -> None:
    generate = AdapterUseCase(session, TextAdapter())
    request = GenerationRequest(model="m", input="x")
    assert_type(await generate(request), GenerationResponse[str])
    assert_type(generate.session, AiohttpSession)


class TextGeneration(
    BaseCompositeAdapter[
        GenerationRequest,
        str,
        GenerationResponse[str],
        bytes,
        GenerationEvent[bytes],
    ],
):
    """Generates from ``GetText`` at once and from ``Download`` streamed."""

    provider = Provider.GENERIC

    def with_options(self, options: NoProviderOptions, /) -> Self:
        checked_options(options, NoProviderOptions, self.provider)
        return self

    def to_method(self, request: GenerationRequest) -> GetText:
        return GetText()

    def to_response(
        self,
        raw: str,
        request: GenerationRequest,
    ) -> GenerationResponse[str]:
        return GenerationResponse[str].model_validate(
            {"provider": "openai", "model": request.model, "raw": raw}
        )

    def to_stream_method(self, request: GenerationRequest) -> Download:
        return Download()

    async def to_events(
        self,
        items: AsyncIterator[bytes],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[bytes]]:
        async for item in items:
            yield OtherEvent[bytes](raw=item)


async def generation_resource_infers_the_types(
    session: AiohttpSession,
) -> None:
    generation = GenerationResource(session, TextGeneration())
    request = GenerationRequest(model="m", input="x", instructions="Be brief")
    assert_type(await generation.create(request), GenerationResponse[str])
    async with generation.stream(request) as events:
        assert_type(events, GenerationStream[bytes])
    assert_type(generation.session, AiohttpSession)
    resource: IGenerationResource[str, bytes] = generation
    assert resource.session is session
    GenerationResource(session, TextAdapter())  # type: ignore[arg-type]


async def openai_responses_infer_their_types(session: AiohttpSession) -> None:
    create = CreateResponse(model="m", input="x", instructions="Be brief")
    assert_type(await session.send_request(create), Response)
    async with session.stream(StreamResponse(model="m", input="x")) as events:
        assert_type(events, Stream[ResponseStreamEvent])
    CreateResponse(model="m", input="x", max_tokens=16)  # type: ignore[call-arg]


async def openai_generation_infers_its_types(session: AiohttpSession) -> None:
    generation = GenerationResource(session, OpenAIGenerationAdapter())
    request = GenerationRequest(model="m", input="x")
    response = await generation.create(request)
    assert_type(response, GenerationResponse[Response])
    assert_type(response.incomplete_reason, IncompleteReason | None)
    assert_type(response.raw_incomplete_reason, str | None)
    assert_type(response.refusal, str | None)
    assert_type(response.to_turn(), Turn)
    async with generation.stream(request) as events:
        assert_type(events, GenerationStream[ResponseStreamEvent])


def turns_are_typed() -> None:
    assert_type(Turn.user("x"), Turn)
    assert_type(Turn.assistant("x").role, Role)
    GenerationRequest(model="m", input=[Turn.user("x")])


async def anthropic_messages_infer_types(session: AiohttpSession) -> None:
    turn = UserMessageParam(role="user", content="x")
    create = CreateMessage(model="m", max_tokens=16, messages=[turn])
    assert_type(await session.send_request(create), Message)
    stream = StreamMessage(model="m", max_tokens=16, messages=[turn])
    async with session.stream(stream) as events:
        assert_type(events, Stream[MessageStreamEvent])
    CreateMessage(model="m", max_tokens=16, messages=[turn], temperature=1)  # type: ignore[call-arg]


async def anthropic_generation_infers_its_types(
    client: AnthropicClient,
) -> None:
    request = GenerationRequest(model="m", input="x")
    response = await client.generation.create(request)
    assert_type(response, GenerationResponse[Message])
    async with client.generation.stream(request) as events:
        assert_type(events, GenerationStream[MessageStreamEvent])
    AnthropicClient("sk", default_max_tokens="1")  # type: ignore[arg-type]


def no_usage(source: Usage, /) -> TokenUsage:
    """A plain function is a conversion of Anthropic's counts."""
    return TokenUsage(input_tokens=source.input_tokens)


def openai_usage(source: ResponseUsage, /) -> TokenUsage:
    """A conversion of OpenAI's counts, not Anthropic's."""
    return TokenUsage(input_tokens=source.input_tokens)


def conversions_are_typed(
    stop: IConversion[Response, GenerationStop],
    response: Response,
) -> None:
    assert_type(stop(response), GenerationStop)
    AnthropicGenerationAdapter(usage_conversion=no_usage)
    OpenAIGenerationAdapter(stop_conversion=stop)
    AnthropicGenerationAdapter(usage_conversion=openai_usage)  # type: ignore[arg-type]
    AnthropicGenerationAdapter(stop_conversion=stop)  # type: ignore[arg-type]


async def generation_takes_keyword_arguments(client: OpenAIClient) -> None:
    response = await client.generation.create(model="m", input="x")
    assert_type(response, GenerationResponse[Response])
    async with client.generation.stream(
        model="m", input=[Turn.user("x")], max_output_tokens=16
    ) as events:
        async for text in events.text():
            assert_type(text, str)
        completed = await events.final()
        assert_type(completed, CompletedEvent[ResponseStreamEvent])
        assert_type(completed.model, str)
        assert_type(completed.text, str | None)
    await client.generation.create(model="m", inputs="x")  # type: ignore[call-overload]
    await client.generation.create(input="x")  # type: ignore[call-overload]
    # None would mean null: an optional field is left out instead.
    await client.generation.create(model="m", input="x", instructions=None)  # type: ignore[call-overload]


async def generation_takes_the_call_options(
    client: OpenAIClient,
    request: GenerationRequest,
) -> None:
    once = RetryPolicy(max_attempts=1)
    response = await client.generation.create(
        model="m",
        input="x",
        headers={"X-A": "1"},
        timeout=5,
        retry_policy=once,
    )
    assert_type(response, GenerationResponse[Response])
    await client.generation.create(request, timeout=5.0, retry_policy=once)
    async with client.generation.stream(request, headers=Headers()) as events:
        assert_type(events, GenerationStream[ResponseStreamEvent])
    await client.generation.create(request, timeout="5")  # type: ignore[call-overload]
    await client.generation.create(request, retry_policy=1)  # type: ignore[call-overload]
    await client.generation.create(request, retries=1)  # type: ignore[call-overload]


def clients_take_session_arguments() -> None:
    OpenAIClient("sk", max_body_size=1_000, trust_env=True)
    AnthropicClient("sk", default_max_tokens=8, follow_redirects=True)
    OpenAIClient("sk", max_body_sizee=1)  # type: ignore[call-arg]
    AnthropicClient("sk", trust_env="yes")  # type: ignore[arg-type]


def any_generation_client(client: IGenerationClient) -> IGenerationClient:
    return client


def clients_are_generation_clients(
    openai: OpenAIClient,
    anthropic: AnthropicClient,
) -> None:
    typed: IGenerationClient[Response, ResponseStreamEvent] = openai
    resource: IGenerationResource[Response, ResponseStreamEvent]
    resource = typed.generation
    assert resource.session is openai.session
    any_generation_client(anthropic)
    any_generation_client(BaseClient(openai.session))  # type: ignore[arg-type]


def session_factories_are_typed() -> None:
    factory: ISessionFactory[AiohttpSession, BearerAuth, ...] = session_factory
    assert_type(factory("sk"), AiohttpSession)
    session: ISession = OpenAIClient.session_factory("sk")
    assert session


def openai_adapter(client: object) -> OpenAIGenerationAdapter:
    return OpenAIGenerationAdapter()


def clients_bind_their_resources() -> None:
    assert_type(
        OpenAIClient("sk").generation,
        GenerationResource[
            Response, ResponseStreamEvent, OpenAIGenerationOptions
        ],
    )
    assert_type(
        AnthropicClient("sk").generation,
        GenerationResource[
            Message, MessageStreamEvent, AnthropicGenerationOptions
        ],
    )
    resource = bind_resource(
        GenerationResource[
            Response, ResponseStreamEvent, OpenAIGenerationOptions
        ],
        OpenAIGenerationAdapter,
        adapter_factory=openai_adapter,
    )
    assert_type(
        resource,
        GenerationResource[
            Response, ResponseStreamEvent, OpenAIGenerationOptions
        ],
    )
    # An adapter of options the resource does not take.
    no_options = GenerationResource[Response, ResponseStreamEvent]
    bind_resource(no_options, OpenAIGenerationAdapter)  # type: ignore[arg-type]
    bind_resource(
        GenerationResource[
            Response, ResponseStreamEvent, OpenAIGenerationOptions
        ],
        OpenAIGenerationAdapter,
        adapter_factory=lambda client: 1,  # type: ignore[arg-type, return-value]
    )
    bind_resource(GenerationResource[Response, ResponseStreamEvent])  # type: ignore[call-arg]
    assert_type(OpenAIClient.provider, Provider)


async def provider_options_are_typed(request: GenerationRequest) -> None:
    generation = OpenAIClient("sk").generation
    response = await generation.create(
        request, provider_options={"store": False, "temperature": 0.2}
    )
    assert_type(response, GenerationResponse[Response])
    await generation.create(
        model="m", input="x", provider_options={"store": None}, timeout=5
    )
    async with generation.stream(
        request, provider_options={"top_p": 0.5}
    ) as events:
        assert_type(events, GenerationStream[ResponseStreamEvent])
    await generation.create(request, provider_options={"stor": False})  # type: ignore[call-overload]
    await generation.create(request, provider_options={"temperature": "hot"})  # type: ignore[call-overload]
    anthropic = AnthropicClient("sk").generation
    await anthropic.create(
        request, provider_options={"stop_sequences": ["END"]}
    )
    await anthropic.create(request, provider_options={"store": False})  # type: ignore[call-overload]
    await neutral_resource_has_no_options(generation, request)


async def effort_is_typed(client: IGenerationClient) -> None:
    await client.generation.create(model="m", input="x", effort=Effort.HIGH)
    await client.generation.create(model="m", input="x", effort="high")  # type: ignore[call-overload]
    GenerationRequest(model="m", input="x", effort=Effort.MAX)
    await client.generation.create(
        model="m", input="x", output_schema={"type": "object", "n": [1, None]}
    )
    bad = {1: "x"}
    await client.generation.create(model="m", input="x", output_schema=bad)  # type: ignore[arg-type]


async def neutral_resource_has_no_options(
    generation: IGenerationResource[Response, ResponseStreamEvent],
    request: GenerationRequest,
) -> None:
    await generation.create(request, provider_options={})  # type: ignore[call-overload]
    await generation.create(request, extra=1)  # type: ignore[call-overload]


class StoreOptions(TypedDict, total=False):
    store: bool


def no_provider_options_is_closed(other: StoreOptions) -> None:
    none: NoProviderOptions = {}
    keyed: NoProviderOptions = {"store": True}  # type: ignore[typeddict-unknown-key]
    taken: NoProviderOptions = other  # type: ignore[assignment]
    assert none == keyed == taken
