"""Block N: resources, use cases and shared types on the fake transport."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_api.adapters.base import (
    BaseAdapter,
    BaseCompositeAdapter,
    BaseStreamAdapter,
)
from ai_api.adapters.protocols import ICompositeAdapter
from ai_api.enums import BodyFormat, Provider, ResponseFormat
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.resources.base import BaseResource, BaseSessionAccess
from ai_api.resources.generation import GenerationResource
from ai_api.resources.protocols import (
    ICompositeUseCase,
    IGenerationResource,
    IResource,
)
from ai_api.resources.usecases import (
    AdapterCompositeUseCase,
    AdapterStreamUseCase,
    AdapterUseCase,
    BaseCompositeUseCase,
    BaseStreamUseCase,
    BaseUseCase,
)
from ai_api.response.buffered import BufferedResponse
from ai_api.response.stream import Stream, map_stream
from ai_api.types.base import BaseAPIType
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import (
    CompletedEvent,
    FailedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)
from ai_api.types.responses.generation import GenerationResponse, TokenUsage
from tests.conftest import FakeStreamResponse, FakeTransport


class RawResponse(BaseAPIType):
    id: str
    output_text: str


class RawEvent(BaseAPIType):
    type: str
    delta: str = ""


class Create(BaseAPIMethod[RawResponse]):
    __request_path__ = "/responses"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    model: str
    input: str


class CreateStream(BaseStreamMethod[RawEvent]):
    __request_path__ = "/responses"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __response_format__ = ResponseFormat.SSE

    model: str
    input: str
    stream: bool = True


def to_event(raw: RawEvent) -> GenerationEvent[RawEvent]:
    match raw.type:
        case "response.output_text.delta":
            return TextDeltaEvent[RawEvent](text=raw.delta, raw=raw)
        case "response.completed":
            return CompletedEvent[RawEvent](raw=raw)
        case _:
            return OtherEvent[RawEvent](raw=raw)


class Generate(
    BaseCompositeUseCase[
        GenerationRequest,
        GenerationResponse[RawResponse],
        GenerationEvent[RawEvent],
        FakeTransport,
    ]
):
    async def __call__(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse[RawResponse]:
        raw = await self.session.send_request(
            Create(model=request.model, input=request.input)
        )
        return GenerationResponse[RawResponse](
            id=raw.id,
            provider=Provider.OPENAI,
            model=request.model,
            text=raw.output_text,
            raw=raw,
        )

    def stream(
        self,
        request: GenerationRequest,
    ) -> AbstractAsyncContextManager[Stream[GenerationEvent[RawEvent]]]:
        method = CreateStream(model=request.model, input=request.input)
        return map_stream(self.session.stream(method), to_event)


class Responses(BaseResource[FakeTransport]):
    def __init__(self, session: FakeTransport) -> None:
        super().__init__(session)
        self.generate = Generate(session)


REQUEST = GenerationRequest(model="gpt", input="Hi")

SSE: Mapping[str, str] = {"Content-Type": "text/event-stream"}


def sse(*events: str) -> list[bytes]:
    return [f"data: {event}\n\n".encode() for event in events]


async def test_use_case_converts_the_response() -> None:
    body = b'{"id": "r-1", "output_text": "Hello"}'
    transport = FakeTransport([BufferedResponse(200, {}, body)])

    response = await Responses(transport).generate(REQUEST)

    assert response.text == "Hello"
    assert response.id == "r-1"
    assert response.provider is Provider.OPENAI
    assert response.raw == RawResponse(id="r-1", output_text="Hello")
    assert transport.execute_calls[0].get_body_map() == {
        "model": "gpt",
        "input": "Hi",
    }


async def test_use_case_streams_shared_events() -> None:
    chunks = sse(
        '{"type": "response.output_text.delta", "delta": "He"}',
        '{"type": "response.output_text.delta", "delta": "llo"}',
        '{"type": "response.created"}',
        '{"type": "response.completed"}',
    )
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(chunks, headers=SSE)]
    )

    async with Generate(transport).stream(REQUEST) as events:
        received = [event async for event in events]

    assert [event.type for event in received] == [
        "text_delta",
        "text_delta",
        "other",
        "completed",
    ]
    assert (
        "".join(
            event.text
            for event in received
            if isinstance(event, TextDeltaEvent)
        )
        == "Hello"
    )
    assert transport.closed_responses == 1


async def test_break_closes_the_provider_stream() -> None:
    chunks = sse(
        '{"type": "response.output_text.delta", "delta": "a"}',
        '{"type": "response.output_text.delta", "delta": "b"}',
    )
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(chunks, headers=SSE)]
    )

    async with Generate(transport).stream(REQUEST) as events:
        async for _ in events:
            break

    assert transport.closed_responses == 1

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await anext(events)


def test_session_keeps_its_type() -> None:
    transport = FakeTransport()
    responses = Responses(transport)

    assert responses.session is transport
    assert responses.generate.session is transport


def test_classes_implement_the_protocols() -> None:
    transport = FakeTransport()
    resource: IResource = Responses(transport)
    use_case: ICompositeUseCase[
        GenerationRequest,
        GenerationResponse[RawResponse],
        GenerationEvent[RawEvent],
    ] = Generate(transport)

    assert resource.session is use_case.session


@pytest.mark.parametrize(
    "base",
    [BaseUseCase, BaseStreamUseCase, BaseCompositeUseCase],
    ids=["buffered", "stream", "composite"],
)
def test_use_cases_are_abstract(base: type[Any]) -> None:
    with pytest.raises(TypeError, match="abstract"):
        base(FakeTransport())


def test_session_access_without_a_subclass() -> None:
    transport = FakeTransport()

    assert BaseSessionAccess(transport).session is transport


class ResponsesAdapter(
    BaseCompositeAdapter[
        GenerationRequest,
        RawResponse,
        GenerationResponse[RawResponse],
        RawEvent,
        GenerationEvent[RawEvent],
    ]
):
    def to_method(self, request: GenerationRequest) -> Create:
        return Create(model=request.model, input=request.input)

    def to_response(
        self,
        raw: RawResponse,
        request: GenerationRequest,
    ) -> GenerationResponse[RawResponse]:
        return GenerationResponse[RawResponse](
            id=raw.id,
            provider=Provider.OPENAI,
            model=request.model,
            text=raw.output_text,
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> CreateStream:
        return CreateStream(model=request.model, input=request.input)

    async def to_events(
        self,
        items: AsyncIterator[RawEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[RawEvent]]:
        # Stateful: deltas are passed on, the rest is dropped, and the end
        # of the stream also reports how many deltas there were.
        deltas = 0

        async for raw in items:
            if raw.type == "response.output_text.delta":
                deltas += 1
                yield TextDeltaEvent[RawEvent](text=raw.delta, raw=raw)

        yield CompletedEvent[RawEvent](
            raw=RawEvent(type=f"{request.model}:{deltas}"),
        )


ADAPTER = ResponsesAdapter()

RESPONSE_BODY = b'{"id": "r-1", "output_text": "Hello"}'

STREAM_CHUNKS = sse(
    '{"type": "response.created"}',
    '{"type": "response.output_text.delta", "delta": "He"}',
    '{"type": "response.output_text.delta", "delta": "llo"}',
)


@pytest.mark.parametrize(
    "use_case",
    [AdapterUseCase, AdapterCompositeUseCase],
    ids=["buffered", "composite"],
)
async def test_adapter_use_case_responds(use_case: type[Any]) -> None:
    transport = FakeTransport([BufferedResponse(200, {}, RESPONSE_BODY)])

    response = await use_case(transport, ADAPTER)(REQUEST)

    assert response == GenerationResponse[RawResponse](
        id="r-1",
        provider=Provider.OPENAI,
        model="gpt",
        text="Hello",
        raw=RawResponse(id="r-1", output_text="Hello"),
    )
    assert transport.execute_calls[0].get_body_map() == {
        "model": "gpt",
        "input": "Hi",
    }


@pytest.mark.parametrize(
    "use_case",
    [AdapterStreamUseCase, AdapterCompositeUseCase],
    ids=["stream", "composite"],
)
async def test_adapter_use_case_streams(use_case: type[Any]) -> None:
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(STREAM_CHUNKS, headers=SSE)]
    )

    async with use_case(transport, ADAPTER).stream(REQUEST) as events:
        received = [event async for event in events]

    assert [event.type for event in received] == [
        "text_delta",
        "text_delta",
        "completed",
    ]
    assert received[-1].raw.type == "gpt:2"
    assert transport.closed_responses == 1


async def test_adapter_stream_closes_both_streams_on_break() -> None:
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(STREAM_CHUNKS, headers=SSE)]
    )
    use_case = AdapterStreamUseCase(transport, ADAPTER)

    async with use_case.stream(REQUEST) as events:
        async for _ in events:
            break

    assert transport.closed_responses == 1

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await anext(events)


def test_adapter_use_cases_keep_session_and_adapter() -> None:
    transport = FakeTransport()
    adapter: ICompositeAdapter[
        GenerationRequest,
        RawResponse,
        GenerationResponse[RawResponse],
        RawEvent,
        GenerationEvent[RawEvent],
    ] = ADAPTER
    use_case: ICompositeUseCase[
        GenerationRequest,
        GenerationResponse[RawResponse],
        GenerationEvent[RawEvent],
    ] = AdapterCompositeUseCase(transport, adapter)

    assert use_case.session is transport
    assert AdapterUseCase(transport, adapter).adapter is adapter


async def test_generation_resource_creates() -> None:
    transport = FakeTransport([BufferedResponse(200, {}, RESPONSE_BODY)])

    response = await GenerationResource(transport, ADAPTER).create(REQUEST)

    assert response.text == "Hello"
    assert response.raw == RawResponse(id="r-1", output_text="Hello")
    assert transport.execute_calls[0].get_body_map() == {
        "model": "gpt",
        "input": "Hi",
    }


async def test_generation_resource_streams_and_closes_on_break() -> None:
    transport = FakeTransport(
        stream_responses=[
            FakeStreamResponse(STREAM_CHUNKS, headers=SSE),
            FakeStreamResponse(STREAM_CHUNKS, headers=SSE),
        ]
    )
    generation = GenerationResource(transport, ADAPTER)

    async with generation.stream(REQUEST) as events:
        received = [event async for event in events]

    assert [event.type for event in received] == [
        "text_delta",
        "text_delta",
        "completed",
    ]

    async with generation.stream(REQUEST) as events:
        async for _ in events:
            break

    assert transport.closed_responses == 2

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await anext(events)


def test_generation_resource_implements_the_protocol() -> None:
    transport = FakeTransport()
    generation = GenerationResource(transport, ADAPTER)
    resource: IGenerationResource[RawResponse, RawEvent] = generation

    assert resource.session is transport
    assert generation.session is transport


@pytest.mark.parametrize(
    "base",
    [BaseAdapter, BaseStreamAdapter, BaseCompositeAdapter],
    ids=["buffered", "stream", "composite"],
)
def test_adapters_are_abstract(base: type[Any]) -> None:
    with pytest.raises(TypeError, match="abstract"):
        base()


def test_request_keeps_instructions() -> None:
    assert REQUEST.instructions is None
    request = GenerationRequest(model="m", input="i", instructions="Be brief")

    assert request.instructions == "Be brief"


def test_request_rejects_unknown_fields_and_bad_limits() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GenerationRequest.model_validate(
            {"model": "m", "input": "i", "temprature": 1}
        )

    with pytest.raises(ValidationError, match="greater_than"):
        GenerationRequest(model="m", input="i", max_output_tokens=0)


def test_response_without_a_parameter_keeps_raw() -> None:
    response: GenerationResponse[Any] = GenerationResponse.model_validate(
        {"provider": "openai", "model": "m", "raw": {"a": 1}}
    )

    assert response.raw == {"a": 1}


def test_parametrized_response_validates_raw() -> None:
    response = GenerationResponse[RawResponse].model_validate(
        {
            "provider": "openai",
            "model": "m",
            "raw": {"id": "r", "output_text": "t"},
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    )

    assert response.raw == RawResponse(id="r", output_text="t")
    assert response.usage == TokenUsage(input_tokens=1, output_tokens=2)

    with pytest.raises(ValidationError):
        GenerationResponse[RawResponse].model_validate(
            {"provider": "openai", "model": "m", "raw": {"a": 1}}
        )


@pytest.mark.parametrize(
    ("data", "event_type"),
    [
        ({"type": "text_delta", "text": "x"}, TextDeltaEvent[RawEvent]),
        ({"type": "completed"}, CompletedEvent[RawEvent]),
        ({"type": "failed", "message": "boom"}, FailedEvent[RawEvent]),
        ({"type": "other"}, OtherEvent[RawEvent]),
    ],
    ids=["text-delta", "completed", "failed", "other"],
)
def test_response_events_are_told_apart_by_type(
    data: dict[str, Any],
    event_type: type[Any],
) -> None:
    adapter: TypeAdapter[GenerationEvent[RawEvent]] = TypeAdapter(
        GenerationEvent[RawEvent]
    )

    event = adapter.validate_python({**data, "raw": {"type": "x"}})

    assert type(event) is event_type
    assert event.raw == RawEvent(type="x")
