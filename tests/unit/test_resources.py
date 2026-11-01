"""Block N: resources, use cases and shared types on the fake transport."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from pydantic import (
    PydanticSchemaGenerationError,
    TypeAdapter,
    ValidationError,
)

from ai_api.adapters.base import (
    BaseAdapter,
    BaseCompositeAdapter,
    BaseStreamAdapter,
)
from ai_api.adapters.protocols import ICompositeAdapter
from ai_api.enums import (
    BodyFormat,
    GenerationEventType,
    IncompleteReason,
    Provider,
    ResponseFormat,
    Role,
)
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.transport import APIStreamInterruptedError
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
from ai_api.types.matchers import EventTypes
from ai_api.types.requests.generation import GenerationRequest, Turn
from ai_api.types.responses.events import (
    GENERATION_TERMINAL,
    CompletedEvent,
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
    input: str | list[Turn]


class CreateStream(BaseStreamMethod[RawEvent]):
    __request_path__ = "/responses"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON
    __response_format__ = ResponseFormat.SSE

    model: str
    input: str | list[Turn]
    stream: bool = True


def to_event(raw: RawEvent) -> GenerationEvent[RawEvent]:
    match raw.type:
        case "response.output_text.delta":
            return TextDeltaEvent[RawEvent](text=raw.delta, raw=raw)
        case "response.completed":
            return CompletedEvent[RawEvent](
                provider=Provider.OPENAI, model="m", raw=raw
            )
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
    provider = Provider.OPENAI

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
            provider=self.provider,
            model=request.model,
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


def test_input_is_text_or_a_non_empty_list_of_turns() -> None:
    request = GenerationRequest.model_validate(
        {
            "model": "m",
            "input": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "More"},
            ],
        }
    )

    assert request.input == [
        Turn.user("Hi"),
        Turn.assistant("Hello"),
        Turn.user("More"),
    ]
    # Empty would fail at the provider after sending; refused here.
    for fields in (
        {"model": "m", "input": ""},
        {"model": "m", "input": []},
        {"model": "", "input": "Hi"},
    ):
        with pytest.raises(ValidationError, match="too_short"):
            GenerationRequest.model_validate(fields)


def test_turn_is_frozen_and_strict() -> None:
    turn = Turn.user("Hi")

    assert turn.role is Role.USER
    assert Turn.assistant("Hey").role is Role.ASSISTANT
    with pytest.raises(ValidationError, match="frozen"):
        turn.content = "Bye"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Turn.model_validate({"role": "user", "content": "Hi", "name": "x"})
    with pytest.raises(ValidationError, match="enum"):
        Turn.model_validate({"role": "system", "content": "Hi"})


def test_response_becomes_an_assistant_turn() -> None:
    response: GenerationResponse[Any] = GenerationResponse(
        id="r", provider=Provider.OPENAI, model="m", text="Hello", raw={}
    )

    assert response.to_turn() == Turn.assistant("Hello")


def test_response_without_text_is_no_turn() -> None:
    response: GenerationResponse[Any] = GenerationResponse(
        id="r",
        provider=Provider.ANTHROPIC,
        model="m",
        incomplete_reason=IncompleteReason.REFUSAL,
        raw={},
    )

    with pytest.raises(ValueError, match="response r has no text"):
        response.to_turn()


def test_a_response_without_an_id_is_not_named_none() -> None:
    response: GenerationResponse[Any] = GenerationResponse(
        provider=Provider.GENERIC, model="m", raw={}
    )

    with pytest.raises(ValueError, match="^the response has no text"):
        response.to_turn()


class Opaque:
    """A class Pydantic has no schema for."""


def test_only_shared_results_take_any_type() -> None:
    # A payload model with such a field is a mistake, caught at once;
    # the raw of a shared result may be any class a method returns.
    with pytest.raises(PydanticSchemaGenerationError):

        class Payload(BaseAPIType):
            value: Opaque

        _ = Payload

    response = GenerationResponse[Opaque](
        provider=Provider.GENERIC, model="m", raw=Opaque()
    )
    event = CompletedEvent[Opaque](
        provider=Provider.GENERIC, model="m", raw=Opaque()
    )

    assert isinstance(response.raw, Opaque)
    assert isinstance(event.raw, Opaque)


def test_event_types_are_enum_members() -> None:
    events: TypeAdapter[GenerationEvent[Any]] = TypeAdapter(
        GenerationEvent[Any]
    )

    event = events.validate_python(
        {"type": "completed", "provider": "generic", "model": "m", "raw": {}}
    )

    assert isinstance(event, CompletedEvent)
    assert event.type is GenerationEventType.COMPLETED
    assert GENERATION_TERMINAL.matches(event)
    assert GENERATION_TERMINAL.matches({"type": "completed"})
    assert TextDeltaEvent[Any](text="x", raw={}).type is (
        GenerationEventType.TEXT_DELTA
    )
    assert OtherEvent[Any](raw={}).type is GenerationEventType.OTHER


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
        (
            {"type": "completed", "provider": "generic", "model": "m"},
            CompletedEvent[RawEvent],
        ),
        ({"type": "other"}, OtherEvent[RawEvent]),
    ],
    ids=["text-delta", "completed", "other"],
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


class ScriptedAdapter(
    BaseStreamAdapter[GenerationRequest, RawEvent, GenerationEvent[RawEvent]]
):
    """Ignores the provider's items and yields the given event types."""

    provider = Provider.GENERIC

    def __init__(self, *types: str) -> None:
        self.types = types
        self.closed = False

    def to_stream_method(self, request: GenerationRequest) -> CreateStream:
        return CreateStream(model=request.model, input=request.input)

    async def to_events(
        self,
        items: AsyncIterator[RawEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[RawEvent]]:
        try:
            async for _ in items:
                pass
            for kind in self.types:
                raw = RawEvent(type=kind)
                if kind == "completed":
                    yield CompletedEvent[RawEvent](
                        provider=self.provider, model="m", raw=raw
                    )
                else:
                    yield OtherEvent[RawEvent](raw=raw)
        finally:
            self.closed = True


def scripted_use_case(*types: str) -> tuple[Any, ScriptedAdapter]:
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(STREAM_CHUNKS, headers=SSE)]
    )
    adapter = ScriptedAdapter(*types)
    use_case = AdapterStreamUseCase(
        transport, adapter, terminal=GENERATION_TERMINAL
    )
    return use_case, adapter


async def test_use_case_accepts_one_final_event_last() -> None:
    use_case, adapter = scripted_use_case("other", "completed")

    async with use_case.stream(REQUEST) as events:
        received = [event.type async for event in events]

    assert received == ["other", "completed"]
    assert adapter.closed


async def test_use_case_rejects_an_event_after_the_final_one() -> None:
    use_case, adapter = scripted_use_case("completed", "other")
    received: list[str] = []

    async with use_case.stream(REQUEST) as events:
        with pytest.raises(APIResponseError, match="after its final"):
            async for event in events:
                received.append(event.type)  # noqa: PERF401 - fails mid-loop

    assert received == ["completed"]
    assert adapter.closed


async def test_use_case_rejects_a_second_final_event() -> None:
    use_case, _ = scripted_use_case("completed", "completed")

    async with use_case.stream(REQUEST) as events:
        with pytest.raises(APIResponseError, match="after its final"):
            async for _ in events:
                pass


async def test_use_case_rejects_a_stream_without_its_final_event() -> None:
    use_case, adapter = scripted_use_case("other", "other")

    async with use_case.stream(REQUEST) as events:
        with pytest.raises(APIStreamInterruptedError) as info:
            async for _ in events:
                pass

    assert info.value.items_received == 2
    assert info.value.__cause__ is None
    assert adapter.closed


class CreateTerminatedStream(CreateStream):
    __stream_terminal__ = EventTypes("response.created")


class TerminatedAdapter(ScriptedAdapter):
    """Its method declares a terminal, so the parser proves the end."""

    def to_stream_method(
        self,
        request: GenerationRequest,
    ) -> CreateTerminatedStream:
        return CreateTerminatedStream(model=request.model, input=request.input)


async def test_complete_source_without_the_final_event_is_a_bug() -> None:
    # The parser saw the provider's terminal, so the stream was not cut:
    # the adapter lost the final event.
    transport = FakeTransport(
        stream_responses=[
            FakeStreamResponse(
                sse('{"type": "response.created"}'), headers=SSE
            )
        ]
    )
    adapter = TerminatedAdapter("other")
    use_case = AdapterStreamUseCase(
        transport, adapter, terminal=GENERATION_TERMINAL
    )

    async with use_case.stream(REQUEST) as events:
        with pytest.raises(
            RuntimeError,
            match="TerminatedAdapter ended a complete stream without",
        ):
            async for _ in events:
                pass

    assert adapter.closed


async def test_use_case_closes_the_adapter_on_break() -> None:
    use_case, adapter = scripted_use_case("other", "other", "completed")

    async with use_case.stream(REQUEST) as events:
        async for _ in events:
            break

    assert adapter.closed


async def test_use_case_without_a_terminal_checks_nothing() -> None:
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(STREAM_CHUNKS, headers=SSE)]
    )
    use_case = AdapterStreamUseCase(transport, ScriptedAdapter("other"))

    async with use_case.stream(REQUEST) as events:
        received = [event.type async for event in events]

    assert use_case.terminal is None
    assert received == ["other"]


class PlainEvents:
    """Converted events from an iterator that is not a generator."""

    def __init__(self, events: list[GenerationEvent[RawEvent]]) -> None:
        self.events = iter(events)

    def __aiter__(self) -> PlainEvents:
        return self

    async def __anext__(self) -> GenerationEvent[RawEvent]:
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration from None


class PlainAdapter(ScriptedAdapter):
    def to_events(  # type: ignore[override]
        self,
        items: AsyncIterator[RawEvent],
        request: GenerationRequest,
    ) -> AsyncIterator[GenerationEvent[RawEvent]]:
        return PlainEvents(
            [
                CompletedEvent[RawEvent](
                    provider=Provider.GENERIC,
                    model="m",
                    raw=RawEvent(type="x"),
                )
            ]
        )


async def test_use_case_checks_events_of_any_iterator() -> None:
    transport = FakeTransport(
        stream_responses=[FakeStreamResponse(STREAM_CHUNKS, headers=SSE)]
    )
    use_case = AdapterStreamUseCase(
        transport, PlainAdapter(), terminal=GENERATION_TERMINAL
    )

    async with use_case.stream(REQUEST) as events:
        received = [event.type async for event in events]

    assert received == ["completed"]
