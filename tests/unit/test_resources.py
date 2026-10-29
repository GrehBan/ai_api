"""Block N: resources, use cases and shared types on the fake transport."""

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_api.client.resources.base import BaseResource, BaseSessionAccess
from ai_api.client.resources.protocols import IResource
from ai_api.client.resources.usecases.base import (
    BaseCompositeUseCase,
    BaseStreamUseCase,
    BaseUseCase,
)
from ai_api.client.resources.usecases.protocols import ICompositeUseCase
from ai_api.client.response.base import BufferedResponse
from ai_api.client.response.stream import Stream, map_stream
from ai_api.enums import Provider
from ai_api.methods.base import BaseAPIMethod, BaseStreamMethod
from ai_api.methods.enums import BodyFormat, ResponseFormat
from ai_api.types.base import BaseAPIType
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import (
    CompletedEvent,
    FailedEvent,
    OtherEvent,
    ResponseEvent,
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


def to_event(raw: RawEvent) -> ResponseEvent[RawEvent]:
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
        ResponseEvent[RawEvent],
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
    ) -> AbstractAsyncContextManager[Stream[ResponseEvent[RawEvent]]]:
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
        ResponseEvent[RawEvent],
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
    adapter: TypeAdapter[ResponseEvent[RawEvent]] = TypeAdapter(
        ResponseEvent[RawEvent]
    )

    event = adapter.validate_python({**data, "raw": {"type": "x"}})

    assert type(event) is event_type
    assert event.raw == RawEvent(type="x")
