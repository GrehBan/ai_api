"""OpenAI generation adapter: shared requests to Responses and back.

The payloads are the reference examples of ``test_openai_responses``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_api.enums import IncompleteReason, Provider
from ai_api.exceptions import RequestValidationError
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import (
    APIStatusError,
    QuotaExceededError,
)
from ai_api.providers.openai import OpenAIClient
from ai_api.providers.openai.adapters import OpenAIGenerationAdapter
from ai_api.providers.openai.client import session_factory
from ai_api.providers.openai.methods import CreateResponse, StreamResponse
from ai_api.providers.openai.types import Response, ResponseStreamEvent
from ai_api.resources import GenerationResource, IGenerationResource
from ai_api.types.requests import GenerationRequest, Turn
from ai_api.types.responses import (
    CompletedEvent,
    OtherEvent,
    TextDeltaEvent,
    TokenUsage,
)
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply
from tests.providers.test_openai_responses import (
    MESSAGE,
    RESPONSE,
    STREAM,
    frame,
    state,
)

EVENTS: TypeAdapter[ResponseStreamEvent] = TypeAdapter(ResponseStreamEvent)

REQUEST = GenerationRequest(
    model="gpt-6-astra",
    input="Hi",
    instructions="Be brief",
    max_output_tokens=16,
    metadata={"user": "u-1"},
)

TEXT = MESSAGE["content"][0]["text"]

adapter = OpenAIGenerationAdapter()


def response(**changes: Any) -> Response:
    """The reference response with some fields changed."""
    return Response.model_validate(RESPONSE | changes)


async def events(payloads: Iterable[dict[str, Any]]) -> AsyncIterator[Any]:
    """Parse payloads into OpenAI events, as the stream would."""
    for payload in payloads:
        yield EVENTS.validate_python(payload)


async def convert(payloads: Iterable[dict[str, Any]]) -> list[Any]:
    """Run ``to_events`` over parsed payloads."""
    return [
        event async for event in adapter.to_events(events(payloads), REQUEST)
    ]


def test_methods_take_the_request_fields() -> None:
    fields = {
        "model": "gpt-6-astra",
        "input": "Hi",
        "instructions": "Be brief",
        "max_output_tokens": 16,
        "metadata": {"user": "u-1"},
    }

    created = adapter.to_method(REQUEST)
    streamed = adapter.to_stream_method(REQUEST)

    assert isinstance(created, CreateResponse)
    assert created.get_body_map() == fields
    assert isinstance(streamed, StreamResponse)
    assert streamed.get_body_map() == fields | {"stream": True}


def test_unset_fields_are_not_sent() -> None:
    method = adapter.to_method(GenerationRequest(model="m", input="Hi"))

    assert method.get_body_map() == {"model": "m", "input": "Hi"}


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"max_output_tokens": 15}, "max_output_tokens"),
        ({"metadata": {f"k{index}": "v" for index in range(17)}}, "metadata"),
        ({"metadata": {"k" * 65: "v"}}, "metadata"),
    ],
)
def test_limits_of_openai_are_checked_before_sending(
    changes: dict[str, Any],
    field: str,
) -> None:
    request = REQUEST.model_copy(update=changes)

    with pytest.raises(RequestValidationError) as caught:
        adapter.to_method(request)
    with pytest.raises(RequestValidationError):
        adapter.to_stream_method(request)

    error = caught.value
    assert error.provider is Provider.OPENAI
    assert error.field == field
    assert isinstance(error, ValueError)
    assert isinstance(error.__cause__, ValidationError)
    assert str(error).startswith(
        f"openai cannot take the request field {field!r}. detail: {field}"
    )


def test_turns_are_sent_as_input_messages() -> None:
    request = REQUEST.model_copy(
        update={
            "input": [
                Turn.user("Hi"),
                Turn.assistant("Hello!"),
                Turn.user("How are you?"),
            ]
        }
    )

    method = adapter.to_stream_method(request)

    assert method.get_body_map()["input"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]


def test_conversation_must_end_with_a_user_turn() -> None:
    request = REQUEST.model_copy(update={"input": [Turn.assistant("Hello")]})

    with pytest.raises(RequestValidationError, match="last turn") as caught:
        adapter.to_method(request)

    assert caught.value.provider is Provider.OPENAI
    assert caught.value.field == "input"
    assert caught.value.__cause__ is None


def test_completed_response_is_converted() -> None:
    raw = response()

    converted = adapter.to_response(raw, REQUEST)

    assert converted.id == RESPONSE["id"]
    assert converted.provider is Provider.OPENAI
    assert converted.model == "gpt-6-astra"
    assert converted.text == TEXT
    assert converted.usage == TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )
    assert converted.incomplete_reason is None
    assert converted.raw_incomplete_reason is None
    assert converted.refusal is None
    assert converted.raw is raw


def test_usage_details_are_optional() -> None:
    usage = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}

    converted = adapter.to_response(response(usage=usage), REQUEST)

    assert converted.usage == TokenUsage(
        input_tokens=1, output_tokens=2, total_tokens=3
    )


def test_response_without_text_or_usage_is_kept_empty() -> None:
    converted = adapter.to_response(response(output=[], usage=None), REQUEST)

    assert converted.text is None
    assert converted.usage is None


@pytest.mark.parametrize(
    ("details", "reason", "raw_reason"),
    [
        (
            {"reason": "max_output_tokens"},
            IncompleteReason.MAX_TOKENS,
            "max_output_tokens",
        ),
        (
            {"reason": "content_filter"},
            IncompleteReason.OTHER,
            "content_filter",
        ),
        ({"reason": "max_messages"}, IncompleteReason.OTHER, "max_messages"),
        ({"reason": "steered"}, IncompleteReason.OTHER, "steered"),
        ({"reason": None}, IncompleteReason.OTHER, None),
        (None, IncompleteReason.OTHER, None),
    ],
)
def test_incomplete_response_keeps_its_text_and_reason(
    details: dict[str, Any] | None,
    reason: IncompleteReason,
    raw_reason: str | None,
) -> None:
    raw = response(status="incomplete", incomplete_details=details)

    converted = adapter.to_response(raw, REQUEST)

    assert converted.text == TEXT
    assert converted.incomplete_reason is reason
    assert converted.raw_incomplete_reason == raw_reason


REFUSED = MESSAGE | {
    "content": [
        {"type": "refusal", "refusal": "I can't "},
        {"type": "refusal", "refusal": "help with that."},
    ]
}


@pytest.mark.parametrize(
    ("status", "details", "raw_reason"),
    [
        ("completed", None, None),
        ("incomplete", {"reason": "max_output_tokens"}, "max_output_tokens"),
    ],
)
def test_refusal_is_a_field_not_an_error(
    status: str,
    details: dict[str, Any] | None,
    raw_reason: str | None,
) -> None:
    raw = response(status=status, incomplete_details=details, output=[REFUSED])

    converted = adapter.to_response(raw, REQUEST)

    assert converted.text is None
    assert converted.refusal == "I can't help with that."
    assert converted.incomplete_reason is IncompleteReason.REFUSAL
    assert converted.raw_incomplete_reason == raw_reason


@pytest.mark.parametrize(
    "status", ["in_progress", "queued", "failed", "cancelled", None]
)
def test_unfinished_response_raises(status: str | None) -> None:
    with pytest.raises(APIResponseError, match="expected 'completed'"):
        adapter.to_response(response(status=status), REQUEST)


async def test_stream_events_are_converted() -> None:
    converted = await convert(STREAM)

    assert [event.type for event in converted] == [
        "other",
        "other",
        "other",
        "other",
        "text_delta",
        "other",
        "other",
        "completed",
    ]
    assert all(isinstance(event, OtherEvent) for event in converted[:4])
    delta = converted[4]
    assert delta.text == "Hi"
    assert delta.raw.sequence_number == 4
    assert converted[5].raw.model_dump() == STREAM[5]
    completed = converted[-1]
    assert completed.usage == TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    )
    assert completed.incomplete_reason is None
    assert completed.raw_incomplete_reason is None
    assert completed.refusal is None
    assert completed.raw.type == "response.completed"
    assert completed.id == completed.raw.response.id
    assert completed.model == completed.raw.response.model
    # The text of the final response, as OpenAI sends it in the event.
    assert completed.text == completed.raw.response.output_text


async def test_incomplete_stream_completes_with_its_reason() -> None:
    incomplete = state(
        "response.incomplete",
        1,
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )

    (completed,) = await convert([incomplete])

    assert isinstance(completed, CompletedEvent)
    assert completed.incomplete_reason is IncompleteReason.MAX_TOKENS
    assert completed.raw_incomplete_reason == "max_output_tokens"


async def test_refused_stream_completes_with_the_refusal() -> None:
    refused = state("response.completed", 1, output=[REFUSED])

    (completed,) = await convert([refused])

    assert isinstance(completed, CompletedEvent)
    assert completed.incomplete_reason is IncompleteReason.REFUSAL
    assert completed.refusal == "I can't help with that."


async def test_adapter_leaves_the_end_to_the_parser_and_use_case() -> None:
    # StreamResponse and GenerationResource check the terminal event.
    converted = await convert(STREAM[:-1])

    assert len(converted) == len(STREAM) - 1


@pytest.fixture
def client(
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> OpenAIClient:
    """``OpenAIClient`` built from a token, sent to the local server."""
    monkeypatch.setattr(session_factory, "api", api)
    return OpenAIClient(TEST_TOKEN)


def test_client_generation_shares_the_client_session(
    client: OpenAIClient,
) -> None:
    resource: IGenerationResource[Response, ResponseStreamEvent] = (
        client.generation
    )

    assert isinstance(client.generation, GenerationResource)
    assert resource.session is client.session


async def test_client_generation_runs_on_responses(
    recorder: Recorder,
    client: OpenAIClient,
) -> None:
    recorder.reply(Reply(json=RESPONSE))
    recorder.reply(
        Reply(
            chunks=[frame(event) for event in STREAM],
            content_type="text/event-stream",
        )
    )

    async with client:
        created = await client.generation.create(REQUEST)
        async with client.generation.stream(REQUEST) as stream:
            received = [event async for event in stream]

    assert created.text == TEXT
    assert [event.type for event in received][-2:] == ["other", "completed"]
    first, second = recorder.requests
    assert first.raw_path == second.raw_path == "/v1/responses"
    assert first.headers["Authorization"] == f"Bearer {TEST_TOKEN}"
    assert json.loads(second.body)["stream"] is True


async def test_client_generation_raises_openai_errors(
    recorder: Recorder,
    client: OpenAIClient,
) -> None:
    quota = {"error": {"code": "insufficient_quota", "message": "No"}}
    recorder.reply(Reply(status=429, json=quota))
    error = {
        "type": "error",
        "code": "server_error",
        "message": "Broken",
        "sequence_number": 1,
    }
    recorder.reply(
        Reply(
            chunks=[frame(STREAM[0]), frame(error)],
            content_type="text/event-stream",
        )
    )

    async with client:
        with pytest.raises(QuotaExceededError):
            await client.generation.create(REQUEST)
        async with client.generation.stream(REQUEST) as stream:
            with pytest.raises(APIStatusError, match="Broken"):
                async for _ in stream:
                    pass

    assert len(recorder.requests) == 2


async def test_client_generation_stops_on_break(
    recorder: Recorder,
    client: OpenAIClient,
) -> None:
    recorder.reply(
        Reply(
            chunks=[frame(event) for event in STREAM],
            content_type="text/event-stream",
        )
    )

    texts: list[str] = []

    async with client:
        async with client.generation.stream(REQUEST) as stream:
            async for event in stream:
                if isinstance(event, TextDeltaEvent):
                    texts.append(event.text)
                    break
        with pytest.raises(RuntimeError):
            await anext(stream)

    assert texts == ["Hi"]
