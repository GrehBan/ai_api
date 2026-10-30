"""OpenAI generation adapter: shared requests to Responses and back.

The payloads are the reference examples of ``test_openai_responses``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ai_api.enums import Provider
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
from ai_api.types.requests import GenerationRequest
from ai_api.types.responses import (
    CompletedEvent,
    FailedEvent,
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
    "changes",
    [
        {"max_output_tokens": 15},
        {"metadata": {f"k{index}": "v" for index in range(17)}},
        {"metadata": {"k" * 65: "v"}},
    ],
)
def test_limits_of_openai_are_checked_before_sending(
    changes: dict[str, Any],
) -> None:
    request = REQUEST.model_copy(update=changes)

    with pytest.raises(ValidationError):
        adapter.to_method(request)
    with pytest.raises(ValidationError):
        adapter.to_stream_method(request)


def test_completed_response_is_converted() -> None:
    raw = response()

    converted = adapter.to_response(raw, REQUEST)

    assert converted.id == RESPONSE["id"]
    assert converted.provider is Provider.OPENAI
    assert converted.model == "gpt-6-astra"
    assert converted.text == TEXT
    assert converted.usage == TokenUsage(
        input_tokens=328, output_tokens=52, total_tokens=380
    )
    assert converted.incomplete_reason is None
    assert converted.raw is raw


def test_response_without_text_or_usage_is_kept_empty() -> None:
    converted = adapter.to_response(response(output=[], usage=None), REQUEST)

    assert converted.text is None
    assert converted.usage is None


@pytest.mark.parametrize(
    ("details", "reason"),
    [
        ({"reason": "max_output_tokens"}, "max_output_tokens"),
        ({"reason": None}, "unknown"),
        (None, "unknown"),
    ],
)
def test_incomplete_response_keeps_its_text_and_reason(
    details: dict[str, Any] | None,
    reason: str,
) -> None:
    raw = response(status="incomplete", incomplete_details=details)

    converted = adapter.to_response(raw, REQUEST)

    assert converted.text == TEXT
    assert converted.incomplete_reason == reason


def test_failed_response_raises_with_its_error() -> None:
    error = {"code": "server_error", "message": "The model failed."}

    with pytest.raises(APIResponseError) as caught:
        adapter.to_response(response(status="failed", error=error), REQUEST)

    assert str(caught.value).endswith(
        "OpenAI response failed (server_error): The model failed."
    )
    assert caught.value.status is None


def test_failed_response_without_an_error_still_raises() -> None:
    with pytest.raises(APIResponseError, match="failed without an error"):
        adapter.to_response(response(status="failed"), REQUEST)


@pytest.mark.parametrize("status", ["in_progress", "queued", None])
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
        input_tokens=328, output_tokens=52, total_tokens=380
    )
    assert completed.incomplete_reason is None
    assert completed.raw.type == "response.completed"


async def test_incomplete_stream_completes_with_its_reason() -> None:
    incomplete = state(
        "response.incomplete",
        1,
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
    )

    (completed,) = await convert([incomplete])

    assert isinstance(completed, CompletedEvent)
    assert completed.incomplete_reason == "max_output_tokens"


async def test_failed_stream_yields_the_failure() -> None:
    failed = state(
        "response.failed",
        1,
        status="failed",
        usage=None,
        error={"code": "server_error", "message": "Failed"},
    )

    (event,) = await convert([failed])

    assert isinstance(event, FailedEvent)
    assert event.message == "Failed"


async def test_stream_cut_before_its_end_raises() -> None:
    received: list[Any] = []

    with pytest.raises(APIResponseError, match="stream ended before"):
        async for event in adapter.to_events(events(STREAM[:-1]), REQUEST):
            received.append(event)  # noqa: PERF401 - kept before the error

    assert len(received) == len(STREAM) - 1


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
