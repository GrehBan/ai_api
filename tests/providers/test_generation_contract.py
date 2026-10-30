"""Shared generation behaves the same for every provider.

Each scenario runs against ``OpenAIClient.generation`` and
``AnthropicClient.generation`` on the local server, with the payloads
of each provider's reference examples. Only the wire format differs per
provider; the shared results and errors must not.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from ai_api.client.factory import SessionFactory
from ai_api.enums import Provider
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import (
    BadRequestError,
    ServiceUnavailableError,
)
from ai_api.providers.anthropic import AnthropicClient
from ai_api.providers.anthropic.client import (
    session_factory as anthropic_factory,
)
from ai_api.providers.openai import OpenAIClient
from ai_api.providers.openai.client import session_factory as openai_factory
from ai_api.resources import IGenerationResource
from ai_api.types.requests import GenerationRequest
from ai_api.types.responses import (
    CompletedEvent,
    TextDeltaEvent,
    TokenUsage,
)
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply
from tests.providers import test_anthropic_messages as anthropic
from tests.providers import test_openai_responses as openai

REQUEST = GenerationRequest(
    model="model-1",
    input="Hi",
    instructions="Be brief",
    max_output_tokens=64,
)


class Client:
    """What the scenarios need of a provider client."""

    generation: IGenerationResource[Any, Any]

    async def __aenter__(self) -> Client: ...  # pragma: no cover

    async def __aexit__(self, *args: object) -> None: ...  # pragma: no cover


@dataclass(frozen=True)
class Case:
    """One provider in the scenarios.

    Attributes:
        provider: The provider.
        client: Builds the client from a token.
        factory: The session factory to send to the local server.
        path: Request path of the endpoint.
        response: A finished buffered response.
        text: Its text.
        usage: Its shared token counts.
        truncated: A response cut off by the token limit.
        stream: Events of a finished stream, its last event terminal.
        deltas: Text deltas of that stream.
        stream_usage: Shared token counts of that stream.
        overloaded: An overloaded error, as the stream sends it.
        bad_request: Body of a 400 response.
    """

    provider: Provider
    client: Callable[[str], Any]
    factory: SessionFactory[Any, Any, ...]
    path: str
    response: dict[str, Any]
    text: str
    usage: TokenUsage
    truncated: dict[str, Any]
    stream: list[dict[str, Any]]
    deltas: list[str]
    stream_usage: TokenUsage
    overloaded: dict[str, Any]
    bad_request: dict[str, Any]


OPENAI = Case(
    provider=Provider.OPENAI,
    client=OpenAIClient,
    factory=openai_factory,
    path="/v1/responses",
    response=openai.RESPONSE,
    text=openai.MESSAGE["content"][0]["text"],
    usage=TokenUsage(input_tokens=328, output_tokens=52, total_tokens=380),
    truncated=openai.RESPONSE
    | {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    stream=openai.STREAM,
    deltas=["Hi"],
    stream_usage=TokenUsage(
        input_tokens=328, output_tokens=52, total_tokens=380
    ),
    overloaded={
        "type": "error",
        "code": "server_is_overloaded",
        "message": "Overloaded",
        "sequence_number": 1,
    },
    bad_request={
        "error": {
            "code": "invalid_value",
            "message": "Bad",
            "type": "invalid_request_error",
        }
    },
)

ANTHROPIC = Case(
    provider=Provider.ANTHROPIC,
    client=AnthropicClient,
    factory=anthropic_factory,
    path="/v1/messages",
    response=anthropic.MESSAGE,
    text="Hi! My name is Claude.",
    usage=TokenUsage(input_tokens=6197, output_tokens=503, total_tokens=6700),
    truncated=anthropic.MESSAGE | {"stop_reason": "max_tokens"},
    stream=anthropic.STREAM,
    deltas=["Hello", "!"],
    stream_usage=TokenUsage(
        input_tokens=25, output_tokens=15, total_tokens=40
    ),
    overloaded={
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    },
    bad_request={
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Bad"},
    },
)


def frame(data: dict[str, Any]) -> bytes:
    """One SSE frame with ``event:`` and ``data:``, as both send it."""
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def events(payloads: list[dict[str, Any]]) -> Reply:
    """A streamed reply of the given events."""
    return Reply(
        chunks=[frame(payload) for payload in payloads],
        content_type="text/event-stream",
    )


@pytest.fixture(params=[OPENAI, ANTHROPIC], ids=["openai", "anthropic"])
def case(request: pytest.FixtureRequest) -> Case:
    """Each provider in turn."""
    return request.param


@pytest.fixture
def client(
    case: Case,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> Client:
    """The provider's client, sent to the local server."""
    monkeypatch.setattr(case.factory, "api", api)
    return case.client(TEST_TOKEN)


async def test_create_returns_the_text_and_usage(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=case.response))

    async with client:
        response = await client.generation.create(REQUEST)

    assert response.provider is case.provider
    assert response.id == case.response["id"]
    assert response.text == case.text
    assert response.usage == case.usage
    assert response.incomplete_reason is None
    assert recorder.last.raw_path == case.path


async def test_truncated_response_keeps_text_and_reason(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=case.truncated))

    async with client:
        response = await client.generation.create(REQUEST)

    assert response.text == case.text
    assert response.incomplete_reason is not None


async def test_stream_yields_deltas_then_completes(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(events(case.stream))

    async with client, client.generation.stream(REQUEST) as stream:
        received = [event async for event in stream]

    deltas = [e.text for e in received if isinstance(e, TextDeltaEvent)]
    assert deltas == case.deltas
    last = received[-1]
    assert isinstance(last, CompletedEvent)
    assert last.usage == case.stream_usage
    assert last.incomplete_reason is None
    assert json.loads(recorder.last.body)["stream"] is True


async def test_stream_error_event_raises_overloaded(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(events([case.stream[0], case.overloaded]))

    async with client, client.generation.stream(REQUEST) as stream:
        with pytest.raises(ServiceUnavailableError) as caught:
            async for _ in stream:
                pass

    assert caught.value.provider is case.provider
    assert "Overloaded" in str(caught.value)


async def test_stream_cut_before_its_end_raises(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(events(case.stream[:-1]))

    async with client, client.generation.stream(REQUEST) as stream:
        with pytest.raises(APIResponseError, match="stream ended before"):
            async for _ in stream:
                pass


async def test_break_closes_the_stream(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(events(case.stream))
    first: list[str] = []

    async with client:
        async with client.generation.stream(REQUEST) as stream:
            async for event in stream:
                if isinstance(event, TextDeltaEvent):
                    first.append(event.text)
                    break
        with pytest.raises(RuntimeError):
            await anext(stream)

    assert first == case.deltas[:1]


async def test_error_status_raises_before_streaming(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(
        Reply(status=400, json=case.bad_request),
        Reply(status=400, json=case.bad_request),
    )

    async with client:
        with pytest.raises(BadRequestError) as created:
            await client.generation.create(REQUEST)
        with pytest.raises(BadRequestError):
            async with client.generation.stream(REQUEST):
                pass  # pragma: no cover

    assert created.value.provider is case.provider
    assert created.value.transient is False
    assert len(recorder.requests) == 2
