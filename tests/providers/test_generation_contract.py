"""Shared generation behaves the same for every provider.

Each scenario runs against ``OpenAIClient.generation`` and
``AnthropicClient.generation`` on the local server, with the payloads
of each provider's reference examples. Only the wire format differs per
provider; the shared results and errors must not.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from typing import Any

import pytest
from aiohttp import ClientTimeout

from ai_api.enums import IncompleteReason, Provider
from ai_api.http.exceptions.status import (
    BadRequestError,
    QuotaExceededError,
    ServiceUnavailableError,
)
from ai_api.http.exceptions.transport import (
    APIConnectionError,
    APIStreamInterruptedError,
    APITimeoutError,
)
from ai_api.methods.base import BaseStreamMethod
from ai_api.providers.anthropic import AnthropicClient
from ai_api.providers.anthropic.client import (
    session_factory as anthropic_factory,
)
from ai_api.providers.anthropic.methods import StreamMessage
from ai_api.providers.openai import OpenAIClient
from ai_api.providers.openai.client import session_factory as openai_factory
from ai_api.providers.openai.methods import StreamResponse
from ai_api.resources import IGenerationResource
from ai_api.response.stream import Stream
from ai_api.session.factory import SessionFactory
from ai_api.types.requests import GenerationRequest, Turn
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

    def stream(
        self, method: BaseStreamMethod[Any]
    ) -> AbstractAsyncContextManager[Stream[Any]]: ...  # pragma: no cover

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
        refused: A response in which the model refused.
        refusal: The refusal's text in it.
        stream: Events of a finished stream, its last event terminal.
        deltas: Text deltas of that stream.
        stream_usage: Shared token counts of that stream.
        overloaded: An overloaded error, as the stream sends it.
        quota: An exhausted quota, as the stream sends it.
        bad_request: Body of a 400 response.
        raw_stream: Builds the provider's own stream method.
    """

    provider: Provider
    client: Callable[..., Any]
    factory: SessionFactory[Any, Any, ...]
    path: str
    response: dict[str, Any]
    text: str
    usage: TokenUsage
    truncated: dict[str, Any]
    refused: dict[str, Any]
    refusal: str
    stream: list[dict[str, Any]]
    deltas: list[str]
    stream_usage: TokenUsage
    overloaded: dict[str, Any]
    quota: dict[str, Any]
    bad_request: dict[str, Any]
    raw_stream: Callable[[], BaseStreamMethod[Any]]

    @property
    def delta(self) -> dict[str, Any]:
        """The first text delta of the stream."""
        return next(e for e in self.stream if "delta" in e["type"])

    @property
    def until_delta(self) -> list[dict[str, Any]]:
        """The stream up to its first text delta, included."""
        return self.stream[: self.stream.index(self.delta) + 1]


OPENAI = Case(
    provider=Provider.OPENAI,
    client=OpenAIClient,
    factory=openai_factory,
    path="/v1/responses",
    response=openai.RESPONSE,
    text=openai.MESSAGE["content"][0]["text"],
    usage=TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    ),
    truncated=openai.RESPONSE
    | {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    refused=openai.RESPONSE
    | {
        "output": [
            openai.MESSAGE
            | {"content": [{"type": "refusal", "refusal": "I can't."}]}
        ]
    },
    refusal="I can't.",
    stream=openai.STREAM,
    deltas=["Hi"],
    stream_usage=TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    ),
    overloaded={
        "type": "error",
        "code": "server_is_overloaded",
        "message": "Overloaded",
        "sequence_number": 1,
    },
    quota={
        "type": "error",
        "code": "insufficient_quota",
        "message": "You exceeded your current quota",
        "sequence_number": 1,
    },
    bad_request={
        "error": {
            "code": "invalid_value",
            "message": "Bad",
            "type": "invalid_request_error",
        }
    },
    raw_stream=lambda: StreamResponse(model="model-1", input="Hi"),
)

ANTHROPIC = Case(
    provider=Provider.ANTHROPIC,
    client=AnthropicClient,
    factory=anthropic_factory,
    path="/v1/messages",
    response=anthropic.MESSAGE,
    text="Hi! My name is Claude.",
    usage=TokenUsage(
        input_tokens=6197,
        output_tokens=503,
        total_tokens=6700,
        cache_read_tokens=2051,
        cache_write_tokens=2051,
        reasoning_tokens=0,
    ),
    truncated=anthropic.MESSAGE | {"stop_reason": "max_tokens"},
    refused=anthropic.MESSAGE | {"stop_reason": "refusal"},
    refusal=anthropic.MESSAGE["stop_details"]["explanation"],
    stream=anthropic.STREAM,
    deltas=["Hello", "!"],
    stream_usage=TokenUsage(
        input_tokens=25, output_tokens=15, total_tokens=40
    ),
    overloaded={
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    },
    quota={
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Spend cap reached",
            "details": {"error_code": "enforced_spend_limit_reached"},
        },
    },
    bad_request={
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Bad"},
    },
    raw_stream=lambda: StreamMessage(**anthropic.request()),
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


@pytest.fixture
def make_client(
    case: Case,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Client]:
    """Build the provider's client with the given session settings."""
    monkeypatch.setattr(case.factory, "api", api)

    def build(**session_kwargs: Any) -> Client:
        session = case.factory(TEST_TOKEN, **session_kwargs)
        client: Client = case.client(session=session)
        return client

    return build


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
    assert response.incomplete_reason is IncompleteReason.MAX_TOKENS


async def test_refusal_is_an_answer(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=case.refused))

    async with client:
        response = await client.generation.create(REQUEST)

    assert response.incomplete_reason is IncompleteReason.REFUSAL
    assert response.refusal == case.refusal


async def test_conversation_continues_with_the_response(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=case.response), Reply(json=case.response))

    async with client:
        first = await client.generation.create(REQUEST)
        request = REQUEST.model_copy(
            update={
                "input": [Turn.user("Hi"), first.to_turn(), Turn.user("More")]
            }
        )
        await client.generation.create(request)

    sent = json.loads(recorder.last.body)
    turns = (
        sent["input"]
        if case.provider is Provider.OPENAI
        else (sent["messages"])
    )
    assert turns == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": case.text},
        {"role": "user", "content": "More"},
    ]


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
    assert last.provider is case.provider
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
    # Only opening is retried: the events already arrived.
    assert len(recorder.requests) == 1


async def test_stream_quota_error_is_a_quota_error(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    # The same class as over HTTP 429: the stream's 200 does not hide it.
    recorder.reply(events([case.stream[0], case.quota]))

    async with client, client.generation.stream(REQUEST) as stream:
        with pytest.raises(QuotaExceededError) as caught:
            async for _ in stream:
                pass

    assert caught.value.provider is case.provider
    assert caught.value.transient is False
    assert caught.value.status == 200
    assert len(recorder.requests) == 1


async def test_stream_cut_before_its_end_raises(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    recorder.reply(events(case.stream[:-1]))

    received: list[Any] = []

    async with client, client.generation.stream(REQUEST) as stream:
        with pytest.raises(APIStreamInterruptedError) as caught:
            async for event in stream:
                received.append(event)  # noqa: PERF401 - fails mid-loop

    assert caught.value.__cause__ is None
    assert caught.value.transient is False
    assert len(received) == len(case.stream) - 1
    assert not any(isinstance(event, CompletedEvent) for event in received)


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


async def test_transient_failures_are_retried(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    unavailable = Reply(status=503, json={"error": {"message": "Busy"}})
    recorder.reply(
        unavailable,
        Reply(json=case.response),
        unavailable,
        events(case.stream),
    )

    async with client:
        response = await client.generation.create(REQUEST)
        async with client.generation.stream(REQUEST) as stream:
            completed = await stream.final()

    assert response.text == case.text
    assert completed.usage == case.stream_usage
    assert len(recorder.requests) == 4


def text_of(events: list[Any]) -> list[str]:
    """The text deltas among shared events."""
    return [e.text for e in events if isinstance(e, TextDeltaEvent)]


async def test_connection_drop_keeps_the_text_and_raises(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    reply = events(case.until_delta)
    recorder.reply(replace(reply, abort=True))
    received: list[Any] = []

    async with client, client.generation.stream(REQUEST) as stream:
        with pytest.raises(APIStreamInterruptedError) as caught:
            async for event in stream:
                received.append(event)  # noqa: PERF401 - fails mid-loop

    assert text_of(received) == case.deltas[:1]
    assert isinstance(caught.value.__cause__, APIConnectionError)
    assert caught.value.transient is False
    assert len(recorder.requests) == 1


async def test_silence_keeps_the_text_and_raises(
    case: Case,
    make_client: Callable[..., Client],
    recorder: Recorder,
) -> None:
    resume = asyncio.Event()
    frames = [frame(payload) for payload in case.until_delta]
    recorder.reply(
        Reply(
            chunks=[*frames, resume, frame(case.stream[-1])],
            content_type="text/event-stream",
        )
    )
    received: list[Any] = []
    client = make_client(stream_timeout=ClientTimeout(sock_read=0.3))

    async with client, client.generation.stream(REQUEST) as stream:
        with pytest.raises(APIStreamInterruptedError) as caught:
            async for event in stream:
                received.append(event)  # noqa: PERF401 - fails mid-loop

    resume.set()

    assert text_of(received) == case.deltas[:1]
    assert isinstance(caught.value.__cause__, APITimeoutError)


def endless(case: Case) -> Reply:
    """A stream of text deltas that never ends."""
    return Reply(
        chunks=itertools.chain(
            [frame(case.stream[0])], itertools.repeat(frame(case.delta))
        ),
        chunk_delay=0.001,
        content_type="text/event-stream",
    )


def opener(
    client: Client, case: Case, mode: str
) -> AbstractAsyncContextManager[Stream[Any]]:
    """Open the raw provider stream or the shared generation stream."""
    if mode == "raw":
        return client.stream(case.raw_stream())
    return client.generation.stream(REQUEST)


MODES = pytest.mark.parametrize("mode", ["raw", "generation"])


@MODES
async def test_cancelling_the_task_while_iterating_releases_the_connection(
    case: Case,
    make_client: Callable[..., Client],
    recorder: Recorder,
    mode: str,
) -> None:
    recorder.reply(endless(case), Reply(json=case.response))
    started = asyncio.Event()

    async def consume(client: Client) -> None:
        async with opener(client, case, mode) as stream:
            async for _ in stream:
                started.set()

    # With one connection, a leaked one would block the next request.
    async with make_client(max_connections=1) as client:
        task = asyncio.create_task(consume(client))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        async with asyncio.timeout(2):
            response = await client.generation.create(REQUEST)

    assert response.text == case.text


@MODES
async def test_cancelling_the_task_while_opening_releases_the_connection(
    case: Case,
    make_client: Callable[..., Client],
    recorder: Recorder,
    mode: str,
) -> None:
    recorder.reply(
        replace(events(case.stream), delay=1),
        Reply(json=case.response),
    )

    async def consume(client: Client) -> None:
        async with opener(client, case, mode) as stream:  # pragma: no cover
            async for _ in stream:
                pass

    async with make_client(max_connections=1) as client:
        task = asyncio.create_task(consume(client))

        async with asyncio.timeout(2):
            while not recorder.requests:
                await asyncio.sleep(0.01)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        async with asyncio.timeout(2):
            response = await client.generation.create(REQUEST)

    assert response.text == case.text
