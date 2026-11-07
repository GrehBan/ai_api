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
from pydantic import MISSING, ValidationError

from aiollm.enums import Effort, IncompleteReason, Provider, ToolChoice
from aiollm.exceptions import RequestValidationError
from aiollm.http.exceptions.status import (
    BadRequestError,
    QuotaExceededError,
    ServiceUnavailableError,
)
from aiollm.http.exceptions.transport import (
    APIConnectionError,
    APIStreamInterruptedError,
    APITimeoutError,
)
from aiollm.methods.base import BaseStreamMethod
from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.anthropic.client import (
    session_factory as anthropic_factory,
)
from aiollm.providers.anthropic.methods import StreamMessage
from aiollm.providers.openai import OpenAIClient
from aiollm.providers.openai.client import session_factory as openai_factory
from aiollm.providers.openai.methods import StreamResponse
from aiollm.resources import GenerationResource, IGenerationResource
from aiollm.response.stream import Stream
from aiollm.session.factory import SessionFactory
from aiollm.session.protocols import ISession
from aiollm.session.retry import RetryPolicy
from aiollm.types.requests import (
    DocumentPart,
    ForcedTool,
    GenerationRequest,
    ImagePart,
    JSONSchema,
    NativeOutput,
    TextPart,
    Tool,
    ToolCall,
    ToolCallTurn,
    ToolResult,
    Turn,
)
from aiollm.types.responses import (
    CompletedEvent,
    TextDeltaEvent,
    TokenUsage,
    ToolCallDeltaEvent,
    ToolCallDoneEvent,
    ToolCallStartEvent,
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

SCHEMA: JSONSchema = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
"""Parameters of the tools of the scenarios."""

CALLS = [
    ("call_1", "get_weather", {"city": "Paris"}),
    ("call_2", "get_time", {}),
]
"""Two calls the model makes in one response: id, tool, arguments."""


def openai_called() -> dict[str, Any]:
    """An OpenAI response with text and ``CALLS``."""
    calls = [
        {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        }
        for n, (call_id, name, arguments) in enumerate(CALLS)
    ]
    return openai.RESPONSE | {"output": [openai.MESSAGE, *calls]}


def openai_called_stream() -> list[dict[str, Any]]:
    """An OpenAI stream of ``CALLS``, the arguments in two pieces."""
    events: list[dict[str, Any]] = [openai.state("response.created", 0)]
    for n, (call_id, name, arguments) in enumerate(CALLS):
        item = {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        text = json.dumps(arguments)
        events += [
            {
                "type": "response.output_item.added",
                "output_index": n,
                "item": item,
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{n}",
                "output_index": n,
                "delta": text[:3],
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{n}",
                "output_index": n,
                "delta": text[3:],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": f"fc_{n}",
                "output_index": n,
                "arguments": text,
            },
            {
                "type": "response.output_item.done",
                "output_index": n,
                "item": item | {"arguments": text, "status": "completed"},
            },
        ]
    events.append(
        openai.state("response.completed", 0, output=openai_called()["output"])
    )
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return events


def anthropic_called() -> dict[str, Any]:
    """An Anthropic message with text and ``CALLS``."""
    blocks = [
        {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
        for call_id, name, arguments in CALLS
    ]
    return anthropic.MESSAGE | {
        "content": [{"type": "text", "text": "Checking."}, *blocks],
        "stop_reason": "tool_use",
        "stop_details": None,
    }


def anthropic_called_stream() -> list[dict[str, Any]]:
    """An Anthropic stream of ``CALLS``: the first call's arguments in
    two pieces, the second's in none, as Anthropic sends ``{}``."""
    start, *_, stop = anthropic.STREAM
    events: list[dict[str, Any]] = [start]
    for index, (call_id, name, arguments) in enumerate(CALLS):
        events.append(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": {},
                },
            }
        )
        if arguments:
            text = json.dumps(arguments)
            events += [
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": part,
                    },
                }
                for part in (text[:3], text[3:])
            ]
        events.append({"type": "content_block_stop", "index": index})
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 15},
        },
        stop,
    ]
    return events


TOOLS = [
    Tool(
        name="get_weather",
        description="Current weather",
        parameters=SCHEMA,
        strict=True,
    ),
    Tool(name="get_time", parameters={"type": "object", "properties": {}}),
]
"""A strict tool with a description, and a bare one."""


PNG = b"\x89PNG\r\n"
PDF = b"%PDF-1.7"
PARTS = [
    TextPart(text="What is in these?"),
    ImagePart(data=PNG, media_type="image/png"),
    ImagePart(url="https://example.com/cat.jpg"),
    DocumentPart(data=PDF, filename="report.pdf"),
    DocumentPart(data=PDF),
    DocumentPart(url="https://example.com/report.pdf"),
]
"""A user turn of every kind of part and source."""


class Client:
    """What the scenarios need of a provider client."""

    generation: IGenerationResource[Any, Any]
    session: ISession

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
        options: Valid provider options, sent as they are.
        other_options: Other valid options, for a concurrent call.
        bad_option: A provider option with a value the method refuses.
        effort: What ``effort=Effort.HIGH`` becomes in the body.
        output_schema: What ``SCHEMA`` with ``Effort.HIGH`` becomes.
        tools: What ``TOOLS`` with ``ToolChoice.REQUIRED`` and no
            parallel calls become.
        forced: What a ``ForcedTool`` becomes.
        only_parallel: What ``parallel_tool_calls=True`` alone becomes.
        called: A finished response with ``CALLS``, text first.
        called_stream: A finished stream with ``CALLS``, text first.
        cut_off: What makes ``called`` stop at the token limit.
        input_field: The body field of the conversation.
        continued: The body's conversation after its first turn, when
            ``called`` comes back with its results.
        continued_shared: The same from the shared calls only.
        continued_streamed: The same when ``called_stream`` comes back.
        parts: What a user turn of ``PARTS`` becomes.
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
    options: dict[str, Any]
    other_options: dict[str, Any]
    bad_option: dict[str, Any]
    effort: dict[str, Any]
    output_schema: dict[str, Any]
    tools: dict[str, Any]
    forced: dict[str, Any]
    only_parallel: dict[str, Any]
    called: dict[str, Any]
    called_stream: list[dict[str, Any]]
    cut_off: dict[str, Any]
    input_field: str
    continued: list[Any]
    continued_shared: list[Any]
    continued_streamed: list[Any]
    parts: dict[str, Any]

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
    options={"store": False, "temperature": 0.2},
    other_options={"store": True, "top_p": 0.5},
    bad_option={"temperature": 5},
    effort={"reasoning": {"effort": "high"}},
    output_schema={
        "reasoning": {"effort": "high"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "output",
                "schema": {"type": "object", "required": ["n"]},
                "strict": True,
            }
        },
    },
    tools={
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather",
                "parameters": SCHEMA,
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
            },
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    },
    forced={"tool_choice": {"type": "function", "name": "get_time"}},
    only_parallel={"parallel_tool_calls": True},
    called=openai_called(),
    called_stream=openai_called_stream(),
    cut_off={
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    input_field="input",
    continued_streamed=[
        *openai_called()["output"],
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    continued=[
        *openai_called()["output"],
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    continued_shared=[
        {"role": "assistant", "content": "Checking."},
        *(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments),
            }
            for call_id, name, arguments in CALLS
        ),
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    parts={
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What is in these?"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBORw0K",
                "detail": "auto",
            },
            {
                "type": "input_image",
                "image_url": "https://example.com/cat.jpg",
                "detail": "auto",
            },
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                "filename": "report.pdf",
            },
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                "filename": "document.pdf",
            },
            {
                "type": "input_file",
                "file_url": "https://example.com/report.pdf",
            },
        ],
    },
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
    options={"stop_sequences": ["END"]},
    other_options={"stop_sequences": ["STOP", "HALT"]},
    bad_option={"stop_sequences": "END"},
    effort={"output_config": {"effort": "high"}},
    output_schema={
        "output_config": {
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "required": ["n"]},
            },
        }
    },
    tools={
        "tools": [
            {
                "name": "get_weather",
                "description": "Current weather",
                "input_schema": SCHEMA,
                "strict": True,
            },
            {
                "name": "get_time",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    },
    forced={"tool_choice": {"type": "tool", "name": "get_time"}},
    only_parallel={
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": False}
    },
    called=anthropic_called(),
    called_stream=anthropic_called_stream(),
    cut_off={"stop_reason": "max_tokens"},
    input_field="messages",
    continued_streamed=[
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": arguments,
                }
                for call_id, name, arguments in CALLS
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    continued=[
        {"role": "assistant", "content": anthropic_called()["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    continued_shared=[
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                *(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": arguments,
                    }
                    for call_id, name, arguments in CALLS
                ),
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    parts={
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in these?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0K",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/cat.jpg",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjc=",
                },
                "title": "report.pdf",
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjc=",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "https://example.com/report.pdf",
                },
            },
        ],
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


def test_generation_shares_the_client_session(client: Client) -> None:
    assert isinstance(client.generation, GenerationResource)
    assert client.generation.session is client.session


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_effort_reaches_the_wire(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    if mode == "create":
        recorder.reply(Reply(json=case.response))
    else:
        recorder.reply(events(case.stream))
    request = REQUEST.model_copy(update={"effort": Effort.HIGH})

    async with client:
        if mode == "create":
            await client.generation.create(request)
        else:
            async with client.generation.stream(request) as stream:
                await stream.final()

    body = json.loads(recorder.last.body)
    assert {key: body[key] for key in case.effort} == case.effort


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_an_output_schema_reaches_the_wire_as_it_is(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    # With effort too: Anthropic sends both in one output_config.
    if mode == "create":
        recorder.reply(Reply(json=case.response))
    else:
        recorder.reply(events(case.stream))
    request = REQUEST.model_copy(
        update={
            "effort": Effort.HIGH,
            "output_schema": {"type": "object", "required": ["n"]},
        }
    )

    async with client:
        if mode == "create":
            await client.generation.create(request)
        else:
            async with client.generation.stream(request) as stream:
                await stream.final()

    body = json.loads(recorder.last.body)
    assert {key: body[key] for key in case.output_schema} == (
        case.output_schema
    )


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_tools_reach_the_wire(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    if mode == "create":
        recorder.reply(Reply(json=case.response))
    else:
        recorder.reply(events(case.stream))
    request = REQUEST.model_copy(
        update={
            "tools": TOOLS,
            "tool_choice": ToolChoice.REQUIRED,
            "parallel_tool_calls": False,
        }
    )

    async with client:
        if mode == "create":
            await client.generation.create(request)
        else:
            async with client.generation.stream(request) as stream:
                await stream.final()

    body = json.loads(recorder.last.body)
    assert {key: body[key] for key in case.tools} == case.tools


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_tool_calls_come_back_alike(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    # The arguments' text is the provider's (OpenAI sends it, Anthropic's
    # object is written as JSON), so the parsed arguments are compared.
    if mode == "create":
        recorder.reply(Reply(json=case.called))
    else:
        recorder.reply(events(case.called_stream))
    request = REQUEST.model_copy(update={"tools": TOOLS})

    async with client:
        if mode == "create":
            result: Any = await client.generation.create(request)
        else:
            async with client.generation.stream(request) as stream:
                result = await stream.final()

    assert [
        (call.id, call.name, call.parse_arguments())
        for call in result.tool_calls
    ] == CALLS
    assert result.incomplete_reason is None
    assert result.raw_incomplete_reason is None


async def test_tool_calls_stream_as_start_pieces_and_done(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    recorder.reply(events(case.called_stream))
    request = REQUEST.model_copy(update={"tools": TOOLS})

    async with client, client.generation.stream(request) as stream:
        received = [event async for event in stream]

    starts = [e for e in received if isinstance(e, ToolCallStartEvent)]
    done = [e.call for e in received if isinstance(e, ToolCallDoneEvent)]
    assert [(e.call_id, e.name) for e in starts] == [
        (call_id, name) for call_id, name, _ in CALLS
    ]
    assert [
        (call.id, call.name, call.parse_arguments()) for call in done
    ] == CALLS
    for call in done:
        pieces = [
            e.arguments
            for e in received
            if isinstance(e, ToolCallDeltaEvent) and e.call_id == call.id
        ]
        assert not pieces or "".join(pieces) == call.arguments
    completed = received[-1]
    assert isinstance(completed, CompletedEvent)
    assert completed.tool_calls == done
    for call in done:
        opened = next(
            n
            for n, e in enumerate(received)
            if isinstance(e, ToolCallStartEvent) and e.call_id == call.id
        )
        closed = next(
            n
            for n, e in enumerate(received)
            if isinstance(e, ToolCallDoneEvent) and e.call.id == call.id
        )
        assert opened < closed


async def test_a_stream_can_fail_after_its_calls_are_done(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    # A done call is a complete call, not a successful generation: the
    # stream can still fail before its CompletedEvent.
    recorder.reply(events([*case.called_stream[:-1], case.overloaded]))
    request = REQUEST.model_copy(update={"tools": TOOLS})
    received: list[Any] = []

    async with client, client.generation.stream(request) as stream:
        with pytest.raises(ServiceUnavailableError):
            async for event in stream:
                received.append(event)  # noqa: PERF401 (kept on error)

    done = [e.call for e in received if isinstance(e, ToolCallDoneEvent)]
    assert [call.id for call in done] == [call_id for call_id, _, _ in CALLS]
    assert not any(isinstance(e, CompletedEvent) for e in received)


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_a_turn_of_parts_sends_text_images_and_documents(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    if mode == "create":
        recorder.reply(Reply(json=case.response))
    else:
        recorder.reply(events(case.stream))
    request = REQUEST.model_copy(update={"input": [Turn.user(PARTS)]})

    async with client:
        if mode == "create":
            await client.generation.create(request)
        else:
            async with client.generation.stream(request) as stream:
                await stream.final()

    assert json.loads(recorder.last.body)[case.input_field] == [case.parts]


async def test_parts_after_tool_results_go_with_them(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    # Anthropic puts a user turn after results into their message; OpenAI
    # keeps it a message of its own.
    recorder.reply(Reply(json=case.response))
    call = ToolCall(id="call_1", name="get_time", arguments="{}")
    request = REQUEST.model_copy(
        update={
            "tools": TOOLS,
            "input": [
                Turn.user("Hi"),
                ToolCallTurn(tool_calls=[call]),
                ToolResult(call_id="call_1", output="noon"),
                Turn.user(PARTS[:2]),
            ],
        }
    )

    async with client:
        await client.generation.create(request)

    *_, results, last = json.loads(recorder.last.body)[case.input_field]
    assert last["content"][-2:] == case.parts["content"][:2]
    if case.provider is Provider.ANTHROPIC:
        # One user message: the result first, then the parts.
        assert last["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "noon",
        }
        assert len(last["content"]) == 3
    else:
        assert results == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "noon",
        }
        assert last["role"] == "user"


async def test_a_streamed_tool_loop_continues_with_the_results(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    # The completed event's turn carries the stream's own output, put
    # together from its events, and the second request sends it back.
    recorder.reply(events(case.called_stream), Reply(json=case.response))
    request = REQUEST.model_copy(update={"tools": TOOLS})

    async with client:
        async with client.generation.stream(request) as stream:
            completed = await stream.final()
        turn = completed.to_turn()
        results = [
            ToolResult(call_id=call.id, output=f"result of {call.name}")
            for call in completed.tool_calls
        ]
        await client.generation.create(
            request.model_copy(
                update={"input": [Turn.user("Hi"), turn, *results]}
            )
        )

    assert isinstance(turn, ToolCallTurn)
    assert turn.native is not MISSING
    assert turn.native.provider is case.provider
    sent = json.loads(recorder.last.body)
    assert sent[case.input_field][1:] == case.continued_streamed


async def test_a_tool_loop_continues_with_the_results(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    # The response's turn of calls goes back with the provider's own
    # output, and the results follow it; the second request ends the
    # loop with text.
    recorder.reply(Reply(json=case.called), Reply(json=case.response))
    request = REQUEST.model_copy(update={"tools": TOOLS})

    async with client:
        called = await client.generation.create(request)
        turn = called.to_turn()
        results = [
            ToolResult(call_id=call.id, output=f"result of {call.name}")
            for call in called.tool_calls
        ]
        answer = await client.generation.create(
            request.model_copy(
                update={"input": [Turn.user("Hi"), turn, *results]}
            )
        )

    assert isinstance(turn, ToolCallTurn)
    assert turn.native is not MISSING
    assert turn.native.provider is case.provider
    assert answer.text == case.text
    sent = json.loads(recorder.last.body)
    assert sent[case.input_field][1:] == case.continued


async def test_a_turn_of_calls_from_another_provider_sends_its_calls(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    # Another provider's output means nothing here: the shared calls go.
    recorder.reply(Reply(json=case.response))
    foreign = (
        Provider.ANTHROPIC
        if case.provider is Provider.OPENAI
        else (Provider.OPENAI)
    )
    turn = ToolCallTurn(
        text="Checking.",
        tool_calls=[
            ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))
            for call_id, name, arguments in CALLS
        ],
        native=NativeOutput(provider=foreign, items=[{"type": "opaque"}]),
    )
    results = [
        ToolResult(call_id=call_id, output=f"result of {name}")
        for call_id, name, _ in CALLS
    ]
    request = REQUEST.model_copy(
        update={"tools": TOOLS, "input": [Turn.user("Hi"), turn, *results]}
    )

    async with client:
        await client.generation.create(request)

    sent = json.loads(recorder.last.body)
    assert sent[case.input_field][1:] == case.continued_shared


@pytest.mark.parametrize(
    ("items", "match"),
    [
        (["call"], "have no result"),
        (["call", "result-1"], "have no result"),
        (["call", "user", "result-1", "result-2"], "comes before the results"),
        (["result-1"], "did not make"),
        (["call", "result-1", "result-1", "result-2"], "did not make"),
        (["twins"], "share an id"),
    ],
    ids=[
        "unanswered",
        "half-answered",
        "text-between",
        "result-alone",
        "result-twice",
        "same-id",
    ],
)
async def test_calls_and_results_out_of_order_are_refused(
    case: Case,
    client: Any,
    items: list[str],
    match: str,
) -> None:
    call = ToolCall(id="call_1", name="get_weather", arguments="{}")
    other = ToolCall(id="call_2", name="get_time", arguments="{}")
    pieces = {
        "call": ToolCallTurn(tool_calls=[call, other]),
        "twins": ToolCallTurn(tool_calls=[call, call]),
        "user": Turn.user("And?"),
        "result-1": ToolResult(call_id="call_1", output="15"),
        "result-2": ToolResult(call_id="call_2", output="noon"),
    }
    conversation = [Turn.user("Hi"), *(pieces[name] for name in items)]
    request = REQUEST.model_copy(update={"input": conversation})

    async with client:
        with pytest.raises(RequestValidationError, match=match) as caught:
            await client.generation.create(request)

    assert caught.value.field == "input"
    assert caught.value.provider is case.provider


async def test_a_response_that_stopped_mid_call_is_no_turn(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=case.called | case.cut_off))

    async with client:
        response = await client.generation.create(
            REQUEST.model_copy(update={"tools": TOOLS})
        )

    assert response.incomplete_reason is IncompleteReason.MAX_TOKENS
    assert response.tool_calls
    with pytest.raises(ValueError, match="last tool call may be cut off"):
        response.to_turn()


async def test_a_response_without_calls_has_none(
    case: Case,
    client: Any,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=case.response))

    async with client:
        response = await client.generation.create(REQUEST)

    assert response.tool_calls == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"tool_choice": ForcedTool(name="get_time")}, "forced"),
        ({"parallel_tool_calls": True}, "only_parallel"),
    ],
    ids=["forced", "only-parallel"],
)
async def test_tool_choices_reach_the_wire(
    case: Case,
    client: Any,
    recorder: Recorder,
    changes: dict[str, Any],
    expected: str,
) -> None:
    recorder.reply(Reply(json=case.response))
    request = GenerationRequest(
        model="model-1", input="Hi", tools=TOOLS, **changes
    )

    async with client:
        await client.generation.create(request)

    body = json.loads(recorder.last.body)
    wanted = getattr(case, expected)
    assert {key: body[key] for key in wanted} == wanted


@pytest.mark.parametrize(
    "changes",
    [
        {"tool_choice": ToolChoice.AUTO},
        {"parallel_tool_calls": False},
        {"tools": [TOOLS[0], TOOLS[0]]},
        {"tools": TOOLS, "tool_choice": ForcedTool(name="other")},
        {
            "tools": TOOLS,
            "tool_choice": ToolChoice.NONE,
            "parallel_tool_calls": True,
        },
        {"tools": []},
    ],
    ids=[
        "choice-without-tools",
        "parallel-without-tools",
        "duplicate-names",
        "forced-unknown-tool",
        "parallel-with-none",
        "no-tools",
    ],
)
def test_tool_settings_no_provider_takes_are_refused(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(model="model-1", input="Hi", **changes)


@pytest.mark.parametrize(
    "name", ["", "has space", "a" * 129, "dots.not.allowed"]
)
def test_tool_names_both_providers_take(name: str) -> None:
    with pytest.raises(ValidationError, match="name"):
        Tool(name=name, parameters={"type": "object"})
    Tool(name="a" * 128, parameters={"type": "object"})


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_provider_options_reach_the_wire_of_their_call_only(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    # The options belong to the call: the next one without them sends none.
    if mode == "create":
        reply = Reply(json=case.response)
    else:
        reply = events(case.stream)
    recorder.reply(reply)
    recorder.reply(reply)

    async with client:
        for options in (case.options, {}):
            if mode == "create":
                await client.generation.create(
                    REQUEST, provider_options=options
                )
            else:
                async with client.generation.stream(
                    REQUEST, provider_options=options
                ) as stream:
                    await stream.final()

    with_options, without = (json.loads(r.body) for r in recorder.requests)
    assert {key: with_options[key] for key in case.options} == case.options
    assert set(case.options).isdisjoint(without)


async def concurrently(
    client: Any,
    mode: str,
    calls: list[dict[str, Any]],
) -> None:
    """Run one call per options at once on the client's one resource."""

    async def call(options: dict[str, Any]) -> None:
        if mode == "create":
            await client.generation.create(REQUEST, provider_options=options)
        else:
            async with client.generation.stream(
                REQUEST, provider_options=options
            ) as stream:
                await stream.final()

    await asyncio.gather(*(call(options) for options in calls))


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_concurrent_calls_keep_their_own_provider_options(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    # The calls share the resource and its adapter; each copy carries
    # only its own options, and a later call without any sends none: an
    # adapter that kept a call's options on itself would leak them there.
    calls = [case.options, case.other_options, case.options]
    for _ in [*calls, None]:
        if mode == "create":
            recorder.reply(Reply(json=case.response))
        else:
            recorder.reply(events(case.stream))

    async with client:
        await concurrently(client, mode, calls)
        if mode == "create":
            await client.generation.create(REQUEST)
        else:
            async with client.generation.stream(REQUEST) as stream:
                await stream.final()

    keys = set(case.options) | set(case.other_options)
    *sent, last = [
        {key: body[key] for key in keys if key in body}
        for body in (json.loads(r.body) for r in recorder.requests)
    ]

    def key(options: dict[str, Any]) -> str:
        return json.dumps(options, sort_keys=True)

    assert sorted(map(key, sent)) == sorted(map(key, calls))
    assert last == {}


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_an_unknown_provider_option_is_refused_before_sending(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    # A shared field is no option: it is set in the request.
    shared = {"model": "m", "input": "x"}

    async with client:
        with pytest.raises(
            RequestValidationError, match="input, model"
        ) as caught:
            if mode == "create":
                await client.generation.create(
                    REQUEST, provider_options=shared
                )
            else:
                async with client.generation.stream(
                    REQUEST, provider_options=shared
                ):
                    pass

    assert caught.value.provider is case.provider
    assert caught.value.field == "input"
    assert recorder.requests == []


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_an_invalid_provider_option_is_refused_before_sending(
    case: Case,
    client: Any,
    recorder: Recorder,
    mode: str,
) -> None:
    async with client:
        with pytest.raises(RequestValidationError) as caught:
            if mode == "create":
                await client.generation.create(
                    REQUEST, provider_options=case.bad_option
                )
            else:
                async with client.generation.stream(
                    REQUEST, provider_options=case.bad_option
                ):
                    pass

    assert caught.value.field == next(iter(case.bad_option))
    assert recorder.requests == []


@pytest.mark.parametrize("mode", ["create", "stream"])
async def test_conversation_must_end_with_a_user_turn(
    case: Case,
    client: Client,
    recorder: Recorder,
    mode: str,
) -> None:
    # Refused before anything is sent, by every provider.
    turns = [Turn.user("Hi"), Turn.assistant("Hello")]

    async with client:
        with pytest.raises(
            RequestValidationError, match="last turn"
        ) as caught:
            if mode == "create":
                await client.generation.create(model="m", input=turns)
            else:
                async with client.generation.stream(model="m", input=turns):
                    pass

    assert caught.value.provider is case.provider
    assert caught.value.field == "input"
    assert recorder.requests == []


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


async def test_call_options_reach_the_session(
    case: Case,
    client: Client,
    recorder: Recorder,
) -> None:
    unavailable = Reply(status=503, json={"error": {"message": "Busy"}})
    recorder.reply(unavailable, unavailable)
    once = RetryPolicy(max_attempts=1)

    async with client:
        with pytest.raises(ServiceUnavailableError):
            await client.generation.create(
                REQUEST, headers={"X-Trace": "t-1"}, retry_policy=once
            )
        with pytest.raises(ServiceUnavailableError):
            async with client.generation.stream(
                model="model-1",
                input="Hi",
                headers={"X-Trace": "t-2"},
                retry_policy=once,
            ):
                pass  # pragma: no cover

    assert [r.headers["X-Trace"] for r in recorder.requests] == ["t-1", "t-2"]


async def test_a_call_timeout_limits_the_silence_of_a_stream(
    case: Case,
    client: Client,
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

    async with client, client.generation.stream(REQUEST, timeout=0.3) as s:
        with pytest.raises(APIStreamInterruptedError) as caught:
            async for _ in s:
                pass

    resume.set()

    assert isinstance(caught.value.__cause__, APITimeoutError)


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
