"""Block L: the README examples run against a local server."""

import ast
import json
import mimetypes
import re
import sys
import types
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from ai_api import (
    APIConnectionError,
    APIRedirectError,
    APIStreamInterruptedError,
    BadRequestError,
    BaseErrorResolver,
    IncompleteReason,
    QuotaExceededError,
    RateLimitError,
    ResponseFormat,
    ServiceUnavailableError,
)
from ai_api.providers.anthropic import AnthropicClient
from ai_api.providers.anthropic.client import (
    session_factory as anthropic_session_factory,
)
from ai_api.providers.openai.client import session_factory
from ai_api.session.aiohttp import AiohttpSession
from ai_api.types.base import BaseAPIType
from ai_api.types.requests import GenerationRequest
from tests.conftest import LocalAPI, Recorder, Reply

README = Path(__file__).parents[2] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text("utf-8"), re.S)

(
    QUICK_START,
    INHERITANCE,
    QUERY,
    ALIASES,
    PATH_PARAMS,
    UPLOADS,
    STREAMING,
    EVENTS,
    RESTART,
    HANDLING,
    RESOLVER,
    AUTH,
    CALL_OPTIONS,
    CLIENT,
    OPENAI_RESPONSES,
    ANTHROPIC_MESSAGES,
    SHARED_GENERATION,
    RESOURCES,
) = range(18)

PNG = b"\x89PNG\r\n\x1a\n"


class ModelList(BaseAPIType):
    data: list[str] = []


class Model(BaseAPIType):
    id: str


class Transcription(BaseAPIType):
    text: str


async def chunks() -> AsyncIterator[bytes]:
    yield b"RIFF"
    yield b"WAVE"


class Examples:
    """Runs README blocks in one module, like a reader pasting them."""

    def __init__(self, module: types.ModuleType) -> None:
        self.module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    async def run(self, *indexes: int) -> list[Any]:
        """Run blocks in order; return values of top-level expressions."""
        values: list[Any] = []

        for index in indexes:
            for statement in ast.parse(BLOCKS[index]).body:
                values.append(await self._run(statement))

        return values

    async def _run(self, statement: ast.stmt) -> Any:
        if is_asyncio_run(statement):
            # The caller awaits ``main()`` itself, inside the test's loop.
            return None

        is_expression = isinstance(statement, ast.Expr)
        tree: ast.Module | ast.Expression = (
            ast.Expression(statement.value)
            if isinstance(statement, ast.Expr)
            else ast.Module([statement], type_ignores=[])
        )
        code = compile(
            tree,
            str(README),
            "eval" if is_expression else "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        result = eval(code, self.module.__dict__)

        if isinstance(result, types.CoroutineType):
            result = await result

        return result


def is_asyncio_run(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == "asyncio.run"
    )


@pytest.fixture
def examples(
    session: AiohttpSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Examples]:
    # Pydantic needs the module of the classes in sys.modules.
    module = types.ModuleType("readme_examples")
    module.__dict__.update(
        ModelList=ModelList,
        Model=Model,
        Transcription=Transcription,
        data=PNG,
        chunks=chunks,
        session=session,
    )
    sys.modules[module.__name__] = module
    monkeypatch.chdir(tmp_path)

    try:
        yield Examples(module)
    finally:
        del sys.modules[module.__name__]


def test_every_example_is_tested() -> None:
    # A new ```python block needs a test here.
    assert len(BLOCKS) == 18


def test_quick_start_ends_with_asyncio_run() -> None:
    last = ast.parse(BLOCKS[QUICK_START]).body[-1]

    assert ast.unparse(last) == "asyncio.run(main())"


async def test_quick_start(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await examples.run(QUICK_START)
    examples.API.base_url = api.base_url
    recorder.reply(Reply(json={"id": "c-1", "text": "Hello back!"}))

    await examples.main()

    request = recorder.last
    assert request.method == "POST"
    assert request.raw_path == "/v1/completions"
    assert request.headers["Authorization"] == "Bearer sk-..."
    assert request.body == (
        b'{"model":"example-1",'
        b'"messages":[{"role":"user","content":"Hello!"}]}'
    )
    assert capsys.readouterr().out == "Hello back!\n"


async def test_inherited_attributes(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, INHERITANCE)
    recorder.reply(Reply(json={"id": "s-1", "text": "Short."}))

    result = await session.send_request(examples.Summarize(text="Long."))

    assert result.text == "Short."
    assert recorder.last.method == "POST"
    assert recorder.last.raw_path == "/v1/summaries"
    assert recorder.last.body == b'{"text":"Long."}'


async def test_query_parameters(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, QUERY)
    recorder.reply(Reply(json={"data": ["example-1"]}))

    result = await session.send_request(examples.ListModels(limit=5))

    assert result == ModelList(data=["example-1"])
    assert recorder.last.raw_path == "/v1/models"
    assert recorder.last.query_string == "limit=5"


async def test_aliased_fields(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, ALIASES)
    recorder.reply(Reply(json={"data": []}))

    await session.send_request(examples.ListFiles(page_size=10))

    assert recorder.last.query_string == "pageSize=10"


async def test_path_parameters(examples: Examples) -> None:
    *_, request_path = await examples.run(QUICK_START, PATH_PARAMS)

    assert request_path == "/models/example-1"


async def test_uploads(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
    tmp_path: Path,
) -> None:
    (tmp_path / "voice.mp3").write_bytes(b"ID3audio")
    *_, from_path, from_bytes, from_stream = await examples.run(
        QUICK_START,
        UPLOADS,
    )
    recorder.reply(Reply(json={"text": "a"}), Reply(json={"text": "b"}))

    for method in (from_path, from_bytes):
        await session.send_request(method)

    parts = [request.parts[0] for request in recorder.requests]
    assert [
        (part.filename, part.content_type, part.data) for part in parts
    ] == [
        ("voice.mp3", "audio/mpeg", b"ID3audio"),
        ("image.png", "image/png", PNG),
    ]
    assert from_stream.file.filename == "s.wav"
    assert (
        from_stream.file.content_type == mimetypes.guess_file_type("s.wav")[0]
    )


async def test_stream_file_upload_is_not_retried(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    *_, from_stream = await examples.run(QUICK_START, UPLOADS)
    recorder.reply(Reply(status=503), Reply(json={"text": "late"}))

    with pytest.raises(ServiceUnavailableError):
        await session.send_request(from_stream)

    assert len(recorder.requests) == 1
    assert recorder.last.parts[0].data == b"RIFFWAVE"


async def test_streaming(
    examples: Examples,
    recorder: Recorder,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder.reply(
        Reply(chunks=[b"binary ", b"content"]),
        Reply(
            chunks=[b"Hi ", b"there ", "😀".encode()[:2], "😀".encode()[2:]],
            content_type="text/plain; charset=utf-8",
        ),
    )

    await examples.run(QUICK_START, STREAMING)

    download, generate = recorder.requests
    assert download.raw_path == "/v1/files/content"
    assert download.query_string == "file_id=f-1"
    assert (tmp_path / "result.bin").read_bytes() == b"binary content"
    assert generate.body == b'{"prompt":"Hi"}'
    assert capsys.readouterr().out == "Hi there 😀"


async def test_events(
    examples: Examples,
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The streaming block runs first: it imports BaseStreamMethod.
    recorder.reply(
        Reply(chunks=[b"file"]),
        Reply(chunks=[b"Text. "], content_type="text/plain"),
        Reply(
            chunks=[
                b'event: text_delta\ndata: {"type": "text_delta", ',
                b'"text": "Hello"}\n\nevent: text_delta\ndata: ',
                b'{"type": "text_delta", "text": " there"}\n\n',
                b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
            ],
            content_type="text/event-stream",
        )
    )

    await examples.run(QUICK_START, STREAMING, EVENTS)

    assert recorder.last.raw_path == "/v1/messages"
    assert recorder.last.body == b'{"prompt":"Hi","stream":true}'
    assert capsys.readouterr().out == "Text. Hello there"
    assert examples.RawEvents.__response_format__ is ResponseFormat.SSE


def stream_replies() -> list[Reply]:
    """Replies to the requests the streaming block itself sends."""
    return [
        Reply(chunks=[b"file"]),
        Reply(chunks=[b"Text. "], content_type="text/plain"),
    ]


async def test_restart_after_a_broken_stream(
    examples: Examples,
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder.reply(
        *stream_replies(),
        Reply(chunks=[b"Hel"], content_type="text/plain", abort=True),
        Reply(chunks=[b"Hello", b"!"], content_type="text/plain"),
    )
    await examples.run(QUICK_START, STREAMING, RESTART)
    capsys.readouterr()

    text = await examples.generate_text("Hi")

    assert text == "Hello!"
    assert len(recorder.requests) == 4


async def test_restart_gives_up_after_the_last_attempt(
    examples: Examples,
    recorder: Recorder,
) -> None:
    recorder.reply(
        *stream_replies(),
        Reply(chunks=[b"Hel"], content_type="text/plain", abort=True),
    )
    await examples.run(QUICK_START, STREAMING, RESTART)

    with pytest.raises(APIStreamInterruptedError) as info:
        await examples.generate_text("Hi", attempts=1)

    assert isinstance(info.value.__cause__, APIConnectionError)

    with pytest.raises(ValueError, match="at least 1"):
        await examples.generate_text("Hi", attempts=0)


async def test_error_handling(
    examples: Examples,
    recorder: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await examples.run(QUICK_START, HANDLING)
    recorder.reply(
        Reply(status=429, headers={"X-Request-Id": "req-1"}),
        Reply(status=400, json={"error": {"code": "bad_model"}}),
        Reply(json={"id": "c-1", "text": "Fine."}),
    )

    assert await examples.ask("Hi") is None
    with pytest.raises(BadRequestError):
        await examples.ask("Hi")
    assert await examples.ask("Hi") == "Fine."

    assert capsys.readouterr().out == (
        "Rate limited, request req-1\n"
        "API error 400, code bad_model\n"
    )
    assert len(recorder.requests) == 3


async def test_error_handling_of_transport_errors(
    examples: Examples,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await examples.run(QUICK_START, HANDLING)

    async def fail(method: Any) -> Any:
        raise APIRedirectError("moved")

    monkeypatch.setattr(examples.session, "send_request", fail)

    with pytest.raises(APIRedirectError):
        await examples.ask("Hi")

    assert capsys.readouterr().out == "Network error, transient: False\n"


async def test_error_resolver(examples: Examples) -> None:
    await examples.run(QUICK_START, RESOLVER)
    resolver = examples.ExampleErrorResolver()

    quota = resolver.resolve(
        status=429,
        body={"error": {"code": "insufficient_quota"}},
        headers={},
    )
    rate = resolver.resolve(
        status=429,
        body={"error": {"code": "rate_limit_exceeded"}},
        headers={},
    )

    assert type(quota) is QuotaExceededError
    assert quota.transient is False
    assert quota.provider == "openai"
    assert type(rate) is RateLimitError


async def test_rotating_auth(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
) -> None:
    await examples.run(QUICK_START, QUERY, AUTH)
    auth = examples.RotatingAuth("sk-old")
    recorder.reply(Reply(json={"data": []}), Reply(json={"data": []}))

    async with AiohttpSession(api, auth, BaseErrorResolver()) as session:
        await session.send_request(examples.ListModels())
        auth.token = "sk-new"
        await session.send_request(examples.ListModels())

    assert [
        request.headers["Authorization"] for request in recorder.requests
    ] == ["Bearer sk-old", "Bearer sk-new"]


async def test_call_options(
    examples: Examples,
    recorder: Recorder,
) -> None:
    await examples.run(QUICK_START, CALL_OPTIONS)
    recorder.reply(Reply(json={"id": "c-1", "text": "Hi."}))

    result = await examples.complete("Hello!", "key-1")

    assert result == examples.Completion(id="c-1", text="Hi.")
    assert recorder.last.raw_path == "/v1/completions"
    assert recorder.last.headers["x-beta-features"] == "long-context"
    assert recorder.last.headers["Idempotency-Key"] == "key-1"


async def test_openai_client(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, QUERY, CLIENT)
    monkeypatch.setattr(session_factory, "api", api)
    recorder.reply(Reply(json={"data": ["gpt-x"]}))

    assert await examples.list_models() == ModelList(data=["gpt-x"])
    assert recorder.last.raw_path == "/v1/models"
    assert recorder.last.headers["Authorization"] == "Bearer sk-..."


async def test_openai_responses(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, CLIENT, OPENAI_RESPONSES)
    message = {
        "type": "message",
        "id": "msg_1",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hello!"}],
    }
    delta = {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 0,
    }
    response = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1741476777,
        "model": "gpt-6-astra",
        "output": [message],
    }
    completed = {
        "type": "response.completed",
        "sequence_number": 2,
        "response": response | {"status": "completed"},
    }
    recorder.reply(
        Reply(json=response),
        Reply(
            chunks=[
                f"data: {json.dumps(data)}\n\n".encode()
                for data in [
                    delta | {"delta": "He"},
                    delta | {"delta": "y"},
                    completed,
                ]
            ],
            content_type="text/event-stream",
        ),
    )

    assert await examples.ask(session) == "Hello!"
    assert await examples.ask_streamed(session) == "Hey"
    assert recorder.requests[-2].raw_path == "/v1/responses"
    assert json.loads(recorder.requests[-2].body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "instructions": "Be brief",
    }


async def test_anthropic_messages(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    await examples.run(QUICK_START, ANTHROPIC_MESSAGES)
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    delta = {"type": "content_block_delta", "index": 0}
    recorder.reply(
        Reply(json=message),
        Reply(
            chunks=[
                f"event: content_block_delta\ndata: {json.dumps(data)}\n\n"
                .encode()
                for data in [
                    delta | {"delta": {"type": "text_delta", "text": "He"}},
                    delta | {"delta": {"type": "signature_delta"}},
                    delta | {"delta": {"type": "text_delta", "text": "y"}},
                ]
            ]
            + [b'event: message_stop\ndata: {"type": "message_stop"}\n\n'],
            content_type="text/event-stream",
        ),
    )

    assert await examples.ask_claude(session) == "Hello!"
    assert await examples.ask_claude_streamed(session) == "Hey"
    first, second = recorder.requests
    assert first.raw_path == "/v1/messages"
    assert first.headers["anthropic-version"] == "2026-10-21"
    assert json.loads(first.body) == {
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "system": "Be brief",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert json.loads(second.body)["stream"] is True


async def test_shared_generation(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, CLIENT, SHARED_GENERATION)
    monkeypatch.setattr(session_factory, "api", api)
    delta = {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "sequence_number": 0,
    }
    completed = {
        "type": "response.completed",
        "sequence_number": 2,
        "response": {
            "id": "resp_2",
            "created_at": 1741476777,
            "status": "completed",
            "model": "gpt-6-astra",
        },
    }
    recorder.reply(
        Reply(
            json={
                "id": "resp_1",
                "created_at": 1741476777,
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "model": "gpt-6-astra",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "incomplete",
                        "content": [{"type": "output_text", "text": "Hel"}],
                    }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 64,
                    "total_tokens": 69,
                },
            }
        ),
        Reply(
            chunks=[
                f"data: {json.dumps(delta | {'delta': 'He'})}\n\n".encode(),
                f"data: {json.dumps(delta | {'delta': 'y'})}\n\n".encode(),
                f"data: {json.dumps(completed)}\n\n".encode(),
            ],
            content_type="text/event-stream",
        ),
    )

    async with examples.OpenAIClient("sk-...") as client:
        response = await examples.generate(client)
        text, completed = await examples.generate_streamed(client)

    assert text == "Hey"
    assert (completed.id, completed.model) == ("resp_2", "gpt-6-astra")

    assert (response.id, response.text) == ("resp_1", "Hel")
    assert response.incomplete_reason is IncompleteReason.MAX_TOKENS
    assert response.raw_incomplete_reason == "max_output_tokens"
    assert response.usage is not None
    assert response.usage.total_tokens == 69
    assert response.raw.status == "incomplete"
    first, second = recorder.requests
    assert first.raw_path == second.raw_path == "/v1/responses"
    assert json.loads(first.body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "instructions": "Be brief",
        "max_output_tokens": 64,
    }
    assert json.loads(second.body) == {
        "model": "gpt-6-astra",
        "input": "Hi",
        "stream": True,
    }


async def test_shared_generation_continues_a_conversation(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, CLIENT, SHARED_GENERATION)
    monkeypatch.setattr(session_factory, "api", api)
    answer = {
        "id": "resp_1",
        "created_at": 1741476777,
        "status": "completed",
        "model": "gpt-6-astra",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
    }
    recorder.reply(Reply(json=answer), Reply(json=answer))

    async with examples.OpenAIClient("sk-...") as client:
        response = await examples.chat(client)

    assert response.text == "Hello!"
    assert json.loads(recorder.last.body)["input"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Tell me more"},
    ]


async def test_shared_generation_works_with_anthropic(
    examples: Examples,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await examples.run(QUICK_START, CLIENT, SHARED_GENERATION)
    monkeypatch.setattr(anthropic_session_factory, "api", api)
    recorder.reply(
        Reply(
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "Hel"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 5, "output_tokens": 64},
            }
        )
    )

    async with AnthropicClient("sk-ant-...") as client:
        response = await examples.generate(client)

    assert response.text == "Hel"
    assert response.incomplete_reason is IncompleteReason.MAX_TOKENS
    assert json.loads(recorder.last.body) == {
        "model": "gpt-6-astra",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 64,
        "system": "Be brief",
    }


async def test_resources(
    examples: Examples,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    recorder.reply(
        *stream_replies(),
        Reply(
            chunks=[b'data: {"type": "message_stop"}\n\n'],
            content_type="text/event-stream",
        ),
    )
    await examples.run(QUICK_START, STREAMING, EVENTS, RESOURCES)
    client = examples.ExampleClient(session)
    request = GenerationRequest(model="example-1", input="Hi")
    recorder.reply(
        Reply(json={"id": "c-1", "text": "Hello!"}),
        Reply(
            chunks=[
                b'data: {"type": "text_delta", "text": "He"}\n\n',
                b'data: {"type": "text_delta", "text": "y"}\n\n',
                b'data: {"type": "message_stop"}\n\n',
            ],
            content_type="text/event-stream",
        ),
    )

    response = await client.generation.create(request)
    async with client.generation.stream(request) as events:
        received = [event async for event in events]

    assert response.text == "Hello!"
    assert response.id == "c-1"
    assert response.raw.text == "Hello!"
    assert [event.type for event in received] == [
        "text_delta",
        "text_delta",
        "completed",
    ]
    assert [getattr(event, "text", None) for event in received] == [
        "He",
        "y",
        None,
    ]
    assert recorder.requests[-2].body == (
        b'{"model":"example-1",'
        b'"messages":[{"role":"user","content":"Hi"}]}'
    )
