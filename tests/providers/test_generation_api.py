"""The user API of generation: keyword arguments and session options.

``client.generation.create``/``stream`` take the fields of
``GenerationRequest`` as keyword arguments, as the providers' SDKs do;
provider clients take the arguments of their ``AiohttpSession``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, get_overloads, get_type_hints

import pytest
from pydantic import ValidationError

from ai_api.providers.anthropic import AnthropicClient
from ai_api.providers.openai import OpenAIClient
from ai_api.providers.openai.client import session_factory as openai_factory
from ai_api.resources import GenerationResource, IGenerationResource
from ai_api.session.aiohttp import AiohttpSession, AiohttpSessionOptions
from ai_api.types.requests import GenerationRequest
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply
from tests.providers.test_openai_responses import RESPONSE, STREAM, frame

MODEL = "gpt-6-astra"


@pytest.fixture
def client(
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> OpenAIClient:
    """``OpenAIClient`` built from a token, sent to the local server."""
    monkeypatch.setattr(openai_factory, "api", api)
    return OpenAIClient(TEST_TOKEN)


async def test_create_and_stream_take_keyword_arguments(
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
        created = await client.generation.create(
            model=MODEL, input="Hi", max_output_tokens=16
        )
        async with client.generation.stream(
            model=MODEL, input="Hi", max_output_tokens=16
        ) as events:
            completed = await events.final()

    assert created.model == RESPONSE["model"]
    assert completed.model == RESPONSE["model"]
    bodies = [json.loads(request.body) for request in recorder.requests]
    assert all(
        body["input"] == "Hi" and body["max_output_tokens"] == 16
        for body in bodies
    )


async def test_request_and_fields_together_raise(
    client: OpenAIClient,
) -> None:
    request = GenerationRequest(model=MODEL, input="Hi")

    with pytest.raises(TypeError, match="not both.*max_output_tokens"):
        await client.generation.create(request, max_output_tokens=32)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="not both"):
        client.generation.stream(request, model="other")  # type: ignore[call-overload]


async def test_misspelled_fields_are_rejected(client: OpenAIClient) -> None:
    with pytest.raises(ValidationError, match="inputs"):
        await client.generation.create(model="m", inputs="Hi")  # type: ignore[call-overload]
    with pytest.raises(ValidationError, match="input"):
        client.generation.stream(model="m")  # type: ignore[call-overload]


@pytest.mark.parametrize(
    "owner",
    [GenerationResource, IGenerationResource],
    ids=["resource", "protocol"],
)
@pytest.mark.parametrize("method", ["create", "stream"])
def test_keyword_overloads_name_the_request_fields(
    owner: type[Any],
    method: str,
) -> None:
    overloads = get_overloads(getattr(owner, method))
    keyword = [
        name
        for overload in overloads
        for name, parameter in inspect.signature(overload).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    required = {
        name
        for name, field in GenerationRequest.model_fields.items()
        if field.is_required()
    }

    assert len(overloads) == 2
    assert keyword == list(GenerationRequest.model_fields)
    signature = inspect.signature(overloads[0])
    assert {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty and name != "self"
    } == required


def test_session_options_match_the_session_arguments() -> None:
    # The autouse no_close_delay fixture patches the module's asyncio.
    names = {"asyncio": asyncio}
    hints = get_type_hints(AiohttpSession.__init__, localns=names)
    keyword = {
        name: hints[name]
        for name, parameter in inspect.signature(
            AiohttpSession.__init__
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert get_type_hints(AiohttpSessionOptions, localns=names) == keyword


@pytest.mark.parametrize(
    "build",
    [OpenAIClient, AnthropicClient],
    ids=["openai", "anthropic"],
)
async def test_clients_pass_session_arguments_on(build: Any) -> None:
    client = build(TEST_TOKEN, max_retry_after=5.0, retry_deadline=None)

    async with client:
        session = client.session
        assert isinstance(session, AiohttpSession)
        assert session.max_retry_after == 5.0
        assert session.retry_deadline is None


@pytest.mark.parametrize(
    "build",
    [OpenAIClient, AnthropicClient],
    ids=["openai", "anthropic"],
)
def test_session_arguments_with_a_session_raise(
    build: Any,
    session: AiohttpSession,
) -> None:
    with pytest.raises(ValueError, match="max_body_size, trust_env"):
        build(session=session, trust_env=True, max_body_size=1)
