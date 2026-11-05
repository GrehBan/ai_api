"""The user API of generation: keyword arguments and session options.

``client.generation.create``/``stream`` take the fields of
``GenerationRequest`` as keyword arguments, as the providers' SDKs do;
provider clients take the arguments of their ``AiohttpSession``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, get_overloads, get_type_hints

import pytest
from pydantic import ValidationError

from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.openai import OpenAIClient
from aiollm.providers.openai.client import session_factory as openai_factory
from aiollm.resources import GenerationResource, IGenerationResource
from aiollm.session.aiohttp import AiohttpSession, AiohttpSessionOptions
from aiollm.session.base import BaseSession
from aiollm.session.protocols import CallOptions, ISession
from aiollm.session.retry import RetryPolicy
from aiollm.types.requests import GenerationRequest
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

    # The concrete resource adds provider_options to both overloads; the
    # protocol, for code that works with any provider, has none.
    own = ["provider_options"] if owner is GenerationResource else []

    assert len(overloads) == 2
    assert keyword == [*GenerationRequest.model_fields, *own, *own]
    signature = inspect.signature(overloads[0])
    assert {
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        and name != "self"
    } == required


@pytest.mark.parametrize(
    "method",
    [ISession.send_request, ISession.stream, BaseSession.send_request],
    ids=["protocol-send", "protocol-stream", "session-send"],
)
def test_call_options_match_the_session_arguments(method: Any) -> None:
    # Compared as written, since both modules defer their annotations;
    # a TypedDict keeps each one as a ForwardRef of that text.
    options = {
        name: hint.__forward_arg__
        for name, hint in CallOptions.__annotations__.items()
    }
    keyword = {
        name: parameter.annotation
        for name, parameter in inspect.signature(method).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert options == keyword


@pytest.mark.parametrize(
    "owner",
    [GenerationResource, IGenerationResource],
    ids=["resource", "protocol"],
)
@pytest.mark.parametrize("method", ["create", "stream"])
def test_generation_takes_the_call_options(
    owner: type[Any], method: str
) -> None:
    parameters = inspect.signature(getattr(owner, method)).parameters
    keyword = {
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    own = {"provider_options"} if owner is GenerationResource else set()

    assert keyword == CallOptions.__optional_keys__ | own


def test_call_options_are_closed() -> None:
    # Only a closed TypedDict lets the resource's provider_options sit
    # beside **options and still match IGenerationResource in pyright.
    assert getattr(CallOptions, "__closed__", None) is True
    assert "provider_options" not in CallOptions.__optional_keys__


def test_no_request_field_is_named_like_a_call_option() -> None:
    # The options and the fields share the keyword arguments of create()
    # and stream(); a field named like an option could not be given.
    names = set(GenerationRequest.model_fields) | {
        field.alias
        for field in GenerationRequest.model_fields.values()
        if field.alias
    }

    assert names.isdisjoint(CallOptions.__optional_keys__)
    assert "provider_options" not in names


def test_session_options_match_the_session_arguments() -> None:
    hints = get_type_hints(AiohttpSession.__init__)
    keyword = {
        name: hints[name]
        for name, parameter in inspect.signature(
            AiohttpSession.__init__
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    assert get_type_hints(AiohttpSessionOptions) == keyword


@pytest.mark.parametrize(
    "build",
    [OpenAIClient, AnthropicClient],
    ids=["openai", "anthropic"],
)
async def test_clients_pass_session_arguments_on(build: Any) -> None:
    policy = RetryPolicy(max_retry_after=5.0, deadline=None)
    client = build(TEST_TOKEN, retry_policy=policy)

    async with client:
        session = client.session
        assert isinstance(session, AiohttpSession)
        assert session.retry_policy is policy


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
