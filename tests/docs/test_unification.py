"""``docs/unification.md`` matches the shared types and their errors.

Every field of the shared types has a row in the document, every shared
event is listed, and every case of its error table raises the error it
names, with the provider and field it names.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

import ai_api
from ai_api.enums import Provider
from ai_api.exceptions import RequestValidationError
from ai_api.providers.anthropic.adapters import AnthropicGenerationAdapter
from ai_api.providers.anthropic.types import Message
from ai_api.providers.openai import OpenAIClient
from ai_api.providers.openai.adapters import OpenAIGenerationAdapter
from ai_api.providers.openai.client import session_factory
from ai_api.providers.openai.types import Response
from ai_api.types.requests import GenerationRequest, Turn
from ai_api.types.responses import (
    CompletedEvent,
    GenerationEvent,
    GenerationResponse,
    TokenUsage,
)
from tests.conftest import TEST_TOKEN, LocalAPI, Recorder, Reply
from tests.providers import test_anthropic_messages as anthropic
from tests.providers import test_openai_responses as openai

DOCUMENT = Path(__file__).parents[2] / "docs" / "unification.md"
TEXT = DOCUMENT.read_text("utf-8")

ROWS = [
    [cell.strip() for cell in line.strip().strip("|").split("|")]
    for line in TEXT.splitlines()
    if line.startswith("| `")
]
"""The cells of every table row that starts with a code name."""

ERRORS = {
    row[0].strip("`"): row[1:]
    for row in ROWS
    if re.fullmatch(r"`[a-z0-9-]+`", row[0])
}
"""Each case of the error table: provider, shared field and error."""

REQUEST = GenerationRequest(model="m", input="Hi")

TURNS = [Turn.user("Hi"), Turn.assistant("Hello")]

PROVIDERS = {Provider.OPENAI: "OpenAI", Provider.ANTHROPIC: "Anthropic"}
"""How the error table names each provider."""


def documented(owner: str) -> set[str]:
    """Collect the fields of a shared type that have a row.

    Args:
        owner: The type's name, such as ``"GenerationRequest"``.

    Returns:
        The field names of the rows ``owner.field``.
    """
    pattern = re.compile(rf"`{owner}\.(\w+)`")
    return {
        match[1] for row in ROWS if (match := pattern.fullmatch(row[0]))
    }


@pytest.mark.parametrize(
    "model",
    [GenerationRequest, GenerationResponse, TokenUsage, CompletedEvent],
    ids=lambda model: model.__name__,
)
def test_every_shared_field_has_a_row(model: type[BaseModel]) -> None:
    assert documented(model.__name__) == set(model.model_fields)


STOP_FIELDS = ("incomplete_reason", "raw_incomplete_reason", "refusal")

CONVERSIONS = {
    "input_conversion": {"GenerationRequest.input"},
    "usage_conversion": {
        "GenerationResponse.usage",
        "CompletedEvent.usage",
        *(f"TokenUsage.{name}" for name in TokenUsage.model_fields),
    },
    "stop_conversion": {
        f"{owner}.{name}"
        for owner in ("GenerationResponse", "CompletedEvent")
        for name in STOP_FIELDS
    },
}
"""The shared fields each conversion of an adapter fills."""

TAKEN_AS_IS = {
    "GenerationRequest.model",
    "GenerationRequest.instructions",
    "GenerationRequest.max_output_tokens",
    "GenerationRequest.metadata",
    "GenerationResponse.id",
    "GenerationResponse.provider",
    "GenerationResponse.model",
    "GenerationResponse.text",
    "GenerationResponse.raw",
    "CompletedEvent.type",
    "CompletedEvent.id",
    "CompletedEvent.provider",
    "CompletedEvent.model",
    "CompletedEvent.text",
    "CompletedEvent.raw",
}
"""The shared fields an adapter copies or sets without a conversion."""


def test_every_documented_field_is_converted_or_taken_as_is() -> None:
    fields = {
        f"{model.__name__}.{name}"
        for model in (
            GenerationRequest,
            GenerationResponse,
            TokenUsage,
            CompletedEvent,
        )
        for name in documented(model.__name__)
    }
    converted = set[str]().union(*CONVERSIONS.values())

    assert converted.isdisjoint(TAKEN_AS_IS)
    assert fields == converted | TAKEN_AS_IS


@pytest.mark.parametrize(
    "adapter",
    [OpenAIGenerationAdapter(), AnthropicGenerationAdapter()],
    ids=["openai", "anthropic"],
)
def test_every_provider_has_every_conversion(adapter: object) -> None:
    for name in CONVERSIONS:
        assert callable(getattr(adapter, name))


def test_every_shared_event_has_a_row() -> None:
    union = get_args(GenerationEvent.__value__)[0]
    events = {
        event.__pydantic_generic_metadata__["origin"].__name__
        for event in get_args(union)
    }

    assert events <= {row[0].strip("`") for row in ROWS}


def test_every_named_error_has_a_case() -> None:
    named = set(re.findall(r"`(\w+Error)`", TEXT))

    assert named == {error.strip("`") for _, _, error in ERRORS.values()}


def raises(case: str) -> type[BaseException]:
    """Look up the error class a case names.

    Args:
        case: The case.

    Returns:
        The class, from ``ai_api``, pydantic or the builtins.
    """
    name = ERRORS[case][2].strip("`")
    if name == "ValidationError":
        return ValidationError
    if name == "ValueError":
        return ValueError
    error: type[BaseException] = getattr(ai_api, name)
    return error


def openai_method(**changes: Any) -> Callable[[], object]:
    """Build OpenAI's method of a changed request."""
    request = REQUEST.model_copy(update=changes)
    return lambda: OpenAIGenerationAdapter().to_method(request)


def anthropic_method(**changes: Any) -> Callable[[], object]:
    """Build Anthropic's method of a changed request."""
    request = REQUEST.model_copy(update=changes)
    return lambda: AnthropicGenerationAdapter().to_method(request)


def text_less_turn() -> object:
    """Turn a response without text into a turn."""
    response = GenerationResponse[Any](
        provider=Provider.OPENAI, model="m", raw={}
    )
    return response.to_turn()


SYNC_CASES: dict[str, Callable[[], object]] = {
    "empty-turn-list": lambda: GenerationRequest(model="m", input=[]),
    "zero-max-output-tokens": lambda: GenerationRequest(
        model="m", input="Hi", max_output_tokens=0
    ),
    "unknown-request-field": lambda: GenerationRequest.model_validate(
        {"model": "m", "input": "Hi", "temperature": 1}
    ),
    "openai-last-assistant-turn": openai_method(input=TURNS),
    "anthropic-last-assistant-turn": anthropic_method(input=TURNS),
    "openai-max-output-tokens-below-16": openai_method(max_output_tokens=15),
    "openai-too-many-metadata-pairs": openai_method(
        metadata={f"k{index}": "v" for index in range(17)}
    ),
    "openai-too-long-metadata-key": openai_method(metadata={"k" * 65: "v"}),
    "anthropic-metadata-key-other-than-user-id": anthropic_method(
        metadata={"team": "t"}
    ),
    "anthropic-too-long-user-id": anthropic_method(
        metadata={"user_id": "u" * 513}
    ),
    "openai-unfinished-response": lambda: (
        OpenAIGenerationAdapter().to_response(
            Response.model_validate(openai.RESPONSE | {"status": "cancelled"}),
            REQUEST,
        )
    ),
    "anthropic-message-without-stop-reason": lambda: (
        AnthropicGenerationAdapter().to_response(
            Message.model_validate(anthropic.MESSAGE | {"stop_reason": None}),
            REQUEST,
        )
    ),
    "turn-from-response-without-text": text_less_turn,
}
"""Cases raised without sending anything."""


async def failed_response(recorder: Recorder) -> None:
    """Create a response that OpenAI reports as failed."""
    failed = openai.RESPONSE | {
        "status": "failed",
        "error": {"code": "server_error", "message": "Boom"},
    }
    recorder.reply(Reply(json=failed))
    async with OpenAIClient(TEST_TOKEN) as client:
        await client.generation.create(REQUEST)


async def stream_without_completion(recorder: Recorder) -> None:
    """Stream a generation whose body ends before its last event."""
    recorder.reply(
        Reply(
            chunks=[
                f"data: {json.dumps(event)}\n\n".encode()
                for event in openai.STREAM[:-1]
            ],
            content_type="text/event-stream",
        )
    )
    async with (
        OpenAIClient(TEST_TOKEN) as client,
        client.generation.stream(REQUEST) as events,
    ):
        async for _ in events:
            pass


ASYNC_CASES: dict[str, Callable[[Recorder], Awaitable[None]]] = {
    "openai-failed-response": failed_response,
    "stream-without-completion": stream_without_completion,
}
"""Cases raised by a request sent to the local server."""


def test_every_case_is_checked() -> None:
    assert set(ERRORS) == SYNC_CASES.keys() | ASYNC_CASES.keys()


def check(case: str, error: BaseException) -> None:
    """Compare a raised error with the provider and field of its row.

    Args:
        case: The case.
        error: What it raised.
    """
    provider, field, _ = ERRORS[case]
    if not isinstance(error, RequestValidationError):
        return
    assert provider == PROVIDERS[error.provider]
    assert field == f"`{error.field}`"


@pytest.mark.parametrize("case", sorted(SYNC_CASES))
def test_documented_request_errors_are_raised(case: str) -> None:
    with pytest.raises(raises(case)) as caught:
        SYNC_CASES[case]()

    check(case, caught.value)


@pytest.mark.parametrize("case", sorted(ASYNC_CASES))
async def test_documented_response_errors_are_raised(
    case: str,
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_factory, "api", api)

    with pytest.raises(raises(case)):
        await ASYNC_CASES[case](recorder)
