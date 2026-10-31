"""The provider methods and types follow the providers' own schemas.

The schemas and examples in ``tests/providers/contracts/`` are a
snapshot of the offline documentation dump (``manifest.json`` says
which SDK versions and documents they come from; ``update.py`` rebuilds
them). The request schemas are closed here: an object that lists its
properties takes no others, so a misspelled or removed field fails
instead of passing as an unknown extra.

A snapshot does not follow the provider by itself: these tests catch
the library drifting from the snapshot, and the provider drifting from
the library only once the snapshot is refreshed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, TypeAdapter

from ai_api.methods import BaseAPIMethod, BaseStreamMethod
from ai_api.providers.anthropic.methods import CreateMessage, StreamMessage
from ai_api.providers.anthropic.types import Message, MessageStreamEvent
from ai_api.providers.openai.methods import CreateResponse, StreamResponse
from ai_api.providers.openai.types import Response, ResponseStreamEvent
from ai_api.response.parse.stream import StreamResponseParser
from ai_api.session.aiohttp import AiohttpSession
from ai_api.types.matchers import AnyOf, EventTypes
from tests.conftest import Recorder, Reply
from tests.providers import test_anthropic_messages as anthropic
from tests.providers import test_openai_responses as openai
from tests.providers.contracts.update import event_tags, type_tags

CONTRACTS = Path(__file__).parent / "contracts"


@dataclass(frozen=True)
class Case:
    """One endpoint and the library's side of it.

    Attributes:
        name: Snapshot folder, ``<provider>/<endpoint>``.
        create: The buffered method.
        stream: The streamed method.
        request: Every request field, set to a valid value.
        response: The response model.
        events: The stream event union.
        documented: Event types the provider documents outside the SDK
            schema, with where.
        examples: The hand-written payloads of the other tests:
            responses and stream events.
    """

    name: str
    create: Callable[..., BaseAPIMethod[Any]]
    stream: Callable[..., BaseStreamMethod[Any]]
    request: dict[str, Any]
    response: type[BaseModel]
    events: TypeAdapter[Any]
    documented: dict[str, str]
    examples: tuple[list[dict[str, Any]], list[dict[str, Any]]]


OPENAI = Case(
    name="openai/responses",
    create=CreateResponse,
    stream=StreamResponse,
    request={
        "model": "gpt-5",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        ],
        "instructions": "Be brief",
        "max_output_tokens": 64,
        "metadata": {"user": "u-1"},
        "previous_response_id": "resp_1",
        "store": False,
        "temperature": 0.5,
        "top_p": 0.9,
    },
    response=Response,
    events=TypeAdapter(ResponseStreamEvent),
    documented={},
    examples=([openai.RESPONSE], openai.STREAM),
)

ANTHROPIC = Case(
    name="anthropic/messages",
    create=CreateMessage,
    stream=StreamMessage,
    request={
        "model": "claude-opus-5-5",
        "messages": [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
            },
            {"role": "user", "content": "Bye"},
        ],
        "max_tokens": 64,
        "system": [{"type": "text", "text": "Be brief"}],
        "metadata": {"user_id": "u-1"},
        "stop_sequences": ["END"],
    },
    response=Message,
    events=TypeAdapter(MessageStreamEvent),
    documented={
        "ping": "https://platform.claude.com/docs/en/build-with-claude/streaming",
        "error": "https://platform.claude.com/docs/en/build-with-claude/streaming",
    },
    examples=([anthropic.MESSAGE], anthropic.STREAM),
)

CASES = [pytest.param(case, id=case.name) for case in (OPENAI, ANTHROPIC)]


@cache
def load(name: str, file: str) -> Any:
    """Read one snapshot file.

    Args:
        name: Snapshot folder, ``<provider>/<endpoint>``.
        file: File name in it.

    Returns:
        The decoded JSON.
    """
    path = CONTRACTS / name / file
    return json.loads(path.read_text(encoding="utf-8"))


def closed(schema: Any) -> Any:
    """Close every object of a schema that lists its properties.

    The SDK-derived request schemas leave ``additionalProperties`` out,
    which JSON Schema reads as "anything goes", while the APIs reject
    unknown fields.

    Args:
        schema: A schema or a node of one.

    Returns:
        A copy with ``additionalProperties: false`` wherever an object
        has ``properties`` and does not say otherwise.
    """
    if isinstance(schema, list):
        return [closed(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: closed(value) for key, value in schema.items()}
    if "properties" in result and "additionalProperties" not in result:
        result["additionalProperties"] = False
    return result


def validator(case: Case, file: str) -> Draft202012Validator:
    """Build a validator of a snapshot schema.

    Args:
        case: The endpoint.
        file: Schema file name.

    Returns:
        The validator; the request schema is closed.
    """
    schema = load(case.name, file)
    if file == "request.schema.json":
        schema = closed(schema)
    return Draft202012Validator(schema)


def errors(validator: Draft202012Validator, value: Any) -> list[str]:
    """List the schema errors of a value with their paths.

    Args:
        validator: The schema's validator.
        value: The value.

    Returns:
        One ``path: message`` line per error.
    """
    return [
        f"{error.json_path}: {error.message[:200]}"
        for error in validator.iter_errors(value)
    ]


def examples(case: Case, file: str) -> list[Any]:
    """Return the documentation examples of a snapshot.

    Args:
        case: The endpoint.
        file: ``responses.json`` or ``events.json``.

    Returns:
        The example bodies.
    """
    return [item["body"] for item in load(case.name, file)]


def frame(data: dict[str, Any]) -> bytes:
    """Encode one SSE frame.

    Args:
        data: The event.

    Returns:
        ``event:`` and ``data:`` lines of the event.
    """
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def test_the_snapshot_has_examples() -> None:
    manifest = json.loads((CONTRACTS / "manifest.json").read_text("utf-8"))

    for case in (OPENAI, ANTHROPIC):
        entry = manifest["endpoints"][case.name]
        assert examples(case, "responses.json")
        assert examples(case, "events.json")
        assert entry["responses"] == len(examples(case, "responses.json"))
        assert entry["events"] == len(examples(case, "events.json"))


@pytest.mark.parametrize("case", CASES)
def test_the_full_request_sets_every_field(case: Case) -> None:
    method = case.create(**case.request)

    assert method.model_fields_set == set(type(method).model_fields) - {
        "stream"
    }


@pytest.mark.parametrize("case", CASES)
async def test_requests_pass_the_request_schema(
    case: Case,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    schema = validator(case, "request.schema.json")
    recorder.reply(
        Reply(json=examples(case, "responses.json")[0]),
        Reply(
            chunks=[frame(event) for event in examples(case, "events.json")],
            content_type="text/event-stream",
        ),
    )

    await session.send_request(case.create(**case.request))
    buffered = json.loads(recorder.last.body)
    # Only the request matters: the examples are not one stream.
    async with session.stream(case.stream(**case.request)):
        pass
    streamed = json.loads(recorder.last.body)

    assert errors(schema, buffered) == []
    assert errors(schema, streamed) == []
    assert streamed == buffered | {"stream": True}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "field",
    [
        pytest.param({"modle": "x"}, id="misspelled"),
        pytest.param({"top_k": 5}, id="unknown"),
    ],
)
def test_the_closed_request_schema_rejects_unknown_fields(
    case: Case,
    field: dict[str, Any],
) -> None:
    body = json.loads(case.create(**case.request).model_dump_json())

    assert errors(validator(case, "request.schema.json"), body | field)


def test_anthropic_takes_no_sampling_parameters() -> None:
    schema = validator(ANTHROPIC, "request.schema.json")
    body = ANTHROPIC.request

    for name in ("temperature", "top_p", "top_k"):
        assert errors(schema, body | {name: 0.5}), name


@pytest.mark.parametrize("case", CASES)
def test_documented_responses_parse(case: Case) -> None:
    for body in examples(case, "responses.json"):
        case.response.model_validate(body)


@pytest.mark.parametrize("case", CASES)
def test_documented_events_parse(case: Case) -> None:
    for body in examples(case, "events.json"):
        event = case.events.validate_python(body)

        assert event.type == body["type"]


@pytest.mark.parametrize("case", CASES)
def test_known_types_exist_upstream(case: Case) -> None:
    stream = load(case.name, "stream.schema.json")
    upstream = event_tags(stream) | case.documented.keys()
    nested = type_tags(load(case.name, "response.schema.json"))
    nested |= type_tags(stream)

    ours = event_tags(case.events.json_schema())

    assert ours - upstream == set()
    assert type_tags(case.response.model_json_schema()) - nested == set()


def matched_types(matcher: object) -> set[str]:
    """Collect the event types a matcher names.

    Args:
        matcher: A matcher, possibly combined with ``|``.

    Returns:
        The ``types`` of every ``EventTypes`` in it.
    """
    if isinstance(matcher, AnyOf):
        return set().union(*map(matched_types, matcher.matchers))
    if isinstance(matcher, EventTypes):
        return set(matcher.types)
    return set()


@pytest.mark.parametrize("case", CASES)
def test_declared_event_types_exist_upstream(case: Case) -> None:
    # A misspelled terminal would never match, and every complete stream
    # would end in APIStreamInterruptedError.
    upstream = event_tags(load(case.name, "stream.schema.json"))
    upstream |= case.documented.keys()
    method = case.stream(**case.request)
    parser = method.stream_parser
    assert isinstance(parser, StreamResponseParser)

    terminal = matched_types(method.stream_terminal)
    errors = matched_types(parser.error_events)

    assert terminal
    assert terminal - upstream == set()
    assert errors - upstream == set()


@pytest.mark.parametrize("case", CASES)
def test_hand_written_examples_pass_the_schemas(case: Case) -> None:
    responses, events = case.examples
    response_schema = validator(case, "response.schema.json")
    event_schema = validator(case, "stream.schema.json")

    for body in responses:
        assert errors(response_schema, body) == []
    for body in events:
        if body["type"] in case.documented:
            continue
        assert errors(event_schema, body) == [], body["type"]
