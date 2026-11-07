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

import importlib
import inspect
import json
import pkgutil
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

import pytest
from jsonschema import Draft202012Validator
from pydantic import MISSING, BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from aiollm.adapters import ICompositeAdapter
from aiollm.enums import Effort, ToolChoice
from aiollm.methods import BaseAPIMethod, BaseStreamMethod
from aiollm.providers.anthropic.adapters import AnthropicGenerationAdapter
from aiollm.providers.anthropic.adapters.conversions import message_text
from aiollm.providers.anthropic.methods import CreateMessage, StreamMessage
from aiollm.providers.anthropic.types import Message, MessageStreamEvent
from aiollm.providers.anthropic.types.options import (
    AnthropicGenerationOptions,
)
from aiollm.providers.openai.adapters import OpenAIGenerationAdapter
from aiollm.providers.openai.adapters.conversions import output_text
from aiollm.providers.openai.methods import CreateResponse, StreamResponse
from aiollm.providers.openai.types import Response, ResponseStreamEvent
from aiollm.providers.openai.types.options import OpenAIGenerationOptions
from aiollm.response.parse.stream import StreamResponseParser
from aiollm.session.aiohttp import AiohttpSession
from aiollm.types.base import BaseAPIType
from aiollm.types.matchers import AnyOf, EventTypes
from aiollm.types.requests import (
    DocumentPart,
    GenerationRequest,
    ImagePart,
    TextPart,
    Tool,
    ToolCallTurn,
    ToolResult,
    Turn,
)
from aiollm.types.unknown import BaseUnknown
from tests.conftest import Recorder, Reply
from tests.providers import test_anthropic_messages as anthropic
from tests.providers import test_openai_responses as openai
from tests.providers.contracts.update import event_tags, type_tags

CONTRACTS = Path(__file__).parent / "contracts"


def modules(
    package: str, *, skip: frozenset[str] = frozenset()
) -> tuple[ModuleType, ...]:
    """Import every module of a package.

    Args:
        package: The package, such as ``aiollm.providers.openai.types``.
        skip: Module names left out, such as the hand-written options.

    Returns:
        The modules, in name order.
    """
    root = importlib.import_module(package)
    return tuple(
        importlib.import_module(f"{package}.{info.name}")
        for info in sorted(pkgutil.iter_modules(root.__path__))
        if info.name not in skip
    )


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
        adapter: The shared generation adapter of the endpoint.
        options: The adapter's provider options.
        models: The modules of the endpoint's response and event models.
        params: The modules of the endpoint's request parameter models.
    """

    name: str
    create: type[BaseAPIMethod[Any]]
    stream: type[BaseStreamMethod[Any]]
    request: dict[str, Any]
    response: type[BaseModel]
    events: TypeAdapter[Any]
    documented: dict[str, str]
    examples: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    adapter: ICompositeAdapter[GenerationRequest, Any, Any, Any, Any]
    options: type
    models: tuple[ModuleType, ...]
    params: tuple[ModuleType, ...]


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
        "safety_identifier": "u-1",
        "reasoning": {
            "effort": "high",
            "summary": "auto",
            "context": "auto",
            "mode": "standard",
        },
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "description": "The answer",
                "strict": True,
            },
            "verbosity": "low",
        },
        "previous_response_id": "resp_1",
        "store": False,
        "temperature": 0.5,
        "top_p": 0.9,
        "include": ["reasoning.encrypted_content"],
        "max_tool_calls": 3,
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather",
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
            }
        ],
        "tool_choice": {"type": "function", "name": "get_weather"},
        "parallel_tool_calls": False,
    },
    response=Response,
    events=TypeAdapter(ResponseStreamEvent),
    documented={},
    examples=([openai.RESPONSE], openai.STREAM),
    adapter=OpenAIGenerationAdapter(),
    options=OpenAIGenerationOptions,
    models=modules(
        "aiollm.providers.openai.types", skip=frozenset({"options"})
    ),
    params=modules("aiollm.providers.openai.methods.params"),
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
        "output_config": {
            "effort": "high",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        },
        "thinking": {"type": "adaptive", "display": "summarized"},
        "stop_sequences": ["END"],
        "tools": [
            {
                "name": "get_weather",
                "description": "Current weather",
                "input_schema": {"type": "object", "properties": {}},
                "strict": True,
            }
        ],
        "tool_choice": {
            "type": "tool",
            "name": "get_weather",
            "disable_parallel_tool_use": True,
        },
    },
    response=Message,
    events=TypeAdapter(MessageStreamEvent),
    documented={
        "ping": "https://platform.claude.com/docs/en/build-with-claude/streaming",
        "error": "https://platform.claude.com/docs/en/build-with-claude/streaming",
    },
    examples=([anthropic.MESSAGE], anthropic.STREAM),
    adapter=AnthropicGenerationAdapter(),
    options=AnthropicGenerationOptions,
    models=modules(
        "aiollm.providers.anthropic.types", skip=frozenset({"options"})
    ),
    params=modules("aiollm.providers.anthropic.methods.params"),
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

    The upstream request schemas leave ``additionalProperties`` out,
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


FULL_REQUEST = GenerationRequest(
    model="m",
    input="Hi",
    instructions="Be brief",
    max_output_tokens=64,
    effort=Effort.HIGH,
    output_schema={"type": "object"},
    end_user_id="u-1",
    tools=[Tool(name="f", parameters={"type": "object"})],
    tool_choice=ToolChoice.REQUIRED,
    parallel_tool_calls=False,
)
"""A shared request with every field set."""


@pytest.mark.parametrize("case", CASES)
def test_the_full_request_sets_every_field_the_library_sends(
    case: Case,
) -> None:
    # The methods take every field of the provider's schema (checked
    # below); the full request covers the ones the shared adapter and the
    # provider options fill, so the body checks see each of them.
    adapter: Any = case.adapter
    sent = {
        name
        for name, value in adapter._fields(FULL_REQUEST).items()
        if value is not MISSING
    }
    method = case.create(**case.request)

    assert sent | get_type_hints(case.options).keys() <= (
        method.model_fields_set
    )


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


CONVERSATION = GenerationRequest(
    model="model-1",
    input=[
        Turn.user("Hi"),
        Turn.assistant("Hello! How can I help?"),
        Turn.user(
            [
                TextPart(text="What is in these?"),
                ImagePart(data=b"\x89PNG", media_type="image/png"),
                ImagePart(url="https://example.com/cat.jpg"),
                DocumentPart(data=b"%PDF-1.7", filename="report.pdf"),
                DocumentPart(data=b"%PDF-1.7"),
                DocumentPart(url="https://example.com/report.pdf"),
            ]
        ),
        Turn.assistant("A cat and a report."),
        Turn.user("Tell me a joke"),
    ],
    instructions="Be brief",
    max_output_tokens=64,
    end_user_id="u-1",
)
"""A shared request with every field set and several turns, one of
them of every kind of part and source."""


@pytest.mark.parametrize("case", CASES)
def test_shared_conversation_passes_the_request_schema(case: Case) -> None:
    schema = validator(case, "request.schema.json")

    for method in (
        case.adapter.to_method(CONVERSATION),
        case.adapter.to_stream_method(CONVERSATION),
    ):
        body = to_jsonable_python(method.get_body_map())
        assert errors(schema, body) == []


@pytest.mark.parametrize("native", [True, False], ids=["native", "shared"])
@pytest.mark.parametrize("case", CASES)
def test_a_tool_loop_passes_the_request_schema(
    case: Case, native: bool
) -> None:
    # The response's own output (or, from another provider, the shared
    # calls) and the results that follow it make a valid body, so the
    # second request of a loop is what the provider takes.
    adapter: Any = case.adapter
    called = adapter.to_response(
        case.response.model_validate(LOOP[case.name]), CONVERSATION
    )
    turn = called.to_turn()
    assert isinstance(turn, ToolCallTurn)
    if not native:
        turn = turn.model_copy(update={"native": MISSING})
    request = GenerationRequest(
        model="model-1",
        input=[
            Turn.user("Weather in Paris?"),
            turn,
            ToolResult(call_id=turn.tool_calls[0].id, output="15 C"),
            Turn.user("And in short?"),
        ],
        tools=[Tool(name="get_weather", parameters={"type": "object"})],
    )
    schema = validator(case, "request.schema.json")

    body = json.loads(json.dumps(adapter.to_method(request).get_body_map()))

    assert errors(schema, body) == []


LOOP: Final = {
    "openai/responses": openai.RESPONSE
    | {
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "gAAAA",
            },
            openai.MESSAGE,
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
                "status": "completed",
            },
        ]
    },
    "anthropic/messages": anthropic.MESSAGE
    | {
        "content": [
            {"type": "thinking", "thinking": "Weather.", "signature": "sig"},
            {"type": "text", "text": "Checking.", "citations": None},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
        "stop_reason": "tool_use",
        "stop_details": None,
    },
}
"""A response of each provider that calls a tool, with reasoning."""


def request_definition(
    case: Case,
    *,
    streamed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the request schema of a buffered or a streamed call.

    OpenAI's schema is a union of a streaming variant (``stream``
    required) and a buffered one; Anthropic's is one object for both.

    Args:
        case: The endpoint.
        streamed: Whether the call is streamed.

    Returns:
        The object schema of the call and the schema's definitions.
    """
    schema = load(case.name, "request.schema.json")
    definitions: dict[str, Any] = schema.get("$defs", {})
    variants = [
        resolve(variant, definitions)
        for variant in schema.get("anyOf", [schema])
    ]

    if len(variants) == 1:
        return variants[0], definitions

    (variant,) = [
        variant
        for variant in variants
        if ("stream" in variant.get("required", ())) is streamed
    ]
    return variant, definitions


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("streamed", [False, True], ids=["create", "stream"])
def test_method_fields_match_the_request_schema(
    case: Case,
    streamed: bool,
) -> None:
    # A field takes None exactly where upstream takes null, and leaves
    # itself out by default only where upstream lets it be absent.
    method = case.stream if streamed else case.create
    definition, definitions = request_definition(case, streamed=streamed)
    properties = definition.get("properties", {})
    required = set(definition.get("required", []))
    problems: list[str] = []

    for name, field in method.model_fields.items():
        wire = field.serialization_alias or name
        where = f"{method.__name__}.{name}"

        if get_origin(field.annotation) is Literal:
            # A constant of the method itself, such as StreamResponse's
            # stream: True; upstream's one schema serves both methods.
            continue
        if wire not in properties:
            problems.append(f"{where}: not in the request schema")
            continue

        nullable = allows_null(properties[wire], definitions)
        takes_none = field.annotation is Any or type(None) in get_args(
            field.annotation
        )

        if nullable and not takes_none:
            problems.append(f"{where}: upstream takes null, the field not")
        if takes_none and not nullable:
            problems.append(f"{where}: the field takes None, upstream not")
        if wire in required and field.default is MISSING:
            problems.append(f"{where}: required upstream, left out here")

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("case", CASES)
def test_a_bare_shared_request_sends_only_what_it_must(case: Case) -> None:
    # Nothing that was not given is sent, not even as null; Anthropic
    # requires max_tokens, so the adapter fills in its default.
    request = GenerationRequest(model="m", input="Hi")
    expected = {
        "openai/responses": {"model", "input"},
        "anthropic/messages": {"model", "messages", "max_tokens"},
    }[case.name]

    for method in (
        case.adapter.to_method(request),
        case.adapter.to_stream_method(request),
    ):
        assert set(method.get_body_map()) - {"stream"} == expected


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


def documented_text(case: Case, body: dict[str, Any]) -> str:
    """Join the text of a documented response straight from its JSON.

    Args:
        case: The endpoint.
        body: The documented response.

    Returns:
        Its text parts, joined, as the provider's SDK reads them.
    """
    if case is OPENAI:
        return "".join(
            part["text"]
            for item in body["output"]
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
    return "".join(
        block["text"] for block in body["content"] if block["type"] == "text"
    )


USAGE_FIELDS: Final = {
    "openai/responses": ("input_tokens", "output_tokens", "total_tokens"),
    "anthropic/messages": ("input_tokens", "output_tokens"),
}
"""The token counts of each endpoint's ``usage``, compared as they are."""


@pytest.mark.parametrize("case", CASES)
def test_documented_responses_parse(case: Case) -> None:
    # Validation alone would pass a field read into the wrong place or
    # dropped by extra="ignore": compare what the model gives with the
    # JSON it came from.
    for body in examples(case, "responses.json"):
        model = case.response.model_validate(body)
        assert isinstance(model, Response | Message)
        text = (
            output_text(model)
            if isinstance(model, Response)
            else message_text(model)
        )

        assert model.id == body["id"]
        assert model.model == body["model"]
        assert (text or "") == documented_text(case, body)

        if body.get("usage") is None:
            assert model.usage is None
            continue

        for name in USAGE_FIELDS[case.name]:
            assert getattr(model.usage, name) == body["usage"][name], name


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


RESPONSE_FILES: Final = ("response.schema.json", "stream.schema.json")
"""The upstream schemas of the response and event models."""

REQUEST_FILES: Final = ("request.schema.json",)
"""The upstream schema of the request parameter models."""


def upstream_definitions(
    name: str, files: tuple[str, ...] = RESPONSE_FILES
) -> dict[str, Any]:
    """Collect the definitions of an endpoint's upstream schemas.

    Args:
        name: Snapshot folder, ``<provider>/<endpoint>``.
        files: The schemas to read: the response and stream ones, or
            the request one.

    Returns:
        Every ``$defs`` entry of the schemas, and each root schema under
        its ``title`` (such as ``Response``), and every object written
        inline in a property under ``<Owner>.<property>``.
    """
    definitions: dict[str, Any] = {}

    for file in files:
        schema = load(name, file)
        definitions.update(schema.get("$defs", {}))

        if title := schema.get("title"):
            definitions.setdefault(title, schema)

    # A nullable object (anyOf of it and null, as the official OpenAPI
    # files write ResponseError) is checked as the object.
    for key, definition in list(definitions.items()):
        options = definition.get("anyOf", ())
        objects = [o for o in options if o.get("type") != "null"]
        if (
            len(options) == 2
            and len(objects) == 1
            and ("properties" in objects[0])
        ):
            definitions[key] = objects[0]

    # An object written inline in a property has no name upstream (the
    # official OpenAPI files do this); it is named <Owner>.<property>.
    for owner, definition in list(definitions.items()):
        for prop, node in definition.get("properties", {}).items():
            for option in [node, *node.get("anyOf", ())]:
                if isinstance(option, dict) and "properties" in option:
                    definitions.setdefault(f"{owner}.{prop}", option)

    return definitions


def resolve(node: Any, definitions: dict[str, Any]) -> Any:
    """Follow a ``$ref`` to its definition.

    Args:
        node: A schema node.
        definitions: The endpoint's definitions by name.

    Returns:
        The referenced definition, or the node itself.
    """
    while isinstance(node, dict) and "$ref" in node:
        node = definitions[str(node["$ref"]).rsplit("/", 1)[-1]]
    return node


def allows_null(node: Any, definitions: dict[str, Any]) -> bool:
    """Tell whether a schema node accepts ``null``.

    Args:
        node: A schema node.
        definitions: The endpoint's definitions by name.

    Returns:
        Whether ``null`` is valid for the node.
    """
    node = resolve(node, definitions)

    if not isinstance(node, dict):
        return False

    constrained = ("type", "anyOf", "oneOf", "allOf", "enum", "const")
    if not any(key in node for key in constrained):
        return True  # {} takes any value, null too

    kind = node.get("type")

    if kind == "null" or (isinstance(kind, list) and "null" in kind):
        return True

    return any(
        allows_null(option, definitions)
        for key in ("anyOf", "oneOf")
        for option in node.get(key, [])
    )


def provider_models(
    case: Case, *, params: bool = False
) -> Iterator[type[BaseAPIType]]:
    """Find the models of an endpoint.

    Args:
        case: The endpoint.
        params: Find the request parameter models instead of the
            response and event ones.

    Yields:
        Every ``BaseAPIType`` subclass defined in the endpoint's model
        modules, except the ``BaseUnknown`` ones, which keep whatever an
        unknown ``type`` sends, and the methods' own parameter mixins,
        whose fields the method checks cover.
    """
    mixins = {*case.create.__mro__, *case.stream.__mro__}
    for module in case.params if params else case.models:
        for _, model in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(model, BaseAPIType)
                and model.__module__ == module.__name__
                and not issubclass(model, BaseUnknown)
                and model not in mixins
            ):
                yield model


def upstream_names(model: type[BaseAPIType]) -> tuple[str, ...]:
    """Read the upstream definitions a model declares it mirrors.

    Args:
        model: A provider model.

    Returns:
        The names; empty for a model without an upstream definition.
    """
    names = model.__dict__["__upstream_schema__"]
    return (names,) if isinstance(names, str) else tuple(names)


def is_constant(annotation: Any, default: Any) -> bool:
    """Tell a single-value tag that defaults to its one value.

    Such a field (``type: Literal["function"] = "function"``) is always
    sent, so a schema that requires it is met without the caller.

    Args:
        annotation: The field's type.
        default: Its default.

    Returns:
        Whether the type is a one-value ``Literal`` of the default.
    """
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return get_origin(annotation) is Literal and get_args(annotation) == (
        default,
    )


SIDES = [pytest.param(False, id="responses"), pytest.param(True, id="params")]
"""The response and event models, and the request parameter models."""


@pytest.mark.parametrize("params", SIDES)
@pytest.mark.parametrize("case", CASES)
def test_every_model_names_its_upstream_schema(
    case: Case, params: bool
) -> None:
    definitions = upstream_definitions(
        case.name, REQUEST_FILES if params else RESPONSE_FILES
    )
    models = list(provider_models(case, params=params))
    missing = [
        model.__name__
        for model in models
        if "__upstream_schema__" not in model.__dict__
    ]
    unknown = [
        f"{model.__name__}: {name}"
        for model in models
        if "__upstream_schema__" in model.__dict__
        for name in upstream_names(model)
        if name not in definitions
    ]

    assert models
    assert not missing, f"set __upstream_schema__ on {missing}"
    assert not unknown, f"no such upstream definition: {unknown}"


@pytest.mark.parametrize("params", SIDES)
@pytest.mark.parametrize("case", CASES)
def test_models_accept_what_upstream_allows(case: Case, params: bool) -> None:
    # Validation of examples sees only the values the docs happen to
    # show; this compares every modeled field with the schema itself.
    # A response field may only be laxer than the schema, to read what
    # is sent; a request field follows it both ways: None only where null
    # is allowed, required only where it is.
    files = REQUEST_FILES if params else RESPONSE_FILES
    definitions = upstream_definitions(case.name, files)
    # upstream_definitions reads a nullable object as the object; its
    # null is read from the definitions as they are.
    nullable = {
        name: definition
        for file in files
        for name, definition in load(case.name, file).get("$defs", {}).items()
    }
    problems: list[str] = []

    for model in provider_models(case, params=params):
        for name in upstream_names(model):
            definition = resolve(definitions[name], definitions)
            properties = definition.get("properties", {})
            required = set(definition.get("required", []))

            for field_name, field in model.model_fields.items():
                wire = field.serialization_alias or field_name
                where = f"{model.__name__}.{field_name} ({name}.{wire})"

                if wire not in properties:
                    problems.append(f"{where}: not in the upstream schema")
                    continue

                takes_none = field.annotation is Any or type(None) in get_args(
                    field.annotation
                )
                null = allows_null(
                    properties[wire], nullable if params else definitions
                )
                if null and not takes_none:
                    problems.append(f"{where}: upstream allows null")
                if params and takes_none and not null:
                    problems.append(f"{where}: upstream takes no null")

                if wire not in required and field.is_required():
                    problems.append(f"{where}: optional upstream")
                if (
                    params
                    and wire in required
                    and not field.is_required()
                    and not is_constant(field.annotation, field.default)
                ):
                    problems.append(f"{where}: required upstream")

    assert not problems, "\n".join(problems)
