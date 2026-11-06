"""Each provider's generation options mirror its method's fields.

An option is a field of the provider's method that the shared request
does not fill, with the method's type (``MISSING`` aside): so the
wire-contract check of the method fields covers the options too.
"""

from __future__ import annotations

import types
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import pytest
from pydantic import MISSING, BaseModel

from aiollm.adapters.validation import checked_options
from aiollm.enums import Effort, Provider
from aiollm.exceptions import RequestValidationError
from aiollm.providers.anthropic.adapters import AnthropicGenerationAdapter
from aiollm.providers.anthropic.methods import CreateMessage
from aiollm.providers.anthropic.types.options import (
    AnthropicGenerationOptions,
)
from aiollm.providers.openai.adapters import OpenAIGenerationAdapter
from aiollm.providers.openai.methods import CreateResponse
from aiollm.providers.openai.types.options import OpenAIGenerationOptions
from aiollm.types.requests import GenerationRequest, NoProviderOptions

CASES = [
    (OpenAIGenerationOptions, CreateResponse, OpenAIGenerationAdapter()),
    (
        AnthropicGenerationOptions,
        CreateMessage,
        AnthropicGenerationAdapter(),
    ),
]
IDS = ["openai", "anthropic"]


def without_missing(annotation: Any) -> set[Any]:
    """The members of a union, ``MISSING`` left out."""
    if get_origin(annotation) in (Union, types.UnionType):
        members = set(get_args(annotation))
    else:
        members = {annotation}
    return {member for member in members if member is not MISSING}


def typed_dict_in(annotation: Any) -> Any:
    """The one ``TypedDict`` of a type, ``None`` and ``MISSING`` aside."""
    members = without_missing(annotation) - {type(None)}
    if len(members) == 1 and is_typeddict(member := next(iter(members))):
        return member
    return None


def object_in(annotation: Any) -> Any:
    """The one object type of a field: a ``TypedDict`` or a model."""
    members = without_missing(annotation) - {type(None)}
    if len(members) != 1:
        return None
    member = next(iter(members))
    if is_typeddict(member) or (
        isinstance(member, type) and issubclass(member, BaseModel)
    ):
        return member
    return None


def keys_of(container: Any) -> dict[str, Any]:
    """The keys of an object type with their types."""
    if is_typeddict(container):
        return get_type_hints(container)
    return {
        name: field.annotation
        for name, field in container.model_fields.items()
    }


def tag(container: Any) -> Any:
    """The ``type`` constant of an object type, if it has one."""
    annotation = keys_of(container).get("type")
    return get_args(annotation) if get_origin(annotation) is Literal else None


def mirrors(option: Any, field: Any, where: str) -> list[str]:
    """Compare an option's type with the method field's type.

    A member of the option's type is a member of the field's type, or a
    ``TypedDict`` that mirrors one of its object members (a model or a
    ``TypedDict``, matched by its ``type`` constant when there are
    several): every key of it is a key of the object, of a mirroring
    type. So a dict literal that type-checks as the option validates as
    the field.

    Args:
        option: The option's type.
        field: The method field's type.
        where: The option's path, for the messages.

    Returns:
        A message per member that does not match; empty if all do.
    """
    wanted = without_missing(field)
    objects = [member for member in wanted if object_in(member) is member]
    problems: list[str] = []
    for member in without_missing(option):
        if member in wanted:
            continue
        if not is_typeddict(member):
            problems.append(f"{where}: {member!r} is not taken")
            continue
        matches = [
            target
            for target in objects
            if len(objects) == 1 or tag(target) == tag(member)
        ]
        if not matches:
            problems.append(f"{where}: no object mirrors {member.__name__}")
            continue
        inner = keys_of(matches[0])
        for key, value in get_type_hints(member).items():
            if key not in inner:
                problems.append(f"{where}.{key}: not a field")
                continue
            problems += mirrors(value, inner[key], f"{where}.{key}")
    return problems


@pytest.mark.parametrize(("options", "method", "adapter"), CASES, ids=IDS)
def test_every_option_is_a_method_field_of_the_same_type(
    options: type,
    method: type[BaseModel],
    adapter: Any,
) -> None:
    # An option is the method's field with its type, or with a TypedDict
    # that mirrors the field's model, so a dict literal type-checks; an
    # option that fills part of an object (OpenAI's reasoning, whose
    # effort is shared) mirrors some of that object's keys.
    hints = get_type_hints(options)
    fields = method.model_fields

    assert hints
    problems: list[str] = []
    for name, annotation in hints.items():
        assert name in fields, name
        problems += mirrors(annotation, fields[name].annotation, name)

    assert not problems, "\n".join(problems)


FULL_REQUEST = GenerationRequest(
    model="m",
    input="Hi",
    instructions="Be brief",
    max_output_tokens=64,
    effort=Effort.HIGH,
    output_schema={"type": "object"},
    end_user_id="u-1",
)
"""A shared request with every field set."""


def paths(value: Any, prefix: str = "") -> set[str]:
    """The keys of a value, one level into nested dicts."""
    found = set()
    for key, item in value.items():
        if item is MISSING:
            continue
        found.add(f"{prefix}{key}")
        if not prefix and isinstance(item, dict):
            found |= {f"{key}.{inner}" for inner in item}
    return found


def option_paths(options: type) -> set[str]:
    """The keys an options TypedDict sets, one level into nested ones."""
    found = set()
    for key, annotation in get_type_hints(options).items():
        nested = typed_dict_in(annotation)
        if nested is None:
            found.add(key)
        else:
            found |= {f"{key}.{inner}" for inner in get_type_hints(nested)}
    return found


@pytest.mark.parametrize(("options", "method", "adapter"), CASES, ids=IDS)
def test_no_option_is_a_field_the_shared_request_fills(
    options: type,
    method: type[BaseModel],
    adapter: Any,
) -> None:
    # The shared field is the one way to set it, so the two never clash;
    # an option may only fill the rest of an object a shared field
    # fills part of.
    shared = paths(adapter._fields(FULL_REQUEST))
    leaves = {
        path
        for path in shared
        if not any(other.startswith(f"{path}.") for other in shared)
    }

    assert option_paths(options).isdisjoint(leaves)


@pytest.mark.parametrize(("options", "method", "adapter"), CASES, ids=IDS)
def test_options_are_all_optional(
    options: Any,
    method: type[BaseModel],
    adapter: Any,
) -> None:
    assert options.__required_keys__ == frozenset()


def test_with_options_leaves_the_adapter_as_it_is() -> None:
    adapter = OpenAIGenerationAdapter()
    tuned = adapter.with_options({"store": False})
    request = GenerationRequest(model="m", input="Hi")

    assert tuned is not adapter
    assert tuned.to_method(request).store is False
    assert adapter.to_method(request).store is MISSING


def test_checked_options_copies_the_known_keys() -> None:
    given = {"store": True}
    checked = checked_options(given, OpenAIGenerationOptions, Provider.OPENAI)

    assert checked == given
    assert checked is not given


def test_checked_options_refuses_anything_but_a_mapping() -> None:
    with pytest.raises(RequestValidationError, match="not list") as caught:
        checked_options([], OpenAIGenerationOptions, Provider.OPENAI)

    assert caught.value.field == "provider_options"


def test_checked_options_needs_a_typed_dict() -> None:
    with pytest.raises(TypeError, match="not a TypedDict"):
        checked_options({}, dict, Provider.OPENAI)


def test_no_provider_options_takes_no_key() -> None:
    with pytest.raises(RequestValidationError, match="store"):
        checked_options({"store": 1}, NoProviderOptions, Provider.GENERIC)
