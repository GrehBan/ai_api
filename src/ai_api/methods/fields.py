"""Helpers for method classes: field types, query values and uploads."""

from __future__ import annotations

import dataclasses
import io
import types
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
)
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    NewType,
    TypeAliasType,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from pydantic import MISSING, BaseModel, PlainSerializer

from ai_api.serialization.traversal import guard_traversal
from ai_api.serialization.values import scalar_text
from ai_api.types.files import InputFile, IOFile

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from ai_api.serialization.protocols import IValueFormatter

_SEQUENCES: Final = (list, tuple, set, frozenset)
"""Values sent as a repeated query key."""

_NOT_DUMPED: Final = object()
"""What ``_see_through`` returns for a type whose serializer decides."""


def is_mapping_type(annotation: Any) -> bool:
    """Check whether a field annotation allows mappings or models.

    Args:
        annotation: The annotation of a field, unions and
            ``Annotated`` included.

    Returns:
        ``True`` if the annotation or a member of its union is a mapping
        or a Pydantic model.
    """
    origin = get_origin(annotation)

    if origin is Annotated:
        return is_mapping_type(get_args(annotation)[0])

    if origin in {Union, types.UnionType}:
        return any(is_mapping_type(arg) for arg in get_args(annotation))

    target = origin or annotation

    return isinstance(target, type) and issubclass(
        target, (Mapping, BaseModel)
    )


def is_container_type(annotation: Any) -> bool:
    """Check whether a field annotation allows containers.

    Args:
        annotation: The annotation of a field, unions and
            ``Annotated`` included.

    Returns:
        ``True`` if the annotation or a member of its union is a mapping,
        a Pydantic model or a collection such as ``list`` or ``set``;
        strings and bytes are scalars.
    """
    origin = get_origin(annotation)

    if origin is Annotated:
        return is_container_type(get_args(annotation)[0])

    if origin in {Union, types.UnionType}:
        return any(is_container_type(arg) for arg in get_args(annotation))

    if is_mapping_type(annotation):
        return True

    target = origin or annotation

    return (
        isinstance(target, type)
        and issubclass(target, Collection)
        and not issubclass(target, (str, bytes, bytearray))
    )


def defaults_to_none(field: FieldInfo) -> bool:
    """Tell whether a field's default is ``None``.

    A default factory is called to see what it gives (factories are
    expected to be pure, as Pydantic calls them for every instance),
    except one that takes the validated data, which cannot be called
    without an instance.

    Args:
        field: A field of a model or a Pydantic dataclass.

    Returns:
        ``True`` if the field defaults to ``None``.
    """
    if field.default is None:
        return True

    factory = field.default_factory

    if factory is None or field.default_factory_takes_validated_data:
        return False

    return cast("Callable[[], Any]", factory)() is None


def none_defaults(owner: type[Any]) -> list[str]:
    """Name the fields of a class of data that default to ``None``.

    Args:
        owner: A Pydantic model or dataclass, a standard dataclass or a
            ``NamedTuple`` class, as ``nested_data_classes`` finds them.

    Returns:
        The names, sorted; fields excluded from dumps are left out.
    """
    fields = _pydantic_fields(owner)

    if fields is not None:
        return sorted(
            name
            for name, field in fields.items()
            if not field.exclude and defaults_to_none(field)
        )

    if dataclasses.is_dataclass(owner):
        return sorted(
            item.name
            for item in dataclasses.fields(owner)
            if item.default is None
            or (
                item.default_factory is not dataclasses.MISSING
                and item.default_factory() is None
            )
        )

    defaults: dict[str, Any] = getattr(owner, "_field_defaults", {})
    return sorted(name for name, value in defaults.items() if value is None)


def names_missing(annotation: Any) -> bool:
    """Tell whether a field's type names ``MISSING``.

    Args:
        annotation: The annotation of a field.

    Returns:
        ``True`` if ``MISSING`` is the type or a member of its union,
        also through ``Annotated`` and type aliases.
    """
    return _names_missing(annotation, set())


def _names_missing(annotation: Any, seen: set[int]) -> bool:
    """Tell whether a type names ``MISSING``.

    Args:
        annotation: A type or a part of one.
        seen: Identities of the type aliases already looked into.

    Returns:
        ``True`` if ``MISSING`` is the type or a member of its union.
    """
    if annotation is MISSING:
        return True

    if isinstance(annotation, TypeAliasType):
        if id(annotation) in seen:
            return False

        seen.add(id(annotation))
        return _names_missing(annotation.__value__, seen)

    origin = get_origin(annotation)

    if origin is Annotated:
        return _names_missing(get_args(annotation)[0], seen)

    if origin in {Union, types.UnionType}:
        return any(_names_missing(arg, seen) for arg in get_args(annotation))

    return False


def serialized_by_hand(metadata: Iterable[Any]) -> bool:
    """Tell whether a type's metadata replaces how it is dumped.

    A ``PlainSerializer`` decides alone what is sent, so the type it
    sits on is not dumped and its defaults are never sent.

    Args:
        metadata: The ``Annotated`` metadata of a type, or a field's.

    Returns:
        ``True`` if a ``PlainSerializer`` is among it.
    """
    return any(isinstance(item, PlainSerializer) for item in metadata)


def nested_data_classes(annotation: Any) -> Iterator[type[Any]]:
    """Find the classes of data an annotation can hold, at any depth.

    A class of data has fields with defaults that are dumped with it:
    a Pydantic model or dataclass, a standard dataclass or a
    ``NamedTuple``. Unions, ``Annotated``, generic containers, type
    aliases, ``NewType``, the keys of a ``TypedDict`` and the fields of
    the classes found are looked into; a type with a
    ``PlainSerializer`` is not, since the serializer decides what is
    sent. Each class, ``TypedDict`` and
    type alias is looked into once, so a recursive one (``type JSON =
    dict[str, JSON] | ...``) ends: a cycle is legal in types, so it
    stops the walk instead of raising.

    Args:
        annotation: The annotation of a field.

    Returns:
        An iterator over every class of data the annotation names,
        directly or nested.
    """
    return _nested_data_classes(annotation, set())


def _nested_data_classes(
    annotation: Any,
    seen: set[int],
) -> Iterator[type[Any]]:
    """Find the classes of data an annotation can hold, at any depth.

    Args:
        annotation: An annotation or a part of one.
        seen: Identities of the classes, ``TypedDict`` classes and type
            aliases already looked into.

    Yields:
        Every class of data not seen before.
    """
    annotation = _see_through(annotation)

    if annotation is _NOT_DUMPED:
        return

    if isinstance(annotation, TypeAliasType) or _is_class(annotation):
        if id(annotation) in seen:
            return

        seen.add(id(annotation))

    if isinstance(annotation, TypeAliasType):
        yield from _nested_data_classes(annotation.__value__, seen)
        return

    members = _member_types(annotation)

    if members is not None:
        if not is_typeddict(annotation):
            yield annotation

        for member in members.values():
            yield from _nested_data_classes(member, seen)
        return

    origin = get_origin(annotation)

    if isinstance(origin, TypeAliasType):
        yield from _nested_data_classes(origin, seen)

    for arg in get_args(annotation):
        yield from _nested_data_classes(arg, seen)


def _see_through(annotation: Any) -> Any:
    """Return the type the walk looks into for an annotation.

    Args:
        annotation: An annotation or a part of one.

    Returns:
        The type behind a ``NewType``, the annotation itself otherwise,
        or ``_NOT_DUMPED`` for a type with a ``PlainSerializer``, whose
        serializer decides what is sent.
    """
    while isinstance(annotation, NewType):
        annotation = annotation.__supertype__

    if get_origin(annotation) is Annotated and serialized_by_hand(
        get_args(annotation)[1:]
    ):
        return _NOT_DUMPED

    return annotation


def _is_class(annotation: Any) -> bool:
    """Tell whether an annotation is a plain class, not a generic alias.

    Args:
        annotation: An annotation or a part of one.

    Returns:
        ``True`` for a class such as ``int`` or a model; ``False`` for
        ``list[int]`` and anything that is not a class.
    """
    return isinstance(annotation, type) and get_origin(annotation) is None


def _member_types(annotation: Any) -> dict[str, Any] | None:
    """Return the field types of a class of data or a ``TypedDict``.

    Args:
        annotation: An annotation or a part of one.

    Returns:
        The type of each field or key; ``None`` for anything else.
    """
    if is_typeddict(annotation):
        return get_type_hints(annotation, include_extras=True)

    if not _is_class(annotation):
        return None

    fields = _pydantic_fields(annotation)

    if fields is not None:
        return {name: field.annotation for name, field in fields.items()}

    if dataclasses.is_dataclass(annotation) or (
        issubclass(annotation, tuple) and hasattr(annotation, "_fields")
    ):
        return get_type_hints(annotation, include_extras=True)

    return None


def _pydantic_fields(owner: type[Any]) -> dict[str, FieldInfo] | None:
    """Return the fields of a Pydantic model or dataclass.

    Args:
        owner: A class.

    Returns:
        The fields by name; ``None`` for a class Pydantic did not build.
    """
    fields = getattr(owner, "__pydantic_fields__", None)
    return fields if isinstance(fields, dict) else None


def _query_item(name: str, value: Any, formatter: IValueFormatter) -> str:
    """Convert one dumped query value to a string.

    Args:
        name: The parameter name, for error messages.
        value: A Python-mode dump of the value.
        formatter: Writes the dates, times and durations.

    Returns:
        The value as a string, by the rule of ``scalar_text``.

    Raises:
        TypeError: If the value is an object or a nested list.
        ValueError: If the value is NaN or infinite, or the formatter
            rejects it.
    """
    if isinstance(value, (Mapping, *_SEQUENCES)):
        raise TypeError(
            f"Query parameter {name!r} must be a scalar or a list of scalars"
        )

    return scalar_text(value, formatter, label=f"Query parameter {name!r}")


def query_value(
    name: str,
    value: Any,
    formatter: IValueFormatter,
) -> str | list[str]:
    """Convert a dumped query field to a string or a list of strings.

    Args:
        name: The parameter name, for error messages.
        value: A Python-mode dump of the field.
        formatter: Writes the dates, times and durations.

    Returns:
        A string, or a list of strings sent as repeated keys.

    Raises:
        TypeError: If the value is an object or contains one.
        ValueError: If the value is ``None`` or a list contains it: a
            query has no ``null`` (dropping it would send another
            request, or shift the other items of a repeated key); if a
            list contains ``MISSING``, which leaves only a whole field
            out; also if a value is NaN or infinite, or the formatter
            rejects one.
    """
    if value is None:
        raise ValueError(
            f"Query parameter {name!r} cannot send None: a query has no "
            "null; leave the field MISSING to leave it out"
        )

    if isinstance(value, _SEQUENCES):
        if any(item is None for item in value):
            raise ValueError(
                f"Query parameter {name!r} cannot send None in a list"
            )

        if any(item is MISSING for item in value):
            raise ValueError(
                f"Query parameter {name!r} cannot send MISSING in a list: "
                "MISSING leaves a whole field out"
            )

        return [_query_item(name, item, formatter) for item in value]

    return _query_item(name, value, formatter)


def _children(value: Any) -> dict[Any, Any] | None:
    """Name the parts of a value the file helpers look into.

    The one definition of what they traverse, so wrapping streams and
    looking for one-shot files never disagree: lists, tuples, mappings
    and Pydantic models, nested at any depth.

    Args:
        value: A field value of a method.

    Returns:
        Each part by its index, key or field name; ``None`` for a value
        with no parts.
    """
    if isinstance(value, BaseModel):
        return dict(value)

    if isinstance(value, Mapping):
        mapping: Mapping[Any, Any] = value
        return dict(mapping)

    if isinstance(value, list | tuple):
        sequence: list[Any] | tuple[Any, ...] = value
        return dict(enumerate(sequence))

    return None


def wrap_streams(value: Any) -> Any:
    """Wrap raw binary streams into ``IOFile``, recursing into containers.

    Args:
        value: A field value of a method.

    Returns:
        The value with every binary stream replaced by an ``IOFile``;
        the same object if it contains no streams. A model is copied
        with the wrapped fields, never changed in place.

    Raises:
        ValueError: If a text stream is found, or the value contains
            itself or is nested deeper than ``MAX_DEPTH``.
        TypeError: If a stream sits in a container this cannot rebuild
            (a subclass of ``list``, ``tuple`` or ``dict``, another
            mapping): left raw, Pydantic would dump it lazily.
    """
    return _wrap_streams(value, set())


def _wrap_streams(
    value: Any,
    ancestors: set[int],
) -> Any:
    """Wrap raw binary streams into ``IOFile``, recursing into containers.

    Args:
        value: A field value of a method.
        ancestors: Identities of containers in the current recursion path.

    Returns:
        The value with every binary stream replaced by an ``IOFile``;
        the same object if it contains no streams. A model is copied
        with the wrapped fields, never changed in place.

    Raises:
        ValueError: If a text stream is found, or the value contains
            itself or is nested deeper than ``MAX_DEPTH``.
        TypeError: If a stream sits in a container this cannot rebuild
            (a subclass of ``list``, ``tuple`` or ``dict``, another
            mapping): left raw, Pydantic would dump it lazily.
    """
    if isinstance(value, (io.BufferedIOBase, io.RawIOBase)):
        return IOFile(value)

    if isinstance(value, io.IOBase):
        raise ValueError("Only binary streams can be uploaded")

    children = _children(value)

    if children is None:
        return value

    with guard_traversal(value, ancestors):
        wrapped = {
            key: _wrap_streams(item, ancestors)
            for key, item in children.items()
        }

    changed = {
        key: item for key, item in wrapped.items() if item is not children[key]
    }

    if not changed:
        return value

    if isinstance(value, BaseModel):
        return value.model_copy(update=changed)

    if type(value) in {list, tuple}:
        return type(value)(wrapped.values())

    if type(value) is dict:
        return wrapped

    message = (
        f"A stream inside a {type(value).__name__} cannot be wrapped; "
        "use a list, tuple or dict"
    )
    raise TypeError(message)


def has_one_shot_file(value: Any) -> bool:
    """Check whether a value contains a file that can be read only once.

    Args:
        value: A method, model, container or field value.

    Returns:
        ``True`` if any nested ``InputFile`` is not reusable.

    Raises:
        ValueError: If the value contains itself or is nested deeper than
            ``MAX_DEPTH``.
    """
    return _has_one_shot_file(value, set())


def _has_one_shot_file(value: Any, ancestors: set[int]) -> bool:
    """Check whether a value contains a file that can be read only once.

    Args:
        value: A method, model, container or field value.
        ancestors: Identities of containers in the current recursion path.

    Returns:
        ``True`` if any nested ``InputFile`` is not reusable.

    Raises:
        ValueError: If the value contains itself or is nested deeper than
            ``MAX_DEPTH``.
    """
    if isinstance(value, InputFile):
        return not value.reusable

    children = _children(value)

    if children is None:
        return False

    with guard_traversal(value, ancestors):
        return any(
            _has_one_shot_file(item, ancestors) for item in children.values()
        )
