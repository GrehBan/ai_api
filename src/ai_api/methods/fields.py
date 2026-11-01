"""Helpers for method classes: field types, query values and uploads."""

from __future__ import annotations

import io
import types
from collections.abc import Collection, Mapping
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

from ai_api.serialization.values import scalar_text
from ai_api.types.files import InputFile, IOFile

if TYPE_CHECKING:
    from ai_api.serialization.protocols import IValueFormatter

_SEQUENCES: Final = (list, tuple, set, frozenset)
"""Values sent as a repeated query key."""


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
        ValueError: If a list contains ``None``, which a repeated key
            cannot express (dropping it would shift the other items), a
            value is NaN or infinite, or the formatter rejects one.
    """
    if isinstance(value, _SEQUENCES):
        if any(item is None for item in value):
            raise ValueError(
                f"Query parameter {name!r} cannot send None in a list"
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
        ValueError: If a text stream is found.
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

    wrapped = {key: wrap_streams(item) for key, item in children.items()}
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
    """
    if isinstance(value, InputFile):
        return not value.reusable

    children = _children(value)

    return children is not None and any(
        has_one_shot_file(item) for item in children.values()
    )
