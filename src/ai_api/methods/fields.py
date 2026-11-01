"""Helpers for method classes: field types, query values and uploads."""

from __future__ import annotations

import io
import types
from collections.abc import Collection, Mapping
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

from ai_api.types.files import InputFile, IOFile


def is_mapping_type(annotation: Any) -> bool:
    """Check whether a field annotation allows mappings or models.

    Args:
        annotation: The annotation of a field, unions included.

    Returns:
        ``True`` if the annotation or a member of its union is a mapping
        or a Pydantic model.
    """
    origin = get_origin(annotation)

    if origin in {Union, types.UnionType}:
        return any(is_mapping_type(arg) for arg in get_args(annotation))

    target = origin or annotation

    return isinstance(target, type) and issubclass(
        target, (Mapping, BaseModel)
    )


def is_container_type(annotation: Any) -> bool:
    """Check whether a field annotation allows containers.

    Args:
        annotation: The annotation of a field, unions included.

    Returns:
        ``True`` if the annotation or a member of its union is a mapping,
        a Pydantic model or a collection such as ``list`` or ``set``;
        strings and bytes are scalars.
    """
    origin = get_origin(annotation)

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


def _query_item(name: str, value: Any) -> str:
    """Convert one dumped query value to a string.

    Args:
        name: The parameter name, for error messages.
        value: A JSON-mode dump of the value.

    Returns:
        The value as a string; booleans as ``"true"``/``"false"``.

    Raises:
        TypeError: If the value is an object or a nested list.
    """
    if isinstance(value, (dict, list)):
        raise TypeError(
            f"Query parameter {name!r} must be a scalar or a list of scalars"
        )

    return str(value).lower() if isinstance(value, bool) else str(value)


def query_value(name: str, value: Any) -> str | list[str]:
    """Convert a dumped query field to a string or a list of strings.

    Args:
        name: The parameter name, for error messages.
        value: A JSON-mode dump of the field.

    Returns:
        A string, or a list of strings sent as repeated keys.

    Raises:
        TypeError: If the value is an object or contains one.
        ValueError: If a list contains ``None``, which a repeated key
            cannot express; dropping it would shift the other items.
    """
    if isinstance(value, list):
        if any(item is None for item in value):
            raise ValueError(
                f"Query parameter {name!r} cannot send None in a list"
            )

        return [_query_item(name, item) for item in value]

    return _query_item(name, value)


def wrap_streams(value: Any) -> Any:
    """Wrap raw binary streams into ``IOFile``, recursing into containers.

    Args:
        value: A field value of a method.

    Returns:
        The value with every binary stream replaced by an ``IOFile``;
        the same object if it contains no streams.

    Raises:
        ValueError: If a text stream is found.
    """
    if isinstance(value, (io.BufferedIOBase, io.RawIOBase)):
        return IOFile(value)

    if isinstance(value, io.IOBase):
        raise ValueError("Only binary streams can be uploaded")

    if type(value) in {list, tuple}:
        items = [wrap_streams(item) for item in value]

        if all(new is old for new, old in zip(items, value, strict=True)):
            return value

        return type(value)(items)

    if type(value) is dict:
        wrapped = {key: wrap_streams(item) for key, item in value.items()}

        if all(wrapped[key] is item for key, item in value.items()):
            return value

        return wrapped

    return value


def has_one_shot_file(value: Any) -> bool:
    """Check whether a value contains a file that can be read only once.

    Args:
        value: A method, model, container or field value.

    Returns:
        ``True`` if any nested ``InputFile`` is not reusable.
    """
    if isinstance(value, InputFile):
        return not value.reusable

    if isinstance(value, BaseModel):
        return any(has_one_shot_file(item) for _, item in value)

    if isinstance(value, Mapping):
        return any(has_one_shot_file(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return any(has_one_shot_file(item) for item in value)

    return False
