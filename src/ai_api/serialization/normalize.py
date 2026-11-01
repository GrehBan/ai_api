"""Conversion of request values into wire-ready values."""

from __future__ import annotations

import io
from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

from ai_api.serialization.values import TEMPORAL_TYPES, format_temporal
from ai_api.types.base import BaseAPIType
from ai_api.types.files import InputFile

if TYPE_CHECKING:
    from ai_api.serialization.protocols import IValueFormatter

MAX_DEPTH: Final = 253
"""Deepest nesting of containers accepted in a body field.

orjson refuses more than 254 nested containers, and the body object is
one of them; a deeper field fails with a clear ``ValueError`` here
instead of an encoder ``TypeError``.
"""


def _key(key: Any) -> Any:
    """Unwrap an enum used as a dict key, as values are unwrapped.

    Args:
        key: A key of a body dict.

    Returns:
        The enum's value, or the key itself.
    """
    while isinstance(key, Enum):
        key = key.value

    return key


def _checked_stream(stream: io.IOBase) -> io.IOBase:
    """Accept a stream as an upload only if it can be read as bytes.

    Args:
        stream: A stream in a body value.

    Returns:
        The stream itself, for the body builder to send.

    Raises:
        TypeError: If it is a text stream: uploads are bytes, and it would
            fail later, in the body builder, with a message about binary
            data.
        ValueError: If it is closed or not readable.
    """
    if not isinstance(stream, (io.BufferedIOBase, io.RawIOBase)):
        raise TypeError(
            "Only binary streams can be uploaded, not a text stream"
        )

    if stream.closed or not stream.readable():
        raise ValueError("Invalid file stream")

    return stream


class BodyNormalizer:
    """Prepares request body values for a transport-specific builder.

    Nested models are dumped by alias, enums (also dict keys) are
    unwrapped, and dates, times and durations are written by the method's
    ``IValueFormatter``. ``None`` fields are dropped from the body and
    from nested models, but kept as ``null`` inside lists and dicts.
    Files and binary streams are passed through unchanged for the body
    builder to handle.
    """

    def normalize_mapping(
        self,
        body: Mapping[str, Any],
        formatter: IValueFormatter,
    ) -> dict[str, Any]:
        """Normalize the fields of a body, dropping ``None`` results.

        Args:
            body: Body fields of a request.
            formatter: Writes the dates, times and durations.

        Returns:
            A new dict with normalized values.

        Raises:
            TypeError: If a value is a text stream.
            ValueError: If a value cannot be normalized.
        """
        return {
            key: normalized
            for key, value in body.items()
            if (normalized := self.normalize(value, formatter)) is not None
        }

    def normalize(self, value: Any, formatter: IValueFormatter) -> Any:
        """Normalize a single value, recursing into containers.

        Args:
            value: Any value of a request body.
            formatter: Writes the dates, times and durations.

        Returns:
            The normalized value; ``None`` for ``None``.

        Raises:
            TypeError: If the value is or contains a text stream.
            ValueError: If a stream is closed or unreadable, the formatter
                rejects a value (a naive datetime, by default), or a
                container contains itself or is nested deeper than
                ``MAX_DEPTH``.
        """
        return self._normalize(value, formatter, set())

    def _normalize(
        self,
        value: Any,
        formatter: IValueFormatter,
        parents: set[int],
    ) -> Any:
        """Normalize a value inside the containers being normalized.

        Args:
            value: The value to normalize.
            formatter: Writes the dates, times and durations.
            parents: Ids of the enclosing containers, to detect cycles.

        Returns:
            The normalized value.

        Raises:
            TypeError: If the value is or contains a text stream.
            ValueError: If a stream is closed or unreadable, the formatter
                rejects a value, a container contains itself or is nested
                deeper than ``MAX_DEPTH``, or two keys of a dict give the
                same value.
        """
        if value is None:
            return None

        if isinstance(value, InputFile):
            return value

        if isinstance(value, io.IOBase):
            return _checked_stream(value)

        if isinstance(value, BaseAPIType):
            return self._normalize(
                value.model_dump(
                    mode="python",
                    by_alias=True,
                    exclude_none=True,
                    warnings=False,
                ),
                formatter,
                parents,
            )

        if isinstance(value, Enum):
            return self._normalize(value.value, formatter, parents)

        if isinstance(value, TEMPORAL_TYPES):
            return format_temporal(value, formatter)

        if isinstance(value, (Mapping, list, tuple)):
            return self._normalize_container(value, formatter, parents)

        return value

    def _normalize_container(
        self,
        value: Mapping[Any, Any] | list[Any] | tuple[Any, ...],
        formatter: IValueFormatter,
        parents: set[int],
    ) -> dict[Any, Any] | list[Any]:
        """Normalize the items of a dict, list or tuple.

        Args:
            value: The container.
            formatter: Writes the dates, times and durations.
            parents: Ids of the enclosing containers, to detect cycles.

        Returns:
            A new dict, or a list for a list or a tuple.

        Raises:
            ValueError: If the container contains itself, is nested deeper
                than ``MAX_DEPTH``, or an item cannot be normalized.
        """
        if id(value) in parents:
            raise ValueError("Request body contains a circular reference")

        if len(parents) >= MAX_DEPTH:
            raise ValueError(
                f"Request body is nested deeper than {MAX_DEPTH} levels"
            )

        parents.add(id(value))

        try:
            if isinstance(value, Mapping):
                return self._normalize_dict(value, formatter, parents)

            return [
                self._normalize(item, formatter, parents) for item in value
            ]
        finally:
            parents.discard(id(value))

    def _normalize_dict(
        self,
        value: Mapping[Any, Any],
        formatter: IValueFormatter,
        parents: set[int],
    ) -> dict[Any, Any]:
        """Normalize the items of a dict, unwrapping enum keys.

        Args:
            value: The dict to normalize.
            formatter: Writes the dates, times and durations.
            parents: Ids of the enclosing containers, this one included.

        Returns:
            A new dict with normalized keys and values.

        Raises:
            ValueError: If two keys give the same value, such as
                ``Color.RED`` and ``"red"``: one of them would be lost.
        """
        result: dict[Any, Any] = {}

        for key, item in value.items():
            name = _key(key)

            if name in result:
                raise ValueError(f"Request body has the key {name!r} twice")

            result[name] = self._normalize(item, formatter, parents)

        return result
