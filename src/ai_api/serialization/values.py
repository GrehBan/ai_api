"""Encoding of single request values: dates, durations and scalars."""

from __future__ import annotations

import datetime
import math
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError, to_jsonable_python

if TYPE_CHECKING:
    from ai_api.serialization.protocols import IValueFormatter

type Temporal = (
    datetime.datetime | datetime.date | datetime.time | datetime.timedelta
)
"""A value that ``IValueFormatter`` writes."""

TEMPORAL_TYPES: Final = (
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
)
"""The classes of ``Temporal``, for ``isinstance`` checks."""

_DATETIME: Final = TypeAdapter(datetime.datetime)
_DATE: Final = TypeAdapter(datetime.date)
_TIME: Final = TypeAdapter(datetime.time)
_TIMEDELTA: Final = TypeAdapter(datetime.timedelta)


def _json_text[T](adapter: TypeAdapter[T], value: T) -> str:
    """Write a value the way Pydantic writes it in JSON.

    Args:
        adapter: The adapter of the value's type.
        value: The value to write.

    Returns:
        Its JSON-mode text.
    """
    text: str = adapter.dump_python(value, mode="json")
    return text


class BaseValueFormatter:
    """Writes dates, times and durations in ISO 8601.

    This is the format of ``date-time``, ``date`` and ``time`` in JSON
    Schema and OpenAPI, and the one Pydantic writes in JSON: a UTC moment
    as ``2026-01-02T03:04:05Z``, another offset as ``+03:00``, a duration
    as ``PT30S``. A provider whose API wants another format subclasses it
    and overrides the methods it needs.
    """

    def format_datetime(self, value: datetime.datetime, /) -> str:
        """Write a moment in time in ISO 8601.

        Args:
            value: A timezone-aware datetime.

        Returns:
            Its text, such as ``2026-01-02T03:04:05Z``.

        Raises:
            ValueError: If the datetime is naive: the moment it names
                depends on the clock of whoever reads it.
        """
        if value.utcoffset() is None:
            raise ValueError("Datetime must be timezone-aware")

        return _json_text(_DATETIME, value)

    def format_date(self, value: datetime.date, /) -> str:
        """Write a calendar date in ISO 8601.

        Args:
            value: The date to write.

        Returns:
            Its text, such as ``2026-01-02``.
        """
        return _json_text(_DATE, value)

    def format_time(self, value: datetime.time, /) -> str:
        """Write a time of day in ISO 8601.

        Args:
            value: The time to write, with or without an offset.

        Returns:
            Its text, such as ``12:00:00`` or ``12:00:00Z``.
        """
        return _json_text(_TIME, value)

    def format_timedelta(self, value: datetime.timedelta, /) -> str:
        """Write a duration in ISO 8601.

        Args:
            value: The duration to write.

        Returns:
            Its text, such as ``PT30S`` or ``-PT30S``.
        """
        return _json_text(_TIMEDELTA, value)


def format_temporal(value: Temporal, formatter: IValueFormatter) -> str:
    """Write a date, time or duration with the method's formatter.

    Args:
        value: The value to write.
        formatter: The formatter of the method.

    Returns:
        The text the formatter gives.
    """
    # datetime is a subclass of date, so it is matched first.
    if isinstance(value, datetime.datetime):
        return formatter.format_datetime(value)

    if isinstance(value, datetime.date):
        return formatter.format_date(value)

    if isinstance(value, datetime.time):
        return formatter.format_time(value)

    return formatter.format_timedelta(value)


def scalar_text(
    value: Any,
    formatter: IValueFormatter,
    *,
    label: str,
) -> str:
    """Write a single value as the text of a query, path or form field.

    One rule for every place a value becomes text: an enum is sent as its
    value, a boolean as ``true`` or ``false``, a date, time or duration
    through the formatter, and other types (``UUID``, ``Decimal``, URLs,
    UTF-8 ``bytes``) as Pydantic writes them in JSON.

    Args:
        value: A scalar, not ``None``.
        formatter: The formatter of the method.
        label: What the value is, for error messages, such as
            ``"Query parameter 'limit'"``.

    Returns:
        The text of the value.

    Raises:
        TypeError: If the value has no text form, such as a class or an
            object Pydantic cannot write, or is written as an object or a
            list.
        ValueError: If the value is NaN or infinite, which has no text an
            API can read, a datetime the formatter rejects, or ``bytes``
            that are not UTF-8.
    """
    while isinstance(value, Enum):
        value = value.value

    if isinstance(value, bool):
        return "true" if value else "false"

    if (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, Decimal) and not value.is_finite()
    ):
        raise ValueError(f"{label} cannot send NaN or infinity")

    if isinstance(value, TEMPORAL_TYPES):
        return format_temporal(value, formatter)

    if isinstance(value, (str, int, float)):
        return str(value)

    try:
        written = to_jsonable_python(value)
    except PydanticSerializationError as exc:
        raise TypeError(f"{label} cannot send {value!r}: {exc}") from exc

    if isinstance(written, (str, int, float)) and not isinstance(
        written, bool
    ):
        return str(written)

    raise TypeError(f"{label} must be a scalar, not {value!r}")
