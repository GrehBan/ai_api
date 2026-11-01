"""Block D: JSON codec and body normalization."""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import io
import math
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from ai_api.serialization.json import dumps, loads
from ai_api.serialization.normalize import MAX_DEPTH, BodyNormalizer
from ai_api.serialization.values import (
    BaseValueFormatter,
    Temporal,
    scalar_text,
)
from ai_api.types.base import BaseAPIType
from ai_api.types.files import BytesFile


@pytest.mark.parametrize(
    "value",
    [
        {"a": 1, "b": [True, None, 1.5], "c": {"d": "é"}},
        [],
        "text",
        0,
        None,
    ],
    ids=["nested", "empty-list", "str", "int", "null"],
)
def test_json_round_trip(value: Any) -> None:
    encoded = dumps(value)

    assert isinstance(encoded, bytes)
    assert loads(encoded) == value
    assert loads(encoded.decode()) == value


@pytest.mark.parametrize("raw", [b"{", b"", "not json", b"\xff"])
def test_loads_invalid_raises_value_error(raw: bytes | str) -> None:
    with pytest.raises(ValueError):
        loads(raw)


class Color(enum.Enum):
    RED = "red"


class Level(enum.IntEnum):
    HIGH = 3


class Wrapped(enum.Enum):
    NESTED = Color.RED


class Inner(BaseAPIType):
    name: str
    note: str | None = None


class Outer(BaseAPIType):
    inner: Inner
    tags: list[str]
    color: Color


ISO = BaseValueFormatter()
PLUS_3 = datetime.timezone(datetime.timedelta(hours=3))


@pytest.fixture
def normalizer() -> BodyNormalizer:
    return BodyNormalizer()


def test_none_is_dropped_only_from_fields(
    normalizer: BodyNormalizer,
) -> None:
    body = {
        "a": None,
        "b": {"c": None, "d": 1},
        "e": [None, 2, {"f": None}],
        "g": (None,),
        "h": Inner(name="n"),
    }

    assert normalizer.normalize_mapping(body, ISO) == {
        "b": {"c": None, "d": 1},
        "e": [None, 2, {"f": None}],
        "g": [None],
        "h": {"name": "n"},
    }


@pytest.mark.parametrize("kind", [dict, list])
def test_circular_reference_is_rejected(
    normalizer: BodyNormalizer,
    kind: type[Any],
) -> None:
    value: Any = kind()

    if isinstance(value, dict):
        value["self"] = value
    else:
        value.append(value)

    with pytest.raises(ValueError, match="circular reference"):
        normalizer.normalize_mapping({"a": value}, ISO)


def nested(depth: int) -> list[Any]:
    value: list[Any] = []

    for _ in range(depth - 1):
        value = [value]

    return value


def test_nesting_at_the_limit_passes(normalizer: BodyNormalizer) -> None:
    assert normalizer.normalize(nested(MAX_DEPTH), ISO) == nested(MAX_DEPTH)


@pytest.mark.parametrize("depth", [MAX_DEPTH + 1, 3000])
def test_deeper_nesting_is_rejected(
    normalizer: BodyNormalizer,
    depth: int,
) -> None:
    with pytest.raises(ValueError, match=f"deeper than {MAX_DEPTH}"):
        normalizer.normalize(nested(depth), ISO)


def test_shared_value_is_not_a_cycle(normalizer: BodyNormalizer) -> None:
    shared = {"x": 1}

    assert normalizer.normalize({"a": shared, "b": [shared, shared]}, ISO) == {
        "a": {"x": 1},
        "b": [{"x": 1}, {"x": 1}],
    }


def test_none_value_normalizes_to_none(normalizer: BodyNormalizer) -> None:
    assert normalizer.normalize(None, ISO) is None


@pytest.mark.parametrize(
    "value",
    [1, 1.5, "s", True, b"raw"],
    ids=["int", "float", "str", "bool", "bytes"],
)
def test_scalars_pass_through(
    normalizer: BodyNormalizer,
    value: Any,
) -> None:
    assert normalizer.normalize(value, ISO) is value


def test_tuple_becomes_list(normalizer: BodyNormalizer) -> None:
    assert normalizer.normalize((1, 2), ISO) == [1, 2]


def test_input_file_passes_through(normalizer: BodyNormalizer) -> None:
    file = BytesFile(b"data")

    assert normalizer.normalize(file, ISO) is file
    assert normalizer.normalize_mapping({"f": [file]}, ISO) == {"f": [file]}


def test_readable_stream_passes_through(normalizer: BodyNormalizer) -> None:
    stream = io.BytesIO(b"data")

    assert normalizer.normalize(stream, ISO) is stream


def test_closed_stream_is_rejected(normalizer: BodyNormalizer) -> None:
    stream = io.BytesIO(b"data")
    stream.close()

    with pytest.raises(ValueError, match="Invalid file stream"):
        normalizer.normalize(stream, ISO)


def test_unreadable_stream_is_rejected(
    normalizer: BodyNormalizer,
    tmp_path: Path,
) -> None:
    with (
        (tmp_path / "out.bin").open("wb") as stream,
        pytest.raises(ValueError, match="Invalid file stream"),
    ):
        normalizer.normalize(stream, ISO)


def test_nested_model_becomes_dict(normalizer: BodyNormalizer) -> None:
    value = Outer(inner=Inner(name="x"), tags=["a"], color=Color.RED)

    assert normalizer.normalize_mapping({"model": value}, ISO) == {
        "model": {"inner": {"name": "x"}, "tags": ["a"], "color": "red"},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(Color.RED, "red"), (Level.HIGH, 3), (Wrapped.NESTED, "red")],
    ids=["str-value", "int-enum", "enum-of-enum"],
)
def test_enum_becomes_value(
    normalizer: BodyNormalizer,
    value: enum.Enum,
    expected: Any,
) -> None:
    result = normalizer.normalize(value, ISO)

    assert result == expected
    assert not isinstance(result, enum.Enum)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC),
            "2026-01-02T03:04:05Z",
        ),
        (
            datetime.datetime(2026, 1, 2, 6, 4, 5, tzinfo=PLUS_3),
            "2026-01-02T06:04:05+03:00",
        ),
        (
            datetime.datetime(2026, 1, 2, 3, 4, 5, 600, tzinfo=datetime.UTC),
            "2026-01-02T03:04:05.000600Z",
        ),
        (datetime.date(2026, 1, 2), "2026-01-02"),
        (datetime.time(12, 30), "12:30:00"),
        (datetime.time(12, 30, tzinfo=datetime.UTC), "12:30:00Z"),
        (datetime.timedelta(seconds=30), "PT30S"),
        (datetime.timedelta(days=1, hours=2), "P1DT2H"),
        (datetime.timedelta(seconds=-30), "-PT30S"),
        (datetime.timedelta(0), "PT0S"),
        (datetime.timedelta(milliseconds=500), "PT0.5S"),
    ],
    ids=[
        "utc",
        "utc+3",
        "microseconds",
        "date",
        "time",
        "time-utc",
        "seconds",
        "days",
        "negative",
        "zero",
        "fraction",
    ],
)
def test_dates_and_durations_are_written_in_iso_8601(
    normalizer: BodyNormalizer,
    value: Temporal,
    expected: str,
) -> None:
    assert normalizer.normalize(value, ISO) == expected


def test_naive_datetime_is_rejected(normalizer: BodyNormalizer) -> None:
    naive = datetime.datetime(2026, 1, 2)  # noqa: DTZ001 - on purpose

    with pytest.raises(ValueError, match="timezone-aware"):
        normalizer.normalize(naive, ISO)


def test_dates_inside_containers(normalizer: BodyNormalizer) -> None:
    at = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
    body = {"at": [at], "in": {"ttl": datetime.timedelta(hours=1)}}

    assert normalizer.normalize_mapping(body, ISO) == {
        "at": ["2026-01-02T03:04:05Z"],
        "in": {"ttl": "PT1H"},
    }


class Unix(BaseValueFormatter):
    """Writes a moment as Unix seconds and keeps ISO for the rest."""

    def format_datetime(self, value: datetime.datetime, /) -> str:
        return str(round(value.timestamp()))


def test_formatter_decides_how_dates_are_written(
    normalizer: BodyNormalizer,
) -> None:
    body = {
        "at": datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC),
        "on": datetime.date(2026, 1, 2),
    }

    assert normalizer.normalize_mapping(body, Unix()) == {
        "at": "1767323045",
        "on": "2026-01-02",
    }


class Aliased(BaseAPIType):
    max_tokens: int = Field(serialization_alias="maxTokens")


def test_nested_model_is_dumped_by_alias(normalizer: BodyNormalizer) -> None:
    # The wire name, as for the method's own fields.
    value = {"inner": Aliased(max_tokens=7)}

    assert normalizer.normalize(value, ISO) == {"inner": {"maxTokens": 7}}


def test_enum_keys_are_unwrapped(normalizer: BodyNormalizer) -> None:
    result = normalizer.normalize({Color.RED: 1, Level.HIGH: [2]}, ISO)

    assert result == {"red": 1, 3: [2]}
    assert not any(isinstance(key, enum.Enum) for key in result)


@pytest.mark.parametrize(
    "value",
    [{Color.RED: 1, "red": 2}, {Color.RED: 1, Wrapped.NESTED: 2}],
    ids=["enum-and-str", "two-enums"],
)
def test_keys_that_become_equal_are_rejected(
    normalizer: BodyNormalizer,
    value: dict[Any, Any],
) -> None:
    # One of the two values would be lost silently.
    with pytest.raises(ValueError, match="key 'red' twice"):
        normalizer.normalize(value, ISO)


@pytest.mark.parametrize(
    "stream",
    [io.StringIO("text"), io.TextIOWrapper(io.BytesIO(b"text"))],
    ids=["string-io", "text-wrapper"],
)
def test_text_stream_is_rejected(
    normalizer: BodyNormalizer,
    stream: io.TextIOBase,
) -> None:
    with pytest.raises(TypeError, match="not a text stream"):
        normalizer.normalize_mapping({"f": [stream]}, ISO)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Color.RED, "red"),
        (Wrapped.NESTED, "red"),
        (Level.HIGH, "3"),
        (True, "true"),
        (False, "false"),
        (1.5, "1.5"),
        (10**30, str(10**30)),
        ("text", "text"),
        (b"bytes", "bytes"),
        (decimal.Decimal("1.50"), "1.50"),
        (uuid.UUID(int=1), "00000000-0000-0000-0000-000000000001"),
        (datetime.timedelta(seconds=30), "PT30S"),
    ],
    ids=[
        "enum",
        "enum-of-enum",
        "int-enum",
        "true",
        "false",
        "float",
        "big-int",
        "str",
        "bytes",
        "decimal",
        "uuid",
        "timedelta",
    ],
)
def test_scalar_text(value: Any, expected: str) -> None:
    assert scalar_text(value, ISO, label="Field 'x'") == expected


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        decimal.Decimal("NaN"),
        decimal.Decimal("Infinity"),
    ],
    ids=["nan", "inf", "-inf", "decimal-nan", "decimal-inf"],
)
def test_scalar_text_rejects_non_finite_numbers(value: Any) -> None:
    with pytest.raises(ValueError, match="'x' cannot send NaN or infinity"):
        scalar_text(value, ISO, label="Field 'x'")


@dataclasses.dataclass
class Pair:
    a: int


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (Color, "cannot send <enum 'Color'>"),
        (object(), "cannot send <object"),
        (Pair(1), "must be a scalar"),
    ],
    ids=["class", "object", "dataclass"],
)
def test_scalar_text_rejects_values_without_text(
    value: Any,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        scalar_text(value, ISO, label="Field 'x'")


def test_scalar_text_rejects_bytes_that_are_not_utf_8() -> None:
    with pytest.raises(ValueError, match="utf-8"):
        scalar_text(b"\xff", ISO, label="Field 'x'")
