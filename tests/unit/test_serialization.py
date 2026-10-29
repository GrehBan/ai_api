"""Block D: JSON codec and body normalization."""

import datetime
import enum
import io
from pathlib import Path
from typing import Any

import pytest

from ai_api.client.serialization.body import MAX_DEPTH, BodyNormalizer
from ai_api.client.serialization.json import dumps, loads
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

    assert normalizer.normalize_mapping(body) == {
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
        normalizer.normalize_mapping({"a": value})


def nested(depth: int) -> list[Any]:
    value: list[Any] = []

    for _ in range(depth - 1):
        value = [value]

    return value


def test_nesting_at_the_limit_passes(normalizer: BodyNormalizer) -> None:
    assert normalizer.normalize(nested(MAX_DEPTH)) == nested(MAX_DEPTH)


@pytest.mark.parametrize("depth", [MAX_DEPTH + 1, 3000])
def test_deeper_nesting_is_rejected(
    normalizer: BodyNormalizer,
    depth: int,
) -> None:
    with pytest.raises(ValueError, match=f"deeper than {MAX_DEPTH}"):
        normalizer.normalize(nested(depth))


def test_shared_value_is_not_a_cycle(normalizer: BodyNormalizer) -> None:
    shared = {"x": 1}

    assert normalizer.normalize({"a": shared, "b": [shared, shared]}) == {
        "a": {"x": 1},
        "b": [{"x": 1}, {"x": 1}],
    }


def test_none_value_normalizes_to_none(normalizer: BodyNormalizer) -> None:
    assert normalizer.normalize(None) is None


@pytest.mark.parametrize(
    "value",
    [1, 1.5, "s", True, b"raw"],
    ids=["int", "float", "str", "bool", "bytes"],
)
def test_scalars_pass_through(
    normalizer: BodyNormalizer,
    value: Any,
) -> None:
    assert normalizer.normalize(value) is value


def test_tuple_becomes_list(normalizer: BodyNormalizer) -> None:
    assert normalizer.normalize((1, 2)) == [1, 2]


def test_input_file_passes_through(normalizer: BodyNormalizer) -> None:
    file = BytesFile(b"data")

    assert normalizer.normalize(file) is file
    assert normalizer.normalize_mapping({"f": [file]}) == {"f": [file]}


def test_readable_stream_passes_through(normalizer: BodyNormalizer) -> None:
    stream = io.BytesIO(b"data")

    assert normalizer.normalize(stream) is stream


def test_closed_stream_is_rejected(normalizer: BodyNormalizer) -> None:
    stream = io.BytesIO(b"data")
    stream.close()

    with pytest.raises(ValueError, match="Invalid file stream"):
        normalizer.normalize(stream)


def test_unreadable_stream_is_rejected(
    normalizer: BodyNormalizer,
    tmp_path: Path,
) -> None:
    with (
        (tmp_path / "out.bin").open("wb") as stream,
        pytest.raises(ValueError, match="Invalid file stream"),
    ):
        normalizer.normalize(stream)


def test_nested_model_becomes_dict(normalizer: BodyNormalizer) -> None:
    value = Outer(inner=Inner(name="x"), tags=["a"], color=Color.RED)

    assert normalizer.normalize_mapping({"model": value}) == {
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
    result = normalizer.normalize(value)

    assert result == expected
    assert not isinstance(result, enum.Enum)


@pytest.mark.parametrize(
    "value",
    [
        datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC),
        datetime.datetime(
            2026,
            1,
            2,
            6,
            4,
            5,
            tzinfo=datetime.timezone(datetime.timedelta(hours=3)),
        ),
    ],
    ids=["utc", "utc+3"],
)
def test_aware_datetime_becomes_unix_timestamp_string(
    normalizer: BodyNormalizer,
    value: datetime.datetime,
) -> None:
    assert normalizer.normalize(value) == "1767323045"


def test_aware_datetime_is_rounded(normalizer: BodyNormalizer) -> None:
    value = datetime.datetime(
        2026, 1, 2, 3, 4, 5, 600_000, tzinfo=datetime.UTC
    )

    assert normalizer.normalize(value) == "1767323046"


def test_naive_datetime_is_rejected(normalizer: BodyNormalizer) -> None:
    naive = datetime.datetime(2026, 1, 2)  # noqa: DTZ001 - on purpose

    with pytest.raises(ValueError, match="timezone-aware"):
        normalizer.normalize(naive)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (datetime.timedelta(hours=1), 3600),
        (datetime.timedelta(0), 0),
        (datetime.timedelta(seconds=-30), -30),
    ],
    ids=["hour", "zero", "negative"],
)
def test_timedelta_is_now_plus_delta(
    normalizer: BodyNormalizer,
    frozen_now: datetime.datetime,
    delta: datetime.timedelta,
    expected: int,
) -> None:
    now = round(frozen_now.timestamp())

    assert normalizer.normalize(delta) == str(now + expected)


def test_datetimes_inside_containers(
    normalizer: BodyNormalizer,
    frozen_now: datetime.datetime,
) -> None:
    body = {"at": [frozen_now], "in": {"delta": datetime.timedelta(1)}}

    assert normalizer.normalize_mapping(body) == {
        "at": ["1767323045"],
        "in": {"delta": str(1767323045 + 86400)},
    }
