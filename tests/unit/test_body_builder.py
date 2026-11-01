"""Block I: the aiohttp body builder.

What ends up inside a multipart ``FormData`` (names, file names, content
types) is checked against a real server in stage 5; here only the public
outcome of ``build`` is checked: the kind of arguments, the rejections,
and the URL-encoded form rendered through the public payload API.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import io
import json
import math
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qsl

import pytest
from aiohttp import BytesPayload, FormData

from ai_api.enums import BodyFormat
from ai_api.http.exceptions.transport import APIRequestError
from ai_api.serialization.json import JsonDumps, dumps, loads
from ai_api.serialization.normalize import BodyNormalizer
from ai_api.session.body.aiohttp import AiohttpBodyBuilder
from ai_api.types.base import BaseAPIType
from ai_api.types.files import BytesFile, PathFile, StreamFile


class Color(enum.Enum):
    RED = "red"


class Point(BaseAPIType):
    x: int
    label: str | None = None


class KeepNone(BodyNormalizer):
    """Normalizer that keeps top-level ``None`` values."""

    def normalize_mapping(self, body: Mapping[str, Any]) -> dict[str, Any]:
        return dict(body)


async def empty() -> AsyncIterator[bytes]:
    yield b""


@pytest.fixture
def builder() -> AiohttpBodyBuilder:
    return AiohttpBodyBuilder()


def form_of(kwargs: dict[str, Any]) -> FormData:
    assert set(kwargs) == {"data"}
    form = kwargs["data"]
    assert isinstance(form, FormData)
    return form


def json_of(kwargs: dict[str, Any]) -> Any:
    assert set(kwargs) == {"data"}
    payload = kwargs["data"]
    assert isinstance(payload, BytesPayload)
    assert payload.content_type == "application/json"
    return loads(payload.decode())


def urlencoded(form: FormData) -> list[tuple[str, str]]:
    return parse_qsl(form().decode(), keep_blank_values=True)


def test_none_format_sends_nothing(builder: AiohttpBodyBuilder) -> None:
    assert builder.build({}, BodyFormat.NONE) == {}


def test_none_format_refuses_a_body(builder: AiohttpBodyBuilder) -> None:
    with pytest.raises(ValueError, match="without a body cannot send: a"):
        builder.build({"a": 1}, BodyFormat.NONE)


def test_json_is_normalized(builder: AiohttpBodyBuilder) -> None:
    at = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC)
    body = {
        "color": Color.RED,
        "point": Point(x=1),
        "at": at,
        "skip": None,
        "items": [1, None, "a"],
    }

    assert json_of(builder.build(body, BodyFormat.JSON)) == {
        "color": "red",
        "point": {"x": 1},
        "at": "1767323045",
        "items": [1, None, "a"],
    }


def test_json_uses_the_custom_encoder() -> None:
    builder = AiohttpBodyBuilder(json_dumps=lambda value: b'"custom"')

    assert json_of(builder.build({"a": 1}, BodyFormat.JSON)) == "custom"


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf, [{"x": math.nan}]],
    ids=["nan", "inf", "-inf", "nested"],
)
@pytest.mark.parametrize(
    "body_format",
    [BodyFormat.JSON, BodyFormat.FORM],
    ids=["json", "form-mapping"],
)
def test_non_finite_floats_are_rejected(
    builder: AiohttpBodyBuilder,
    value: Any,
    body_format: BodyFormat,
) -> None:
    with pytest.raises(ValueError, match="NaN and infinity"):
        builder.build({"meta": {"v": value}}, body_format)


@pytest.mark.parametrize(
    "value",
    [{1, 2}, decimal.Decimal("1.5"), 2**64, {1: "a"}],
    ids=["set", "decimal", "big-int", "int-key"],
)
@pytest.mark.parametrize(
    "body_format",
    [BodyFormat.JSON, BodyFormat.FORM],
    ids=["json", "form-mapping"],
)
def test_unsupported_json_values_are_rejected(
    builder: AiohttpBodyBuilder,
    value: Any,
    body_format: BodyFormat,
) -> None:
    with pytest.raises(TypeError, match="cannot be encoded as JSON"):
        builder.build({"meta": {"v": value}}, body_format)


def test_missing_upload_file_is_rejected(
    builder: AiohttpBodyBuilder,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(APIRequestError, match="missing.txt") as info:
        builder.build({"f": PathFile(missing)}, BodyFormat.MULTIPART)

    assert info.value.transient is False
    assert isinstance(info.value.__cause__, FileNotFoundError)


def test_directory_upload_is_rejected(
    builder: AiohttpBodyBuilder,
    tmp_path: Path,
) -> None:
    with pytest.raises(APIRequestError, match="Is a directory") as info:
        builder.build({"f": PathFile(tmp_path)}, BodyFormat.MULTIPART)

    assert isinstance(info.value.__cause__, IsADirectoryError)


def test_json_dumps_returning_str_is_encoded() -> None:
    # json.dumps returns str; the type only promises bytes.
    builder = AiohttpBodyBuilder(json_dumps=cast(JsonDumps, json.dumps))

    assert json_of(builder.build({"a": "é"}, BodyFormat.JSON)) == {"a": "é"}


@pytest.mark.parametrize(
    ("body_format", "multipart"),
    [(BodyFormat.FORM, False), (BodyFormat.MULTIPART, True)],
    ids=["form", "multipart"],
)
def test_forms_produce_form_data(
    builder: AiohttpBodyBuilder,
    body_format: BodyFormat,
    multipart: bool,
) -> None:
    form = form_of(builder.build({"a": "1"}, body_format))

    assert form.is_multipart is multipart


def test_multipart_accepts_files(builder: AiohttpBodyBuilder) -> None:
    body = {
        "file": BytesFile(b"x", filename="a.txt"),
        "stream": StreamFile(empty()),
        "raw": b"raw",
        "io": io.BytesIO(b"io"),
        "files": [BytesFile(b"1"), BytesFile(b"2")],
        "name": "n",
    }

    assert form_of(builder.build(body, BodyFormat.MULTIPART)).is_multipart


def test_unknown_format_is_rejected(builder: AiohttpBodyBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported body format"):
        builder.build({"a": 1}, cast(BodyFormat, "xml"))


@pytest.mark.parametrize(
    "body",
    [
        {"file": BytesFile(b"x")},
        {"raw": b"x"},
        {"raw": bytearray(b"x")},
        {"raw": memoryview(b"x")},
        {"stream": io.BytesIO(b"x")},
        {"nested": {"deep": [b"x"]}},
        {"items": [1, (BytesFile(b"x"),)]},
    ],
    ids=[
        "file",
        "bytes",
        "bytearray",
        "memoryview",
        "stream",
        "nested-dict",
        "nested-list",
    ],
)
def test_binary_data_in_json_is_rejected(
    builder: AiohttpBodyBuilder,
    body: dict[str, Any],
) -> None:
    with pytest.raises(TypeError, match="cannot be sent as JSON"):
        builder.build(body, BodyFormat.JSON)


@pytest.mark.parametrize(
    "value",
    [BytesFile(b"x"), b"x", io.BytesIO(b"x"), [BytesFile(b"x")]],
    ids=["file", "bytes", "stream", "list"],
)
def test_files_in_urlencoded_form_are_rejected(
    builder: AiohttpBodyBuilder,
    value: Any,
) -> None:
    with pytest.raises(TypeError, match="Files require multipart"):
        builder.build({"f": value}, BodyFormat.FORM)


@pytest.mark.parametrize(
    "body_format",
    [BodyFormat.FORM, BodyFormat.MULTIPART],
    ids=["form", "multipart"],
)
def test_text_stream_is_rejected(
    builder: AiohttpBodyBuilder,
    body_format: BodyFormat,
) -> None:
    with pytest.raises(TypeError, match="Only binary streams"):
        builder.build({"f": io.StringIO("text")}, body_format)


def test_binary_data_inside_form_mapping_is_rejected(
    builder: AiohttpBodyBuilder,
) -> None:
    with pytest.raises(TypeError, match="cannot be sent as JSON"):
        builder.build({"meta": {"raw": b"x"}}, BodyFormat.MULTIPART)


def test_form_field_encoding(builder: AiohttpBodyBuilder) -> None:
    body = {
        "tags": ["a", "b"],
        "meta": {"k": [1, True], "color": Color.RED},
        "on": True,
        "off": False,
        "n": 5,
        "ratio": 0.5,
        "text": "plain",
    }

    form = form_of(builder.build(body, BodyFormat.FORM))

    assert urlencoded(form) == [
        ("tags", "a"),
        ("tags", "b"),
        ("meta", dumps({"k": [1, True], "color": "red"}).decode()),
        ("on", "true"),
        ("off", "false"),
        ("n", "5"),
        ("ratio", "0.5"),
        ("text", "plain"),
    ]


def test_custom_json_dumps_for_form_mappings() -> None:
    builder = AiohttpBodyBuilder(json_dumps=lambda value: b"<json>")

    form = form_of(builder.build({"meta": {"a": 1}}, BodyFormat.FORM))

    assert urlencoded(form) == [("meta", "<json>")]


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (["a", None], ValueError, "'tags' cannot send None"),
        ([["a", "b"], ["c"]], TypeError, "'tags' cannot send nested lists"),
    ],
    ids=["none", "nested"],
)
@pytest.mark.parametrize(
    "body_format", [BodyFormat.FORM, BodyFormat.MULTIPART]
)
def test_form_lists_it_cannot_express_are_rejected(
    builder: AiohttpBodyBuilder,
    body_format: BodyFormat,
    value: list[Any],
    error: type[Exception],
    match: str,
) -> None:
    # A repeated field has no null, and nested lists would be flattened.
    with pytest.raises(error, match=match):
        builder.build({"tags": value}, body_format)


def test_none_fields_are_skipped() -> None:
    builder = AiohttpBodyBuilder(normalizer=KeepNone())

    form = form_of(builder.build({"a": None, "b": "1"}, BodyFormat.FORM))

    assert urlencoded(form) == [("b", "1")]
