"""Block G2: path parameters."""

from __future__ import annotations

import datetime
import enum
import math
from typing import Any, Literal

import pytest

from ai_api.enums import BodyFormat
from ai_api.methods.base import BaseAPIMethod
from tests.conftest import make_method


class Region(enum.StrEnum):
    EU = "eu-west"


def build_path(
    path: str,
    values: dict[str, Any],
    **class_vars: Any,
) -> str:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        path=path,
        fields=dict.fromkeys(values, Any),
        **class_vars,
    )
    return method_cls.model_validate(values).request_path


def test_without_params_path_is_returned_as_is() -> None:
    method_cls = make_method(BaseAPIMethod[bytes], path="/models")

    assert method_cls.__request_path_params__ == frozenset()
    assert method_cls().request_path == "/models"


def test_several_params() -> None:
    path = build_path(
        "/orgs/{org}/models/{model_id}",
        {"org": "acme", "model_id": 42},
    )

    assert path == "/orgs/acme/models/42"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(Region.EU, "eu-west"), (True, "true"), (False, "false")],
    ids=["enum", "true", "false"],
)
def test_value_conversion(value: Any, expected: str) -> None:
    assert build_path("/x/{value}", {"value": value}) == f"/x/{expected}"


@pytest.mark.parametrize(
    "annotation",
    [list[str], tuple[int, ...], set[str], dict[str, int], Region | list[str]],
    ids=["list", "tuple", "set", "dict", "union-with-list"],
)
def test_container_path_field_is_rejected(annotation: Any) -> None:
    # It has no single path segment; str() would send its Python repr.
    with pytest.raises(TypeError, match="'ids' must be a scalar"):
        make_method(
            BaseAPIMethod[bytes],
            path="/items/{ids}",
            fields={"ids": annotation},
        )


@pytest.mark.parametrize(
    "annotation",
    [str, int, float, bool, Region, str | None, bytes, Literal["a", "b"]],
    ids=[
        "str",
        "int",
        "float",
        "bool",
        "enum",
        "optional",
        "bytes",
        "literal",
    ],
)
def test_scalar_path_field_is_accepted(annotation: Any) -> None:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        path="/items/{item}",
        fields={"item": annotation},
    )

    assert method_cls.__request_path_params__ == {"item"}


@pytest.mark.parametrize(
    "value",
    [["a", "b"], {"a": 1}],
    ids=["list", "dict"],
)
def test_container_in_an_any_path_field_is_rejected(value: Any) -> None:
    with pytest.raises(TypeError, match="'ids' must be a scalar"):
        build_path("/items/{ids}", {"ids": value})


def test_literal_braces() -> None:
    assert build_path("/{{literal}}/{x}", {"x": "a"}) == "/{literal}/a"


def test_params_are_parsed_once_per_class() -> None:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        path="/a/{x}/{y}/{x}",
        fields={"x": str, "y": str},
    )

    assert method_cls.__request_path_params__ == frozenset({"x", "y"})
    assert (
        method_cls.model_validate({"x": "1", "y": "2"}).request_path
        == "/a/1/2/1"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("../admin", "..%2Fadmin"),
        ("a/b?x=1#f", "a%2Fb%3Fx%3D1%23f"),
        ("модель", "%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D1%8C"),
        ("a b", "a%20b"),
        ("50%", "50%25"),
    ],
    ids=["traversal", "query-and-fragment", "cyrillic", "space", "percent"],
)
def test_values_are_escaped(value: str, expected: str) -> None:
    assert build_path("/m/{value}", {"value": value}) == f"/m/{expected}"


def test_path_safe_keeps_slash() -> None:
    path = build_path(
        "/m/{model}",
        {"model": "org/model name"},
        __request_path_safe__="/",
    )

    assert path == "/m/org/model%20name"


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (None, "is not set"),
        ("", "has an unsafe value"),
        (".", "has an unsafe value"),
        ("..", "has an unsafe value"),
        (math.nan, "cannot send NaN or infinity"),
        (math.inf, "cannot send NaN or infinity"),
    ],
    ids=["none", "empty", "dot", "dot-dot", "nan", "inf"],
)
def test_unsafe_values(value: Any, match: str) -> None:
    with pytest.raises(ValueError, match=f"'value' {match}"):
        build_path("/m/{value}", {"value": value})


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_float_field_is_rejected(value: float) -> None:
    # Typed as float: the JSON dump gave "nan" here, not null.
    method_cls = make_method(
        BaseAPIMethod[bytes],
        path="/m/{value}",
        fields={"value": float},
    )

    with pytest.raises(ValueError, match="cannot send NaN or infinity"):
        _ = method_cls.model_validate({"value": value}).request_path


@pytest.mark.parametrize(
    ("annotation", "value", "expected"),
    [
        (
            datetime.datetime,
            datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC),
            "2026-01-02T03%3A04%3A05Z",
        ),
        (datetime.date, datetime.date(2026, 1, 2), "2026-01-02"),
        (datetime.timedelta, datetime.timedelta(seconds=30), "PT30S"),
    ],
    ids=["datetime", "date", "timedelta"],
)
def test_dates_are_written_in_iso_8601(
    annotation: Any,
    value: Any,
    expected: str,
) -> None:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        path="/m/{value}",
        fields={"value": annotation},
    )

    path = method_cls.model_validate({"value": value}).request_path

    assert path == f"/m/{expected}"


def test_naive_datetime_is_rejected() -> None:
    naive = datetime.datetime(2026, 1, 2)  # noqa: DTZ001 - on purpose

    with pytest.raises(ValueError, match="timezone-aware"):
        build_path("/m/{value}", {"value": naive})


@pytest.mark.parametrize(
    "value",
    ["a/../b", "a//b", "../x", "x/", "/x", "a/./b"],
    ids=[
        "inner-dot-dot",
        "double-slash",
        "leading-dot-dot",
        "trailing",
        "leading-slash",
        "inner-dot",
    ],
)
def test_unsafe_segments_with_path_safe(value: str) -> None:
    with pytest.raises(ValueError, match="unsafe value"):
        build_path("/m/{value}", {"value": value}, __request_path_safe__="/")


def test_dot_segments_are_escaped_without_path_safe() -> None:
    assert build_path("/m/{value}", {"value": "a/../b"}) == "/m/a%2F..%2Fb"


def test_missing_field() -> None:
    with pytest.raises(TypeError, match=r"placeholders \['model_id'\]"):
        make_method(BaseAPIMethod[bytes], path="/m/{model_id}")


@pytest.mark.parametrize(
    "path",
    [
        "/m/{}",
        "/m/{a.b}",
        "/m/{a[0]}",
        "/m/{a:>5}",
        "/m/{a!r}",
        "/m/{a",
        "/m/a}",
    ],
    ids=[
        "empty",
        "attribute",
        "index",
        "format-spec",
        "conversion",
        "unclosed",
        "unopened",
    ],
)
def test_invalid_templates(path: str) -> None:
    with pytest.raises(TypeError, match="invalid __request_path__"):
        make_method(BaseAPIMethod[bytes], path=path, fields={"a": str})


def test_path_and_query_overlap_in_same_class() -> None:
    with pytest.raises(TypeError, match=r"fields \['a'\] cannot be both"):
        make_method(
            BaseAPIMethod[bytes],
            path="/m/{a}",
            fields={"a": str},
            __request_query__={"a"},
        )


def test_path_and_query_overlap_in_subclass() -> None:
    base = make_method(BaseAPIMethod[bytes], path="/m/{a}", fields={"a": str})

    with pytest.raises(TypeError, match=r"fields \['a'\] cannot be both"):
        make_method(base, path=None, __request_query__={"a"})


def test_subclass_inherits_path_params() -> None:
    base = make_method(BaseAPIMethod[bytes], path="/m/{a}", fields={"a": str})
    sub = make_method(
        base,
        path=None,
        fields={"b": (int, 0)},
        __request_body_format__=BodyFormat.JSON,
    )

    assert sub.__request_path_params__ == frozenset({"a"})
    assert sub.model_validate({"a": "x/y"}).request_path == "/m/x%2Fy"


def test_subclass_with_own_path_replaces_params() -> None:
    base = make_method(BaseAPIMethod[bytes], path="/m/{a}", fields={"a": str})
    # Without its placeholder, "a" is a body field of the subclass.
    sub = make_method(
        base, path="/static", __request_body_format__=BodyFormat.JSON
    )

    assert sub.__request_path_params__ == frozenset()
