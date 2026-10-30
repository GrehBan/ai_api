"""Block G2: path parameters."""

from __future__ import annotations

import enum
import math
from typing import Any

import pytest

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
        (math.nan, "has an unsafe value: nan"),
        (math.inf, "has an unsafe value: inf"),
    ],
    ids=["none", "empty", "dot", "dot-dot", "nan", "inf"],
)
def test_unsafe_values(value: Any, match: str) -> None:
    with pytest.raises(ValueError, match=f"'value' {match}"):
        build_path("/m/{value}", {"value": value})


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
    sub = make_method(base, path=None, fields={"b": (int, 0)})

    assert sub.__request_path_params__ == frozenset({"a"})
    assert sub.model_validate({"a": "x/y"}).request_path == "/m/x%2Fy"


def test_subclass_with_own_path_replaces_params() -> None:
    base = make_method(BaseAPIMethod[bytes], path="/m/{a}", fields={"a": str})
    sub = make_method(base, path="/static")

    assert sub.__request_path_params__ == frozenset()
