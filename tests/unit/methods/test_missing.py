"""Optional fields: ``MISSING`` leaves a field out, ``None`` sends null."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

import pytest
from pydantic import MISSING, BaseModel, Field

from ai_api.enums import BodyFormat
from ai_api.methods.base import BaseAPIMethod
from ai_api.types.base import BaseAPIType
from tests.conftest import make_method


class Loose(BaseModel):
    """Request data whose field defaults to None: it would send null."""

    name: str
    note: str | None = None


class Holder(BaseModel):
    loose: Loose


class LooseKeys(TypedDict):
    loose: NotRequired[Loose]


type Looses = list[Loose]
type Many[T] = list[T]


class Strict(BaseModel):
    """Request data that says what it sends: null only if given."""

    name: str
    note: str | None


class Node(BaseModel):
    name: str
    children: list[Node] = Field(default_factory=list)


class Tree(TypedDict):
    label: str
    note: NotRequired[str | None]
    kids: NotRequired[list[Tree]]


class Reply(BaseAPIType):
    """A response item, sent back in the next request."""

    text: str
    note: str | None = None
    seed: int | None


class Send(BaseAPIMethod[bytes]):
    __request_path__ = "/send"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    count: int | MISSING = MISSING
    cursor: str | None | MISSING = MISSING
    options: dict[str, Any] | MISSING = MISSING
    item: Any = MISSING


class Search(BaseAPIMethod[bytes]):
    __request_path__ = "/search"
    __request_method__ = "GET"
    __request_query__ = frozenset({"q", "tags"})

    q: str | None | MISSING = MISSING
    tags: list[str] | MISSING = MISSING


@pytest.mark.parametrize("annotation", [int | None, Any])
def test_a_none_default_is_rejected(annotation: Any) -> None:
    with pytest.raises(TypeError, match="'x' defaults to None, which is sent"):
        make_method(
            BaseAPIMethod[bytes],
            fields={"x": (annotation, None)},
            __request_body_format__=BodyFormat.JSON,
        )


def test_an_excluded_none_default_is_never_sent() -> None:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        fields={"note": (str | None, Field(default=None, exclude=True))},
        __request_body_format__=BodyFormat.JSON,
    )

    assert method_cls().get_body_map() == {}


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(Loose, id="model"),
        pytest.param(list[Loose], id="list"),
        pytest.param(dict[str, Loose], id="dict"),
        pytest.param(Loose | int | MISSING, id="union"),
        pytest.param(Annotated[Loose, "meta"], id="annotated"),
        pytest.param(Looses, id="type-alias"),
        pytest.param(Many[Loose], id="generic-alias"),
        pytest.param(Holder, id="nested-model"),
        pytest.param(LooseKeys, id="typeddict"),
    ],
)
def test_a_model_with_a_none_default_in_the_type_is_rejected(
    annotation: Any,
) -> None:
    # The model's own defaults are dumped with it, so its None default
    # would send null just as a field's.
    with pytest.raises(
        TypeError, match=r"'data' can hold Loose, whose fields \['note'\]"
    ):
        make_method(
            BaseAPIMethod[bytes],
            fields={"data": annotation},
            __request_body_format__=BodyFormat.JSON,
        )


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(Strict, id="model"),
        pytest.param(Node, id="recursive-model"),
        pytest.param(Tree, id="recursive-typeddict"),
        pytest.param(list[Tree] | MISSING, id="typeddict-list"),
    ],
)
def test_request_data_without_none_defaults_is_accepted(
    annotation: Any,
) -> None:
    make_method(
        BaseAPIMethod[bytes],
        fields={"data": annotation},
        __request_body_format__=BodyFormat.JSON,
    )


def test_missing_is_left_out_and_none_is_sent_as_null() -> None:
    assert Send().get_body_map() == {}
    assert Send(
        cursor=None, options={"k": None}, item=[None]
    ).get_body_map() == {
        "cursor": None,
        "options": {"k": None},
        "item": [None],
    }
    assert Send(count=1, cursor="c").get_body_map() == {
        "count": 1,
        "cursor": "c",
    }


def test_typeddict_keys_not_given_are_left_out() -> None:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        fields={"tree": Tree},
        __request_body_format__=BodyFormat.JSON,
    )

    method = method_cls.model_validate(
        {"tree": {"label": "a", "kids": [{"label": "b", "note": None}]}}
    )

    assert method.get_body_map() == {
        "tree": {"label": "a", "kids": [{"label": "b", "note": None}]}
    }


def test_a_model_sent_back_keeps_the_shape_it_came_in() -> None:
    # A field that is None by default was absent from the response;
    # a required one that is None was null in it.
    absent = Reply.model_validate({"text": "t", "seed": None})
    given = Reply(text="t", note="n", seed=1)

    assert Send(item=absent).get_body_map() == {
        "item": {"text": "t", "seed": None}
    }
    assert Send(item=[given]).get_body_map() == {
        "item": [{"text": "t", "note": "n", "seed": 1}]
    }
    # Outside a request the model dumps as Pydantic does.
    assert absent.model_dump() == {"text": "t", "note": None, "seed": None}


def test_missing_is_left_out_of_the_query() -> None:
    assert Search().get_query_map() == {}
    assert Search(q="cats", tags=["a"]).get_query_map() == {
        "q": "cats",
        "tags": ["a"],
    }


def test_none_in_the_query_is_refused() -> None:
    # A query has no null; skipping it would send another request.
    with pytest.raises(ValueError, match="'q' cannot send None"):
        Search(q=None).get_query_map()


def test_a_path_parameter_cannot_default_to_missing() -> None:
    with pytest.raises(TypeError, match="'item_id' cannot default to MISSING"):
        make_method(
            BaseAPIMethod[bytes],
            path="/items/{item_id}",
            fields={"item_id": (str | MISSING, MISSING)},
        )
