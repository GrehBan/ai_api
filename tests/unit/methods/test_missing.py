"""Optional fields: ``MISSING`` leaves a field out, ``None`` sends null."""

from __future__ import annotations

import dataclasses
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    NamedTuple,
    NewType,
    NotRequired,
    TypedDict,
)

import pytest
from pydantic import (
    MISSING,
    BaseModel,
    Field,
    PlainSerializer,
    PydanticUserError,
    WrapSerializer,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from ai_api.enums import BodyFormat
from ai_api.methods.base import BaseAPIMethod
from ai_api.methods.fields import names_missing
from ai_api.serialization.normalize import BodyNormalizer
from ai_api.serialization.values import BaseValueFormatter
from ai_api.types.base import BaseAPIType
from tests.conftest import make_method

if TYPE_CHECKING:
    from decimal import Decimal


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


type JSON = dict[str, JSON] | list[JSON] | str | int | float | bool | None
type Ping = list[Pong] | int
type Pong = dict[str, Ping]
type Branches[T] = list[Branches[T]] | T


class Early(BaseAPIMethod[bytes]):
    """Names models declared further down: checked at its first use."""

    __request_path__ = "/early"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    late: LateLoose | MISSING = MISSING


class EarlyStrict(BaseAPIMethod[bytes]):
    __request_path__ = "/early"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    late: LateStrict


class LateLoose(BaseModel):
    note: str | None = None


class LateStrict(BaseModel):
    note: str | None


class Unresolved(TypedDict):
    amount: Decimal  # imported only for type checkers


class Pending(BaseAPIMethod[bytes]):
    """Names a type that never resolves: Pydantic keeps it incomplete."""

    __request_path__ = "/pending"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.JSON

    data: Unresolved


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
        pytest.param(Loose | int, id="union"),
        pytest.param(Annotated[Loose, "meta"], id="annotated"),
        pytest.param(Looses, id="type-alias"),
        pytest.param(Many[Loose], id="generic-alias"),
        pytest.param(Branches[Loose], id="recursive-generic-alias"),
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
        pytest.param(list[Tree], id="typeddict-list"),
        # A cycle is legal in a type: the walk stops at it.
        pytest.param(JSON, id="recursive-alias"),
        pytest.param(Ping, id="mutually-recursive-aliases"),
        pytest.param(Branches[Strict], id="recursive-generic"),
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


def test_a_model_in_the_body_sends_none_as_null() -> None:
    # No exception for models: None is null here too, also for a field
    # that was absent from the response the model came from.
    reply = Reply.model_validate({"text": "t", "seed": None})

    assert Send(item=reply).get_body_map() == {
        "item": {"text": "t", "note": None, "seed": None}
    }


def test_exclude_unset_sends_back_what_came() -> None:
    # The documented way to send a response item back as it came: the
    # absent field stays absent, an explicit null stays null.
    absent = Reply.model_validate({"text": "t", "seed": None})
    explicit = Reply.model_validate({"text": "t", "note": None, "seed": 1})

    assert Send(
        item=[
            absent.model_dump(by_alias=True, exclude_unset=True),
            explicit.model_dump(by_alias=True, exclude_unset=True),
        ]
    ).get_body_map() == {
        "item": [
            {"text": "t", "seed": None},
            {"text": "t", "note": None, "seed": 1},
        ]
    }


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


def test_a_forward_reference_is_checked_at_first_use() -> None:
    # The class completes, and its types are looked into, only once
    # LateLoose exists; every instance raises, not only the first.
    for _ in range(2):
        with pytest.raises(
            TypeError, match=r"Early: field 'late' can hold LateLoose"
        ):
            Early()

    assert Early.__definition_error__ is not None


def test_a_clean_forward_reference_is_accepted() -> None:
    method = EarlyStrict(late=LateStrict(note=None))

    assert method.get_body_map() == {"late": {"note": None}}
    assert EarlyStrict.__definition_error__ is None


def test_an_unresolved_name_does_not_fail_the_definition() -> None:
    # Like any Pydantic model, the class waits for the name; it is not
    # looked into before, so the definition does not raise NameError.
    assert Pending.__pydantic_complete__ is False

    with pytest.raises(PydanticUserError, match="not fully defined"):
        Pending.model_validate({"data": {"amount": 1}})


@dataclasses.dataclass
class LooseRecord:
    name: str
    note: str | None = None


@dataclasses.dataclass
class FactoryRecord:
    tags: list[str] | None = dataclasses.field(default_factory=lambda: None)


@dataclasses.dataclass
class CleanRecord:
    name: str
    tags: list[str] = dataclasses.field(default_factory=list)


@pydantic_dataclass
class LooseData:
    note: str | None = None


class LoosePair(NamedTuple):
    name: str
    note: str | None = None


class FactoryModel(BaseModel):
    note: str | None = Field(default_factory=lambda: None)


class FromData(BaseModel):
    name: str
    # Takes the validated data: it cannot be called without an instance.
    alias: str | None = Field(default_factory=lambda data: data["name"])


class HoldsRecord(BaseModel):
    record: LooseRecord


LooseId = NewType("LooseId", Loose)
StrictId = NewType("StrictId", Strict)
type Spin = list[Spun] | MISSING
type Spun = Spin | int
type MaybeCount = int | MISSING
# A cycle through unions only: Python takes it, type checkers do not.
type Round = Trip | int  # pyright: ignore[reportGeneralTypeIssues]
type Trip = Round | str


@pytest.mark.parametrize(
    ("annotation", "owner"),
    [
        pytest.param(LooseRecord, "LooseRecord", id="dataclass"),
        pytest.param(FactoryRecord, "FactoryRecord", id="dataclass-factory"),
        pytest.param(LooseData, "LooseData", id="pydantic-dataclass"),
        pytest.param(LoosePair, "LoosePair", id="namedtuple"),
        pytest.param(LooseId, "Loose", id="newtype"),
        pytest.param(FactoryModel, "FactoryModel", id="model-factory"),
        pytest.param(HoldsRecord, "LooseRecord", id="dataclass-in-model"),
    ],
)
def test_every_class_of_data_is_looked_into(
    annotation: Any,
    owner: str,
) -> None:
    # Each of them is dumped with its defaults, so a None default would
    # send null just as a model's.
    with pytest.raises(TypeError, match=rf"'data' can hold {owner}, whose"):
        make_method(
            BaseAPIMethod[bytes],
            fields={"data": annotation},
            __request_body_format__=BodyFormat.JSON,
        )


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(CleanRecord, id="dataclass"),
        pytest.param(FromData, id="factory-taking-data"),
        pytest.param(StrictId, id="newtype"),
        pytest.param(type[int], id="class"),
    ],
)
def test_classes_without_none_defaults_are_accepted(annotation: Any) -> None:
    make_method(
        BaseAPIMethod[bytes],
        fields={"data": annotation},
        __request_body_format__=BodyFormat.JSON,
    )


def test_a_default_factory_giving_none_is_a_none_default() -> None:
    with pytest.raises(TypeError, match="'tags' defaults to None"):
        make_method(
            BaseAPIMethod[bytes],
            fields={
                "tags": (list[str] | None, Field(default_factory=lambda: None))
            },
            __request_body_format__=BodyFormat.JSON,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        pytest.param(
            (int | MISSING, ...),
            "takes MISSING but does not default to it",
            id="required",
        ),
        pytest.param(
            (int | MISSING, 5),
            "takes MISSING but does not default to it",
            id="other-default",
        ),
        pytest.param(
            (int, MISSING),
            "defaults to MISSING, which its type does not take",
            id="type-without-missing",
        ),
    ],
)
def test_missing_goes_with_its_default(
    field: tuple[Any, Any],
    message: str,
) -> None:
    # A required field that takes MISSING could be left out of a request
    # it is required in; a MISSING default the type does not take is
    # not a value of the field.
    with pytest.raises(TypeError, match=rf"'x' {message}"):
        make_method(
            BaseAPIMethod[bytes],
            fields={"x": field},
            __request_body_format__=BodyFormat.JSON,
        )


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(Any, id="any"),
        pytest.param(Annotated[int | MISSING, "meta"], id="annotated"),
        pytest.param(MaybeCount, id="alias"),
        pytest.param(Spin, id="recursive-alias"),
    ],
)
def test_types_that_take_missing_default_to_it(annotation: Any) -> None:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        fields={"x": (annotation, MISSING)},
        __request_body_format__=BodyFormat.JSON,
    )

    assert method_cls().get_body_map() == {}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1, MISSING], id="list-item"),
        pytest.param({"k": MISSING}, id="dict-value"),
        pytest.param(MISSING, id="field"),
    ],
)
def test_missing_inside_a_value_cannot_be_sent(value: Any) -> None:
    # It leaves a whole field out; as an item it would reach the encoder
    # and fail there with a message about an unknown type.
    with pytest.raises(ValueError, match="MISSING cannot be sent"):
        BodyNormalizer().normalize_mapping(
            {"item": value}, BaseValueFormatter()
        )


def test_missing_in_a_query_list_cannot_be_sent() -> None:
    method_cls = make_method(
        BaseAPIMethod[bytes],
        http_method="GET",
        fields={"ids": (list[Any] | MISSING, MISSING)},
        __request_query__={"ids"},
    )

    with pytest.raises(ValueError, match="'ids' cannot send MISSING"):
        method_cls.model_validate({"ids": ["a", MISSING]}).get_query_map()


def test_a_cycle_of_aliases_ends_the_missing_lookup() -> None:
    # The lookup stops at an alias it has seen instead of recursing.
    assert names_missing(Spun)
    assert not names_missing(Round)


ByName = Annotated[Loose, PlainSerializer(lambda value: value.name)]


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(ByName, id="field"),
        pytest.param(list[ByName], id="nested"),
    ],
)
def test_a_plain_serializer_decides_what_is_sent(annotation: Any) -> None:
    # The model is never dumped, so its None default is never sent.
    method_cls = make_method(
        BaseAPIMethod[bytes],
        fields={"data": annotation},
        __request_body_format__=BodyFormat.JSON,
    )

    loose = Loose(name="n")
    value = loose if annotation is ByName else [loose]
    expected = "n" if annotation is ByName else ["n"]
    method = method_cls.model_validate({"data": value})
    assert method.get_body_map() == {"data": expected}


def test_a_wrap_serializer_is_still_looked_into() -> None:
    # It may call the default dump, which sends the model's defaults.
    with pytest.raises(TypeError, match="can hold Loose"):
        make_method(
            BaseAPIMethod[bytes],
            fields={
                "data": Annotated[
                    Loose,
                    WrapSerializer(lambda value, handler: handler(value)),
                ]
            },
            __request_body_format__=BodyFormat.JSON,
        )
