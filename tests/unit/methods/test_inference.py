"""Block G1: inference of the response type and format."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, TypeVar

import pytest
from pydantic import Field

from ai_api.enums import ResponseFormat
from ai_api.methods.base import BaseAPIMethod, BaseMethod, BaseStreamMethod
from ai_api.response.parse.buffered import BufferedResponseParser
from ai_api.response.parse.sse import ServerSentEvent
from ai_api.response.parse.stream import StreamResponseParser
from ai_api.types.base import BaseAPIType
from tests.conftest import make_method


class Model(BaseAPIType):
    value: int


class Event(BaseAPIType):
    data: str


T = TypeVar("T")


@pytest.mark.parametrize(
    ("argument", "returning", "response_format"),
    [
        (Model, Model, ResponseFormat.JSON),
        (str, str, ResponseFormat.TEXT),
        (bytes, bytes, ResponseFormat.BYTES),
        (None, type(None), ResponseFormat.NONE),
    ],
    ids=["model", "str", "bytes", "none"],
)
def test_buffered_inference(
    argument: Any,
    returning: type[Any],
    response_format: ResponseFormat,
) -> None:
    method_cls = make_method(BaseAPIMethod[argument])

    assert method_cls.__returning__ is returning
    assert method_cls.__response_format__ is response_format

    method = method_cls()

    assert method.response_type is returning
    assert method.response_format is response_format


def test_explicit_format_beats_inference() -> None:
    # Only an explicit parser may parse into another type than declared.
    class Raw(BaseAPIMethod[str]):
        __request_path__ = "/raw"
        __request_method__ = "GET"
        __response_format__ = ResponseFormat.BYTES
        __response_parser__ = BufferedResponseParser()

    assert Raw.__response_format__ is ResponseFormat.BYTES
    assert Raw.__returning__ is str


def test_explicit_returning_beats_inference() -> None:
    class Custom(BaseAPIMethod[Model]):
        __request_path__ = "/custom"
        __request_method__ = "GET"
        __returning__: ClassVar[type[Any] | None] = Event

    assert Custom.__returning__ is Event
    assert Custom.__response_format__ is ResponseFormat.JSON


def test_format_from_intermediate_generic_base_is_inherited() -> None:
    class TextBase[R](BaseAPIMethod[R]):
        __response_format__ = ResponseFormat.TEXT
        __response_parser__ = BufferedResponseParser()

    class Leaf(TextBase[bytes]):
        __request_path__ = "/leaf"
        __request_method__ = "GET"

    assert Leaf.__response_format__ is ResponseFormat.TEXT
    assert Leaf.__returning__ is bytes


@pytest.mark.parametrize(
    "argument",
    [list[int], int | str, T],
    ids=["generic-alias", "union", "typevar"],
)
def test_non_class_argument_keeps_defaults(argument: Any) -> None:
    method_cls = make_method(BaseAPIMethod[argument])

    assert method_cls.__returning__ is None
    assert method_cls.__response_format__ is ResponseFormat.JSON
    assert method_cls().response_type is None


def test_unknown_class_keeps_default_format() -> None:
    method_cls = make_method(
        BaseAPIMethod[int],
        __response_parser__=BufferedResponseParser(),
    )

    assert method_cls.__returning__ is int
    assert method_cls.__response_format__ is ResponseFormat.JSON


@pytest.mark.parametrize(
    ("argument", "response_format", "message"),
    [
        (int, None, "JSON responses are parsed into a model, not int"),
        (bytes, ResponseFormat.JSON, "parsed into a model, not bytes"),
        (
            Model,
            ResponseFormat.BYTES,
            "BYTES responses are parsed into bytes, not Model",
        ),
        (
            int,
            ResponseFormat.TEXT,
            "TEXT responses are parsed into str, not int",
        ),
        (str, ResponseFormat.NONE, "parsed into None, not str"),
        (None, ResponseFormat.TEXT, "parsed into str, not None"),
    ],
    ids=[
        "int",
        "bytes-json",
        "model-bytes",
        "int-text",
        "str-none",
        "none-text",
    ],
)
def test_buffered_format_must_match_the_result(
    argument: Any,
    response_format: ResponseFormat | None,
    message: str,
) -> None:
    class_vars: dict[str, Any] = (
        {}
        if response_format is None
        else {"__response_format__": response_format}
    )

    with pytest.raises(TypeError, match=f"^Method: .*{message}"):
        make_method(BaseAPIMethod[argument], **class_vars)


@pytest.mark.parametrize(
    "response_format",
    [ResponseFormat.SSE, ResponseFormat.NDJSON],
    ids=["sse", "ndjson"],
)
def test_buffered_method_rejects_stream_formats(
    response_format: ResponseFormat,
) -> None:
    with pytest.raises(TypeError, match="can only be streamed"):
        make_method(
            BaseAPIMethod[bytes],
            __response_format__=response_format,
        )


def test_buffered_check_uses_explicit_returning() -> None:
    with pytest.raises(TypeError, match="parsed into str, not Model"):
        make_method(
            BaseAPIMethod[str],
            __returning__=Model,
        )


def test_buffered_check_on_leaf_of_generic_base() -> None:
    class BytesBase[R](BaseAPIMethod[R]):
        __response_format__ = ResponseFormat.BYTES

    with pytest.raises(TypeError, match="parsed into bytes, not Model"):

        class Leaf(BytesBase[Model]):  # pyright: ignore[reportUnusedClass]
            __request_path__ = "/leaf"
            __request_method__ = "GET"


@pytest.mark.parametrize(
    ("argument", "response_format", "message"),
    [
        (
            str,
            ResponseFormat.BYTES,
            "BYTES streams yield bytes items, not str",
        ),
        (
            bytes,
            ResponseFormat.TEXT,
            "TEXT streams yield str items, not bytes",
        ),
        (
            ServerSentEvent,
            ResponseFormat.BYTES,
            "yield bytes items, not ServerSentEvent",
        ),
        (bytes, ResponseFormat.JSON, "JSON is not a stream format"),
        (bytes, ResponseFormat.NONE, "NONE is not a stream format"),
        (Model, ResponseFormat.TEXT, "yield str items, not Model"),
        (list[int], ResponseFormat.TEXT, r"not list\[int\]"),
    ],
    ids=[
        "str-bytes",
        "bytes-text",
        "event-bytes",
        "json",
        "none",
        "model-text",
        "alias-text",
    ],
)
def test_stream_format_must_match_the_item(
    argument: Any,
    response_format: ResponseFormat,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=f"^Method: .*{message}"):
        make_method(
            BaseStreamMethod[argument],
            __response_format__=response_format,
        )


@pytest.mark.parametrize(
    ("argument", "response_format"),
    [
        (str, ResponseFormat.TEXT),
        (bytes, ResponseFormat.BYTES),
        (str, ResponseFormat.NDJSON),
        (bytes, ResponseFormat.SSE),
    ],
    ids=["text", "bytes", "ndjson-str", "sse-bytes"],
)
def test_stream_format_matching_the_item_is_accepted(
    argument: Any,
    response_format: ResponseFormat,
) -> None:
    method_cls = make_method(
        BaseStreamMethod[argument],
        __response_format__=response_format,
    )

    assert method_cls.__response_format__ is response_format


def test_base_classes_keep_their_defaults() -> None:
    make_method(BaseAPIMethod[str])
    make_method(BaseStreamMethod[str])

    assert BaseMethod.__returning__ is None
    assert BaseAPIMethod.__returning__ is None
    assert BaseAPIMethod.__response_format__ is ResponseFormat.JSON
    assert BaseStreamMethod.__returning__ is None
    assert BaseStreamMethod.__response_format__ is ResponseFormat.BYTES


@pytest.mark.parametrize(
    ("argument", "response_format"),
    [(bytes, ResponseFormat.BYTES), (str, ResponseFormat.TEXT)],
    ids=["bytes", "str"],
)
def test_stream_inference(
    argument: Any,
    response_format: ResponseFormat,
) -> None:
    method_cls = make_method(BaseStreamMethod[argument])

    assert method_cls.__returning__ is argument
    assert method_cls.__response_format__ is response_format
    assert method_cls().response_format is response_format


@pytest.mark.parametrize(
    ("argument", "type_name"),
    [(Model, "Model"), (None, "None"), (int, "int")],
    ids=["model", "none", "int"],
)
def test_stream_rejects_unsupported_items(
    argument: Any,
    type_name: str,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"Method: stream items of type {type_name} are not supported",
    ):
        make_method(BaseStreamMethod[argument])


def test_stream_non_class_argument_needs_a_format() -> None:
    with pytest.raises(TypeError, match=r"items of type list\[int\]"):
        make_method(BaseStreamMethod[list[int]])


def test_stream_non_class_argument_with_format() -> None:
    method_cls = make_method(
        BaseStreamMethod[list[int]],
        __response_format__=ResponseFormat.NDJSON,
    )

    assert method_cls.__returning__ is None
    assert method_cls.__stream_item__ == list[int]


def test_stream_explicit_format_allows_any_item() -> None:
    method_cls = make_method(
        BaseStreamMethod[Event],
        __response_format__=ResponseFormat.SSE,
    )

    assert method_cls.__response_format__ is ResponseFormat.SSE


def test_stream_explicit_parser_allows_any_item() -> None:
    parser = StreamResponseParser()
    method_cls = make_method(BaseStreamMethod[Event], __stream_parser__=parser)

    assert method_cls().stream_parser is parser
    assert method_cls.__response_format__ is ResponseFormat.BYTES


def test_stream_format_from_own_generic_base() -> None:
    class SseBase[R](BaseStreamMethod[R]):
        __response_format__ = ResponseFormat.SSE

    class Events(SseBase[Event]):
        __request_path__ = "/events"
        __request_method__ = "GET"

    assert Events.__response_format__ is ResponseFormat.SSE
    assert Events.__returning__ is Event


def test_stream_parser_from_own_generic_base() -> None:
    class ParsedBase[R](BaseStreamMethod[R]):
        __stream_parser__ = StreamResponseParser()

    class Events(ParsedBase[Event]):
        __request_path__ = "/events"
        __request_method__ = "GET"

    assert Events.__returning__ is Event


def test_returning_from_generic_base_beats_inference() -> None:
    class EventBase[R](BaseAPIMethod[R]):
        __returning__: ClassVar[type[Any] | None] = Event

    class Leaf(EventBase[Model]):
        __request_path__ = "/leaf"
        __request_method__ = "GET"

    assert Leaf.__returning__ is Event
    assert Leaf.__response_format__ is ResponseFormat.JSON


def test_buffered_method_exposes_its_parser() -> None:
    method_cls = make_method(BaseAPIMethod[str])

    assert method_cls().response_parser is BaseAPIMethod.__response_parser__


class Start(BaseAPIType):
    type: Literal["start"]


class Stop(BaseAPIType):
    type: Literal["stop"]


type Union = Annotated[Start | Stop, Field(discriminator="type")]


class Opaque:
    """A class Pydantic cannot validate."""


def test_server_sent_event_items_infer_sse() -> None:
    method_cls = make_method(BaseStreamMethod[ServerSentEvent])

    assert method_cls.__response_format__ is ResponseFormat.SSE
    assert method_cls.__returning__ is ServerSentEvent
    assert method_cls().response_adapter is None


@pytest.mark.parametrize(
    "response_format",
    [ResponseFormat.SSE, ResponseFormat.NDJSON],
    ids=["sse", "ndjson"],
)
def test_event_formats_build_an_adapter(
    response_format: ResponseFormat,
) -> None:
    method_cls = make_method(
        BaseStreamMethod[Union],
        __response_format__=response_format,
    )
    adapter = method_cls().response_adapter

    assert adapter is not None
    assert adapter.validate_json('{"type": "stop"}') == Stop(type="stop")
    assert method_cls.__returning__ is None


def test_server_sent_event_items_of_ndjson_are_validated() -> None:
    method_cls = make_method(
        BaseStreamMethod[ServerSentEvent],
        __response_format__=ResponseFormat.NDJSON,
    )

    assert method_cls().response_adapter is not None


def test_other_formats_have_no_adapter() -> None:
    method_cls = make_method(
        BaseStreamMethod[Event],
        __stream_parser__=StreamResponseParser(),
    )

    assert method_cls().response_adapter is None


def test_union_without_format_is_rejected() -> None:
    with pytest.raises(TypeError, match="items of type Union are not"):
        make_method(BaseStreamMethod[Union])


def test_item_pydantic_cannot_validate_is_rejected() -> None:
    with pytest.raises(TypeError, match="cannot validate stream items"):
        make_method(
            BaseStreamMethod[Opaque],
            __response_format__=ResponseFormat.SSE,
        )


def test_adapter_from_own_generic_base() -> None:
    class NdjsonBase[R](BaseStreamMethod[R]):
        __response_format__ = ResponseFormat.NDJSON

    class Events(NdjsonBase[Union]):
        __request_path__ = "/events"
        __request_method__ = "GET"

    adapter = Events().response_adapter

    assert NdjsonBase.__response_adapter__ is None
    assert adapter is not None
    assert adapter.validate_python({"type": "start"}) == Start(type="start")
