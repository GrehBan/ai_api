"""Block G3: query, body, stream wrapping and retries."""

from __future__ import annotations

import enum
import io
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import Field, ValidationError

from ai_api.enums import BodyFormat
from ai_api.methods.base import BaseAPIMethod
from ai_api.types.base import BaseAPIType
from ai_api.types.files import BytesFile, InputFile, IOFile, StreamFile
from tests.conftest import make_method


class Order(enum.StrEnum):
    ASC = "asc"


class ListItems(BaseAPIMethod[bytes]):
    __request_path__ = "/orgs/{org}/items"
    __request_method__ = "GET"
    __request_query__ = {"limit", "page_size", "order", "deleted", "after"}

    org: str
    limit: int | None = None
    page_size: int | None = Field(
        default=None,
        validation_alias="pageSize",
        serialization_alias="pageSize",
    )
    order: Order | None = None
    deleted: bool | None = None
    after: str | None = None
    name: str | None = None
    display_name: str | None = Field(
        default=None,
        validation_alias="displayName",
        serialization_alias="displayName",
    )


class Upload(BaseAPIMethod[bytes]):
    __request_path__ = "/upload"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART

    payload: Any = None


class Attachment(BaseAPIType):
    file: InputFile


class RetryableUpload(Upload):
    __retryable__ = True


async def produce() -> AsyncIterator[bytes]:
    yield b"x"


def test_query_map_converts_values() -> None:
    method = ListItems(
        org="acme",
        limit=5,
        page_size=10,
        order=Order.ASC,
        deleted=False,
    )

    assert method.get_query_map() == {
        "limit": "5",
        "pageSize": "10",
        "order": "asc",
        "deleted": "false",
    }


def test_query_map_true_and_skipped_none() -> None:
    method = ListItems(org="acme", deleted=True)

    assert method.get_query_map() == {"deleted": "true"}


def test_query_map_populated_by_alias() -> None:
    assert ListItems.model_validate(
        {"org": "a", "pageSize": 3}
    ).get_query_map() == {"pageSize": "3"}


class Search(BaseAPIMethod[bytes]):
    __request_path__ = "/search"
    __request_method__ = "GET"
    __request_query__ = {"ids", "tags", "flags", "raw"}

    ids: list[str] | None = None
    tags: tuple[Order, ...] | None = None
    flags: set[bool] | None = None
    raw: Any = None


def test_lists_become_repeated_keys() -> None:
    method = Search(ids=["a", "b"], tags=(Order.ASC,), flags={True})

    assert method.get_query_map() == {
        "ids": ["a", "b"],
        "tags": ["asc"],
        "flags": ["true"],
    }


def test_none_items_of_a_list_are_dropped() -> None:
    assert Search(raw=[1, None, 2]).get_query_map() == {"raw": ["1", "2"]}


@pytest.mark.parametrize(
    "raw",
    [{"a": 1}, [{"a": 1}], [[1]], Order],
    ids=["dict", "list-of-dicts", "nested-list", "not-json"],
)
def test_objects_in_query_are_rejected(raw: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        Search(raw=raw).get_query_map()


def test_object_in_query_names_the_parameter() -> None:
    with pytest.raises(TypeError, match="'raw' must be a scalar"):
        Search(raw={"a": 1}).get_query_map()


def test_unknown_query_name_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"names \['limt'\] have no"):
        make_method(
            BaseAPIMethod[bytes],
            fields={"limit": int},
            __request_query__={"limt"},
        )


@pytest.mark.parametrize(
    "annotation",
    [dict[str, int], dict[str, int] | None, Attachment, BaseAPIType | int],
    ids=["dict", "optional-dict", "model", "union-with-model"],
)
def test_mapping_query_field_is_rejected(annotation: Any) -> None:
    with pytest.raises(TypeError, match="'q' cannot be a mapping"):
        make_method(
            BaseAPIMethod[bytes],
            fields={"q": annotation},
            __request_query__={"q"},
        )


def test_no_query_fields_gives_none() -> None:
    assert Upload().get_query_map() is None


def test_body_map_excludes_path_query_and_none() -> None:
    method = ListItems(org="acme", limit=5, name="n", display_name="Display")

    assert method.get_body_map() == {"name": "n", "displayName": "Display"}


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="metadat"):
        ListItems.model_validate({"org": "acme", "metadat": {"a": 1}})


def test_body_map_is_empty_without_body_fields() -> None:
    assert ListItems(org="acme", limit=1).get_body_map() == {}


@pytest.mark.parametrize(
    ("payload", "location"),
    [
        (io.BytesIO(b"raw"), ()),
        ([io.BytesIO(b"raw")], (0,)),
        ((io.BytesIO(b"raw"),), (0,)),
        ({"file": io.BytesIO(b"raw")}, ("file",)),
        ([{"files": [io.BytesIO(b"raw")]}], (0, "files", 0)),
    ],
    ids=["field", "list", "tuple", "dict", "nested"],
)
def test_raw_binary_streams_become_io_files(
    payload: Any,
    location: tuple[int | str, ...],
) -> None:
    value = Upload(payload=payload).payload

    for key in location:
        value = value[key]

    assert isinstance(value, IOFile)


def test_text_stream_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Only binary streams"):
        Upload(payload=io.StringIO("text"))


def test_non_stream_values_are_kept() -> None:
    file = BytesFile(b"x")
    payload = {"file": file, "items": [1, "a"]}

    method = Upload(payload=payload)

    assert method.payload is payload
    assert method.payload["file"] is file


def test_wrapping_records_start_position() -> None:
    stream = io.BytesIO(b"skipbody")
    stream.seek(4)

    method = Upload(payload=stream)

    assert isinstance(method.payload, IOFile)
    assert method.payload.file is stream


def test_regression_no_serialization_iterator_in_body() -> None:
    # model_dump() used to turn raw streams into SerializationIterator.
    method = Upload(payload=[io.BytesIO(b"a"), {"b": io.BytesIO(b"b")}])

    body = method.get_body_map()

    assert isinstance(body["payload"][0], IOFile)
    assert isinstance(body["payload"][1]["b"], IOFile)
    assert "SerializationIterator" not in repr(body)


def test_input_file_survives_dump() -> None:
    file = BytesFile(b"x")

    assert Upload(payload=file).get_body_map()["payload"] is file


def test_not_retryable_by_default() -> None:
    assert Upload(payload=BytesFile(b"x")).retryable is False


def test_retryable_when_enabled() -> None:
    assert RetryableUpload(payload=BytesFile(b"x")).retryable is True
    assert RetryableUpload().retryable is True


@pytest.mark.parametrize(
    "payload",
    [
        StreamFile(produce()),
        [BytesFile(b"x"), StreamFile(produce())],
        {"files": (StreamFile(produce()),)},
        Attachment(file=StreamFile(produce())),
    ],
    ids=["field", "list", "dict-tuple", "nested-model"],
)
def test_one_shot_file_disables_retries(payload: Any) -> None:
    assert RetryableUpload(payload=payload).retryable is False


def test_non_seekable_stream_disables_retries() -> None:
    class OneWay(io.BytesIO):
        def seekable(self) -> bool:
            return False

    assert RetryableUpload(payload=OneWay(b"x")).retryable is False


def test_reusable_nested_model_keeps_retries() -> None:
    payload = Attachment(file=BytesFile(b"x"))

    assert RetryableUpload(payload=payload).retryable is True


def test_method_properties() -> None:
    method = Upload()

    assert method.request_method == "POST"
    assert method.request_body_format is BodyFormat.MULTIPART
