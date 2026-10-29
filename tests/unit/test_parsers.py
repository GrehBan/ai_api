"""Block H1: the buffered response parser."""

import datetime
import enum
import uuid
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from pydantic import Strict

from ai_api.client.exceptions.transport import APIResponseError
from ai_api.client.response.base import BufferedResponse
from ai_api.client.response.parse.buffered import BufferedResponseParser
from ai_api.client.serialization.json import loads
from ai_api.methods.base import BaseAPIMethod
from ai_api.methods.enums import ResponseFormat
from ai_api.types.base import BaseAPIType
from tests.conftest import make_method


class Status(enum.StrEnum):
    DONE = "done"


class Job(BaseAPIType):
    id: uuid.UUID
    created: datetime.datetime
    status: Status


class StrictJob(BaseAPIType):
    id: Annotated[uuid.UUID, Strict()]
    created: Annotated[datetime.datetime, Strict()]
    status: Annotated[Status, Strict()]


JOB_ID = "6f1c6d2e-3a4b-4c5d-8e9f-0a1b2c3d4e5f"
JOB_JSON = (
    f'{{"id": "{JOB_ID}", "created": "2026-01-02T03:04:05Z", '
    '"status": "done", "extra": 1}'
).encode()


def response(
    body: bytes,
    *,
    status: int = 200,
    charset: str | None = None,
) -> BufferedResponse:
    return BufferedResponse(
        status=status, headers={}, body=body, charset=charset
    )


def parse(
    argument: Any,
    body: bytes,
    *,
    charset: str | None = None,
    json_loads: Any = None,
    **class_vars: Any,
) -> Any:
    method = make_method(BaseAPIMethod[argument], **class_vars)()
    return BufferedResponseParser().parse(
        method,
        response(body, status=201, charset=charset),
        json_loads,
    )


@pytest.mark.parametrize("model", [Job, StrictJob], ids=["lax", "strict"])
def test_default_decoder_uses_json_mode(model: type[BaseAPIType]) -> None:
    # Regression: strict datetime/UUID/StrEnum fields must accept the JSON
    # strings, which only Pydantic's JSON mode does.
    job = parse(model, JOB_JSON)

    assert isinstance(job, model)
    assert isinstance(job, Job | StrictJob)
    assert job.id == uuid.UUID(JOB_ID)
    assert job.created == datetime.datetime(
        2026, 1, 2, 3, 4, 5, tzinfo=datetime.UTC
    )
    assert job.status is Status.DONE


def test_session_default_decoder_is_recognized() -> None:
    assert isinstance(parse(StrictJob, JOB_JSON, json_loads=loads), StrictJob)


def test_custom_decoder_is_called_in_python_mode() -> None:
    custom = MagicMock(wraps=loads)

    job = parse(Job, JOB_JSON, json_loads=custom)

    custom.assert_called_once_with(JOB_JSON)
    assert job.status is Status.DONE


def test_custom_decoder_rejects_strict_fields() -> None:
    # Documented behavior: a custom decoder's result is validated in
    # Python mode, where strict fields reject JSON strings.
    with pytest.raises(APIResponseError, match="Failed to validate") as info:
        parse(StrictJob, JOB_JSON, json_loads=MagicMock(wraps=loads))

    assert info.value.status == 201


def test_parser_level_decoder_is_used_by_default() -> None:
    custom = MagicMock(wraps=loads)
    method = make_method(BaseAPIMethod[Job])()

    BufferedResponseParser(json_loads=custom).parse(method, response(JOB_JSON))

    custom.assert_called_once_with(JOB_JSON)


@pytest.mark.parametrize(
    "body",
    [b"{", b"", b'{"id": "not-a-uuid"}', b"[]"],
    ids=["broken", "empty", "schema", "wrong-type"],
)
def test_invalid_json_body(body: bytes) -> None:
    with pytest.raises(APIResponseError, match="Failed to validate") as info:
        parse(Job, body)

    assert info.value.status == 201
    assert info.value.__cause__ is not None


def test_custom_decoder_errors_propagate() -> None:
    # The session turns this ValueError into APIResponseError.
    with pytest.raises(ValueError):
        parse(Job, b"{", json_loads=MagicMock(wraps=loads))


@pytest.mark.parametrize(
    "argument",
    [list[int], int],
    ids=["generic-alias", "not-a-model"],
)
def test_json_without_model_is_rejected(argument: Any) -> None:
    # An explicit parser skips the class-definition check, so the parser's
    # own guard is reached.
    with pytest.raises(TypeError, match="requires a response model"):
        parse(
            argument,
            b"[1]",
            __response_parser__=BufferedResponseParser(),
        )


def test_text_uses_response_charset() -> None:
    assert parse(str, "привет".encode("cp1251"), charset="cp1251") == "привет"


def test_text_defaults_to_utf8() -> None:
    assert parse(str, "привет €".encode()) == "привет €"


@pytest.mark.parametrize(
    ("body", "charset"),
    [(b"\xff\xfe\xfa", None), (b"abc", "no-such-charset")],
    ids=["invalid-bytes", "unknown-charset"],
)
def test_text_decoding_errors(body: bytes, charset: str | None) -> None:
    with pytest.raises(APIResponseError, match="Failed to decode") as info:
        parse(str, body, charset=charset)

    assert info.value.status == 201


def test_bytes_are_returned_as_is() -> None:
    body = b"\x00\xffbinary"

    assert parse(bytes, body) is body


def test_none_for_empty_body() -> None:
    assert parse(None, b"") is None


def test_none_rejects_non_empty_body() -> None:
    with pytest.raises(APIResponseError, match="Expected an empty") as info:
        parse(None, b"{}")

    assert info.value.status == 201


@pytest.mark.parametrize(
    "response_format",
    [ResponseFormat.SSE, ResponseFormat.NDJSON],
    ids=["sse", "ndjson"],
)
def test_stream_formats_are_unsupported(
    response_format: ResponseFormat,
) -> None:
    with pytest.raises(ValueError, match="Unsupported buffered"):
        parse(
            bytes,
            b"",
            __response_format__=response_format,
            __response_parser__=BufferedResponseParser(),
        )
