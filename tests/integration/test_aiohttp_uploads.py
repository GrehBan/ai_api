"""Block K: multipart uploads of ``AiohttpSession`` against a real server."""

import hashlib
import io
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from ai_api.client.exceptions.http import ServiceUnavailableError
from ai_api.client.session.aiohttp import AiohttpSession
from ai_api.methods.base import BaseAPIMethod
from ai_api.methods.enums import BodyFormat
from ai_api.types.files import (
    DEFAULT_CHUNK_SIZE,
    BytesFile,
    InputFile,
    IOFile,
    PathFile,
    StreamFile,
)
from tests.conftest import RecordedPart, Recorder, Reply

pytestmark = pytest.mark.integration

CONTENT = bytes(range(256)) * 1200
assert len(CONTENT) > DEFAULT_CHUNK_SIZE


class Upload(BaseAPIMethod[bytes]):
    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART

    purpose: str = "test"
    file: Any = None


class RetryableUpload(Upload):
    __retryable__ = True


class TypedUpload(BaseAPIMethod[bytes]):
    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART

    file: InputFile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_part(recorder: Recorder, name: str = "file") -> RecordedPart:
    parts = [part for part in recorder.last.parts if part.name == name]
    assert len(parts) == 1
    return parts[0]


async def produce(data: bytes, size: int = 50_000) -> AsyncIterator[bytes]:
    for start in range(0, len(data), size):
        yield data[start : start + size]


def make_path_file(tmp_path: Path) -> PathFile:
    path = tmp_path / "speech.mp3"
    path.write_bytes(CONTENT)
    return PathFile(path)


def make_io_file(_: Path) -> IOFile:
    return IOFile(io.BytesIO(CONTENT), filename="speech.mp3")


def make_raw_bytes_io(_: Path) -> io.BytesIO:
    return io.BytesIO(CONTENT)


FILE_FACTORIES: dict[str, Callable[[Path], Any]] = {
    "path-file": make_path_file,
    "bytes-file": lambda _: BytesFile(CONTENT, filename="speech.mp3"),
    "io-file": make_io_file,
    "stream-file": lambda _: StreamFile(
        produce(CONTENT),
        filename="speech.mp3",
    ),
}


@pytest.mark.parametrize(
    "kind", list(FILE_FACTORIES), ids=list(FILE_FACTORIES)
)
async def test_file_arrives_whole_with_metadata(
    session: AiohttpSession,
    recorder: Recorder,
    tmp_path: Path,
    kind: str,
) -> None:
    await session.send_request(Upload(file=FILE_FACTORIES[kind](tmp_path)))

    part = file_part(recorder)
    assert digest(part.data) == digest(CONTENT)
    assert part.filename == "speech.mp3"
    assert part.content_type == "audio/mpeg"
    assert file_part(recorder, "purpose").data == b"test"


@pytest.mark.parametrize(
    "value",
    [CONTENT, io.BytesIO(CONTENT)],
    ids=["raw-bytes", "raw-bytes-io"],
)
async def test_raw_values_arrive_whole(
    session: AiohttpSession,
    recorder: Recorder,
    value: Any,
) -> None:
    await session.send_request(Upload(file=value))

    part = file_part(recorder)
    assert digest(part.data) == digest(CONTENT)
    assert part.filename == "file"
    assert part.content_type == "application/octet-stream"


async def test_raw_path_in_input_file_field(
    session: AiohttpSession,
    recorder: Recorder,
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(b"hello")

    await session.send_request(TypedUpload.model_validate({"file": path}))

    part = file_part(recorder)
    assert part.data == b"hello"
    assert part.filename == "notes.txt"
    assert part.content_type == "text/plain"


async def test_list_of_files_in_one_field(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    files = [
        BytesFile(b"first", filename="a.txt"),
        BytesFile(b"second", filename="b.json"),
    ]

    await session.send_request(Upload(file=files))

    parts = [part for part in recorder.last.parts if part.name == "file"]
    assert [(part.filename, part.data) for part in parts] == [
        ("a.txt", b"first"),
        ("b.json", b"second"),
    ]
    assert parts[1].content_type == "application/json"


@pytest.mark.parametrize(
    "make_file",
    [
        make_path_file,
        make_io_file,
        make_raw_bytes_io,
    ],
    ids=["path-file", "io-file", "raw-bytes-io"],
)
async def test_retry_resends_full_file(
    session: AiohttpSession,
    recorder: Recorder,
    tmp_path: Path,
    make_file: Callable[[Path], Any],
) -> None:
    recorder.reply(Reply(status=503), Reply(status=503), Reply(body=b"ok"))

    result = await session.send_request(
        RetryableUpload(file=make_file(tmp_path))
    )

    assert result == b"ok"
    assert len(recorder.requests) == 3
    for request in recorder.requests:
        (part,) = [part for part in request.parts if part.name == "file"]
        assert digest(part.data) == digest(CONTENT)


async def test_stream_file_is_sent_once(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(status=503), Reply(body=b"ok"))
    method = RetryableUpload(file=StreamFile(produce(CONTENT)))

    with pytest.raises(ServiceUnavailableError):
        await session.send_request(method)

    assert len(recorder.requests) == 1
