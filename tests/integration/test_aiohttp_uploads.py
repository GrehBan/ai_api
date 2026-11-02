"""Block K: multipart uploads of ``AiohttpSession`` against a real server."""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import MISSING

from ai_api.enums import BodyFormat
from ai_api.http.exceptions.status import ServiceUnavailableError
from ai_api.http.exceptions.transport import APIRequestError
from ai_api.methods.base import BaseAPIMethod
from ai_api.session.aiohttp import AiohttpSession
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
    file: Any = MISSING


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
    recorder.reply(Reply())
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
    recorder.reply(Reply())
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
    recorder.reply(Reply())
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
    recorder.reply(Reply())
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


class Chunks:
    """An async iterator that is not a generator and cannot be closed."""

    def __init__(self, *chunks: bytes) -> None:
        self.chunks = list(chunks)
        self.calls: list[str] = []

    def __aiter__(self) -> Chunks:
        return self

    async def __anext__(self) -> bytes:
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class SyncClosed(Chunks):
    def close(self) -> None:
        self.calls.append("close")


class BothClosed(Chunks):
    def close(self) -> None:
        self.calls.append("close")

    async def aclose(self) -> None:
        self.calls.append("aclose")


class FailingClose(Chunks):
    async def aclose(self) -> None:
        raise OSError("disk gone")


class Source(InputFile):
    """A file whose ``open()`` returns a given iterator, of any kind."""

    def __init__(self, iterator: AsyncIterator[bytes]) -> None:
        super().__init__("a.bin")
        self.iterator = iterator

    def open(self) -> AsyncIterator[bytes]:
        return self.iterator


@pytest.mark.parametrize(
    ("source", "calls"),
    [
        (Chunks(b"a", b"b"), []),
        (SyncClosed(b"a", b"b"), ["close"]),
        (BothClosed(b"a", b"b"), ["aclose"]),
    ],
    ids=["no-close", "close", "both"],
)
async def test_any_source_is_sent_and_closed_once(
    session: AiohttpSession,
    recorder: Recorder,
    source: Chunks,
    calls: list[str],
) -> None:
    # open() may return any async iterator; aclose() wins over close().
    recorder.reply(Reply())

    await session.send_request(Upload(file=Source(source)))

    (part,) = [part for part in recorder.last.parts if part.name == "file"]
    assert part.data == b"ab"
    assert source.calls == calls


async def test_a_source_that_fails_to_close_is_a_request_error(
    session: AiohttpSession,
) -> None:
    # The upload may be incomplete: such a failure is never dropped.
    method = Upload(file=Source(FailingClose(b"a")))

    with pytest.raises(APIRequestError, match="disk gone"):
        await session.send_request(method)


class FailingRead(Chunks):
    async def __anext__(self) -> bytes:
        raise OSError("read failed")

    async def aclose(self) -> None:
        raise OSError("close failed")


class NeverReady(Chunks):
    """A source whose first chunk never comes; it fails to close."""

    async def __anext__(self) -> bytes:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        self.calls.append("aclose")
        raise OSError("close failed")


class Unopenable(InputFile):
    def open(self) -> AsyncIterator[bytes]:
        raise OSError("cannot open")


class UploadWithCount(Upload):
    count: Any = MISSING


async def test_a_stream_file_source_is_closed_after_the_upload(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    recorder.reply(Reply())
    source = SyncClosed(b"a", b"b")

    await session.send_request(Upload(file=StreamFile(source)))

    assert file_part(recorder).data == b"ab"
    assert source.calls == ["close"]


async def test_a_file_that_cannot_be_opened_is_a_request_error(
    session: AiohttpSession,
    recorder: Recorder,
) -> None:
    with pytest.raises(APIRequestError, match="Cannot open upload 'a.bin'"):
        await session.send_request(Upload(file=Unopenable("a.bin")))

    assert recorder.requests == []


async def test_a_failing_close_never_hides_a_read_error(
    session: AiohttpSession,
) -> None:
    with pytest.raises(APIRequestError, match="read failed") as info:
        await session.send_request(Upload(file=Source(FailingRead())))

    cause = info.value.__cause__
    assert isinstance(cause, OSError)
    assert cause.__notes__ == [
        "Closing the source also failed: OSError('close failed')"
    ]


@pytest.mark.parametrize(
    ("source", "notes"),
    [(SyncClosed(b"a"), 0), (FailingClose(b"a"), 1)],
    ids=["closed", "close-fails"],
)
async def test_a_body_that_fails_to_build_closes_its_sources(
    session: AiohttpSession,
    recorder: Recorder,
    source: Chunks,
    notes: int,
) -> None:
    # The file is opened before the next field fails: nothing is sent,
    # and the source is closed all the same.
    method = UploadWithCount(
        file=StreamFile(source, filename="a.bin"), count=math.nan
    )

    with pytest.raises(ValueError, match="NaN or infinity") as info:
        await session.send_request(method)

    assert recorder.requests == []
    if isinstance(source, SyncClosed):
        assert source.calls == ["close"]
    found = getattr(info.value, "__notes__", [])
    assert len(found) == notes
    assert all(note.startswith("Closing upload 'a.bin'") for note in found)


async def test_a_cancelled_upload_stays_cancelled(
    session: AiohttpSession,
) -> None:
    source = NeverReady()
    task = asyncio.create_task(
        session.send_request(Upload(file=StreamFile(source)))
    )
    await asyncio.sleep(0.2)

    task.cancel()

    async with asyncio.timeout(5):
        with pytest.raises(asyncio.CancelledError):
            await task

    assert source.calls == ["aclose"]
