"""Block F: upload files."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from ai_api.types.files import (
    BytesFile,
    InputFile,
    IOFile,
    PathFile,
    StreamFile,
    guess_content_type,
)


async def read_all(file: InputFile) -> bytes:
    return b"".join([chunk async for chunk in file.open()])


async def chunks_of(file: InputFile) -> list[bytes]:
    return [chunk async for chunk in file.open()]


class OneWayStream(io.RawIOBase):
    """A readable, non-seekable binary stream, like a pipe or socket."""

    def __init__(self, data: bytes) -> None:
        self._data = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def readinto(self, buffer: Any) -> int:
        data = self._data.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


async def produce(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("photo.png", "image/png"),
        ("dir/voice.mp3", "audio/mpeg"),
        ("data.unknown-ext", None),
        ("no-extension", None),
        (None, None),
    ],
    ids=["png", "with-dir", "unknown", "no-extension", "none"],
)
def test_guess_content_type(
    filename: str | None, expected: str | None
) -> None:
    assert guess_content_type(filename) == expected


def test_from_value_keeps_input_file() -> None:
    file = BytesFile(b"x")

    assert InputFile.from_value(file) is file


@pytest.mark.parametrize(
    "value",
    [b"data", bytearray(b"data"), memoryview(b"data")],
    ids=["bytes", "bytearray", "memoryview"],
)
async def test_from_value_bytes_like(value: bytes | bytearray) -> None:
    file = InputFile.from_value(value)

    assert isinstance(file, BytesFile)
    assert type(file.data) is bytes
    assert await read_all(file) == b"data"


def test_from_value_path(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"

    file = InputFile.from_value(path)

    assert isinstance(file, PathFile)
    assert file.path == path


def test_from_value_bytes_io() -> None:
    assert isinstance(InputFile.from_value(io.BytesIO(b"x")), IOFile)


def test_from_value_file_io(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")

    with io.FileIO(path) as stream:
        assert isinstance(InputFile.from_value(stream), IOFile)


@pytest.mark.parametrize(
    "value",
    [io.StringIO("text"), "a.txt", 1, None],
    ids=["text-stream", "str", "int", "none"],
)
def test_from_value_non_files(value: Any) -> None:
    assert InputFile.from_value(value) is None


class AnyFileModel(BaseModel):
    file: InputFile


class PathFileModel(BaseModel):
    file: PathFile


def test_input_file_field_accepts_raw_values(tmp_path: Path) -> None:
    def validate(value: Any) -> InputFile:
        return AnyFileModel.model_validate({"file": value}).file

    assert isinstance(validate(b"x"), BytesFile)
    assert isinstance(validate(tmp_path / "a.txt"), PathFile)
    assert isinstance(validate(io.BytesIO()), IOFile)


def test_typed_file_field_rejects_other_kinds() -> None:
    with pytest.raises(ValidationError, match="Expected PathFile"):
        PathFileModel.model_validate({"file": b"x"})


def test_file_field_rejects_non_files() -> None:
    with pytest.raises(ValidationError, match="Expected InputFile"):
        AnyFileModel.model_validate({"file": "a.txt"})


def test_model_dump_keeps_the_file_object() -> None:
    file = BytesFile(b"x")

    assert AnyFileModel(file=file).model_dump()["file"] is file


async def test_bytes_file_reads_full_content_every_time() -> None:
    file = BytesFile(b"content", filename="a.txt")

    assert await read_all(file) == b"content"
    assert await read_all(file) == b"content"
    assert file.content_type == "text/plain"
    assert file.reusable is True


def test_explicit_content_type_wins() -> None:
    file = BytesFile(b"x", filename="a.txt", content_type="x/custom")

    assert file.content_type == "x/custom"


async def test_path_file_reads_full_content_every_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "voice.mp3"
    path.write_bytes(b"0123456789")
    file = PathFile(path, chunk_size=4)

    assert await chunks_of(file) == [b"0123", b"4567", b"89"]
    assert await read_all(file) == b"0123456789"
    assert file.filename == "voice.mp3"
    assert file.content_type == "audio/mpeg"
    assert file.reusable is True


def test_path_file_accepts_str_and_filename_override(tmp_path: Path) -> None:
    file = PathFile(str(tmp_path / "a.bin"), filename="b.png")

    assert file.path == tmp_path / "a.bin"
    assert file.filename == "b.png"
    assert file.content_type == "image/png"


async def test_io_file_reads_from_initial_position_every_time() -> None:
    stream = io.BytesIO(b"headerbody")
    stream.seek(6)
    file = IOFile(stream, chunk_size=2)

    assert await chunks_of(file) == [b"bo", b"dy"]
    assert await read_all(file) == b"body"
    assert file.reusable is True


async def test_non_seekable_io_file_reads_once() -> None:
    file = IOFile(io.BufferedReader(OneWayStream(b"once")))

    assert file.reusable is False
    assert await read_all(file) == b"once"

    with pytest.raises(RuntimeError, match="only once"):
        await read_all(file)


class NonBlockingStream(io.RawIOBase):
    """A raw stream whose data is not ready: ``read()`` gives ``None``."""

    def __init__(self, reads: list[bytes | None]) -> None:
        self._reads = reads

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes | None:
        return self._reads.pop(0)


async def test_non_blocking_stream_without_data_raises() -> None:
    # None means "no data yet", not the end: reading on would send an
    # empty or truncated file without an error.
    file = IOFile(NonBlockingStream([b"he", None, b"llo", b""]))

    with pytest.raises(BlockingIOError, match="non-blocking"):
        await read_all(file)


def test_closed_stream_is_rejected() -> None:
    stream = io.BytesIO(b"x")
    stream.close()

    with pytest.raises(ValueError, match="Invalid file stream"):
        IOFile(stream)


def test_unreadable_stream_is_rejected(tmp_path: Path) -> None:
    with (
        io.FileIO(tmp_path / "out.bin", "w") as stream,
        pytest.raises(ValueError, match="Invalid file stream"),
    ):
        IOFile(stream)


def test_io_file_filename_from_name(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF")

    with path.open("rb") as stream:
        file = IOFile(stream)

    assert file.filename == "report.pdf"
    assert file.content_type == "application/pdf"


def test_io_file_without_name() -> None:
    file = IOFile(io.BytesIO(b"x"))

    assert file.filename is None
    assert file.content_type is None


def test_io_file_explicit_filename_wins(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF")

    with path.open("rb") as stream:
        assert IOFile(stream, filename="other.txt").filename == "other.txt"


def test_io_file_ignores_non_str_name(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")

    # FileIO opened from a descriptor has an int ``name``.
    with (
        path.open("rb") as source,
        io.FileIO(source.fileno(), closefd=False) as stream,
    ):
        assert IOFile(stream).filename is None


async def test_stream_file_reads_once() -> None:
    file = StreamFile(produce(b"a", b"b"), filename="a.bin")

    assert file.reusable is False
    assert await chunks_of(file) == [b"a", b"b"]

    with pytest.raises(RuntimeError, match="only once"):
        await read_all(file)


@pytest.mark.parametrize(
    "file",
    [
        BytesFile(b"secret-content", filename="a.txt"),
        IOFile(io.BytesIO(b"secret-content"), filename="a.txt"),
        StreamFile(produce(b"secret-content"), filename="a.txt"),
    ],
    ids=["bytes", "io", "stream"],
)
def test_repr_hides_content(file: InputFile) -> None:
    text = repr(file)

    assert "secret-content" not in text
    assert text == (
        f"{type(file).__name__}(filename='a.txt', content_type='text/plain')"
    )


def test_path_file_repr(tmp_path: Path) -> None:
    assert repr(PathFile(tmp_path / "a.txt")) == (
        "PathFile(filename='a.txt', content_type='text/plain')"
    )
