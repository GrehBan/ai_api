"""Files to upload in multipart requests."""

from __future__ import annotations

import asyncio
import errno
import io
import mimetypes
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any, Final, Self

import aiofiles
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

DEFAULT_CHUNK_SIZE: Final[int] = 64 * 1024
"""Number of bytes read from disk or a stream at a time."""

type BinaryStream = io.BufferedIOBase | io.RawIOBase


def guess_content_type(filename: str | None) -> str | None:
    """Guess a MIME type from a file name.

    Args:
        filename: File name or path.

    Returns:
        The MIME type, or ``None`` if it is unknown or there is no name.
    """
    if filename is None:
        return None

    content_type, _ = mimetypes.guess_file_type(filename)
    return content_type


class InputFile(ABC):
    """A file to upload, independent of the HTTP client.

    Fields annotated as ``InputFile`` accept ``bytes``, paths and binary
    streams as well and convert them to the matching subclass.

    Attributes:
        filename: File name sent in the multipart part.
        content_type: MIME type sent in the multipart part; guessed from
            ``filename`` when not given.

    Example:
        >>> class Transcribe(BaseAPIMethod[Transcription]):
        ...     file: InputFile
        >>> Transcribe(file=Path("voice.mp3")).file
        PathFile(filename='voice.mp3', content_type='audio/mpeg')
    """

    def __init__(
        self,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None:
        """Store the file metadata.

        Args:
            filename: File name sent in the multipart part.
            content_type: MIME type; guessed from ``filename`` if omitted.
        """
        self.filename = filename
        self.content_type = content_type or guess_content_type(filename)

    @property
    def reusable(self) -> bool:
        """Whether the content can be read more than once.

        A request carrying a non-reusable file is never retried.
        """
        return True

    def check(self) -> None:  # noqa: B027
        """Check, before sending, that the content can be read.

        Called once per request; the default accepts every file.

        Raises:
            OSError: If the content cannot be read.
        """

    @abstractmethod
    def open(self) -> AsyncIterator[bytes]:
        """Read the content from its beginning.

        Called once per upload attempt.

        Returns:
            An iterator over chunks of the content.
        """
        raise NotImplementedError

    @classmethod
    def from_value(cls, value: Any) -> InputFile | None:
        """Convert a raw value into a file.

        Args:
            value: An ``InputFile``, bytes-like object, path or binary
                stream.

        Returns:
            The matching ``InputFile``, or ``None`` if the value is not a
            file.
        """
        if isinstance(value, InputFile):
            return value

        if isinstance(value, (bytes, bytearray, memoryview)):
            return BytesFile(bytes(value))

        if isinstance(value, os.PathLike):
            return PathFile(Path(value))

        if isinstance(value, (io.BufferedIOBase, io.RawIOBase)):
            return IOFile(value)

        return None

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Let Pydantic validate fields annotated with a file class.

        The instance is kept as is when dumping, so it reaches the body
        builder instead of being converted to a dict.

        Args:
            source_type: The annotated type.
            handler: Pydantic's schema handler.

        Returns:
            A schema that converts raw values with ``from_value``.
        """
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value: Any) -> Self:
        """Validate a field value against this file class.

        Args:
            value: The raw field value.

        Returns:
            The file instance.

        Raises:
            ValueError: If the value cannot be converted to ``cls``.
        """
        file = InputFile.from_value(value)

        if isinstance(file, cls):
            return file

        raise ValueError(
            f"Expected {cls.__name__}, bytes, a path or a binary stream"
        )

    def __repr__(self) -> str:
        """Return the class name with the file metadata.

        Returns:
            A representation that does not include the content.
        """
        return (
            f"{type(self).__name__}(filename={self.filename!r}, "
            f"content_type={self.content_type!r})"
        )


class BytesFile(InputFile):
    """Content held in memory.

    Sent with a known length, which servers that reject chunked uploads
    require.

    Attributes:
        data: The file content.
    """

    def __init__(
        self,
        data: bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None:
        """Create the file.

        Args:
            data: The file content.
            filename: File name sent in the multipart part.
            content_type: MIME type; guessed from ``filename`` if omitted.
        """
        super().__init__(filename, content_type)
        self.data = data

    async def open(self) -> AsyncIterator[bytes]:
        """Yield the whole content.

        Yields:
            The content as a single chunk.
        """
        yield self.data


class PathFile(InputFile):
    """A file on disk, read in chunks with ``aiofiles``.

    Attributes:
        path: Location of the file.
        chunk_size: Number of bytes read at a time.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        filename: str | None = None,
        content_type: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Create the file; it is not opened until upload.

        Args:
            path: Location of the file.
            filename: File name to send; the path's name by default.
            content_type: MIME type; guessed from the file name if
                omitted.
            chunk_size: Number of bytes read at a time.
        """
        self.path = Path(path)
        super().__init__(filename or self.path.name, content_type)
        self.chunk_size = chunk_size

    def check(self) -> None:
        """Check that the path is an existing file.

        Raises:
            IsADirectoryError: If ``path`` is a directory.
            FileNotFoundError: If there is no file at ``path``.
        """
        if self.path.is_dir():
            raise IsADirectoryError(
                errno.EISDIR,
                "Is a directory",
                str(self.path),
            )

        if not self.path.is_file():
            raise FileNotFoundError(
                errno.ENOENT,
                "No such file",
                str(self.path),
            )

    async def open(self) -> AsyncIterator[bytes]:
        """Read the file from the beginning.

        Yields:
            Chunks of at most ``chunk_size`` bytes.
        """
        async with aiofiles.open(self.path, "rb") as file:
            while chunk := await file.read(self.chunk_size):
                yield chunk


class IOFile(InputFile):
    """An already open binary stream.

    Seekable streams are read from the position they had when the file
    was created, on every attempt; non-seekable ones can be read once.

    Attributes:
        file: The wrapped stream.
        chunk_size: Number of bytes read at a time.
    """

    def __init__(
        self,
        file: BinaryStream,
        filename: str | None = None,
        content_type: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Wrap the stream and remember its current position.

        Args:
            file: An open binary stream.
            filename: File name to send; the stream's ``name`` by
                default.
            content_type: MIME type; guessed from the file name if
                omitted.
            chunk_size: Number of bytes read at a time.

        Raises:
            ValueError: If the stream is closed or not readable.
        """
        if file.closed or not file.readable():
            raise ValueError("Invalid file stream")

        name = getattr(file, "name", None)

        if filename is None and isinstance(name, str):
            filename = Path(name).name

        super().__init__(filename, content_type)
        self.file = file
        self.chunk_size = chunk_size
        self._start = file.tell() if file.seekable() else None
        self._opened = False

    @property
    def reusable(self) -> bool:
        """Whether the stream is seekable."""
        return self._start is not None

    async def open(self) -> AsyncIterator[bytes]:
        """Read the stream from its start position.

        Blocking reads run in a worker thread.

        Yields:
            Chunks of at most ``chunk_size`` bytes.

        Raises:
            RuntimeError: If a non-seekable stream is read a second time.
        """
        if self._start is not None:
            await asyncio.to_thread(self.file.seek, self._start)
        elif self._opened:
            raise RuntimeError("Non-seekable stream can be read only once")

        self._opened = True

        while chunk := await asyncio.to_thread(
            self.file.read, self.chunk_size
        ):
            yield chunk


class StreamFile(InputFile):
    """Content produced by an async iterator; it can be read only once.

    A request carrying it is never retried.

    Attributes:
        stream: The source of the content.
    """

    def __init__(
        self,
        stream: AsyncIterable[bytes],
        filename: str | None = None,
        content_type: str | None = None,
    ) -> None:
        """Create the file.

        Args:
            stream: The source of the content.
            filename: File name sent in the multipart part.
            content_type: MIME type; guessed from ``filename`` if omitted.
        """
        super().__init__(filename, content_type)
        self.stream = stream
        self._opened = False

    @property
    def reusable(self) -> bool:
        """Always ``False``: the iterator cannot be restarted."""
        return False

    async def open(self) -> AsyncIterator[bytes]:
        """Forward the iterator's chunks.

        Yields:
            Chunks as produced by the source.

        Raises:
            RuntimeError: If the file is read a second time.
        """
        if self._opened:
            raise RuntimeError("StreamFile can be read only once")

        self._opened = True

        async for chunk in self.stream:
            yield chunk
