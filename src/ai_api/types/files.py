"""Files to upload in multipart requests."""

from __future__ import annotations

import asyncio
import errno
import inspect
import io
import mimetypes
import os
import weakref
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal, Self, cast

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

DEFAULT_CHUNK_SIZE: Final[int] = 64 * 1024
"""Number of bytes read from disk or a stream at a time."""

type BinaryStream = io.BufferedIOBase | io.RawIOBase


@asynccontextmanager
async def closing_source[T](
    source: AsyncIterator[T],
) -> AsyncGenerator[AsyncIterator[T]]:
    """Close an async iterator when the block ends, however it ends.

    The iterator is closed by ``aclose()`` when it has one (a generator,
    for example), else by ``close()``, whose result is awaited when it is
    awaitable; never by both, and not at all when it has neither.

    A failing close never hides why the block ended: when the block
    raised, a cancellation too, that exception goes on with the close
    error added as a note. Only after a block that ended normally, or was
    closed as a generator, is the close error itself raised.

    Args:
        source: The iterator to close.

    Yields:
        The same iterator.
    """
    try:
        yield source
    except GeneratorExit:
        await _close(source)
        raise
    except BaseException as exc:
        try:
            await _close(source)
        except Exception as close_error:  # noqa: BLE001 - kept as a note
            exc.add_note(f"Closing the source also failed: {close_error!r}")
        raise
    else:
        await _close(source)


async def _close(source: object) -> None:
    """Close an iterator by ``aclose()``, else by ``close()``, if it can be.

    Args:
        source: The iterator to close.
    """
    for name in ("aclose", "close"):
        close = getattr(source, name, None)

        if callable(close):
            result = close()

            if inspect.isawaitable(result):
                await result

            return


class ThreadWork:
    """Blocking calls on one resource, run in a worker thread one by one.

    A thread cannot be stopped: when the coroutine awaiting a call is
    cancelled, the call still runs to its end. ``ThreadWork`` keeps that
    call, so the resource is released only after it (``finish``) and the
    next user waits for it (``wait_idle``) instead of using the resource
    under it.
    """

    def __init__(self) -> None:
        """Create the helper with no call running."""
        self._running: asyncio.Task[Any] | None = None

    @property
    def busy(self) -> bool:
        """Whether a call is still running in its thread."""
        return self._running is not None and not self._running.done()

    async def run[**P, R](
        self,
        func: Callable[P, R],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Run a blocking call in a worker thread.

        Args:
            func: The blocking callable.
            *args: Its positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            What the call returned.

        Raises:
            RuntimeError: If a call is still running: one resource takes
                one call at a time.
        """
        if self.busy:
            raise RuntimeError("ThreadWork runs one call at a time")

        task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
        self._running = task
        # A cancelled caller leaves the call running and kept in _running.
        return await asyncio.shield(task)

    async def wait_idle(self) -> None:
        """Wait until no call is running; a failed call does not raise."""
        if self._running is not None:
            await asyncio.wait({self._running})

    def finish(
        self,
        cleanup: Callable[[], object],
        *,
        error: BaseException | None = None,
    ) -> None:
        """Release the resource now, or as soon as the running call ends.

        A cleanup put off until the call ends runs in the event loop,
        which reports its failure through the loop's exception handler.

        Args:
            cleanup: Releases the resource, such as a file's ``close``.
            error: The exception that ends the work, if any; a failing
                cleanup is added to it as a note instead of hiding it.

        Raises:
            Exception: Whatever the cleanup raises when it runs now and
                there is no ``error`` to keep it, or that error is a
                ``GeneratorExit``, which nobody would see.
        """
        running = self._running

        if running is not None and not running.done():
            running.add_done_callback(lambda _: cleanup())
            return

        try:
            cleanup()
        except Exception as exc:
            if error is None or isinstance(error, GeneratorExit):
                raise
            error.add_note(f"Releasing the resource also failed: {exc!r}")


async def open_file(
    path: str | os.PathLike[str],
    mode: Literal["rb", "wb"],
) -> BinaryIO:
    """Open a binary file without blocking the event loop.

    The opening runs in a worker thread, which goes on when the caller is
    cancelled; a file it opens after that is closed at once, so a
    cancelled open leaves no file open.

    Args:
        path: The file to open.
        mode: ``"rb"`` to read, ``"wb"`` to create or overwrite.

    Returns:
        The open file.
    """
    opening = asyncio.create_task(
        asyncio.to_thread(Path(path).open, mode),
    )

    try:
        return await asyncio.shield(opening)
    except asyncio.CancelledError:
        opening.add_done_callback(_close_opened)
        raise


def _close_opened(opening: asyncio.Task[BinaryIO]) -> None:
    """Close the file an abandoned ``open_file`` opened, if it did.

    Args:
        opening: The task of the opening.
    """
    if opening.cancelled() or opening.exception() is not None:
        return

    opening.result().close()


@dataclass(slots=True)
class _Lease:
    """One ``IOFile`` read of a stream, kept in ``_READERS``.

    Attributes:
        reader: The ``repr`` of the reading ``IOFile``; a string, since
            the ``IOFile`` holds the stream, which would keep its weak key
            alive.
        work: The read's calls in the worker thread.
        ended: Whether the read has ended; its last call may still run.
    """

    reader: str
    work: ThreadWork = field(default_factory=ThreadWork)
    ended: bool = False


_READERS: Final[weakref.WeakKeyDictionary[BinaryStream, _Lease]] = (
    weakref.WeakKeyDictionary()
)
"""The read of each stream now.

A stream has one position, so two reads at once would take chunks from
under each other and send both uploads cut and mixed. The state belongs
to the stream, not to the ``IOFile``: one raw stream given to two
methods becomes two ``IOFile``s. A read that has ended keeps its entry
until its last call in the worker thread is over, so the next read waits
for that call instead of seeking under it.
"""


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
            ValueError: If the value cannot be converted to ``cls``. A
                ``str`` is never taken: it could be a path or the content
                itself, so a path must be a ``pathlib.Path``.
        """
        message = (
            f"Expected {cls.__name__}, bytes, a pathlib.Path (a str is not "
            "taken: it may be a path or the content) or a binary stream"
        )

        try:
            file = InputFile.from_value(value)
        except TypeError as exc:  # a path-like Path() rejects, bytes paths
            raise ValueError(message) from exc

        if isinstance(file, cls):
            return file

        raise ValueError(message)

    def __repr__(self) -> str:
        """Return the class name with the file metadata.

        Returns:
            A representation that does not include the content.
        """
        return (
            f"{type(self).__name__}(filename={self.filename!r}, "
            f"content_type={self.content_type!r})"
        )


class ChunkedFile(InputFile):
    """Content read in chunks, never held in memory as a whole.

    The base of ``PathFile`` and ``IOFile``, and of a file of your own
    that reads its source piece by piece.

    Attributes:
        chunk_size: Number of bytes read at a time.
    """

    def __init__(
        self,
        filename: str | None = None,
        content_type: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Store the file metadata and chunk size.

        Args:
            filename: File name sent in the multipart part.
            content_type: MIME type; guessed from ``filename`` if omitted.
            chunk_size: Number of bytes read at a time.

        Raises:
            ValueError: If ``chunk_size`` is below one: ``read(0)`` gives
                ``b""``, which would end the upload before it starts.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        super().__init__(filename, content_type)
        self.chunk_size = chunk_size


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


class PathFile(ChunkedFile):
    """A file on disk, read in chunks in a worker thread.

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
        super().__init__(
            filename or self.path.name,
            content_type,
            chunk_size,
        )

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

        The file is closed when the iterator ends or is closed; after a
        cancellation, once the read running in the worker thread is over.

        Yields:
            Chunks of at most ``chunk_size`` bytes.
        """
        file = await open_file(self.path, "rb")
        work = ThreadWork()

        try:
            while chunk := await work.run(file.read, self.chunk_size):
                yield chunk
        except BaseException as exc:
            work.finish(file.close, error=exc)
            raise

        work.finish(file.close)


class IOFile(ChunkedFile):
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

        super().__init__(filename, content_type, chunk_size)
        self.file = file
        self._start = file.tell() if file.seekable() else None
        self._opened = False

    @property
    def reusable(self) -> bool:
        """Whether the stream is seekable."""
        return self._start is not None

    async def open(self) -> AsyncIterator[bytes]:
        """Read the stream from its start position.

        Blocking calls run in a worker thread. The stream counts as being
        read until the iterator ends or is closed; the session closes it
        after every attempt, so a retry can read it again. A read that was
        cancelled leaves its last call running in the thread; the next
        read waits for that call before it seeks.

        Yields:
            Chunks of at most ``chunk_size`` bytes.

        Raises:
            RuntimeError: If the stream is being read by another upload at
                the moment, or a non-seekable stream is read a second time.
            BlockingIOError: If the stream is non-blocking and has no data
                ready: its ``read()`` returns ``None``, which is not the
                end of the stream, so the upload would be cut short.
        """
        while (lease := _READERS.get(self.file)) is not None:
            if not lease.ended:
                message = (
                    f"The stream is already being read by {lease.reader}; "
                    "two uploads cannot read one stream at once"
                )
                raise RuntimeError(message)

            await lease.work.wait_idle()

        if self._start is None and self._opened:
            raise RuntimeError("Non-seekable stream can be read only once")

        lease = _Lease(repr(self))
        _READERS[self.file] = lease
        self._opened = True

        try:
            if self._start is not None:
                await lease.work.run(self.file.seek, self._start)
            # A raw stream in non-blocking mode returns None, not b"".
            read = cast("Callable[[int], bytes | None]", self.file.read)

            while True:
                chunk = await lease.work.run(read, self.chunk_size)

                if chunk is None:
                    raise BlockingIOError(
                        errno.EAGAIN,
                        "The stream is non-blocking and has no data ready; "
                        "IOFile needs a blocking stream",
                    )
                if not chunk:
                    return

                yield chunk
        finally:
            lease.ended = True
            stream = self.file
            lease.work.finish(lambda: _READERS.pop(stream))


class StreamFile(InputFile):
    """Content produced by an async iterator; it can be read only once.

    A request carrying it is never retried. ``StreamFile`` takes ownership
    of the source: the upload closes it when it ends, however it ends
    (read to the end, cut short, cancelled or failed), by ``aclose()``,
    else by ``close()``.

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

    def open(self) -> AsyncIterator[bytes]:
        """Hand the source over to the upload.

        The returned iterator is not a generator, so closing it closes the
        source also before the first chunk is read.

        Returns:
            An iterator over the source's chunks, whose ``aclose()``
            closes the source.

        Raises:
            RuntimeError: If the file is read a second time.
        """
        if self._opened:
            raise RuntimeError("StreamFile can be read only once")

        self._opened = True
        return _OwnedSource(aiter(self.stream))


class _OwnedSource(AsyncIterator[bytes]):
    """A ``StreamFile`` source handed over to an upload."""

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        """Take the source.

        Args:
            source: The iterator over the content.
        """
        self._source = source

    async def __anext__(self) -> bytes:
        """Return the source's next chunk.

        Returns:
            The chunk.
        """
        return await anext(self._source)

    async def aclose(self) -> None:
        """Close the source by ``aclose()``, else ``close()``, if it can."""
        await _close(self._source)
