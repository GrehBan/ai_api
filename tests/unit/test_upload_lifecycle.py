"""How uploads and saved streams let go of what they read and write.

A worker thread cannot be stopped, so a cancelled blocking call runs on.
These tests check that nothing is closed or read again under it, that
every source is closed once, by ``aclose()`` else ``close()``, and that
no exception, a cancellation least of all, is lost on the way.
"""

from __future__ import annotations

import asyncio
import gc
import io
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest

from ai_api.enums import BodyFormat
from ai_api.http.exceptions.transport import (
    APIConnectionError,
    APIRequestError,
    APIStreamInterruptedError,
)
from ai_api.response.stream import Stream
from ai_api.serialization.values import BaseValueFormatter
from ai_api.session.body.aiohttp import AiohttpBodyBuilder
from ai_api.types.files import (
    IOFile,
    PathFile,
    StreamFile,
    ThreadWork,
    closing_source,
    open_file,
)

WAIT = 5.0
"""Seconds a test waits for a thread before it fails."""


class Gate:
    """Holds a blocking call in its thread until the test lets it go."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()

    def wait(self) -> None:
        self.entered.set()
        assert self.released.wait(WAIT)

    async def reached(self) -> None:
        assert await asyncio.to_thread(self.entered.wait, WAIT)


async def until(condition: Callable[[], object]) -> None:
    """Wait for something a thread or a loop callback does."""
    async with asyncio.timeout(WAIT):
        while not condition():
            await asyncio.sleep(0.01)


async def cancelled(task: asyncio.Task[Any]) -> None:
    """Cancel a task and check that it ends cancelled."""
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


class Chunks:
    """An async iterator that records how it is closed."""

    def __init__(self, *chunks: bytes) -> None:
        self.chunks = list(chunks)
        self.calls: list[str] = []

    def __aiter__(self) -> Chunks:
        return self

    async def __anext__(self) -> bytes:
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class SyncClose(Chunks):
    def close(self) -> None:
        self.calls.append("close")


class AsyncClose(Chunks):
    async def close(self) -> None:
        self.calls.append("async close")


class BothClose(Chunks):
    def close(self) -> None:
        self.calls.append("close")

    async def aclose(self) -> None:
        self.calls.append("aclose")


class FailingClose(Chunks):
    async def aclose(self) -> None:
        self.calls.append("aclose")
        raise OSError("close failed")


# closing_source


@pytest.mark.parametrize(
    ("source", "calls"),
    [
        (Chunks(), []),
        (SyncClose(), ["close"]),
        (AsyncClose(), ["async close"]),
        (BothClose(), ["aclose"]),
    ],
    ids=["neither", "close", "awaitable-close", "both"],
)
async def test_a_source_is_closed_once_aclose_first(
    source: Chunks,
    calls: list[str],
) -> None:
    async with closing_source(source):
        pass

    assert source.calls == calls


async def test_a_failing_close_after_a_normal_end_is_raised() -> None:
    with pytest.raises(OSError, match="close failed"):
        async with closing_source(FailingClose()):
            pass


async def test_a_failing_close_never_hides_the_error() -> None:
    with pytest.raises(ValueError, match="read failed") as info:
        async with closing_source(FailingClose()):
            raise ValueError("read failed")

    assert info.value.__notes__ == [
        "Closing the source also failed: OSError('close failed')"
    ]


async def test_a_cancellation_survives_a_failing_close() -> None:
    source = FailingClose()
    notes: list[str] = []

    async def read() -> None:
        try:
            async with closing_source(source):
                await asyncio.sleep(WAIT)
        except asyncio.CancelledError as exc:
            notes.extend(exc.__notes__)
            raise

    task = asyncio.create_task(read())
    await asyncio.sleep(0)

    await cancelled(task)

    assert source.calls == ["aclose"]
    assert notes == ["Closing the source also failed: OSError('close failed')"]


async def forward(source: AsyncIterator[bytes]) -> AsyncGenerator[bytes]:
    async with closing_source(source):
        async for chunk in source:
            yield chunk


async def test_a_generator_closed_early_closes_its_source() -> None:
    source = SyncClose(b"a", b"b")
    reader = forward(source)
    assert await anext(reader) == b"a"

    await reader.aclose()

    assert source.calls == ["close"]


async def test_a_generator_closed_early_raises_a_failing_close() -> None:
    reader = forward(FailingClose(b"a", b"b"))
    assert await anext(reader) == b"a"

    with pytest.raises(OSError, match="close failed"):
        await reader.aclose()


# ThreadWork


async def test_thread_work_returns_what_the_call_returns() -> None:
    work = ThreadWork()
    await work.wait_idle()

    assert await work.run(sum, [1, 2]) == 3
    assert not work.busy
    await work.wait_idle()


class Reported:
    """Collects what the event loop's exception handler is given."""

    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

    def __call__(
        self, _: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        self.contexts.append(context)

    @property
    def messages(self) -> list[str]:
        return [context["message"] for context in self.contexts]


@pytest.fixture
async def reported() -> AsyncIterator[Reported]:
    """Catch the reports of failures that have no caller any more."""
    collector = Reported()
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(collector)

    try:
        yield collector
    finally:
        gc.collect()  # an unretrieved task exception is reported here
        loop.set_exception_handler(None)


async def test_a_cancelled_call_runs_on_and_the_cleanup_waits(
    reported: Reported,
) -> None:
    gate = Gate()
    events: list[str] = []
    work = ThreadWork()

    def call() -> None:
        gate.wait()
        events.append("call ended")

    def cleanup() -> None:
        on_loop = threading.current_thread() is threading.main_thread()
        events.append("cleanup on the loop" if on_loop else "cleanup")

    task = asyncio.create_task(work.run(call))
    await gate.reached()

    await cancelled(task)

    assert work.busy
    await work.finish(cleanup)  # put off: it returns at once
    assert events == []

    with pytest.raises(RuntimeError, match="one call at a time"):
        await work.run(call)

    gate.released.set()
    await work.wait_idle()

    # The cleanup ran after the call, in a worker thread.
    assert events == ["call ended", "cleanup"]
    assert reported.contexts == []


@pytest.mark.parametrize(
    ("blocking", "thread"),
    [(True, "worker"), (False, "loop")],
    ids=["blocking", "not-blocking"],
)
async def test_a_cleanup_runs_where_it_may(
    blocking: bool,  # noqa: FBT001
    thread: str,
) -> None:
    where: list[str] = []

    def cleanup() -> None:
        on_loop = threading.current_thread() is threading.main_thread()
        where.append("loop" if on_loop else "worker")

    await ThreadWork().finish(cleanup, blocking=blocking)

    assert where == [thread]


def fail() -> None:
    raise OSError("close failed")


@pytest.mark.parametrize(
    "error",
    [None, GeneratorExit()],
    ids=["no-error", "generator-exit"],
)
async def test_a_failing_cleanup_is_raised_when_nothing_would_show_it(
    error: BaseException | None,
) -> None:
    with pytest.raises(OSError, match="close failed"):
        await ThreadWork().finish(fail, error=error)


async def test_a_failing_cleanup_is_noted_on_the_error() -> None:
    error = ValueError("read failed")

    await ThreadWork().finish(fail, error=error)

    assert error.__notes__ == [
        "Releasing the resource also failed: OSError('close failed')"
    ]


@pytest.mark.parametrize("noted", [True, False], ids=["error", "no-error"])
async def test_a_failing_put_off_cleanup_is_noted_and_reported(
    reported: Reported,
    noted: bool,  # noqa: FBT001
) -> None:
    # Nobody awaits a put-off cleanup: its failure goes on the error that
    # ended the work and to the loop, never to a later caller.
    gate = Gate()
    work = ThreadWork()
    task = asyncio.create_task(work.run(gate.wait))
    await gate.reached()
    await cancelled(task)
    error = asyncio.CancelledError()

    await work.finish(fail, error=error if noted else None)
    gate.released.set()
    await work.wait_idle()
    await asyncio.sleep(0)  # the done callback of the cleanup

    notes = ["Releasing the resource also failed: OSError('close failed')"]
    assert getattr(error, "__notes__", []) == (notes if noted else [])
    assert reported.messages == [
        "ThreadWork: Releasing the resource also failed"
    ]
    assert isinstance(reported.contexts[0]["exception"], OSError)


async def test_cancelled_work_retrieves_late_exception(
    reported: Reported,
) -> None:
    gate = Gate()
    work = ThreadWork()

    def read() -> bytes:
        gate.wait()
        raise OSError("read failed")

    task = asyncio.create_task(work.run(read))
    await gate.reached()

    await cancelled(task)
    gate.released.set()
    await work.wait_idle()
    await asyncio.sleep(0)  # the done callback of the call

    # Reported once, as a failure nobody could receive, and never as
    # "Task exception was never retrieved".
    assert reported.messages == [
        "ThreadWork: A call whose caller was cancelled failed"
    ]
    assert str(reported.contexts[0]["exception"]) == "read failed"


async def test_a_put_off_cleanup_cancelled_before_it_ran_is_reported(
    reported: Reported,
) -> None:
    # As at the end of asyncio.run, which cancels every task left.
    gate = Gate()
    work = ThreadWork()
    task = asyncio.create_task(work.run(gate.wait))
    await gate.reached()
    await cancelled(task)
    error = asyncio.CancelledError()
    before = asyncio.all_tasks()

    await work.finish(fail, error=error)
    (finishing,) = asyncio.all_tasks() - before
    finishing.cancel()
    await asyncio.wait({finishing})
    await asyncio.sleep(0)
    gate.released.set()
    await work.wait_idle()

    message = "Releasing the resource was cancelled before it ran"
    assert reported.messages == [f"ThreadWork: {message}"]
    assert error.__notes__[0].startswith(message)


# open_file


def patch_open(
    monkeypatch: pytest.MonkeyPatch,
    opener: Callable[[Path, str], Any],
) -> None:
    """Make ``Path.open`` call ``opener`` with the path and the mode."""

    def fake_open(path: Path, mode: str = "r") -> Any:
        return opener(path, mode)

    monkeypatch.setattr(Path, "open", fake_open)


class SlowOpen:
    """``Path.open`` that waits for the test before it opens."""

    def __init__(self, *, fails: bool = False) -> None:
        self.gate = Gate()
        self.fails = fails
        self.opened: list[Any] = []
        self.real = Path.open

    def __call__(self, path: Path, mode: str = "r") -> Any:
        self.gate.wait()

        if self.fails:
            raise OSError("open failed")

        file = self.real(path, mode)
        self.opened.append(file)
        return file


async def test_open_file_opens_a_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")

    with await open_file(path, "rb") as file:
        assert file.read() == b"x"


@pytest.mark.parametrize("mode", ["rb", "wb"])
async def test_a_cancelled_open_closes_what_the_thread_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Any,
) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")  # before the patch: it opens the file too
    slow_open = SlowOpen()
    patch_open(monkeypatch, slow_open)
    task = asyncio.create_task(open_file(path, mode))
    await slow_open.gate.reached()

    await cancelled(task)
    slow_open.gate.released.set()

    await until(lambda: slow_open.opened and slow_open.opened[0].closed)


async def test_a_cancelled_open_that_fails_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported: Reported,
) -> None:
    opener = SlowOpen(fails=True)
    patch_open(monkeypatch, opener)
    task = asyncio.create_task(open_file(tmp_path / "a.bin", "rb"))
    await opener.gate.reached()

    await cancelled(task)
    opener.gate.released.set()
    await until(lambda: asyncio.all_tasks() == {asyncio.current_task()})
    await asyncio.sleep(0)

    assert reported.messages == [
        "ThreadWork: A call whose caller was cancelled failed"
    ]


async def test_an_open_cancelled_with_its_thread_reports_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported: Reported,
) -> None:
    # As at the end of asyncio.run, which cancels every task left.
    slow_open = SlowOpen()
    patch_open(monkeypatch, slow_open)
    task = asyncio.create_task(open_file(tmp_path / "a.bin", "wb"))
    await slow_open.gate.reached()
    (opening,) = asyncio.all_tasks() - {asyncio.current_task(), task}

    await cancelled(task)
    opening.cancel()
    await asyncio.wait({opening})
    slow_open.gate.released.set()

    assert opening.cancelled()
    await until(lambda: slow_open.opened)
    slow_open.opened[0].close()
    await until(lambda: asyncio.all_tasks() == {asyncio.current_task()})
    assert reported.contexts == []


# PathFile


class GatedFile:
    """A file whose first read waits for the test; it records its close."""

    def __init__(self, data: bytes) -> None:
        self.gate = Gate()
        self.data = io.BytesIO(data)
        self.first = True
        self.reading = False
        self.closed = False
        self.closed_while_reading = False

    def read(self, size: int) -> bytes:
        self.reading = True
        try:
            if self.first:
                self.first = False
                self.gate.wait()
            return self.data.read(size)
        finally:
            self.reading = False

    def write(self, data: bytes) -> int:
        self.reading = True
        try:
            self.gate.wait()
            return len(data)
        finally:
            self.reading = False

    def close(self) -> None:
        self.closed_while_reading = self.reading
        self.closed = True


async def test_a_cancelled_read_closes_the_file_after_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = GatedFile(b"abc")
    patch_open(monkeypatch, lambda _path, _mode: file)
    reader = PathFile(tmp_path / "a.bin").open()

    async def step() -> bytes:
        return await anext(reader)

    task = asyncio.create_task(step())
    await file.gate.reached()

    await cancelled(task)

    assert not file.closed  # the read runs on in its thread
    file.gate.released.set()
    await until(lambda: file.closed)
    assert not file.closed_while_reading


class Opened:
    """Records every file ``Path.open`` opens."""

    def __init__(self) -> None:
        self.files: list[Any] = []
        self.real = Path.open

    def __call__(self, path: Path, mode: str = "r") -> Any:
        file = self.real(path, mode)
        self.files.append(file)
        return file


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    recorder = Opened()
    patch_open(monkeypatch, recorder)
    return recorder


async def test_a_path_file_closes_its_file_at_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    opened = Opened()
    patch_open(monkeypatch, opened)
    file = PathFile(path, chunk_size=1)

    assert [chunk async for chunk in file.open()] == [b"a", b"b", b"c"]
    reader = file.open()
    assert isinstance(reader, AsyncGenerator)
    assert await anext(reader) == b"a"
    await reader.aclose()

    assert [f.closed for f in opened.files] == [True, True]


# IOFile


class SlowStream(io.RawIOBase):
    """A seekable stream whose first read waits for the test.

    It records a seek or a read that starts while another one runs.
    """

    def __init__(self, data: bytes) -> None:
        self.data = io.BytesIO(data)
        self.gate = Gate()
        self.first = True
        self.busy = False
        self.seeks = 0
        self.overlaps: list[str] = []

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.data.tell()

    def seek(self, pos: int, whence: int = 0) -> int:
        if self.busy:
            self.overlaps.append("seek")
        self.seeks += 1
        return self.data.seek(pos, whence)

    def readinto(self, buffer: Any) -> int:
        if self.busy:
            self.overlaps.append("read")
        self.busy = True
        try:
            if self.first:
                self.first = False
                self.gate.wait()
            chunk = self.data.read(len(buffer))
            buffer[: len(chunk)] = chunk
            return len(chunk)
        finally:
            self.busy = False


async def test_the_next_read_waits_for_a_cancelled_one() -> None:
    content = bytes(range(100))
    stream = SlowStream(content)
    file = IOFile(stream, chunk_size=10)
    first = file.open()

    async def step() -> bytes:
        return await anext(first)

    task = asyncio.create_task(step())
    await stream.gate.reached()
    await cancelled(task)

    async def read_all() -> bytes:
        return b"".join([chunk async for chunk in file.open()])

    second = asyncio.create_task(read_all())
    for _ in range(10):
        await asyncio.sleep(0)
    assert stream.seeks == 1  # only the first read's: the second waits

    stream.gate.released.set()

    assert await second == content
    assert stream.overlaps == []


# StreamFile


async def test_a_stream_file_source_is_closed_before_it_is_read() -> None:
    source = SyncClose(b"a")

    async with closing_source(StreamFile(source).open()):
        pass

    assert source.calls == ["close"]


async def test_a_stream_file_generator_is_closed_when_cut_short() -> None:
    closed: list[bool] = []

    async def produce() -> AsyncIterator[bytes]:
        try:
            yield b"a"
            yield b"b"
        finally:
            closed.append(True)

    async with closing_source(StreamFile(produce()).open()) as reader:
        assert await anext(reader) == b"a"

    assert closed == [True]


async def test_a_never_sent_upload_closes_its_source() -> None:
    # The session closes the attempt's stack also when nothing was sent.
    source = SyncClose(b"a")
    uploads = AsyncExitStack()
    AiohttpBodyBuilder().build(
        {"file": StreamFile(source)},
        BodyFormat.MULTIPART,
        BaseValueFormatter(),
        uploads=uploads,
    )

    await uploads.aclose()

    assert source.calls == ["close"]


async def test_a_never_sent_upload_that_fails_to_close_raises() -> None:
    uploads = AsyncExitStack()
    AiohttpBodyBuilder().build(
        {"file": StreamFile(FailingClose(b"a"), filename="a.bin")},
        BodyFormat.MULTIPART,
        BaseValueFormatter(),
        uploads=uploads,
    )

    with pytest.raises(APIRequestError, match="close failed") as info:
        await uploads.aclose()

    assert "Failed to close upload 'a.bin'" in str(info.value)


async def test_a_failing_close_is_noted_on_the_attempts_error() -> None:
    uploads = AsyncExitStack()
    error = APIConnectionError("connection refused")
    AiohttpBodyBuilder().build(
        {"file": StreamFile(FailingClose(b"a"), filename="a.bin")},
        BodyFormat.MULTIPART,
        BaseValueFormatter(),
        uploads=uploads,
    )

    await uploads.__aexit__(type(error), error, None)

    (note,) = error.__notes__
    assert note.startswith("Closing upload 'a.bin' also failed: APIRequest")


# Stream.save


async def chunks(*items: bytes, error: bool = False) -> AsyncIterator[bytes]:
    for item in items:
        yield item
    if error:
        raise APIConnectionError("connection lost")


async def test_a_cancelled_save_closes_the_file_after_the_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = GatedFile(b"")
    patch_open(monkeypatch, lambda _path, _mode: file)
    task = asyncio.create_task(Stream(chunks(b"a")).save(tmp_path / "x"))
    await file.gate.reached()

    await cancelled(task)

    assert not file.closed  # the write runs on in its thread
    file.gate.released.set()
    await until(lambda: file.closed)
    assert not file.closed_while_reading


async def test_a_broken_save_closes_the_file_and_keeps_what_came(
    tmp_path: Path,
    opened: Opened,
) -> None:
    target = tmp_path / "out.bin"

    with pytest.raises(APIStreamInterruptedError):
        await Stream(chunks(b"a", b"b", error=True)).save(target)

    assert [f.closed for f in opened.files] == [True]
    assert target.read_bytes() == b"ab"
