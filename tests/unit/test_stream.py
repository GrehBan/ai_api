"""Block H2: the stream parser and ``Stream``."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from ai_api.enums import ResponseFormat
from ai_api.http.exceptions.resolver import BaseErrorResolver
from ai_api.http.exceptions.response import APIResponseError
from ai_api.methods.base import BaseStreamMethod
from ai_api.response.parse.stream import StreamResponseParser
from ai_api.response.stream import Stream, map_stream, transform_stream
from ai_api.serialization.json import loads
from tests.conftest import FakeStreamResponse, make_method


def parse(
    argument: Any,
    chunks: list[bytes],
    *,
    charset: str | None = None,
    **class_vars: Any,
) -> AsyncIterator[Any]:
    method = make_method(BaseStreamMethod[argument], **class_vars)()
    return StreamResponseParser().parse(
        method,
        FakeStreamResponse(chunks, status=206, charset=charset),
        loads,
        BaseErrorResolver(),
    )


async def collect(items: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in items]


async def test_bytes_chunks_are_unchanged_and_ordered() -> None:
    chunks = [b"\x00\x01", b"", b"\xff" * 3, b"tail"]

    assert await collect(parse(bytes, chunks)) == chunks


SPLIT_CASES = [
    (char, position)
    for char in ("é", "€", "😀")
    for position in range(1, len(char.encode()))
]


@pytest.mark.parametrize(
    ("char", "position"),
    SPLIT_CASES,
    ids=[
        f"{len(char.encode())}-bytes-at-{position}"
        for char, position in SPLIT_CASES
    ],
)
async def test_text_survives_split_characters(
    char: str,
    position: int,
) -> None:
    encoded = f"a{char}b".encode()
    cut = 1 + position

    pieces = await collect(parse(str, [encoded[:cut], encoded[cut:]]))

    assert "".join(pieces) == f"a{char}b"
    assert all(pieces)


async def test_text_character_split_over_many_chunks() -> None:
    encoded = "😀".encode()

    pieces = await collect(parse(str, [bytes([byte]) for byte in encoded]))

    assert pieces == ["😀"]


async def test_text_uses_response_charset() -> None:
    chunks = ["при".encode("cp1251"), "вет".encode("cp1251")]

    assert await collect(parse(str, chunks, charset="cp1251")) == [
        "при",
        "вет",
    ]


async def test_text_skips_empty_pieces() -> None:
    assert await collect(parse(str, [b"", b"a", b""])) == ["a"]


async def test_text_flushes_decoder_at_the_end() -> None:
    # The idna decoder holds text back until a label is complete, so the
    # final ``decode(b"", final=True)`` is what produces it.
    assert await collect(parse(str, [b"abc"], charset="idna")) == ["abc"]


async def test_text_unknown_charset() -> None:
    with pytest.raises(
        APIResponseError, match="Unknown response charset"
    ) as info:
        await collect(parse(str, [b"a"], charset="no-such-charset"))

    assert info.value.status == 206


@pytest.mark.parametrize(
    "chunks",
    [[b"ok", "€".encode()[:2]], [b"\xff"]],
    ids=["truncated-at-end", "invalid-byte"],
)
async def test_text_decoding_errors(chunks: list[bytes]) -> None:
    with pytest.raises(APIResponseError, match="Failed to decode") as info:
        await collect(parse(str, chunks))

    assert info.value.status == 206
    assert isinstance(info.value.__cause__, UnicodeError)


@pytest.mark.parametrize(
    "response_format",
    [ResponseFormat.JSON, ResponseFormat.NONE],
    ids=["json", "none"],
)
def test_unsupported_format_fails_on_parse_call(
    response_format: ResponseFormat,
) -> None:
    # An explicit parser skips the class-definition check.
    with pytest.raises(ValueError, match="Unsupported stream"):
        parse(
            bytes,
            [b"x"],
            __response_format__=response_format,
            __stream_parser__=StreamResponseParser(),
        )


async def test_stream_iterates_items() -> None:
    stream = Stream(parse(bytes, [b"a", b"b"]))

    assert [chunk async for chunk in stream] == [b"a", b"b"]


async def test_stream_aclose_closes_generator() -> None:
    closed = False

    async def items() -> AsyncIterator[int]:
        nonlocal closed
        try:
            yield 1
            yield 2
        finally:
            closed = True

    stream = Stream(items())

    async for _ in stream:
        break

    await stream.aclose()

    assert closed


async def test_stream_aclose_on_plain_iterator() -> None:
    class Countdown:
        def __init__(self) -> None:
            self.left = 2

        def __aiter__(self) -> Countdown:
            return self

        async def __anext__(self) -> int:
            if not self.left:
                raise StopAsyncIteration
            self.left -= 1
            return self.left

    countdown = Countdown()
    stream = Stream(countdown)

    await stream.aclose()

    assert countdown.left == 2


async def test_closed_stream_cannot_be_iterated(tmp_path: Path) -> None:
    stream = Stream(parse(bytes, [b"a"]))

    await stream.aclose()

    with pytest.raises(RuntimeError, match="Stream is closed"):
        aiter(stream)

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await stream.save(tmp_path / "out.bin")

    assert not (tmp_path / "out.bin").exists()


async def test_iterator_taken_inside_fails_after_close() -> None:
    stream = Stream(parse(bytes, [b"a", b"b"]))
    items = aiter(stream)
    assert await anext(items) == b"a"

    await stream.aclose()

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await anext(items)


async def test_stream_save(tmp_path: Path) -> None:
    chunks = [b"first", b"", b"\x00second"]
    target = tmp_path / "out.bin"
    target.write_bytes(b"old content that is longer")

    written = await Stream(parse(bytes, chunks)).save(target)

    assert written == len(b"first\x00second")
    assert target.read_bytes() == b"first\x00second"


async def test_stream_save_accepts_str_path(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"

    assert await Stream(parse(bytes, [b"x"])).save(str(target)) == 1
    assert target.read_bytes() == b"x"


class Source:
    """A source stream that records when its parts are closed."""

    def __init__(self, items: list[int]) -> None:
        self.items = items
        self.log: list[str] = []

    async def generate(self) -> AsyncGenerator[int]:
        try:
            for item in self.items:
                yield item
        finally:
            self.log.append("source items closed")

    @asynccontextmanager
    async def open(self) -> AsyncGenerator[Stream[int]]:
        stream = Stream(self.generate())

        try:
            yield stream
        finally:
            await stream.aclose()
            self.log.append("source closed")


async def test_map_stream_converts_every_item() -> None:
    source = Source([1, 2, 3])

    async with map_stream(source.open(), str) as stream:
        assert await collect(stream) == ["1", "2", "3"]

    assert source.log == ["source items closed", "source closed"]


async def test_map_stream_closes_both_layers_on_break() -> None:
    source = Source([1, 2, 3])
    log = source.log

    def convert(item: int) -> int:
        log.append(f"item {item}")
        return item * 10

    async with map_stream(source.open(), convert) as stream:
        async for item in stream:
            assert item == 10
            break

        log.append("block ends")

    # The converted stream is closed before the source, inside the block.
    assert log == [
        "item 1",
        "block ends",
        "source items closed",
        "source closed",
    ]


async def test_map_stream_closes_both_layers_on_error() -> None:
    source = Source([1, 2])

    def convert(item: int) -> int:
        if item == 2:
            raise ValueError("bad item")

        return item

    with pytest.raises(ValueError, match="bad item"):
        async with map_stream(source.open(), convert) as stream:
            await collect(stream)

    assert source.log == ["source items closed", "source closed"]


async def test_map_stream_fails_after_the_block() -> None:
    source = Source([1, 2])

    async with map_stream(source.open(), str) as stream:
        items = aiter(stream)
        assert await anext(items) == "1"

    with pytest.raises(RuntimeError, match="Stream is closed"):
        await anext(items)


async def test_transform_stream_sees_the_whole_source() -> None:
    source = Source([1, 2, 3, 4])

    async def evens_twice(items: AsyncIterator[int]) -> AsyncGenerator[int]:
        async for item in items:
            if item % 2 == 0:
                yield item
                yield item

    async with transform_stream(source.open(), evens_twice) as stream:
        assert await collect(stream) == [2, 2, 4, 4]

    assert source.log == ["source items closed", "source closed"]


async def test_transform_stream_keeps_state_between_items() -> None:
    source = Source([1, 2, 3])

    async def total(items: AsyncIterator[int]) -> AsyncGenerator[int]:
        sum_ = 0

        async for item in items:
            sum_ += item

        yield sum_

    async with transform_stream(source.open(), total) as stream:
        assert await collect(stream) == [6]


async def test_transform_stream_closes_the_transform_first() -> None:
    source = Source([1, 2, 3])
    log = source.log

    async def logged(items: AsyncIterator[int]) -> AsyncGenerator[int]:
        try:
            async for item in items:
                yield item
        finally:
            log.append("transform closed")

    async with transform_stream(source.open(), logged) as stream:
        async for _ in stream:
            break

    assert log == ["transform closed", "source items closed", "source closed"]
