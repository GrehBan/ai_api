"""Block H2: the stream parser and ``Stream``."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from ai_api.client.exceptions.transport import APIResponseError
from ai_api.client.response.parse.stream import StreamResponseParser
from ai_api.client.response.stream import Stream
from ai_api.methods.base import BaseStreamMethod
from ai_api.methods.enums import ResponseFormat
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
    with pytest.raises(ValueError, match="Unsupported stream"):
        parse(bytes, [b"x"], __response_format__=response_format)


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
