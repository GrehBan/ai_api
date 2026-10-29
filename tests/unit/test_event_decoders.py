"""SSE stage 1: splitting bytes into lines and lines into events."""

import pytest

from ai_api.client.response.parse.lines import LineDecoder, LineTooLongError
from ai_api.client.response.parse.sse import (
    EventTooLargeError,
    ServerSentEvent,
    SSEDecoder,
)

BODY = "event: delta\r\ndata: при вет 😀\r\rdata: [DONE]\n\n".encode()


def split_lines(chunks: list[bytes]) -> list[str]:
    decoder = LineDecoder()
    lines = [line for chunk in chunks for line in decoder.decode(chunk)]
    return lines + decoder.flush()


def events_of(lines: list[str]) -> list[ServerSentEvent]:
    decoder = SSEDecoder()
    return [
        event for line in lines if (event := decoder.feed(line)) is not None
    ]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"a\nb\n", ["a", "b"]),
        (b"a\r\nb\r\n", ["a", "b"]),
        (b"a\rb\r", ["a", "b"]),
        (b"a\n\nb", ["a", "", "b"]),
        (b"a\r\n\r\nb", ["a", "", "b"]),
        (b"a\r\rb", ["a", "", "b"]),
        (b"a\n\rb", ["a", "", "b"]),
        (b"no newline", ["no newline"]),
        (b"", []),
    ],
    ids=[
        "lf",
        "crlf",
        "cr",
        "empty-lf",
        "empty-crlf",
        "empty-cr",
        "lf-then-cr",
        "last-line-without-ending",
        "empty-body",
    ],
)
def test_line_endings(body: bytes, expected: list[str]) -> None:
    assert split_lines([body]) == expected


@pytest.mark.parametrize("cut", range(len(BODY) + 1))
def test_lines_survive_any_split_point(cut: int) -> None:
    assert split_lines([BODY[:cut], BODY[cut:]]) == [
        "event: delta",
        "data: при вет 😀",
        "",
        "data: [DONE]",
        "",
    ]


def test_lines_survive_one_byte_chunks() -> None:
    chunks = [BODY[i : i + 1] for i in range(len(BODY))]

    assert split_lines(chunks) == split_lines([BODY])


def test_cr_and_lf_in_separate_chunks_end_one_line() -> None:
    decoder = LineDecoder()

    assert decoder.decode(b"a\r") == ["a"]
    assert decoder.decode(b"") == []
    assert decoder.decode(b"\nb\n") == ["b"]


def test_long_line_is_joined_from_many_chunks() -> None:
    decoder = LineDecoder()

    for _ in range(3):
        assert decoder.decode(b"ab") == []

    assert decoder.decode(b"c\n") == ["abababc"]


@pytest.mark.parametrize(
    "chunks",
    [
        [b"12345\nabc\n"],
        [b"12", b"345", b"\n"],
        [b"1234", b"5\r", b"\n"],
    ],
    ids=["one-chunk", "held-back", "split-crlf"],
)
def test_lines_at_the_limit_pass(chunks: list[bytes]) -> None:
    decoder = LineDecoder(max_line_size=5)
    lines = [line for chunk in chunks for line in decoder.decode(chunk)]

    assert lines[0] == "12345"


@pytest.mark.parametrize(
    "chunks",
    [
        [b"123456\n"],
        [b"a\n123456"],
        [b"123", b"456"],
        [b"12345", b"6\n"],
    ],
    ids=["complete-line", "held-back-tail", "growing", "completed-later"],
)
def test_lines_over_the_limit_raise(chunks: list[bytes]) -> None:
    decoder = LineDecoder(max_line_size=5)

    with pytest.raises(LineTooLongError, match="exceeds 5 characters"):
        for chunk in chunks:
            decoder.decode(chunk)


def test_limit_counts_characters_not_bytes() -> None:
    decoder = LineDecoder(max_line_size=2)

    assert decoder.decode("😀😀\n".encode()) == ["😀😀"]


def test_limit_resets_after_flush() -> None:
    decoder = LineDecoder(max_line_size=3)
    decoder.decode(b"abc")

    assert decoder.flush() == ["abc"]
    assert decoder.decode(b"def\n") == ["def"]


def test_leading_bom_is_dropped_once() -> None:
    bom = "﻿".encode()

    assert split_lines([bom[:1], bom[1:] + b"a\n" + bom + b"b"]) == [
        "a",
        "﻿b",
    ]


@pytest.mark.parametrize(
    "chunks",
    [[b"a\xff\n"], [b"\xd0"]],
    ids=["invalid-byte", "truncated-character"],
)
def test_invalid_utf8_raises(chunks: list[bytes]) -> None:
    with pytest.raises(UnicodeDecodeError):
        split_lines(chunks)


def test_event_fields() -> None:
    events = events_of(
        ["event: delta", "id: 7", "retry: 1500", "data: {}", ""],
    )

    assert events == [
        ServerSentEvent(data="{}", event="delta", id="7", retry=1500),
    ]


def test_defaults() -> None:
    assert events_of(["data: x", ""]) == [ServerSentEvent(data="x")]


def test_multiline_data_is_joined_with_newlines() -> None:
    events = events_of(["data: a", "data:", "data:  b", ""])

    assert [event.data for event in events] == ["a\n\n b"]


@pytest.mark.parametrize(
    ("line", "data"),
    [("data:x", "x"), ("data: x", "x"), ("data:  x", " x"), ("data", "")],
    ids=["no-space", "one-space", "two-spaces", "no-colon"],
)
def test_one_leading_space_is_removed(line: str, data: str) -> None:
    assert events_of([line, ""]) == [ServerSentEvent(data=data)]


def test_comments_and_unknown_fields_are_ignored() -> None:
    events = events_of([": keep-alive", "foo: bar", "data: x", ""])

    assert events == [ServerSentEvent(data="x")]


@pytest.mark.parametrize(
    "lines",
    [[""], ["event: ping", ""], ["id: 1", ""], [": comment", ""]],
    ids=["blank", "event-only", "id-only", "comment-only"],
)
def test_event_without_data_is_not_dispatched(lines: list[str]) -> None:
    assert events_of(lines) == []


def test_event_type_does_not_leak_into_next_event() -> None:
    events = events_of(["event: ping", "", "data: x", ""])

    assert events == [ServerSentEvent(data="x")]


def test_last_event_id_carries_over() -> None:
    events = events_of(
        ["id: 1", "data: a", "", "data: b", "", "id:", "data: c", ""],
    )

    assert [event.id for event in events] == ["1", "1", ""]


def test_id_with_nul_is_ignored() -> None:
    events = events_of(["id: 1", "data: a", "", "id: 2\0", "data: b", ""])

    assert [event.id for event in events] == ["1", "1"]


@pytest.mark.parametrize(
    "value",
    ["", "1.5", "-1", "²", "abc"],
    ids=["empty", "fraction", "negative", "superscript", "text"],
)
def test_invalid_retry_is_ignored(value: str) -> None:
    assert events_of([f"retry: {value}", "data: x", ""]) == [
        ServerSentEvent(data="x"),
    ]


def test_retry_applies_to_its_event_only() -> None:
    events = events_of(["retry: 10", "data: a", "", "data: b", ""])

    assert [event.retry for event in events] == [10, None]


def test_incomplete_last_event_is_dropped() -> None:
    assert events_of(["data: a", "", "data: b"]) == [
        ServerSentEvent(data="a"),
    ]


def test_event_at_the_size_limit_passes() -> None:
    decoder = SSEDecoder(max_event_size=5)
    decoder.feed("data: ab")
    decoder.feed("data: cd")  # "ab\ncd": the newline counts

    assert decoder.feed("") == ServerSentEvent(data="ab\ncd")


def test_event_over_the_size_limit_raises() -> None:
    decoder = SSEDecoder(max_event_size=4)
    decoder.feed("data: ab")

    with pytest.raises(EventTooLargeError, match="exceeds 4"):
        decoder.feed("data: cd")


def test_event_size_resets_after_dispatch() -> None:
    decoder = SSEDecoder(max_event_size=3)

    for _ in range(3):
        decoder.feed("data: abc")
        assert decoder.feed("") == ServerSentEvent(data="abc")
