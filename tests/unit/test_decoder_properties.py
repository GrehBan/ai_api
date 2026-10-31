"""T3: property tests of the line and SSE decoders.

Whatever the body and however it is cut into chunks, the decoders give
the same lines and events as for the whole body at once.
"""

from __future__ import annotations

import itertools
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from ai_api.response.parse.lines import LineDecoder
from ai_api.response.parse.sse import ServerSentEvent, SSEDecoder

NEWLINES = st.sampled_from(["\n", "\r\n", "\r"])

TEXT = st.text(
    st.characters(blacklist_categories=["Cs"], blacklist_characters="\r\n"),
    max_size=20,
)
"""Text of one line: any character, multi-byte ones included."""


@st.composite
def bodies(draw: st.DrawFn) -> str:
    """Lines with mixed endings, the last one maybe without, maybe a BOM."""
    lines = draw(st.lists(st.tuples(TEXT, NEWLINES), max_size=12))
    text = "".join(line + newline for line, newline in lines)
    text += draw(st.one_of(st.just(""), TEXT))
    return ("﻿" if draw(st.booleans()) else "") + text


@st.composite
def event_streams(draw: st.DrawFn) -> str:
    """An SSE body: fields, comments and blank lines, mixed endings."""
    field = st.one_of(
        TEXT.map("data: {}".format),
        TEXT.map("data:{}".format),
        st.just("data"),
        TEXT.map("event: {}".format),
        TEXT.map("id: {}".format),
        st.integers(0, 10**6).map("retry: {}".format),
        TEXT.map(": {}".format),
        TEXT.map("unknown: {}".format),
        st.just(""),
    )
    lines = draw(st.lists(st.tuples(field, NEWLINES), max_size=20))
    return "".join(line + newline for line, newline in lines)


def split(body: bytes, cuts: list[int]) -> list[bytes]:
    """Cut a body at the given positions.

    Args:
        body: The body.
        cuts: Positions; out-of-range ones are clamped, repeats give
            empty chunks.

    Returns:
        The chunks, in order.
    """
    points = sorted(min(cut, len(body)) for cut in cuts)
    edges = [0, *points, len(body)]
    return [body[start:end] for start, end in itertools.pairwise(edges)]


def lines_of(chunks: list[bytes]) -> list[str]:
    """Decode chunks into lines.

    Args:
        chunks: The body.

    Returns:
        Every line, the unterminated last one included.
    """
    decoder = LineDecoder()
    lines = [line for chunk in chunks for line in decoder.decode(chunk)]
    return lines + decoder.flush()


def events_of(chunks: list[bytes]) -> list[ServerSentEvent]:
    """Decode chunks into server-sent events.

    Args:
        chunks: The body.

    Returns:
        The completed events.
    """
    decoder = SSEDecoder()
    return [
        event
        for line in lines_of(chunks)
        if (event := decoder.decode(line)) is not None
    ]


CUTS = st.lists(st.integers(min_value=0, max_value=400), max_size=10)


@settings(max_examples=300, deadline=None)
@given(text=bodies(), cuts=CUTS)
def test_lines_do_not_depend_on_chunks(text: str, cuts: list[int]) -> None:
    body = text.encode()

    whole = lines_of([body])

    assert lines_of(split(body, cuts)) == whole
    assert lines_of([body[i : i + 1] for i in range(len(body))]) == whole


@settings(max_examples=300, deadline=None)
@given(text=bodies())
def test_lines_follow_the_sse_line_endings(text: str) -> None:
    parts = re.split(r"\r\n|\r|\n", text.removeprefix("﻿"))
    expected = parts[:-1] + ([parts[-1]] if parts[-1] else [])

    assert lines_of([text.encode()]) == expected


@settings(max_examples=300, deadline=None)
@given(text=event_streams(), cuts=CUTS)
def test_events_do_not_depend_on_chunks(text: str, cuts: list[int]) -> None:
    body = text.encode()

    whole = events_of([body])

    assert events_of(split(body, cuts)) == whole
    assert events_of([body[i : i + 1] for i in range(len(body))]) == whole
