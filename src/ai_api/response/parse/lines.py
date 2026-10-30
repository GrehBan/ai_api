"""Incremental splitting of a byte stream into text lines."""

from __future__ import annotations

import codecs
import re
from typing import Final

_NEWLINE: Final = re.compile(r"\r\n|\r|\n")


class LineTooLongError(ValueError):
    """A line is longer than the decoder's limit."""


class LineDecoder:
    r"""Decodes UTF-8 chunks and splits them into lines.

    Lines end with ``\n``, ``\r\n`` or ``\r``, as in server-sent events;
    other Unicode line breaks stay inside the line. A line, a multi-byte
    character or a ``\r\n`` pair split between two chunks is held back
    until the rest arrives. A leading byte order mark is dropped.

    Attributes:
        max_line_size: Longest line accepted, in characters; ``None`` for
            no limit.

    Example:
        >>> decoder = LineDecoder()
        >>> decoder.decode(b"a\r\nb")
        ['a']
        >>> decoder.flush()
        ['b']
    """

    def __init__(self, max_line_size: int | None = None) -> None:
        """Create a decoder at the start of a stream.

        Args:
            max_line_size: Longest line accepted, in characters; ``None``
                for no limit.
        """
        self.max_line_size = max_line_size
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")()
        self._parts: list[str] = []
        self._held = 0
        self._pending_cr = False

    def decode(self, chunk: bytes) -> list[str]:
        """Feed a chunk of the body.

        Args:
            chunk: The next bytes of the body.

        Returns:
            The lines completed by this chunk, without line endings.

        Raises:
            UnicodeDecodeError: If the body is not valid UTF-8.
            LineTooLongError: If a line is longer than ``max_line_size``.
        """
        return self._split(self._decoder.decode(chunk))

    def flush(self) -> list[str]:
        """Finish the stream.

        Returns:
            The lines still held back, including a last line that has no
            line ending.

        Raises:
            UnicodeDecodeError: If the body ends inside a character.
        """
        lines = self._split(self._decoder.decode(b"", final=True))

        if self._parts:
            lines.append("".join(self._parts))
            self._parts = []
            self._held = 0

        return lines

    def _split(self, text: str) -> list[str]:
        """Split decoded text, joining it with the held-back line.

        Args:
            text: Newly decoded text.

        Returns:
            The completed lines.

        Raises:
            LineTooLongError: If a line is longer than ``max_line_size``.
        """
        if not text:
            return []

        if self._pending_cr:
            # The previous text ended with "\r": a "\n" here completes the
            # same line ending, not an empty line.
            self._pending_cr = False
            text = text.removeprefix("\n")

        parts = _NEWLINE.split(text)

        if len(parts) == 1:
            self._parts.append(text)
            self._held += len(text)
            self._check(self._held)
            return []

        self._pending_cr = text.endswith("\r")
        lines = ["".join([*self._parts, parts[0]]), *parts[1:-1]]
        self._parts = [parts[-1]] if parts[-1] else []
        self._held = len(parts[-1])

        if self.max_line_size is not None:
            for size in (*map(len, lines), self._held):
                self._check(size)

        return lines

    def _check(self, size: int) -> None:
        """Enforce ``max_line_size``.

        Args:
            size: Length of a line or of the held-back part of one.

        Raises:
            LineTooLongError: If ``size`` exceeds the limit.
        """
        if self.max_line_size is not None and size > self.max_line_size:
            raise LineTooLongError(
                f"Line exceeds {self.max_line_size} characters"
            )
