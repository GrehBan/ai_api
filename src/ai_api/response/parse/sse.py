"""Server-sent events, assembled from lines as the HTML standard says."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SSE_DONE: Final[str] = "[DONE]"
"""Data of the event that ends an OpenAI-style stream."""


class EventTooLargeError(ValueError):
    """The data of an event is larger than the decoder's limit."""


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    r"""One server-sent event.

    Use it as the item type of a stream method to receive events as they
    are, without validating their data.

    Attributes:
        data: The ``data`` lines of the event, joined with ``\n``.
        event: The event type; ``"message"`` when the event has none.
        id: The last event ID seen so far in the stream, if any.
        retry: The reconnection time in milliseconds, if the event sets
            one.
    """

    data: str
    event: str = "message"
    id: str | None = None
    retry: int | None = None


class SSEDecoder:
    """Builds events from the lines of a ``text/event-stream`` body.

    Example:
        >>> decoder = SSEDecoder()
        >>> decoder.decode("event: ping")
        >>> decoder.decode("data: {}")
        >>> decoder.decode("")
        ServerSentEvent(data='{}', event='ping', id=None, retry=None)
    """

    def __init__(self, max_event_size: int | None = None) -> None:
        """Create a decoder at the start of a stream.

        Args:
            max_event_size: Largest event data accepted, in characters;
                ``None`` for no limit. Without it, a server that never
                ends an event makes the decoder buffer without bound,
                however short each line is.
        """
        self._max_event_size = max_event_size
        self._size = 0
        self._data: list[str] = []
        self._event = ""
        self._retry: int | None = None
        self._last_id: str | None = None

    def decode(self, line: str) -> ServerSentEvent | None:
        """Process one line.

        Args:
            line: A line of the body, without its line ending.

        Returns:
            The event completed by an empty line, or ``None``. An event
            without ``data`` lines is not returned.

        Raises:
            EventTooLargeError: If the event data exceeds
                ``max_event_size``.
        """
        if not line:
            return self._dispatch()

        if line.startswith(":"):
            return None

        name, colon, value = line.partition(":")

        if colon:
            value = value.removeprefix(" ")

        match name:
            case "data":
                # Joined with "\n", so every line after the first adds one.
                self._size += len(value) + bool(self._data)
                self._data.append(value)

                if (
                    self._max_event_size is not None
                    and self._size > self._max_event_size
                ):
                    raise EventTooLargeError(
                        f"Event exceeds {self._max_event_size} characters"
                    )
            case "event":
                self._event = value
            case "id" if "\0" not in value:
                self._last_id = value
            case "retry" if value.isascii() and value.isdigit():
                self._retry = int(value)
            case _:
                pass

        return None

    def _dispatch(self) -> ServerSentEvent | None:
        """Finish the current event and reset its fields.

        The last event ID is kept: it applies to later events too.

        Returns:
            The event, or ``None`` if it has no ``data``.
        """
        data, event, retry = self._data, self._event, self._retry
        self._data, self._event, self._retry = [], "", None
        self._size = 0

        if not data:
            return None

        return ServerSentEvent(
            data="\n".join(data),
            event=event or "message",
            id=self._last_id,
            retry=retry,
        )
