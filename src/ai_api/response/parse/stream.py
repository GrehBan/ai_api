"""Parser for streamed responses."""

from __future__ import annotations

import codecs
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import TypeAdapter

from ai_api.enums import ResponseFormat
from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import APIStatusError
from ai_api.http.exceptions.transport import APIStreamInterruptedError
from ai_api.http.headers import CONTENT_TYPE, EVENT_STREAM_CONTENT_TYPE
from ai_api.response.parse.lines import LineDecoder, LineTooLongError
from ai_api.response.parse.sse import SSE_DONE, EventTooLargeError, SSEDecoder
from ai_api.response.protocols import IStreamResponse
from ai_api.serialization.json import JsonLoads, loads
from ai_api.types.matchers import DONE_MARKER, EventTypes, TopLevelError
from ai_api.types.protocols import IEventMatcher

if TYPE_CHECKING:
    from ai_api.methods.protocols import IStreamMethod


ERROR_EVENTS: Final = EventTypes("error") | TopLevelError()
"""Default ``error_events`` of ``StreamResponseParser``.

An event with a ``type`` is an error only if the type is ``"error"``, so
a regular event may carry an ``error`` field (a failed tool call, for
example). An object without a ``type`` is an error if its ``error`` is
``true`` or a non-empty object or string, as in OpenAI and Ollama.
"""


def _text_decoder(response: IStreamResponse) -> codecs.IncrementalDecoder:
    """Build the incremental decoder of a text stream's charset.

    Called when the stream is opened, like the ``Content-Type`` check of
    SSE, so a bad charset fails before the first item is awaited.

    Args:
        response: The streamed response.

    Returns:
        A decoder for the response charset, UTF-8 by default.

    Raises:
        APIResponseError: If the charset is unknown or is not a text
            encoding.
    """
    charset = response.charset or "utf-8"

    try:
        codec = codecs.lookup(charset)
    except LookupError as exc:
        raise APIResponseError(
            "Unknown response charset",
            status=response.status,
        ) from exc

    # The server picks the charset, and codecs such as base64, zlib
    # or rot13 do not turn bytes into text (bytes.decode rejects them
    # by this flag too).
    if not getattr(codec, "_is_text_encoding", True):
        raise APIResponseError(
            f"Unknown response charset: {charset} is not a text encoding",
            status=response.status,
        )

    return codec.incrementaldecoder()


def _check_event_stream(response: IStreamResponse) -> None:
    """Ensure an SSE response is an event stream.

    Args:
        response: The streamed response.

    Raises:
        APIResponseError: If its ``Content-Type`` is not
            ``text/event-stream``.
    """
    content_type = response.headers.get(CONTENT_TYPE, "")

    media_type = content_type.partition(";")[0].strip().lower()

    if media_type != EVENT_STREAM_CONTENT_TYPE:
        raise APIResponseError(
            f"Expected an event stream, got Content-Type {content_type!r}",
            status=response.status,
        )


class StreamResponseParser:
    """Turns a streamed body into chunks, text pieces or events."""

    def __init__(
        self,
        *,
        max_line_size: int | None = None,
        max_event_size: int | None = None,
        require_event_stream: bool = True,
        error_events: IEventMatcher = ERROR_EVENTS,
    ) -> None:
        """Create the parser.

        Args:
            max_line_size: Longest SSE or NDJSON line accepted, in
                characters; ``None`` for no limit.
            max_event_size: Largest SSE event data accepted, in
                characters; ``None`` for no limit.
            require_event_stream: Whether an SSE response must have
                ``Content-Type: text/event-stream``. A proxy or captive
                portal answering 200 with an HTML page would otherwise
                give a silently empty stream; turn it off only for a
                server that sends another type.
            error_events: Recognizes the decoded JSON of an SSE event or
                an NDJSON line that reports an error; such an event is
                raised as the ``APIStatusError`` the session's error
                resolver builds, instead of being yielded. An SSE event
                of type ``error`` is always raised.
        """
        self._max_line_size = max_line_size
        self._max_event_size = max_event_size
        self._require_event_stream = require_event_stream
        self._error_events = error_events

    @property
    def error_events(self) -> IEventMatcher:
        """Recognizes the decoded event data that is raised as an error."""
        return self._error_events

    def parse[T](
        self,
        method: IStreamMethod[T],
        response: IStreamResponse,
        json_loads: JsonLoads,
        error_resolver: IErrorResolver,
    ) -> AsyncIterator[T]:
        """Start parsing according to ``method.response_format``.

        Args:
            method: The method that was sent.
            response: The successful streamed response.
            json_loads: JSON decoder of the session.
            error_resolver: Error resolver of the session, for errors
                reported inside the stream.

        Returns:
            Raw chunks for BYTES, decoded text pieces for TEXT, and events
            validated into the method's item type for SSE and NDJSON
            (``ServerSentEvent`` items are returned as they are). For SSE
            and NDJSON, ``method.stream_terminal`` is enforced as
            ``BaseStreamMethod.__stream_terminal__`` describes.

        Raises:
            ValueError: If the format is not supported for streams.
        """
        match method.response_format:
            case ResponseFormat.BYTES:
                return cast(AsyncIterator[T], response.iter_chunks())

            case ResponseFormat.TEXT:
                decoder = _text_decoder(response)
                return cast(
                    AsyncIterator[T], self._iter_text(response, decoder)
                )

            case ResponseFormat.SSE | ResponseFormat.NDJSON:
                events = _EventReader(
                    response,
                    method.response_adapter,
                    json_loads,
                    error_resolver,
                    error_events=self._error_events,
                    terminal=method.stream_terminal,
                    max_line_size=self._max_line_size,
                    max_event_size=self._max_event_size,
                )

                if method.response_format is ResponseFormat.SSE:
                    if self._require_event_stream:
                        _check_event_stream(response)

                    return events.iter_sse()

                return events.iter_ndjson()

            case _:
                raise ValueError(
                    "Unsupported stream response format: "
                    f"{method.response_format}"
                )

    async def _iter_text(
        self,
        response: IStreamResponse,
        decoder: codecs.IncrementalDecoder,
    ) -> AsyncIterator[str]:
        """Decode the body incrementally.

        A multi-byte character split between two chunks is held back until
        the rest of it arrives.

        Args:
            response: The streamed response.
            decoder: The decoder of its charset.

        Yields:
            Non-empty pieces of decoded text.

        Raises:
            APIResponseError: If the body is not valid in its charset.
        """
        try:
            async for chunk in response.iter_chunks():
                if text := decoder.decode(chunk):
                    yield text

            if text := decoder.decode(b"", final=True):
                yield text
        except UnicodeError as exc:
            raise APIResponseError(
                "Failed to decode text stream",
                status=response.status,
            ) from exc


class _Ending:
    """Enforces the terminal item of one stream.

    Attributes:
        ended: Whether the terminal item has been seen.
        received: Items yielded so far.
    """

    def __init__(self, terminal: IEventMatcher | None, status: int) -> None:
        """Start at the beginning of a stream.

        Args:
            terminal: Recognizes the terminal item; ``None`` checks
                nothing.
            status: HTTP status of the stream, for errors.
        """
        self._terminal = terminal
        self._status = status
        self.ended = False
        self.received = 0

    def check_open(self) -> None:
        """Ensure no item comes after the terminal one.

        Raises:
            APIResponseError: If the terminal item has been seen.
        """
        if self.ended:
            raise APIResponseError(
                "Stream sent an item after its terminal event",
                status=self._status,
            )

    def see(self, item: object) -> None:
        """Count an item about to be yielded.

        Args:
            item: The item.
        """
        self.received += 1

        if self._terminal is not None and self._terminal.matches(item):
            self.ended = True

    def see_done(self) -> bool:
        """Handle ``data: [DONE]``, which is not an item.

        Returns:
            Whether to stop reading: without a declared terminal
            ``[DONE]`` ends the stream; with one, it only counts as the
            terminal if the terminal matches ``DONE_MARKER``, and the
            body is read to its end.
        """
        if self._terminal is None:
            return True

        if self._terminal.matches(DONE_MARKER):
            self.ended = True

        return False

    def finish(self) -> None:
        """Check the end of the body.

        Raises:
            APIStreamInterruptedError: If a terminal is declared and has
                not been seen.
        """
        if self._terminal is not None and not self.ended:
            raise APIStreamInterruptedError(
                "Stream ended before its terminal event",
                items_received=self.received,
            )


class _EventReader[T]:
    """Reads SSE or NDJSON events of one response.

    An event that signals an error is raised as the ``APIStatusError`` the
    error resolver builds: an SSE event of type ``error``, or data that
    ``error_events`` matches. A declared terminal is enforced as
    ``BaseStreamMethod.__stream_terminal__`` describes.
    """

    def __init__(
        self,
        response: IStreamResponse,
        adapter: TypeAdapter[T] | None,
        json_loads: JsonLoads,
        error_resolver: IErrorResolver,
        *,
        error_events: IEventMatcher,
        terminal: IEventMatcher | None = None,
        max_line_size: int | None = None,
        max_event_size: int | None = None,
    ) -> None:
        """Bind the reader to a response.

        Args:
            response: The streamed response.
            adapter: Validator of the item type. Without one, SSE events
                are returned as ``ServerSentEvent`` and NDJSON lines as
                decoded JSON.
            json_loads: JSON decoder; with the default one, data is
                validated by Pydantic directly in JSON mode.
            error_resolver: Builds the error for an error event.
            error_events: Recognizes decoded data that is an error.
            terminal: Recognizes the item that ends the stream; ``None``
                checks nothing.
            max_line_size: Longest line accepted, in characters; ``None``
                for no limit.
            max_event_size: Largest SSE event data accepted, in
                characters; ``None`` for no limit.
        """
        self._response = response
        self._adapter = adapter
        self._json_loads = json_loads
        self._error_resolver = error_resolver
        self._error_events = error_events
        self._terminal = terminal
        self._max_line_size = max_line_size
        self._max_event_size = max_event_size

    async def iter_sse(self) -> AsyncIterator[T]:
        """Parse a ``text/event-stream`` body.

        Yields:
            Validated events, or ``ServerSentEvent`` without an adapter.
            ``data: [DONE]`` is not yielded; without a declared terminal
            it ends the stream.

        Raises:
            APIStatusError: For an error event.
            APIResponseError: If the body is not valid UTF-8, an event is
                larger than the limit or does not match the item type, or
                an event comes after the terminal one.
            APIStreamInterruptedError: If the body ends before the
                declared terminal event.
        """
        decoder = SSEDecoder(self._max_event_size)
        ending = _Ending(self._terminal, self._response.status)

        async for line in self._iter_lines():
            try:
                event = decoder.decode(line)
            except EventTooLargeError as exc:
                raise APIResponseError(
                    str(exc),
                    status=self._response.status,
                ) from exc

            if event is None:
                continue

            if event.data == SSE_DONE:
                if ending.see_done():
                    return

                continue

            if event.event == "error":
                raise self._resolve(self._loads_or_none(event.data))

            item: T

            if self._adapter is None:
                ending.check_open()
                item = cast(T, event)
            else:
                body = self._decode(event.data)
                ending.check_open()
                item = self._validate(event.data, body)

            ending.see(item)
            yield item

        ending.finish()

    async def iter_ndjson(self) -> AsyncIterator[T]:
        """Parse a newline-delimited JSON body.

        Yields:
            One validated item per non-empty line.

        Raises:
            APIStatusError: For an error object.
            APIResponseError: If the body is not valid UTF-8, a line does
                not match the item type, or a line comes after the
                terminal one.
            APIStreamInterruptedError: If the body ends before the
                declared terminal line.
        """
        ending = _Ending(self._terminal, self._response.status)

        async for line in self._iter_lines():
            if line.strip():
                body = self._decode(line)
                ending.check_open()
                item = self._validate(line, body)
                ending.see(item)
                yield item

        ending.finish()

    async def _iter_lines(self) -> AsyncIterator[str]:
        """Split the body into lines.

        Yields:
            Lines without their endings; the last one may have none.

        Raises:
            APIResponseError: If the body is not valid UTF-8 or a line is
                longer than the limit.
        """
        decoder = LineDecoder(self._max_line_size)

        try:
            async for chunk in self._response.iter_chunks():
                for line in decoder.decode(chunk):
                    yield line

            for line in decoder.flush():
                yield line
        except UnicodeError as exc:
            raise APIResponseError(
                "Failed to decode event stream",
                status=self._response.status,
            ) from exc
        except LineTooLongError as exc:
            raise APIResponseError(
                str(exc),
                status=self._response.status,
            ) from exc

    def _decode(self, data: str) -> Any:
        """Decode the data of one event and raise it if it is an error.

        Every event is decoded, so ``error_events`` sees the same JSON
        the error resolver gets, whatever the item type accepts.

        Args:
            data: JSON text of the event.

        Returns:
            The decoded JSON.

        Raises:
            APIStatusError: If ``error_events`` matches the data.
            APIResponseError: If the data is not valid JSON.
        """
        try:
            body = self._json_loads(data)
        except ValueError as exc:
            raise APIResponseError(
                "Failed to decode stream event",
                status=self._response.status,
            ) from exc

        if self._error_events.matches(body):
            raise self._resolve(body, data)

        return body

    def _validate(self, data: str, body: Any) -> T:
        """Turn the data of one event into an item.

        Args:
            data: JSON text of the event.
            body: The same data, decoded.

        Returns:
            The validated item, or the decoded JSON without an adapter.

        Raises:
            APIResponseError: If the data does not match the item type.
        """
        if self._adapter is None:
            return cast(T, body)

        try:
            # The default decoder means JSON mode, as for buffered
            # responses; a custom decoder's result is validated as it is.
            if self._json_loads is loads:
                return self._adapter.validate_json(data)

            return self._adapter.validate_python(body)
        except ValueError as exc:  # ValidationError included
            raise APIResponseError(
                "Failed to validate stream event",
                status=self._response.status,
            ) from exc

    def _loads_or_none(self, data: str) -> Any:
        """Decode JSON, tolerating invalid data.

        Args:
            data: JSON text.

        Returns:
            The decoded value, or ``None`` if it is not valid JSON.
        """
        try:
            return self._json_loads(data)
        except ValueError:
            return None

    def _resolve(self, body: Any, data: str | None = None) -> APIStatusError:
        """Build the error for an error event.

        Args:
            body: The decoded event data, or ``None``.
            data: JSON text of the event, if it may fit the item type;
                the validated item is then kept as the error's ``raw``.

        Returns:
            The error built by the resolver from the stream's status and
            headers.
        """
        error = self._error_resolver.resolve(
            status=self._response.status,
            body=body,
            headers=self._response.headers,
        )

        # An error event that does not fit the item type (OpenAI's flat
        # "error" event, for one) is still raised; only raw stays None.
        if data is not None and self._adapter is not None:
            with suppress(APIResponseError):
                error.raw = self._validate(data, body)

        return error
