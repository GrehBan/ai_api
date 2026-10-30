"""Parser for streamed responses."""

from __future__ import annotations

import codecs
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from pydantic import TypeAdapter

from ai_api.enums import ResponseFormat
from ai_api.http.exceptions.protocols import IErrorResolver
from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.status import APIStatusError
from ai_api.http.headers import CONTENT_TYPE, EVENT_STREAM_CONTENT_TYPE
from ai_api.response.parse.lines import LineDecoder, LineTooLongError
from ai_api.response.parse.sse import SSE_DONE, EventTooLargeError, SSEDecoder
from ai_api.response.protocols import IStreamResponse
from ai_api.serialization.json import JsonLoads, loads

if TYPE_CHECKING:
    from ai_api.methods.protocols import IStreamMethod


def _is_error(value: Any) -> bool:
    """Check whether the ``error`` key of an event reports an error.

    Args:
        value: The value of the top-level ``error`` key.

    Returns:
        ``True`` for ``true`` and a non-empty object or string. ``null``,
        ``false`` and empty values mean "no error", as in OpenAI batch
        output lines.
    """
    return value is True or (isinstance(value, (Mapping, str)) and bool(value))


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
        """
        self._max_line_size = max_line_size
        self._max_event_size = max_event_size
        self._require_event_stream = require_event_stream

    def is_error_event(self, body: Any) -> bool:
        """Decide whether decoded event data reports an error.

        An event with a ``type`` is an error only if the type is
        ``"error"``, so a regular event may carry an ``error`` field (a
        failed tool call, for example). An event without a ``type`` is an
        error if its ``error`` key is ``true`` or a non-empty object or
        string, as in OpenAI and Ollama. Override it for a provider with
        other rules.

        Args:
            body: The decoded JSON of an SSE event or an NDJSON line.

        Returns:
            Whether the event is raised as an ``APIStatusError``.
        """
        if not isinstance(body, Mapping):
            return False

        body = cast(Mapping[str, Any], body)

        if "type" in body:
            kind: object = body["type"]
            return kind == "error"

        return _is_error(body.get("error"))

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
            (``ServerSentEvent`` items are returned as they are).

        Raises:
            ValueError: If the format is not supported for streams.
        """
        match method.response_format:
            case ResponseFormat.BYTES:
                return cast(AsyncIterator[T], response.iter_chunks())

            case ResponseFormat.TEXT:
                return cast(AsyncIterator[T], self._iter_text(response))

            case ResponseFormat.SSE | ResponseFormat.NDJSON:
                events = _EventReader(
                    response,
                    method.response_adapter,
                    json_loads,
                    error_resolver,
                    is_error_event=self.is_error_event,
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
    ) -> AsyncIterator[str]:
        """Decode the body incrementally.

        A multi-byte character split between two chunks is held back until
        the rest of it arrives.

        Args:
            response: The streamed response.

        Yields:
            Non-empty pieces of decoded text.

        Raises:
            APIResponseError: If the charset is unknown, is not a text
                encoding, or the body is not valid in it.
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

        decoder = codec.incrementaldecoder()

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


class _EventReader[T]:
    """Reads SSE or NDJSON events of one response.

    An event that signals an error is raised as the ``APIStatusError`` the
    error resolver builds: an SSE event of type ``error``, or data that
    ``is_error_event`` accepts.
    """

    def __init__(
        self,
        response: IStreamResponse,
        adapter: TypeAdapter[T] | None,
        json_loads: JsonLoads,
        error_resolver: IErrorResolver,
        *,
        is_error_event: Callable[[Any], bool],
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
            is_error_event: Decides whether decoded data is an error.
            max_line_size: Longest line accepted, in characters; ``None``
                for no limit.
            max_event_size: Largest SSE event data accepted, in
                characters; ``None`` for no limit.
        """
        self._response = response
        self._adapter = adapter
        self._json_loads = json_loads
        self._error_resolver = error_resolver
        self._max_line_size = max_line_size
        self._is_error_event = is_error_event
        self._max_event_size = max_event_size

    async def iter_sse(self) -> AsyncIterator[T]:
        """Parse a ``text/event-stream`` body.

        Yields:
            Validated events, or ``ServerSentEvent`` without an adapter.
            ``data: [DONE]`` ends the stream and is not yielded.

        Raises:
            APIStatusError: For an error event.
            APIResponseError: If the body is not valid UTF-8, an event is
                larger than the limit or does not match the item type.
        """
        decoder = SSEDecoder(self._max_event_size)

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
                return

            if event.event == "error":
                raise self._resolve(self._loads_or_none(event.data))

            if self._adapter is None:
                yield cast(T, event)
            else:
                yield self._validate(event.data)

    async def iter_ndjson(self) -> AsyncIterator[T]:
        """Parse a newline-delimited JSON body.

        Yields:
            One validated item per non-empty line.

        Raises:
            APIStatusError: For an error object.
            APIResponseError: If the body is not valid UTF-8 or a line
                does not match the item type.
        """
        async for line in self._iter_lines():
            if line.strip():
                yield self._validate(line)

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

    def _validate(self, data: str) -> T:
        """Turn the data of one event into an item.

        Args:
            data: JSON text of the event.

        Returns:
            The validated item, or the decoded JSON without an adapter.

        Raises:
            APIStatusError: If the data is an error object.
            APIResponseError: If the data is not valid JSON or does not
                match the item type.
        """
        # Only data that may be an error object is decoded twice.
        if '"error"' in data:
            body = self._loads_or_none(data)

            if self._is_error_event(body):
                raise self._resolve(body)

        try:
            if self._adapter is None:
                return cast(T, self._json_loads(data))

            if self._json_loads is loads:
                return self._adapter.validate_json(data)

            return self._adapter.validate_python(self._json_loads(data))
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

    def _resolve(self, body: Any) -> APIStatusError:
        """Build the error for an error event.

        Args:
            body: The decoded event data, or ``None``.

        Returns:
            The error built by the resolver from the stream's status and
            headers.
        """
        return self._error_resolver.resolve(
            status=self._response.status,
            body=body,
            headers=self._response.headers,
        )
