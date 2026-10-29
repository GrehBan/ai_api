"""Enumerations describing how a method's request and response look."""

from enum import StrEnum, unique


@unique
class BodyFormat(StrEnum):
    """Encoding of the request body.

    Attributes:
        JSON: ``application/json``; files and binary data are rejected.
        FORM: ``application/x-www-form-urlencoded``; files are rejected.
        MULTIPART: ``multipart/form-data``; the only format for files.
        NONE: The request has no body.
    """

    JSON = "json"
    FORM = "form"
    MULTIPART = "multipart"
    NONE = "none"


@unique
class ResponseFormat(StrEnum):
    """How the response body is parsed.

    Attributes:
        JSON: Validated into the method's response model.
        TEXT: Decoded to ``str`` using the response charset.
        BYTES: Returned as raw ``bytes``.
        NONE: The body must be empty; ``None`` is returned.
        SSE: Server-sent events (``text/event-stream``); streams only.
        NDJSON: Newline-delimited JSON, one value per line; streams only.
    """

    JSON = "json"
    TEXT = "text"
    BYTES = "bytes"
    NONE = "none"
    SSE = "sse"
    NDJSON = "ndjson"


@unique
class ResponseMode(StrEnum):
    """Whether the response is read at once or consumed as a stream.

    Attributes:
        BUFFERED: The whole body is read before parsing.
        STREAM: The body is parsed incrementally while it arrives.
    """

    BUFFERED = "buffered"
    STREAM = "stream"
