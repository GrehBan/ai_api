"""Identifiers shared by all layers: providers and request/response formats."""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class Provider(StrEnum):
    """API provider an error or integration belongs to.

    Attributes:
        OPENAI: OpenAI.
        ANTHROPIC: Anthropic.
        GENERIC: Not tied to a provider, e.g. the base error resolver.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GENERIC = "generic"


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
