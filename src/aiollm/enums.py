"""Identifiers shared by all layers: providers, formats and shared values."""

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


@unique
class Role(StrEnum):
    """Author of a turn of a conversation.

    Attributes:
        USER: The end user.
        ASSISTANT: The model.
    """

    USER = "user"
    ASSISTANT = "assistant"


@unique
class IncompleteReason(StrEnum):
    """Why a generation stopped before its end, in terms every provider has.

    The provider's own reason is kept apart, as
    ``raw_incomplete_reason``.

    Attributes:
        MAX_TOKENS: The output reached the token limit of the request.
        REFUSAL: The model refused to answer.
        OTHER: Any other reason, such as a content filter or a full
            context window.
    """

    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    OTHER = "other"


@unique
class GenerationEventType(StrEnum):
    """``type`` of a shared generation event.

    Attributes:
        TEXT_DELTA: A piece of generated text, ``TextDeltaEvent``.
        COMPLETED: The end of the generation, ``CompletedEvent``.
        OTHER: A provider event without a shared counterpart,
            ``OtherEvent``.
    """

    TEXT_DELTA = "text_delta"
    COMPLETED = "completed"
    OTHER = "other"
