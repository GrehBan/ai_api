from enum import StrEnum, unique


@unique
class BodyFormat(StrEnum):
    JSON = "json"
    FORM = "form"
    MULTIPART = "multipart"
    NONE = "none"


@unique
class ResponseFormat(StrEnum):
    JSON = "json"
    TEXT = "text"
    BYTES = "bytes"
    NONE = "none"
    SSE = "sse"
    NDJSON = "ndjson"


@unique
class ResponseMode(StrEnum):
    BUFFERED = "buffered"
    STREAM = "stream"