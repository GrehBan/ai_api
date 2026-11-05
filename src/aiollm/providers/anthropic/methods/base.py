"""Request fields and headers shared by Anthropic methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal, TypedDict

from pydantic import (
    MISSING,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    with_config,
)

from aiollm.http.headers import ANTHROPIC_VERSION
from aiollm.providers.anthropic.api import API_VERSION

VERSION_HEADERS: Final[Mapping[str, str]] = {ANTHROPIC_VERSION: API_VERSION}
"""``__request_headers__`` of every Anthropic method.

Anthropic requires ``anthropic-version`` on every request, and the
version fixes the request and response format, so a method written
against ``API_VERSION`` sends it itself. A method of your own sets
``__request_headers__ = VERSION_HEADERS`` too.
"""


@with_config(ConfigDict(extra="forbid"))
class MessageParam(TypedDict):
    """One turn of the conversation.

    A ``TypedDict``, so a plain dict literal type-checks; unknown keys
    are still rejected when the method is built.

    Attributes:
        role: Who wrote the turn.
        content: Text, or a list of content blocks as Anthropic defines
            them.
    """

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


@with_config(ConfigDict(extra="forbid"))
class Metadata(TypedDict, total=False):
    """Metadata of a request; Anthropic takes only ``user_id``.

    Any other key is rejected before the request is sent.

    Attributes:
        user_id: Opaque identifier of the end user, such as a hash; up
            to 512 characters.
    """

    user_id: Annotated[str, StringConstraints(max_length=512)] | None


@with_config(ConfigDict(extra="forbid"))
class JSONOutputFormat(TypedDict):
    """Output that follows a JSON Schema.

    Attributes:
        type: Always ``json_schema``.
        schema: The JSON Schema.
    """

    type: Literal["json_schema"]
    schema: dict[str, Any]


@with_config(ConfigDict(extra="forbid"))
class OutputConfig(TypedDict, total=False):
    """Configuration of the output.

    Any other key is rejected before the request is sent.

    Attributes:
        effort: How much work the model puts into the response; thinking
            and the reply both count.
        format: A JSON Schema the response must follow.
    """

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None
    format: JSONOutputFormat | None


class MessageParams(BaseModel):
    """Request fields of creating a message, buffered or streamed.

    A mixin of ``CreateMessage`` and ``StreamMessage``: it holds only
    fields, and the method classes add the endpoint. Only the fields
    needed for text generation are modeled so far. There are no
    ``temperature``, ``top_p`` and ``top_k``: API version 2026-10-21
    removed them and answers 400 to a request with any of them. The
    optional fields are left out until they are given; Anthropic's
    schema takes no ``null`` for them.

    Attributes:
        model: Model identifier, such as ``"claude-opus-5-5"``.
        messages: The conversation; at least one turn.
        max_tokens: Upper bound on generated tokens; required, at least
            0 (0 only fills the prompt cache).
        system: System prompt: text, or a list of text blocks.
        metadata: Metadata of the request.
        output_config: Configuration of the output, such as its effort.
        stop_sequences: Texts that stop generation.
    """

    model: str
    messages: list[MessageParam] = Field(min_length=1)
    max_tokens: int = Field(ge=0)
    system: str | list[dict[str, Any]] | MISSING = MISSING
    metadata: Metadata | MISSING = MISSING
    output_config: OutputConfig | MISSING = MISSING
    stop_sequences: list[str] | MISSING = MISSING
