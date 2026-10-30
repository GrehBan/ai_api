"""Request fields shared by OpenAI methods."""

from __future__ import annotations

from typing import Annotated, Any

from annotated_types import MaxLen
from pydantic import BaseModel, Field, StringConstraints

type Metadata = Annotated[
    dict[
        Annotated[str, StringConstraints(max_length=64)],
        Annotated[str, StringConstraints(max_length=512)],
    ],
    MaxLen(16),
]
"""OpenAI metadata: up to 16 pairs, keys up to 64 and values up to 512
characters, checked before the request is sent."""


class ResponseParams(BaseModel):
    """Request fields of creating a response, buffered or streamed.

    A mixin of ``CreateResponse`` and ``StreamResponse``: it holds only
    fields, and the method classes add the endpoint. Only the fields
    needed for text generation are modeled so far.

    Attributes:
        model: Model identifier, such as ``"gpt-5"``.
        input: Text, taken as a ``user`` message, or a list of input
            items as OpenAI defines them.
        instructions: System (developer) message for this response; not
            carried over by ``previous_response_id``.
        max_output_tokens: Upper bound on output tokens, reasoning
            included; at least 16.
        metadata: Key-value pairs stored with the response.
        previous_response_id: Response to continue from.
        store: Whether OpenAI stores the response; ``True`` by default.
        temperature: Sampling temperature, 0 to 2.
        top_p: Nucleus sampling mass, 0 to 1.
    """

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=16)
    metadata: Metadata | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
