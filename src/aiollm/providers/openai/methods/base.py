"""Request fields shared by OpenAI methods."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from annotated_types import MaxLen
from pydantic import (
    MISSING,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    with_config,
)

type Metadata = Annotated[
    dict[
        Annotated[str, StringConstraints(max_length=64)],
        Annotated[str, StringConstraints(max_length=512)],
    ],
    MaxLen(16),
]
"""OpenAI metadata: up to 16 pairs, keys up to 64 and values up to 512
characters, checked before the request is sent."""


@with_config(ConfigDict(extra="forbid"))
class Reasoning(TypedDict, total=False):
    """Configuration of a reasoning model.

    Any other key is rejected before the request is sent.

    Attributes:
        effort: How much the model reasons; ``none`` and ``minimal``
            only for the models that take them.
        summary: A summary of the reasoning to return.
        context: Which earlier reasoning the model sees.
        mode: The reasoning mode, such as ``standard`` or ``pro``.
    """

    effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        | None
    )
    summary: Literal["auto", "concise", "detailed"] | None
    context: Literal["auto", "current_turn", "all_turns"] | None
    mode: str


class ResponseParams(BaseModel):
    """Request fields of creating a response, buffered or streamed.

    A mixin of ``CreateResponse`` and ``StreamResponse``: it holds only
    fields, and the method classes add the endpoint. Only the fields
    needed for text generation are modeled so far. Every optional field
    is left out until it is given and takes ``None``, sent as ``null``,
    as OpenAI's schema allows.

    Attributes:
        model: Model identifier, such as ``"gpt-5"``.
        input: Text, taken as a ``user`` message, or a list of input
            items as OpenAI defines them.
        instructions: System (developer) message for this response; not
            carried over by ``previous_response_id``.
        max_output_tokens: Upper bound on output tokens, reasoning
            included; at least 16.
        metadata: Key-value pairs stored with the response.
        reasoning: Configuration of a reasoning model, such as its
            effort.
        previous_response_id: Response to continue from.
        store: Whether OpenAI stores the response; ``True`` by default.
        temperature: Sampling temperature, 0 to 2.
        top_p: Nucleus sampling mass, 0 to 1.
    """

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None | MISSING = MISSING
    max_output_tokens: int | None | MISSING = Field(default=MISSING, ge=16)
    metadata: Metadata | None | MISSING = MISSING
    reasoning: Reasoning | None | MISSING = MISSING
    previous_response_id: str | None | MISSING = MISSING
    store: bool | None | MISSING = MISSING
    temperature: float | None | MISSING = Field(default=MISSING, ge=0, le=2)
    top_p: float | None | MISSING = Field(default=MISSING, ge=0, le=1)
