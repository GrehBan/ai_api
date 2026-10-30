"""Shared request for text generation."""

from __future__ import annotations

from pydantic import Field

from ai_api.types.requests.base import BaseRequest


class GenerationRequest(BaseRequest):
    """Generate text from an input.

    Attributes:
        model: Model identifier of the provider.
        input: Text to respond to.
        instructions: Instructions for the model given apart from the
            input, such as a system prompt; none if ``None``.
        max_output_tokens: Upper bound on generated tokens; positive.
        metadata: Free-form key-value pairs stored with the request.
    """

    model: str
    input: str
    instructions: str | None = None

    max_output_tokens: int | None = Field(
        default=None,
        gt=0,
    )

    metadata: dict[str, str] | None = None
