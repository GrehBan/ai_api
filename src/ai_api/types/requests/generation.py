"""Shared request for text generation."""

from pydantic import Field

from ai_api.types.requests.base import BaseRequest


class GenerationRequest(BaseRequest):
    """Generate text from an input.

    Attributes:
        model: Model identifier of the provider.
        input: Text to respond to.
        temperature: Sampling temperature; the provider's default if
            ``None``.
        max_output_tokens: Upper bound on generated tokens; positive.
        metadata: Free-form key-value pairs stored with the request.
    """

    model: str
    input: str

    temperature: float | None = None
    max_output_tokens: int | None = Field(
        default=None,
        gt=0,
    )

    metadata: dict[str, str] | None = None
