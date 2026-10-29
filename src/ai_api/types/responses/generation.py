"""Shared response of text generation."""

from ai_api.enums import Provider
from ai_api.types.base import BaseAPIType
from ai_api.types.responses.base import BaseResponse


class TokenUsage(BaseAPIType):
    """Token counts reported by the provider.

    Attributes:
        input_tokens: Tokens in the input.
        output_tokens: Tokens generated.
        total_tokens: Both together.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GenerationResponse[RawT](BaseResponse):
    """Generated text with the provider's own response.

    ``RawT`` is the provider's response type. It has no bound: an
    unparametrized ``GenerationResponse`` keeps ``raw`` as it is given,
    while a bound such as ``BaseAPIType`` would turn a dict into an
    empty model, because unknown fields are ignored.

    Attributes:
        id: Response identifier from the provider, if any.
        provider: Provider that generated the response.
        model: Model that generated it.
        text: Generated text, if any.
        usage: Token counts, if reported.
        raw: The provider's response.
    """

    id: str | None = None
    provider: Provider
    model: str

    text: str | None = None
    usage: TokenUsage | None = None

    raw: RawT
