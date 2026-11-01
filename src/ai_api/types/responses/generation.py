"""Shared response of text generation."""

from __future__ import annotations

from ai_api.enums import IncompleteReason, Provider
from ai_api.types.base import BaseAPIType
from ai_api.types.requests.generation import Turn
from ai_api.types.responses.base import BaseResponse


class TokenUsage(BaseAPIType):
    """Token counts reported by the provider.

    The cache and reasoning counts are parts of the input and output
    counts, not added to them. ``None`` means the provider did not
    report the count.

    Attributes:
        input_tokens: Tokens in the input, the cached ones included.
        output_tokens: Tokens generated, reasoning included.
        total_tokens: Both together.
        cache_read_tokens: Input tokens read from the prompt cache.
        cache_write_tokens: Input tokens written to the prompt cache.
        reasoning_tokens: Output tokens spent on reasoning (thinking).
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


class GenerationResponse[RawT](BaseResponse):
    """Generated text with the provider's own response.

    ``RawT`` is the provider's response type. It has no bound: an
    unparametrized ``GenerationResponse`` keeps ``raw`` as it is given,
    while a bound such as ``BaseAPIType`` would turn a dict into an
    empty model, because unknown fields are ignored.

    A refusal is an answer, as in the providers' SDKs, not an error: it
    sets ``incomplete_reason`` to ``IncompleteReason.REFUSAL``.

    Attributes:
        id: Response identifier from the provider, if any.
        provider: Provider that generated the response.
        model: Model that generated it.
        text: Generated text, if any.
        usage: Token counts, if reported.
        incomplete_reason: Why the provider stopped before the end, in
            terms every provider has; ``None`` if the generation
            finished. ``text`` then holds what was generated until the
            stop.
        raw_incomplete_reason: The provider's own reason, such as
            OpenAI's ``"content_filter"``, if it gave one.
        refusal: Why the model refused, if it did and the provider says
            why.
        raw: The provider's response.
    """

    id: str | None = None
    provider: Provider
    model: str

    text: str | None = None
    usage: TokenUsage | None = None
    incomplete_reason: IncompleteReason | None = None
    raw_incomplete_reason: str | None = None
    refusal: str | None = None

    raw: RawT

    def to_turn(self) -> Turn:
        """Turn the generated text into a turn of the conversation.

        Returns:
            An ``assistant`` turn with ``text``.

        Raises:
            ValueError: The response has no text, such as a refusal
                without one.

        Example:
            >>> request = GenerationRequest(
            ...     model=model,
            ...     input=[*turns, response.to_turn(), Turn.user("More")],
            ... )
        """
        if self.text is None:
            name = "the response" if self.id is None else f"response {self.id}"
            message = f"{name} has no text to continue the conversation with"
            raise ValueError(message)
        return Turn.assistant(self.text)
