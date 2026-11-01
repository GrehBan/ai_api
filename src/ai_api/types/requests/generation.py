"""Shared request for text generation."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import ConfigDict, Field

from ai_api.enums import Role
from ai_api.types.base import BaseAPIType
from ai_api.types.requests.base import BaseRequest


class Turn(BaseAPIType):
    """One turn of a conversation: who wrote it and its text.

    Immutable, and unknown fields are rejected. System prompts are not
    turns: they go in ``GenerationRequest.instructions``.

    Attributes:
        role: Who wrote the turn.
        content: Text of the turn.

    Example:
        >>> Turn.user("Hi")
        Turn(role=<Role.USER: 'user'>, content='Hi')
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: str

    @classmethod
    def user(cls, content: str) -> Self:
        """Create a turn of the end user.

        Args:
            content: Text of the turn.

        Returns:
            The turn.
        """
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Self:
        """Create a turn of the model, such as an earlier answer.

        Args:
            content: Text of the turn.

        Returns:
            The turn.
        """
        return cls(role=Role.ASSISTANT, content=content)


class GenerationRequest(BaseRequest):
    """Generate text from an input.

    Attributes:
        model: Model identifier of the provider; not empty.
        input: Text to respond to, taken as one ``user`` turn, or the
            conversation so far, a list of at least one turn; an empty
            text or list is refused before anything is sent. Every
            provider requires the last turn to be a ``user`` one; the
            adapter checks it and raises ``RequestValidationError``
            otherwise.
        instructions: Instructions for the model given apart from the
            input, such as a system prompt; none if ``None``.
        max_output_tokens: Upper bound on generated tokens; positive.
        metadata: Free-form key-value pairs stored with the request.

    Example:
        >>> GenerationRequest(
        ...     model="gpt-5",
        ...     input=[Turn.user("Hi"), Turn.assistant("Hello!"),
        ...            Turn.user("How are you?")],
        ... )
    """

    model: str = Field(min_length=1)
    input: (
        Annotated[str, Field(min_length=1)]
        | Annotated[list[Turn], Field(min_length=1)]
    )
    instructions: str | None = None

    max_output_tokens: int | None = Field(
        default=None,
        gt=0,
    )

    metadata: dict[str, str] | None = None
