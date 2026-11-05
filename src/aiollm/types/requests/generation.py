"""Shared request for text generation."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import MISSING, ConfigDict, Field
from typing_extensions import TypedDict

from aiollm.enums import Effort, Role
from aiollm.types.base import BaseAPIType
from aiollm.types.requests.base import BaseRequest


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


class NoProviderOptions(TypedDict, total=False, closed=True):
    """Provider options of a generation resource that takes none.

    The default of ``GenerationResource``'s ``OptionsT``. It is closed
    (PEP 728): only ``{}`` matches it, not even a ``TypedDict`` with keys
    of its own, which an open empty ``TypedDict`` would take. An adapter
    without options still checks what it is given at run time with
    ``checked_options(options, NoProviderOptions, provider)``.
    """


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
            input, such as a system prompt.
        max_output_tokens: Upper bound on generated tokens; positive.
        metadata: Free-form key-value pairs stored with the request.
        effort: How much work the model puts into the response; the
            model's own default when left out.

    An optional field is left out until it is given; ``None`` is not
    taken (it would mean ``null``, which not every provider accepts),
    so leave a field out rather than set it to ``None``.

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
    instructions: str | MISSING = MISSING
    max_output_tokens: int | MISSING = Field(default=MISSING, gt=0)
    metadata: dict[str, str] | MISSING = MISSING
    effort: Effort | MISSING = MISSING
