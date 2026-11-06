"""Shared request for text generation."""

from __future__ import annotations

from typing import Annotated, Final, Self

from pydantic import MISSING, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from aiollm.enums import Effort, Role, ToolChoice
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


type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)
"""A JSON value, as ``json.loads`` gives it."""

type JSONSchema = dict[str, JSONValue]
"""A JSON Schema, as a JSON object."""


TOOL_NAME_PATTERN: Final = r"^[a-zA-Z0-9_-]{1,128}$"
"""The tool names both providers take: Anthropic's pattern, and the 128
characters OpenAI took in a live check (2026-11-06)."""


class Tool(BaseAPIType):
    """A function in your code that the model may call.

    Immutable, and unknown fields are rejected.

    Attributes:
        name: Name the model calls it by: letters, digits, ``_`` and
            ``-``, up to 128 characters; unique in a request.
        description: What the tool does and when to use it.
        parameters: JSON Schema of its arguments, an object schema; a
            tool without arguments takes ``{"type": "object",
            "properties": {}}``. Sent as it is.
        strict: Whether the arguments must follow ``parameters``
            exactly; the schema then needs ``additionalProperties:
            false`` and every property in ``required``. Left out, neither
            provider is strict.

    Example:
        >>> Tool(
        ...     name="get_weather",
        ...     description="Current weather in a city.",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"city": {"type": "string"}},
        ...         "required": ["city"],
        ...     },
        ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str | MISSING = MISSING
    parameters: JSONSchema
    strict: bool | MISSING = MISSING


class ForcedTool(BaseAPIType):
    """Make the model call this tool.

    The ``tool_choice`` that names one tool, apart from the
    ``ToolChoice`` modes, so a tool named ``auto`` is no mode.

    Attributes:
        name: Name of a tool of the request.

    Example:
        >>> ForcedTool(name="get_weather")
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=TOOL_NAME_PATTERN)


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
        end_user_id: An opaque id of the end user the request is made
            for, such as a hash, never personal data; providers use it
            to detect abuse. Each provider limits its length.
        effort: How much work the model puts into the response; the
            model's own default when left out.
        output_schema: A JSON Schema the response text must follow; the
            text is then JSON. Each provider takes its own subset of
            JSON Schema and answers 400 to the rest; the schema is sent
            as it is, never rewritten.
        tools: Functions the model may call, at least one, with unique
            names.
        tool_choice: Whether the model calls a tool (a ``ToolChoice``) or
            which one (a ``ForcedTool`` naming one of ``tools``); the
            providers' default is ``ToolChoice.AUTO``. Only with
            ``tools``.
        parallel_tool_calls: Whether the model may call several tools in
            one response; both providers allow it by default. Only with
            ``tools``, and not with ``ToolChoice.NONE``.

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
    end_user_id: str | MISSING = Field(default=MISSING, min_length=1)
    effort: Effort | MISSING = MISSING
    output_schema: JSONSchema | MISSING = MISSING
    tools: Annotated[list[Tool], Field(min_length=1)] | MISSING = MISSING
    tool_choice: ToolChoice | ForcedTool | MISSING = MISSING
    parallel_tool_calls: bool | MISSING = MISSING

    @model_validator(mode="after")
    def _check_tools(self) -> Self:
        """Refuse tool settings that no provider takes.

        Returns:
            The request as it is.

        Raises:
            ValueError: ``tool_choice`` or ``parallel_tool_calls`` comes
                without ``tools``; two tools share a name; a
                ``ForcedTool`` names no tool of the request; or
                ``parallel_tool_calls`` comes with ``ToolChoice.NONE``.
        """
        if self.tools is MISSING:
            if self.tool_choice is not MISSING:
                message = "tool_choice needs tools"
                raise ValueError(message)
            if self.parallel_tool_calls is not MISSING:
                message = "parallel_tool_calls needs tools"
                raise ValueError(message)
            return self
        names = [tool.name for tool in self.tools]
        if len(set(names)) != len(names):
            message = f"tool names must be unique, got {names}"
            raise ValueError(message)
        choice = self.tool_choice
        if isinstance(choice, ForcedTool) and choice.name not in names:
            message = f"tool_choice names {choice.name!r}, not a tool"
            raise ValueError(message)
        if (
            choice is ToolChoice.NONE
            and self.parallel_tool_calls is not MISSING
        ):
            message = "parallel_tool_calls means nothing with tool_choice none"
            raise ValueError(message)
        return self
