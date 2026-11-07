"""Shared request for text generation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated, Final, Literal, Self, overload

from pydantic import (
    MISSING,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    model_validator,
)
from typing_extensions import TypedDict

from aiollm.enums import Effort, Provider, Role, ToolChoice
from aiollm.types.base import BaseAPIType
from aiollm.types.requests.base import BaseRequest
from aiollm.types.requests.content import ContentPart


class Turn(BaseAPIType):
    """One turn of a conversation: who wrote it and what.

    Immutable, and unknown fields are rejected. System prompts are not
    turns: they go in ``GenerationRequest.instructions``.

    Attributes:
        type: Always ``"text"``; tells the turn apart from the other
            items of a conversation (a turn of parts has it too).
        role: Who wrote the turn.
        content: Text of the turn, or its parts in order: text, images
            and documents, at least one. Only a ``user`` turn takes
            parts; an ``assistant`` turn is text.

    Example:
        >>> Turn.user("Hi")
        Turn(type='text', role=<Role.USER: 'user'>, content='Hi')
        >>> Turn.user([TextPart(text="What is it?"), await
        ...            ImagePart.load("cat.png")])
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    type: Literal["text"] = "text"
    role: Role
    content: str | Annotated[list[ContentPart], Field(min_length=1)]

    @model_validator(mode="after")
    def _parts_from_the_user(self) -> Self:
        """Check that only a ``user`` turn has parts.

        Returns:
            The turn.

        Raises:
            ValueError: A turn of another role has parts.
        """
        if not isinstance(self.content, str) and self.role is not Role.USER:
            message = (
                f"only a {Role.USER.value!r} turn takes parts; "
                f"give a {self.role.value!r} turn its text"
            )
            raise ValueError(message)
        return self

    @classmethod
    def user(cls, content: str | Sequence[ContentPart]) -> Self:
        """Create a turn of the end user.

        Args:
            content: Text of the turn, or its parts.

        Returns:
            The turn.
        """
        if not isinstance(content, str):
            content = list(content)
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

    @classmethod
    def from_model(
        cls,
        name: str,
        model: type[BaseModel],
        *,
        description: str | MISSING = MISSING,
        strict: bool | MISSING = MISSING,
    ) -> Self:
        """Create a tool whose arguments are a Pydantic model.

        ``parameters`` is the model's JSON Schema
        (``model.model_json_schema()``), sent as it is, and
        ``ToolCall.parse_arguments(model)`` reads the arguments back
        into the model. For ``strict=True`` the model forbids extra
        fields (``extra="forbid"``) and every field is required, or the
        providers answer 400.

        Args:
            name: Name the model calls it by.
            model: The model of the arguments.
            description: What the tool does and when to use it.
            strict: Whether the arguments must follow the schema
                exactly.

        Returns:
            The tool.

        Example:
            >>> class City(BaseModel):
            ...     model_config = ConfigDict(extra="forbid")
            ...     city: str
            >>> tool = Tool.from_model("get_weather", City, strict=True)
        """
        return cls(
            name=name,
            description=description,
            parameters=model.model_json_schema(),
            strict=strict,
        )


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


class ToolCall(BaseAPIType):
    """A call of one of the request's tools.

    Immutable, and unknown fields are rejected. The arguments are the
    JSON text the model wrote, as OpenAI sends them and as both stream
    them; Anthropic's object is written as JSON. Without ``strict`` the
    text may be invalid JSON, so it is parsed only on
    ``parse_arguments()``.

    Attributes:
        id: The call's id, which its result refers to (OpenAI's
            ``call_id``, Anthropic's ``tool_use.id``).
        name: The tool called.
        arguments: The arguments, a JSON object as text.

    Example:
        >>> for call in response.tool_calls:
        ...     result = run(call.name, call.parse_arguments())
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str
    arguments: str

    @overload
    def parse_arguments(self) -> dict[str, JSONValue]: ...

    @overload
    def parse_arguments[ModelT: BaseModel](
        self, model: type[ModelT]
    ) -> ModelT: ...

    def parse_arguments(
        self, model: type[BaseModel] | None = None
    ) -> dict[str, JSONValue] | BaseModel:
        """Parse the arguments, and validate them with a model if given.

        Without a model only the JSON is checked, not the tool's
        ``parameters``: a model may write other arguments unless the
        tool is ``strict``. With one (the model of
        ``Tool.from_model``), the arguments are validated by it.

        Args:
            model: The Pydantic model the arguments must follow.

        Returns:
            The arguments as a JSON object, or the model made from them.

        Raises:
            ValueError: The arguments are not a JSON object, or do not
                follow the model (a ``pydantic.ValidationError``, which
                is a ``ValueError``).

        Example:
            >>> call.parse_arguments(City).city
            'Paris'
        """
        value = self._parse_object()
        if model is None:
            return value
        # From the text, so JSON's rules apply (a date from its string).
        return model.model_validate_json(self.arguments)

    def _parse_object(self) -> dict[str, JSONValue]:
        """Parse the arguments as a JSON object.

        Returns:
            The arguments.

        Raises:
            ValueError: The arguments are not a JSON object.
        """
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError as error:
            message = f"arguments of tool call {self.id} are not JSON"
            raise ValueError(message) from error
        if not isinstance(value, dict):
            message = f"arguments of tool call {self.id} are not an object"
            raise ValueError(message)
        return value


class NativeOutput(BaseAPIType):
    """A response's output as its provider's next request takes it.

    The provider's own record of the turn: OpenAI's output items, its
    reasoning items with ``encrypted_content`` among them, or Anthropic's
    content blocks, thinking with its ``signature`` among them, each as
    the response sent it. The provider's adapter sends it in place of the
    shared fields of the turn, so the model sees its reasoning again;
    another provider's adapter sends the shared fields only, since it
    cannot read it.

    Attributes:
        provider: The provider whose output it is.
        items: The items, in output order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    items: Annotated[list[dict[str, JSONValue]], Field(min_length=1)]


class ToolCallTurn(BaseAPIType):
    """A turn of the model that calls tools.

    Made by ``GenerationResponse.to_turn()`` from a response with calls;
    the next items of the conversation are a ``ToolResult`` for each of
    its calls.

    Attributes:
        type: Always ``"tool_calls"``.
        text: Text the model wrote with the calls, if any.
        tool_calls: The calls; at least one.
        native: The response's own output, which its provider gets back
            in place of ``text`` and ``tool_calls``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_calls"] = "tool_calls"
    text: str | MISSING = MISSING
    tool_calls: Annotated[list[ToolCall], Field(min_length=1)]
    native: NativeOutput | MISSING = MISSING


class ToolResult(BaseAPIType):
    """What a tool returned for one call.

    Immutable, and unknown fields are rejected. Describe a failure in
    ``output``: OpenAI has no error flag, and both read the text.

    Attributes:
        type: Always ``"tool_result"``.
        call_id: ``ToolCall.id`` of the call it answers.
        output: The result as text, such as JSON.

    Example:
        >>> ToolResult(call_id=call.id, output='{"celsius": 15}')
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_result"] = "tool_result"
    call_id: str = Field(min_length=1)
    output: str


def _item_tag(value: object) -> object:
    """Tell which item of a conversation a value is.

    A dict without ``type`` is a text turn, as the conversations written
    before the other items were.

    Args:
        value: An item, or a dict of one.

    Returns:
        Its ``type``.
    """
    if isinstance(value, dict):
        return value.get("type", "text")
    return getattr(value, "type", None)


type InputItem = Annotated[
    Annotated[Turn, Tag("text")]
    | Annotated[ToolCallTurn, Tag("tool_calls")]
    | Annotated[ToolResult, Tag("tool_result")],
    Discriminator(_item_tag),
]
"""An item of a conversation: a text turn, a turn of calls or a result."""


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
            conversation so far, a list of at least one item: text turns,
            turns of tool calls and tool results; an empty text or list
            is refused before anything is sent. The conversation ends
            with a ``user`` turn or a tool result, and the results of a
            turn of calls follow it, one per call; the adapter checks it
            and raises ``RequestValidationError`` otherwise.
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
        | Annotated[Sequence[InputItem], Field(min_length=1)]
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
