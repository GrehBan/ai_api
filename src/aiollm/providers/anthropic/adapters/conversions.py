"""Conversions of Anthropic's Messages types into shared types."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import MISSING, BaseModel

from aiollm.adapters.base import BaseConversion
from aiollm.adapters.stop import Stop
from aiollm.enums import IncompleteReason, Role
from aiollm.http.exceptions.response import APIResponseError
from aiollm.providers.anthropic.api import PROVIDER
from aiollm.providers.anthropic.types import (
    Delta,
    Message,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from aiollm.types.requests.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
)
from aiollm.types.requests.generation import (
    InputItem,
    NativeOutput,
    ToolCallTurn,
    ToolResult,
    Turn,
)
from aiollm.types.responses.generation import TokenUsage, ToolCall

FINISHED: Final = frozenset({"end_turn", "stop_sequence", "tool_use"})
"""Stop reasons of a generation that reached its end.

``tool_use`` is one: the model waits for the tool results, as an
OpenAI response with function calls is ``completed``. Any other reason
(``max_tokens``, ``refusal``, ``model_context_window_exceeded``,
``pause_turn``, a reason Anthropic adds later) stopped it early.
"""

INCOMPLETE_REASONS: Final[Mapping[str, IncompleteReason]] = {
    "max_tokens": IncompleteReason.MAX_TOKENS,
    "refusal": IncompleteReason.REFUSAL,
}
"""Shared reason of each early stop reason that has one.

Every other early stop is ``IncompleteReason.OTHER``.
"""

REFUSAL: Final = "refusal"
"""The stop reason of a refusal."""


class InputConversion(
    BaseConversion[str | Sequence[InputItem], list[dict[str, Any]]],
):
    """Converts the shared ``input`` into ``messages``.

    A text becomes one ``user`` turn, and so does a text turn of a list.
    A turn of calls becomes an ``assistant`` message: its Anthropic
    content blocks when it has them (``native``, thinking with it), else
    a ``text`` block and a ``tool_use`` block per call. The tool results
    after it become one ``user`` message of ``tool_result`` blocks, and a
    ``user`` turn right after them joins that message, after the results,
    as Anthropic requires. The adapter checks first that the arguments
    of the calls are JSON objects.
    """

    def __call__(
        self, source: str | Sequence[InputItem], /
    ) -> list[dict[str, Any]]:
        """Convert the input.

        Args:
            source: The text or the conversation of the shared request.

        Returns:
            The messages.
        """
        if isinstance(source, str):
            return [{"role": Role.USER.value, "content": source}]
        messages: list[dict[str, Any]] = []
        results: list[dict[str, Any]] | None = None
        for item in source:
            match item:
                case ToolResult():
                    block = {
                        "type": "tool_result",
                        "tool_use_id": item.call_id,
                        "content": item.output,
                    }
                    if results is None:
                        results = [block]
                        messages.append(
                            {"role": Role.USER.value, "content": results}
                        )
                    else:
                        results.append(block)
                    continue
                case Turn(role=Role.USER) if results is not None:
                    results += _blocks(item.content)
                case Turn(content=str()):
                    messages.append(
                        {"role": item.role.value, "content": item.content}
                    )
                case Turn():
                    messages.append(
                        {
                            "role": item.role.value,
                            "content": _blocks(item.content),
                        }
                    )
                case _:
                    messages.append(
                        {
                            "role": Role.ASSISTANT.value,
                            "content": _assistant_blocks(item),
                        }
                    )
            results = None
        return messages


def _blocks(content: str | Sequence[ContentPart]) -> list[dict[str, Any]]:
    """Convert a turn's content into content blocks.

    Args:
        content: The text, or the parts.

    Returns:
        The blocks: one ``text`` block for a text, else one per part.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [_block(part) for part in content]


def _block(part: ContentPart) -> dict[str, Any]:
    """Convert a part of a turn into a content block.

    Text becomes a ``text`` block, an image an ``image`` block and a
    document a ``document`` block, with a ``base64`` source for bytes
    and a ``url`` source for a URL.

    Args:
        part: The part.

    Returns:
        The block.
    """
    match part:
        case ImagePart() | DocumentPart():
            if isinstance(part.data, bytes):
                source = {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": base64.b64encode(part.data).decode(),
                }
            else:
                source = {"type": "url", "url": part.url}
            return {"type": part.type, "source": source}
        case _:
            return {"type": "text", "text": part.text}


def _assistant_blocks(turn: ToolCallTurn) -> list[dict[str, Any]]:
    """Build the content of a turn of calls.

    Args:
        turn: The turn.

    Returns:
        Its Anthropic content blocks when it has them, else a ``text``
        block and a ``tool_use`` block per call, the arguments parsed.
    """
    native = turn.native
    if native is not MISSING and native.provider is PROVIDER:
        return [dict(block) for block in native.items]
    blocks: list[dict[str, Any]] = []
    if turn.text is not MISSING:
        blocks.append({"type": "text", "text": turn.text})
    blocks += [
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.parse_arguments(),
        }
        for call in turn.tool_calls
    ]
    return blocks


class UsageConversion(BaseConversion[Usage, TokenUsage]):
    """Converts Anthropic's token counts.

    Anthropic counts input read from and written to the cache apart from
    the rest; the shared ``input_tokens`` is their sum, and
    ``total_tokens`` adds the output.
    """

    def __call__(self, source: Usage, /) -> TokenUsage:
        """Convert the counts of a message.

        Args:
            source: Token counts of a message.

        Returns:
            The shared token counts.
        """
        cache_write = source.cache_creation_input_tokens
        cache_read = source.cache_read_input_tokens
        input_tokens = (
            source.input_tokens + (cache_write or 0) + (cache_read or 0)
        )
        details = source.output_tokens_details
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=source.output_tokens,
            total_tokens=input_tokens + source.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=(
                None if details is None else details.thinking_tokens
            ),
        )


class StopConversion(BaseConversion[Message | Delta, Stop]):
    """Tells how a message stopped.

    It takes a ``Message``, or for a stream the ``Delta`` that
    ``StreamState`` collects: both carry ``stop_reason`` and
    ``stop_details``.
    """

    def __call__(self, source: Message | Delta, /) -> Stop:
        """Convert the stop of a finished message.

        Args:
            source: The message, or the collected changes of a stream.

        Returns:
            Nothing set if the message reached its end; else the shared and
            Anthropic's reason, and for a refusal its explanation, else its
            category.

        Raises:
            APIResponseError: There is no stop reason, which Anthropic
                sends on every finished message.
        """
        stop_reason = source.stop_reason
        if stop_reason is None:
            message = "Anthropic message finished without a stop_reason"
            raise APIResponseError(message)
        if stop_reason in FINISHED:
            return Stop()
        details = source.stop_details
        refusal = None
        if stop_reason == REFUSAL and details is not None:
            refusal = details.explanation or details.category
        return Stop(
            INCOMPLETE_REASONS.get(stop_reason, IncompleteReason.OTHER),
            stop_reason,
            refusal,
        )


def message_text(message: Message) -> str | None:
    """Join the text of a message.

    Args:
        message: A message.

    Returns:
        The text of all ``text`` blocks, or ``None`` if there are none.
    """
    texts = [
        block.text for block in message.content if isinstance(block, TextBlock)
    ]
    return "".join(texts) if texts else None


def tool_call(block: ToolUseBlock) -> ToolCall:
    """Turn a ``tool_use`` block into a shared call.

    Args:
        block: The block.

    Returns:
        The call, its ``input`` written as JSON text.
    """
    arguments = json.dumps(block.input, ensure_ascii=False)
    return ToolCall(id=block.id, name=block.name, arguments=arguments)


def message_tool_calls(message: Message) -> list[ToolCall]:
    """Collect the tool calls of a message.

    Args:
        message: A message.

    Returns:
        A shared call per ``tool_use`` block, in content order; empty
        without calls.
    """
    return [
        tool_call(block)
        for block in message.content
        if isinstance(block, ToolUseBlock)
    ]


def message_native(message: Message) -> NativeOutput | None:
    """Keep a message's content as Anthropic's next request takes it.

    Args:
        message: A message.

    Returns:
        Its content blocks with only the fields Anthropic sent (thinking
        with its ``signature``, text, ``tool_use``), which an
        ``assistant`` message takes back; ``None`` without content.
    """
    blocks = [
        block.model_dump(mode="json", by_alias=True, exclude_unset=True)
        for block in message.content
    ]
    return NativeOutput(provider=PROVIDER, items=blocks) if blocks else None


def message_thinking(message: Message) -> str | None:
    """Join the thinking of a message.

    Args:
        message: A message.

    Returns:
        The text of all ``thinking`` blocks, or ``None`` if there are
        none or all are empty (thinking whose display is ``omitted``).
    """
    thinking = "".join(
        block.thinking
        for block in message.content
        if isinstance(block, ThinkingBlock)
    )
    return thinking or None


INPUT_CONVERSION: Final = InputConversion()
"""The default conversion of the shared ``input``."""

USAGE_CONVERSION: Final = UsageConversion()
"""The default conversion of token counts."""

STOP_CONVERSION: Final = StopConversion()
"""The default conversion of a message's stop."""


@dataclass
class StreamState:
    """What a message stream has told so far about the whole message.

    ``message_start`` carries the message's ``Usage``, and every
    ``message_delta`` the stop reason and the cumulative counts, so the
    counts are replaced, never added up. At the end the state gives a
    real ``Usage`` and ``Delta``, so a stream goes through the
    same conversions as a message, and the message's id, model and
    whole text. It also keeps every content block, from its
    ``content_block_start`` and its deltas, to give the message's
    content back as ``native`` (thinking with its ``signature`` among
    it) and each tool call once its block stops.

    Attributes:
        delta: The stop reason and details sent so far.
    """

    delta: Delta = field(default_factory=Delta)
    _message: Message | None = field(default=None, init=False, repr=False)
    _texts: list[str] = field(default_factory=list, init=False, repr=False)
    _thinking: list[str] = field(default_factory=list, init=False, repr=False)
    _calls: dict[int, ToolUseBlock] = field(
        default_factory=dict[int, ToolUseBlock], init=False, repr=False
    )
    _arguments: dict[int, list[str]] = field(
        default_factory=dict[int, list[str]], init=False, repr=False
    )
    _blocks: dict[int, dict[str, Any]] = field(
        default_factory=dict[int, dict[str, Any]], init=False, repr=False
    )
    _pieces: dict[int, dict[str, list[str]]] = field(
        default_factory=dict[int, dict[str, list[str]]],
        init=False,
        repr=False,
    )

    def start(self, event: RawMessageStartEvent) -> None:
        """Take the message and the counts of ``message_start``.

        Args:
            event: The first event of the stream.
        """
        self._message = event.message

    def add_text(self, text: str) -> None:
        """Keep a text delta for the whole text.

        Args:
            text: The new text of a ``text_delta``.
        """
        self._texts.append(text)

    def start_block(self, index: int, block: BaseModel) -> None:
        """Keep a block that a ``content_block_start`` opened.

        Args:
            index: The block's index in the message.
            block: The block as it starts: a ``tool_use`` with an empty
                ``input``, text and thinking without their deltas yet.
        """
        self._blocks[index] = block.model_dump(
            mode="json", by_alias=True, exclude_unset=True
        )
        self._pieces[index] = {}
        if isinstance(block, ToolUseBlock):
            self._calls[index] = block
            self._arguments[index] = []

    def add_piece(self, index: int, key: str, piece: str) -> None:
        """Keep a piece of a block's text, thinking or signature.

        A block whose ``content_block_start`` did not come (some of the
        documented streams leave it out) is opened from the piece: a
        ``text`` block for text, a ``thinking`` block for thinking or a
        signature, so its content still comes back whole.

        Args:
            index: The block's index.
            key: The field the piece belongs to: ``text``, ``thinking``
                or ``signature``.
            piece: The piece.
        """
        if index not in self._pieces:
            kind = "text" if key == "text" else "thinking"
            self._blocks[index] = {"type": kind}
            self._pieces[index] = {}
        self._pieces[index].setdefault(key, []).append(piece)

    def call(self, index: int) -> ToolCall | None:
        """Give the tool call of a block, once its block stops.

        Args:
            index: The block's index.

        Returns:
            The call, or ``None`` if the block is no ``tool_use``.
        """
        if index not in self._calls:
            return None
        return self._call(index)

    def _call(self, index: int) -> ToolCall:
        """Build the call of a ``tool_use`` block.

        Args:
            index: The block's index.

        Returns:
            The call, its arguments the pieces joined, or the block's own
            ``input`` when no piece came.
        """
        block = self._calls[index]
        pieces = self._arguments[index]
        if not pieces:
            return tool_call(block)
        return ToolCall(
            id=block.id, name=block.name, arguments="".join(pieces)
        )

    def native(self) -> NativeOutput | None:
        """Give the streamed content as Anthropic's next request takes it.

        Returns:
            Each block as it started, with its text, thinking and
            signature pieces joined into it and a ``tool_use`` block's
            ``input`` parsed from its pieces; ``None`` without blocks.

        Raises:
            APIResponseError: A ``tool_use`` block's pieces do not join
                into a JSON object.
        """
        if not self._blocks:
            return None
        blocks: list[dict[str, Any]] = []
        for index in sorted(self._blocks):
            block = dict(self._blocks[index])
            for key, pieces in self._pieces[index].items():
                block[key] = block.get(key, "") + "".join(pieces)
            if index in self._calls:
                block["input"] = _input(self._call(index))
            blocks.append(block)
        return NativeOutput(provider=PROVIDER, items=blocks)

    def add_arguments(self, index: int, partial: str) -> str:
        """Keep a piece of a call's arguments.

        Args:
            index: The index of the call's block.
            partial: The ``partial_json`` of an ``input_json_delta``.

        Returns:
            The id of the call the piece belongs to.

        Raises:
            APIResponseError: No ``tool_use`` block opened at the index.
        """
        if index not in self._arguments:
            message = (
                f"Anthropic stream sent input_json_delta for block {index} "
                "without its tool_use"
            )
            raise APIResponseError(message)
        self._arguments[index].append(partial)
        return self._calls[index].id

    def tool_calls(self) -> list[ToolCall]:
        """Give the tool calls streamed so far.

        Returns:
            A shared call per ``tool_use`` block, in block order, its
            arguments the pieces joined, or the block's own ``input``
            when no piece came.
        """
        return [self._call(index) for index in sorted(self._calls)]

    def add_thinking(self, thinking: str) -> None:
        """Keep a thinking delta for the whole reasoning.

        Args:
            thinking: The new text of a ``thinking_delta``.
        """
        self._thinking.append(thinking)

    def thinking(self) -> str | None:
        """Join the thinking deltas.

        Returns:
            The thinking received, or ``None`` if there was none.
        """
        return "".join(self._thinking) or None

    def update(self, event: RawMessageDeltaEvent) -> None:
        """Take the stop reason and counts of a ``message_delta``.

        Args:
            event: A ``message_delta`` event.

        Raises:
            APIResponseError: No ``message_start`` came before it to
                take its counts into.
        """
        changes = _sent(event.delta)
        if changes:
            self.delta = self.delta.model_copy(update=changes)
        if self._message is None:
            message = (
                "Anthropic stream sent message_delta before message_start"
            )
            raise APIResponseError(message)
        counts = self._message.usage.model_copy(update=_sent(event.usage))
        self._message = self._message.model_copy(update={"usage": counts})

    def usage(self) -> Usage:
        """Give the counts of the whole message.

        Returns:
            The counts of ``message_start`` with the last cumulative
            counts of ``message_delta``.

        Raises:
            APIResponseError: The stream had no ``message_start``.
        """
        return self.message().usage

    def message(self) -> Message:
        """Give the message of ``message_start``.

        Returns:
            The message as the stream started it, for its id and model,
            with the counts of the last ``message_delta``.

        Raises:
            APIResponseError: The stream had no ``message_start``.
        """
        if self._message is None:
            message = "Anthropic stream finished without message_start"
            raise APIResponseError(message)
        return self._message

    def text(self) -> str | None:
        """Give the whole text streamed so far.

        Returns:
            The text deltas joined, or ``None`` if none came, as
            ``message_text`` gives for a message without text.
        """
        return "".join(self._texts) if self._texts else None


def _input(call: ToolCall) -> dict[str, Any]:
    """Parse a streamed call's arguments into ``tool_use.input``.

    Args:
        call: The call.

    Returns:
        Its arguments as an object.

    Raises:
        APIResponseError: The arguments are not a JSON object.
    """
    try:
        return call.parse_arguments()
    except ValueError as error:
        raise APIResponseError(str(error)) from error


def _sent(model: BaseModel) -> dict[str, object]:
    """Take the fields of a model that have a value.

    The values stay as they are, nested models included, so they can
    update another model with ``model_copy``.

    Args:
        model: A partial update, such as a ``message_delta``'s counts.

    Returns:
        Each field that is not ``None``, with its value.
    """
    return {
        name: value
        for name in type(model).model_fields
        if (value := getattr(model, name)) is not None
    }
