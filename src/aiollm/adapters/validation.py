"""Checks adapters run on a shared request before building a method."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from pydantic import MISSING, ValidationError
from typing_extensions import is_typeddict

from aiollm.enums import Role
from aiollm.exceptions import RequestValidationError
from aiollm.types.requests.generation import ToolCallTurn, ToolResult, Turn

if TYPE_CHECKING:
    from aiollm.enums import Provider
    from aiollm.types.requests.generation import (
        GenerationRequest,
        InputItem,
    )


def checked_options(
    options: object,
    options_type: type,
    provider: Provider,
) -> dict[str, Any]:
    """Check the keys of a call's provider options at run time.

    A type checker sees them only in typed code; a ``dict`` built at run
    time reaches the adapter unchecked. The values are checked later by
    the method the options go into.

    Args:
        options: The ``provider_options`` of a call.
        options_type: The provider's ``TypedDict`` of its options.
        provider: The provider, for the error.

    Returns:
        A copy of the options as a plain ``dict``.

    Raises:
        TypeError: ``options_type`` is not a ``TypedDict``.
        RequestValidationError: ``options`` is not a mapping, or has a
            key that is not one of the provider's options; ``field`` is
            the first such key.
    """
    # typing_extensions' check: typing's does not know a TypedDict made
    # by typing_extensions (a closed one, such as NoProviderOptions).
    if not is_typeddict(options_type):
        message = f"{options_type!r} is not a TypedDict"
        raise TypeError(message)

    if not isinstance(options, Mapping):
        message = (
            f"provider_options must be a mapping, not {type(options).__name__}"
        )
        raise RequestValidationError(
            message, provider=provider, field="provider_options"
        )

    # A TypedDict class keeps its keys here; type checkers see a plain
    # type, so they are read with getattr.
    allowed = frozenset(getattr(options_type, "__required_keys__", ())) | (
        frozenset(getattr(options_type, "__optional_keys__", ()))
    )
    unknown = sorted(str(key) for key in options if key not in allowed)

    if unknown:
        message = (
            f"Not options of {provider.value}: {', '.join(unknown)}; "
            "the shared fields are set in the request"
        )
        raise RequestValidationError(
            message, provider=provider, field=unknown[0]
        )

    return dict(options)


def merge_options(
    fields: Mapping[str, Any],
    options: Mapping[str, Any],
    provider: Provider,
) -> dict[str, Any]:
    """Merge a call's provider options into the fields of a method.

    A shared field may fill part of an object, such as OpenAI's
    ``reasoning.effort``, while the options fill the rest of it
    (``reasoning.summary``): such objects are merged one level deep.
    A key both sides set is refused, so the two never disagree.

    Args:
        fields: The method fields the shared request gives; ``MISSING``
            for a field it leaves out.
        options: The checked provider options.
        provider: The provider, for the error.

    Returns:
        The merged fields.

    Raises:
        RequestValidationError: The options set a key of an object that
            a shared field sets too; ``field`` is its dotted path.
    """
    merged = dict(fields)

    for key, value in options.items():
        given = merged.get(key, MISSING)

        if isinstance(given, Mapping) and isinstance(value, Mapping):
            if clash := sorted(set(given) & set(value)):
                path = f"{key}.{clash[0]}"
                message = (
                    f"{path} is set by a shared field of the request; "
                    "set it there, not in the provider options"
                )
                raise RequestValidationError(
                    message, provider=provider, field=path
                )
            merged[key] = {**given, **value}
        else:
            merged[key] = value

    return merged


@contextmanager
def request_errors(
    provider: Provider,
    fields: Mapping[str, str] | None = None,
) -> Generator[None]:
    """Report a method the provider's limits reject as the request's fault.

    A provider method checks the provider's limits when it is built, and
    raises pydantic's ``ValidationError``. The caller gave a shared
    request, so the error speaks of the shared fields: ``field`` is the
    shared name of the first failing field, and the message lists every
    error, each under its shared name.

    Args:
        provider: Provider of the method built inside.
        fields: Shared field name of each method field whose name
            differs, such as ``{"max_tokens": "max_output_tokens"}``.

    Yields:
        Nothing; the method is built in the ``with`` block.

    Raises:
        RequestValidationError: Building the method failed validation;
            the ``ValidationError`` is its ``__cause__``.

    Example:
        >>> with request_errors(Provider.OPENAI):
        ...     method = CreateResponse(**fields)
    """
    names = fields or {}

    def shared(location: tuple[int | str, ...]) -> str:
        """Write an error location with the shared name of its field.

        Args:
            location: The location pydantic gives, field first.

        Returns:
            The dotted location, such as ``max_output_tokens`` or
            ``input.0.content``; empty for an error of the whole method.
        """
        parts = [str(part) for part in location]
        if not parts:
            return ""
        parts[0] = names.get(parts[0], parts[0])
        return ".".join(parts)

    def line(location: tuple[int | str, ...], text: str) -> str:
        """Write one error, after its location when it has one.

        Args:
            location: The location pydantic gives.
            text: The error message.

        Returns:
            ``"<location>: <message>"``, or the message alone.
        """
        where = shared(location)
        return f"{where}: {text}" if where else text

    try:
        yield
    except ValidationError as error:
        details = error.errors()
        field = shared(details[0]["loc"][:1])
        message = "; ".join(
            line(detail["loc"], detail["msg"]) for detail in details
        )
        raise RequestValidationError(
            message, provider=provider, field=field
        ) from error


def check_conversation(request: GenerationRequest, provider: Provider) -> None:
    """Check the order of a conversation that every provider requires.

    Anthropic rejects a last ``assistant`` turn (prefill) on current
    models, and OpenAI gives it a meaning of its own, so a conversation
    ends with a ``user`` turn or a tool result. Both answer 400 to a call
    without its result and to a result without its call (seen in a live
    check, 2026-11-06), and Anthropic to results that do not follow
    their call at once, so the results of a turn of calls come right
    after it, one per call.

    Args:
        request: The shared request.
        provider: Provider the request is for.

    Raises:
        RequestValidationError: ``input`` breaks one of these rules.
    """
    if isinstance(request.input, str):
        return

    problem = _order_problem(request.input)
    last = request.input[-1]
    if (
        problem is None
        and isinstance(last, Turn)
        and last.role is not Role.USER
    ):
        problem = (
            f"the last turn must be a {Role.USER.value!r} turn or a tool "
            f"result, got {last.role.value!r}"
        )
    if problem is not None:
        raise RequestValidationError(problem, provider=provider, field="input")


def _order_problem(items: Sequence[InputItem]) -> str | None:
    """Find the first item out of order between calls and results.

    Args:
        items: The conversation.

    Returns:
        What is wrong, or ``None`` if every turn of calls is followed at
        once by one result per call and no result stands elsewhere.
    """
    open_calls: set[str] = set()
    for index, item in enumerate(items):
        if isinstance(item, ToolResult):
            if item.call_id not in open_calls:
                return (
                    f"item {index} answers tool call {item.call_id!r}, "
                    "which the turn of calls before it did not make"
                )
            open_calls.discard(item.call_id)
            continue
        if open_calls:
            return (
                f"item {index} comes before the results of tool calls "
                f"{sorted(open_calls)}"
            )
        if isinstance(item, ToolCallTurn):
            ids = [call.id for call in item.tool_calls]
            if len(set(ids)) != len(ids):
                return f"item {index} has tool calls that share an id"
            open_calls = set(ids)
    if open_calls:
        return f"tool calls {sorted(open_calls)} have no result"
    return None
