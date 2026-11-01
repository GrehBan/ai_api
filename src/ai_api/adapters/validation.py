"""Checks adapters run on a shared request before building a method."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ai_api.enums import Role
from ai_api.exceptions import RequestValidationError

if TYPE_CHECKING:
    from ai_api.enums import Provider
    from ai_api.types.requests.generation import GenerationRequest


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


def check_last_turn(request: GenerationRequest, provider: Provider) -> None:
    """Check that a conversation ends with a ``user`` turn.

    Anthropic rejects a last ``assistant`` turn (prefill) on current
    models, and OpenAI gives it a meaning of its own, so no adapter
    sends one.

    Args:
        request: The shared request.
        provider: Provider the request is for.

    Raises:
        RequestValidationError: ``input`` is a list of turns whose last
            turn is not a ``user`` one.
    """
    if isinstance(request.input, str):
        return
    last = request.input[-1]
    if last.role is not Role.USER:
        message = (
            f"the last turn must be a {Role.USER.value!r} turn, "
            f"got {last.role.value!r}"
        )
        raise RequestValidationError(message, provider=provider, field="input")
