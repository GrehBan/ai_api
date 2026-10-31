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
    request, so the error names the shared field instead.

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
    try:
        yield
    except ValidationError as error:
        details = error.errors()
        name = "".join(str(part) for part in details[0]["loc"][:1])
        field = (fields or {}).get(name, name)
        message = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in details
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
