"""Root of the library's exception hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ai_api.enums import Provider


def _rebuild(cls: type[AIAPIError], args: tuple[Any, ...]) -> AIAPIError:
    """Create an error without running its ``__init__``.

    Args:
        cls: The error class.
        args: The ``args`` of the original error.

    Returns:
        The error; its attributes are restored by unpickling.
    """
    return cls.__new__(cls, *args)


class AIAPIError(Exception):
    """Base class for every error raised by the library.

    The final message is built from the class-level ``default_message``
    (formatted with the keyword arguments) followed by the per-instance
    detail, so callers get a stable prefix plus the API's own explanation.

    Attributes:
        default_message: Message template shared by all instances of the
            class. ``None`` means the instance message is used alone.
        message: The composed message that is also passed to
            ``Exception``.
    """

    default_message: ClassVar[str | None] = None

    def __init__(
        self,
        message: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Compose the error message.

        Args:
            message: Detail specific to this occurrence, appended as
                ``"detail: <message>"``.
            *args: Extra positional arguments forwarded to ``Exception``.
            **kwargs: Values substituted into ``default_message``.
        """
        parts: list[str] = []

        if self.default_message:
            parts.append(self.default_message.format_map(kwargs))

        if message:
            parts.append(f"detail: {message}")

        self.message = " ".join(parts)
        super().__init__(self.message, *args)

    def __reduce__(self) -> tuple[Any, ...]:
        """Support ``pickle`` and ``copy``.

        Subclasses take other constructor arguments, and ``__init__``
        composes the message, so a copy is rebuilt from ``args`` and the
        attributes instead of calling ``__init__`` again.

        Returns:
            The rebuild function, its arguments and the attributes.
        """
        return (_rebuild, (type(self), self.args), self.__dict__)


class RequestValidationError(AIAPIError, ValueError):
    """A shared request cannot be sent to a provider as it is.

    Raised by an adapter before anything is sent, when a field of a
    shared request breaks a rule of the provider, such as OpenAI's
    minimum ``max_output_tokens`` or Anthropic's only metadata key. A
    ``ValueError``, like the validation error of the request itself.

    Attributes:
        provider: Provider that cannot take the request.
        field: Name of the shared request field at fault.
    """

    default_message: ClassVar[str | None] = (
        "{provider} cannot take the request field {field!r}."
    )

    def __init__(
        self,
        message: str,
        *,
        provider: Provider,
        field: str,
    ) -> None:
        """Store the provider and the field.

        Args:
            message: What is wrong with the field. A pydantic error
                behind it goes in ``__cause__``.
            provider: Provider that cannot take the request.
            field: Name of the shared request field at fault.
        """
        super().__init__(message, provider=provider.value, field=field)
        self.provider = provider
        self.field = field
