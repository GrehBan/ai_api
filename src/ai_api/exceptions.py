"""Root of the library's exception hierarchy."""

from typing import Any, ClassVar


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
