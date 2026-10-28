from typing import Any, ClassVar


class AIAPIError(Exception):
    default_message: ClassVar[str | None] = None

    def __init__(
        self,
        message: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        parts: list[str] = []

        if self.default_message:
            parts.append(self.default_message.format_map(kwargs))

        if message:
            parts.append(f"detail: {message}")

        super().__init__(" ".join(parts), *args)