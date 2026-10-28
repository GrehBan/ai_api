from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BearerAuth:
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("Bearer token cannot be empty")

        if any(char.isspace() for char in self.token):
            raise ValueError("Bearer token cannot contain whitespace")

    def encode(self) -> str:
        return f"Bearer {self.token}"