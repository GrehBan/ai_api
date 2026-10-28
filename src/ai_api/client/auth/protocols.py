from typing import Protocol


class IAuth(Protocol):
    def encode(self) -> str:
        ...
