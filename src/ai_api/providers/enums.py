from enum import StrEnum, unique


@unique
class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"