"""Identifiers shared by all layers: the supported API providers."""

from enum import StrEnum, unique


@unique
class Provider(StrEnum):
    """API provider an error or integration belongs to."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"
