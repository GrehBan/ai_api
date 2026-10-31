"""How a generation stopped, as the shared types tell it."""

from __future__ import annotations

from dataclasses import dataclass

from ai_api.enums import IncompleteReason


@dataclass(frozen=True, slots=True)
class Stop:
    """How a generation stopped, in shared terms.

    What a provider's stop conversion returns: the three stop fields of
    ``GenerationResponse`` and ``CompletedEvent``.

    Attributes:
        incomplete_reason: Shared reason of an early stop; ``None`` if
            the generation reached its end.
        raw_incomplete_reason: The provider's own reason, if it gives one.
        refusal: What the provider says about a refusal, if anything.
    """

    incomplete_reason: IncompleteReason | None = None
    raw_incomplete_reason: str | None = None
    refusal: str | None = None
