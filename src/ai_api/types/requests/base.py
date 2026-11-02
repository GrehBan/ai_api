"""Base class of shared requests."""

from __future__ import annotations

from pydantic import ConfigDict

from ai_api.types.base import BaseAPIType


class BaseRequest(BaseAPIType):
    """Base class of requests in shared types.

    Unlike responses, requests reject unknown fields, so a typo fails
    validation instead of being dropped, and validate assignments, so a
    value set after construction (``None`` where a field takes none,
    say) fails as it would in the constructor. ``model_copy(update=...)``
    validates nothing, as everywhere in Pydantic.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
