"""Base class for API data models."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class BaseAPIType(BaseModel):
    """Base class for request and response payload models.

    Unknown fields are ignored, so new fields added by the provider do
    not break validation. Fields can be populated by name or alias. Give
    a wire name with ``validation_alias`` and ``serialization_alias``
    rather than ``alias``: then the constructor takes the field name in
    both mypy and pyright (with ``alias`` they disagree).

    Attributes:
        __upstream_schema__: Name of the definition in the provider's
            upstream JSON schema that the model mirrors, several names
            for a model that covers several definitions, or ``()`` for a
            model without one. ``None`` (the default) means not set; a
            provider's contract test requires it on every provider model
            and checks that the model accepts what that schema allows.
    """

    __upstream_schema__: ClassVar[str | tuple[str, ...] | None] = None

    model_config = ConfigDict(
        extra="ignore",
        validate_by_name=True,
        validate_by_alias=True,
        arbitrary_types_allowed=True,
    )
