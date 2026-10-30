"""Base class for API data models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseAPIType(BaseModel):
    """Base class for request and response payload models.

    Unknown fields are ignored, so new fields added by the provider do
    not break validation. Fields can be populated by name or alias. Give
    a wire name with ``validation_alias`` and ``serialization_alias``
    rather than ``alias``: then the constructor takes the field name in
    both mypy and pyright (with ``alias`` they disagree).
    """

    model_config = ConfigDict(
        extra="ignore",
        validate_by_name=True,
        validate_by_alias=True,
        arbitrary_types_allowed=True,
    )
