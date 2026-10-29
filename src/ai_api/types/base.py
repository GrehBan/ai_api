"""Base class for API data models."""

from pydantic import BaseModel, ConfigDict


class BaseAPIType(BaseModel):
    """Base class for request and response payload models.

    Unknown fields are ignored, so new fields added by the provider do
    not break validation. Fields can be populated by name or alias.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )
