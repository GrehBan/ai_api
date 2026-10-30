"""Base class of shared requests."""

from __future__ import annotations

from pydantic import ConfigDict

from ai_api.types.base import BaseAPIType


class BaseRequest(BaseAPIType):
    """Base class of requests in shared types.

    Unlike responses, requests reject unknown fields, so a typo fails
    validation instead of being dropped.
    """

    model_config = ConfigDict(extra="forbid")
