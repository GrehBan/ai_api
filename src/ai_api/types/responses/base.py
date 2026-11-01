"""Base class of shared responses."""

from __future__ import annotations

from pydantic import ConfigDict

from ai_api.types.base import BaseAPIType


class BaseResponse(BaseAPIType):
    """Base class of responses in shared types.

    Its ``raw`` holds the provider's own result, which may be any class a
    custom method returns, so any type is allowed here, and only here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
