"""Resources and use cases that adapt provider methods to shared types."""

from __future__ import annotations

from ai_api.resources.base import (
    BaseResource,
    BaseSessionAccess,
)
from ai_api.resources.protocols import (
    ICompositeUseCase,
    IResource,
    ISessionAccess,
    IStreamUseCase,
    IUseCase,
)
from ai_api.resources.usecases import (
    BaseCompositeUseCase,
    BaseStreamUseCase,
    BaseUseCase,
)

__all__ = [
    "BaseCompositeUseCase",
    "BaseResource",
    "BaseSessionAccess",
    "BaseStreamUseCase",
    "BaseUseCase",
    "ICompositeUseCase",
    "IResource",
    "ISessionAccess",
    "IStreamUseCase",
    "IUseCase",
]
