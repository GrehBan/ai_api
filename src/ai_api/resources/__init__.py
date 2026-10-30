"""Resources and use cases that adapt provider methods to shared types."""

from __future__ import annotations

from ai_api.resources.base import (
    BaseResource,
    BaseSessionAccess,
)
from ai_api.resources.generation import (
    GenerationResource,
)
from ai_api.resources.protocols import (
    ICompositeUseCase,
    IGenerationResource,
    IResource,
    ISessionAccess,
    IStreamUseCase,
    IUseCase,
)
from ai_api.resources.usecases import (
    AdapterCompositeUseCase,
    AdapterStreamUseCase,
    AdapterUseCase,
    BaseCompositeUseCase,
    BaseStreamUseCase,
    BaseUseCase,
)

__all__ = [
    "AdapterCompositeUseCase",
    "AdapterStreamUseCase",
    "AdapterUseCase",
    "BaseCompositeUseCase",
    "BaseResource",
    "BaseSessionAccess",
    "BaseStreamUseCase",
    "BaseUseCase",
    "GenerationResource",
    "ICompositeUseCase",
    "IGenerationResource",
    "IResource",
    "ISessionAccess",
    "IStreamUseCase",
    "IUseCase",
]
