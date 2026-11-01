"""Async-first toolkit for building typed HTTP API clients.

The names most clients need are re-exported here; everything else is
imported from its layer package (``ai_api.http``, ``ai_api.methods``, ...).
Transport code, such as ``ai_api.session.aiohttp``, and provider
packages, such as ``ai_api.providers.openai``, are imported on their own,
so importing ``ai_api`` does not import a transport.
"""

from __future__ import annotations

from ai_api.__meta__ import (
    __version__,
)
from ai_api.client.base import (
    BaseClient,
    BaseProviderClient,
)
from ai_api.enums import (
    BodyFormat,
    GenerationEventType,
    IncompleteReason,
    Provider,
    ResponseFormat,
    Role,
)
from ai_api.exceptions import (
    AIAPIError,
    RequestValidationError,
)
from ai_api.http.api import (
    BaseAPI,
)
from ai_api.http.auth import (
    APIKeyAuth,
    BearerAuth,
)
from ai_api.http.exceptions.resolver import (
    BaseErrorResolver,
)
from ai_api.http.exceptions.response import (
    APIResponseError,
)
from ai_api.http.exceptions.status import (
    APIStatusError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    BillingError,
    ConflictError,
    ContentTooLargeError,
    GatewayTimeoutError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    RequestTimeoutError,
    ServiceUnavailableError,
    UnprocessableEntityError,
)
from ai_api.http.exceptions.transport import (
    APIConnectionError,
    APIRedirectError,
    APIRequestError,
    APIStreamInterruptedError,
    APITimeoutError,
    APITLSError,
    APITransportError,
)
from ai_api.http.headers import (
    Headers,
)
from ai_api.methods.base import (
    BaseAPIMethod,
    BaseStreamMethod,
)
from ai_api.resources.protocols import (
    IGenerationClient,
    IGenerationResource,
)
from ai_api.resources.stream import (
    GenerationStream,
)
from ai_api.response.parse.sse import (
    ServerSentEvent,
)
from ai_api.response.stream import (
    Stream,
)
from ai_api.serialization.protocols import (
    IValueFormatter,
)
from ai_api.serialization.values import (
    BaseValueFormatter,
)
from ai_api.session.base import (
    BaseSession,
)
from ai_api.types.base import (
    BaseAPIType,
)
from ai_api.types.files import (
    BytesFile,
    InputFile,
    IOFile,
    PathFile,
    StreamFile,
)
from ai_api.types.requests.generation import (
    GenerationRequest,
    Turn,
)
from ai_api.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)
from ai_api.types.responses.generation import (
    GenerationResponse,
    TokenUsage,
)

__all__ = [
    "AIAPIError",
    "APIConnectionError",
    "APIKeyAuth",
    "APIRedirectError",
    "APIRequestError",
    "APIResponseError",
    "APIStatusError",
    "APIStreamInterruptedError",
    "APITLSError",
    "APITimeoutError",
    "APITransportError",
    "AuthenticationError",
    "BadGatewayError",
    "BadRequestError",
    "BaseAPI",
    "BaseAPIMethod",
    "BaseAPIType",
    "BaseClient",
    "BaseErrorResolver",
    "BaseProviderClient",
    "BaseSession",
    "BaseStreamMethod",
    "BaseValueFormatter",
    "BearerAuth",
    "BillingError",
    "BodyFormat",
    "BytesFile",
    "CompletedEvent",
    "ConflictError",
    "ContentTooLargeError",
    "GatewayTimeoutError",
    "GenerationEvent",
    "GenerationEventType",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationStream",
    "Headers",
    "IGenerationClient",
    "IGenerationResource",
    "IOFile",
    "IValueFormatter",
    "IncompleteReason",
    "InputFile",
    "InternalServerError",
    "NotFoundError",
    "OtherEvent",
    "PathFile",
    "PermissionDeniedError",
    "Provider",
    "QuotaExceededError",
    "RateLimitError",
    "RequestTimeoutError",
    "RequestValidationError",
    "ResponseFormat",
    "Role",
    "ServerSentEvent",
    "ServiceUnavailableError",
    "Stream",
    "StreamFile",
    "TextDeltaEvent",
    "TokenUsage",
    "Turn",
    "UnprocessableEntityError",
    "__version__",
]
