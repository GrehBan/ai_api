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
from ai_api.response.parse.sse import (
    ServerSentEvent,
)
from ai_api.response.stream import (
    Stream,
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
    "BaseSession",
    "BaseStreamMethod",
    "BearerAuth",
    "BillingError",
    "BodyFormat",
    "BytesFile",
    "ConflictError",
    "GatewayTimeoutError",
    "GenerationEventType",
    "Headers",
    "IOFile",
    "IncompleteReason",
    "InputFile",
    "InternalServerError",
    "NotFoundError",
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
    "UnprocessableEntityError",
    "__version__",
]
