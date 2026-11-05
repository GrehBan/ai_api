"""Async-first toolkit for building typed HTTP API clients.

The names most clients need are re-exported here; everything else is
imported from its layer package (``aiollm.http``, ``aiollm.methods``, ...).
Transport code, such as ``aiollm.session.aiohttp``, and provider
packages, such as ``aiollm.providers.openai``, are imported on their own,
so importing ``aiollm`` does not import a transport.
"""

from __future__ import annotations

import logging

from aiollm.__meta__ import (
    __version__,
)
from aiollm.client.base import (
    BaseClient,
    BaseProviderClient,
)
from aiollm.client.binding import (
    bind_resource,
)
from aiollm.enums import (
    BodyFormat,
    GenerationEventType,
    IncompleteReason,
    Provider,
    ResponseFormat,
    Role,
)
from aiollm.exceptions import (
    AIAPIError,
    RequestValidationError,
    SessionMismatchWarning,
)
from aiollm.http.api import (
    BaseAPI,
)
from aiollm.http.auth import (
    APIKeyAuth,
    BearerAuth,
)
from aiollm.http.exceptions.resolver import (
    BaseErrorResolver,
)
from aiollm.http.exceptions.response import (
    APIResponseError,
)
from aiollm.http.exceptions.status import (
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
from aiollm.http.exceptions.transport import (
    APIConnectionError,
    APIRedirectError,
    APIRequestError,
    APIStreamInterruptedError,
    APITimeoutError,
    APITLSError,
    APITransportError,
)
from aiollm.http.headers import (
    Headers,
)
from aiollm.methods.base import (
    BaseAPIMethod,
    BaseStreamMethod,
)
from aiollm.resources.protocols import (
    IGenerationClient,
    IGenerationResource,
)
from aiollm.resources.stream import (
    GenerationStream,
)
from aiollm.response.parse.sse import (
    ServerSentEvent,
)
from aiollm.response.stream import (
    Stream,
)
from aiollm.serialization.protocols import (
    IValueFormatter,
)
from aiollm.serialization.values import (
    BaseValueFormatter,
)
from aiollm.session.base import (
    BaseSession,
)
from aiollm.session.retry.policy import (
    FullJitterBackoff,
    RetryPolicy,
)
from aiollm.session.retry.state import (
    RetryState,
)
from aiollm.types.base import (
    BaseAPIType,
    BaseAPITypeExtra,
)
from aiollm.types.files import (
    BytesFile,
    ChunkedFile,
    InputFile,
    IOFile,
    PathFile,
    StreamFile,
)
from aiollm.types.requests.generation import (
    GenerationRequest,
    Turn,
)
from aiollm.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    TextDeltaEvent,
)
from aiollm.types.responses.generation import (
    GenerationResponse,
    TokenUsage,
)

# The library configures no logging: its records reach the handlers
# the application sets up, and without them nothing is printed (as
# the logging HOWTO advises libraries).
logging.getLogger(__name__).addHandler(logging.NullHandler())

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
    "BaseAPITypeExtra",
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
    "ChunkedFile",
    "CompletedEvent",
    "ConflictError",
    "ContentTooLargeError",
    "FullJitterBackoff",
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
    "RetryPolicy",
    "RetryState",
    "Role",
    "ServerSentEvent",
    "ServiceUnavailableError",
    "SessionMismatchWarning",
    "Stream",
    "StreamFile",
    "TextDeltaEvent",
    "TokenUsage",
    "Turn",
    "UnprocessableEntityError",
    "__version__",
    "bind_resource",
]
