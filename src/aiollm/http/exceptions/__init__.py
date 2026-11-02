"""HTTP and transport errors and the error resolver."""

from __future__ import annotations

from aiollm.http.exceptions.protocols import (
    IErrorResolver,
)
from aiollm.http.exceptions.resolver import (
    BaseErrorResolver,
)
from aiollm.http.exceptions.response import (
    APIResponseError,
)
from aiollm.http.exceptions.status import (
    HTTP_ERRORS,
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

__all__ = [
    "APIConnectionError",
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
    "BaseErrorResolver",
    "BillingError",
    "ConflictError",
    "ContentTooLargeError",
    "GatewayTimeoutError",
    "HTTP_ERRORS",
    "IErrorResolver",
    "InternalServerError",
    "NotFoundError",
    "PermissionDeniedError",
    "QuotaExceededError",
    "RateLimitError",
    "RequestTimeoutError",
    "ServiceUnavailableError",
    "UnprocessableEntityError",
]
