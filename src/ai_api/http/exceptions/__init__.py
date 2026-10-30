"""HTTP and transport errors and the error resolver."""

from __future__ import annotations

from ai_api.http.exceptions.protocols import (
    IErrorResolver,
)
from ai_api.http.exceptions.resolver import (
    BaseErrorResolver,
)
from ai_api.http.exceptions.response import (
    APIResponseError,
)
from ai_api.http.exceptions.status import (
    HTTP_ERRORS,
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
    "APITLSError",
    "APITimeoutError",
    "APITransportError",
    "AuthenticationError",
    "BadGatewayError",
    "BadRequestError",
    "BaseErrorResolver",
    "BillingError",
    "ConflictError",
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
