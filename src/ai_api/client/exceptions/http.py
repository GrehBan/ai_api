from typing import Final

from ai_api.exceptions import AIAPIError


class APIError(AIAPIError):
  def __init__(
        self,
        *,
        status: int,
        message: str | None = None,
        code: str | None = None,
        error_type: str | None = None,
        provider: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        transient: bool = False,
    ) -> None:
        self.status = status
        self.message = message or self.default_message
        self.code = code
        self.error_type = error_type
        self.provider = provider
        self.request_id = request_id
        self.retry_after = retry_after
        self.transient = transient

        super().__init__(self.default_message)



class BadRequestError(APIError):
    default_message = (
        "Bad request: Invalid request parameters "
        "or malformed data."
    )


class AuthenticationError(APIError):
    default_message = (
        "Authentication failed: "
        "Invalid or missing credentials."
    )


class BillingError(APIError):
    default_message = (
        "Billing error: Payment is required "
        "to access this resource."
    )


class PermissionDeniedError(APIError):
    default_message = (
        "Permission denied: Insufficient permissions "
        "to access this resource."
    )


class NotFoundError(APIError):
    default_message = (
        "Not found: The requested resource "
        "does not exist or is unavailable."
    )


class RequestTimeoutError(APIError):
    default_message = (
        "Request timeout: The server did not respond "
        "within the allowed time."
    )


class ConflictError(APIError):
    default_message = (
        "Conflict: The request conflicts "
        "with the current resource state."
    )


class UnprocessableEntityError(APIError):
    default_message = (
        "Unprocessable entity: The request contains "
        "invalid or unsupported data."
    )


class RateLimitError(APIError):
    default_message = (
        "Rate limit exceeded: "
        "Too many requests or tokens."
    )


class QuotaExceededError(APIError):
    default_message = (
        "Quota exceeded: Available credits "
        "or usage limits have been exhausted."
    )


class InternalServerError(APIError):
    default_message = (
        "Internal server error: "
        "The API encountered an unexpected error."
    )


class BadGatewayError(APIError):
    default_message = (
        "Bad gateway: The upstream server "
        "returned an invalid response."
    )


class ServiceUnavailableError(APIError):
    default_message = (
        "Service unavailable: "
        "The API is temporarily unavailable."
    )


class GatewayTimeoutError(APIError):
    default_message = (
        "Gateway timeout: The upstream server "
        "did not respond in time."
    )


HTTP_ERRORS: Final[dict[int, type[APIError]]] = {
    400: BadRequestError,
    401: AuthenticationError,
    402: BillingError,
    403: PermissionDeniedError,
    404: NotFoundError,
    408: RequestTimeoutError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
    502: BadGatewayError,
    503: ServiceUnavailableError,
    504: GatewayTimeoutError,
}
