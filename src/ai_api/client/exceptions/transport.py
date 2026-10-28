from ai_api.exceptions import AIAPIError


class APITransportError(AIAPIError):
    def __init__(
        self,
        message: str,
        *,
        transient: bool = True,
    ) -> None:
        super().__init__(message)
        self.transient = transient


class APIConnectionError(APITransportError):
    def __init__(
        self,
        message: str = "Failed to connect to the API.",
    ) -> None:
        super().__init__(message, transient=True)


class APITimeoutError(APITransportError):
    def __init__(
        self,
        message: str = "API request timed out.",
    ) -> None:
        super().__init__(message, transient=True)


class APITLSError(APITransportError):
    def __init__(
        self,
        message: str = "TLS connection verification failed.",
    ) -> None:
        super().__init__(message, transient=False)


class APIResponseError(AIAPIError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
    ) -> None:
        self.status = status
        super().__init__(message)
