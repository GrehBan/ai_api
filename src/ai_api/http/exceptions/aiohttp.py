"""Translation of ``aiohttp`` exceptions into library errors."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from aiohttp import (
    ClientConnectionError,
    ClientConnectorCertificateError,
    ClientPayloadError,
    ClientResponseError,
    ServerFingerprintMismatch,
    TooManyRedirects,
)

from ai_api.http.exceptions.response import APIResponseError
from ai_api.http.exceptions.transport import (
    APIConnectionError,
    APIRedirectError,
    APIRequestError,
    APITimeoutError,
    APITLSError,
)


def _find_request_error(exc: BaseException) -> APIRequestError | None:
    """Find a failed upload source among the causes of an exception.

    Args:
        exc: The exception raised by aiohttp.

    Returns:
        The ``APIRequestError`` raised while reading an upload, if any.
    """
    seen: set[int] = set()
    current = exc.__cause__ or exc.__context__

    while current is not None and id(current) not in seen:
        if isinstance(current, APIRequestError):
            return current

        seen.add(id(current))
        current = current.__cause__ or current.__context__

    return None


@contextmanager
def transport_errors() -> Generator[None]:
    """Translate aiohttp exceptions raised in the block into library ones.

    Yields:
        Control to the wrapped block.

    Raises:
        APITimeoutError: On any timeout.
        APITLSError: If the server's certificate fails verification or
            does not match the pinned fingerprint.
        APIRedirectError: If the API redirects too many times.
        APIRequestError: If an upload source fails while it is sent.
        APIConnectionError: On other connection failures, including a
            TLS handshake that breaks off, which another attempt may
            complete.
        APIResponseError: If the response is not valid HTTP.
        APIRequestError: If aiohttp rejects a header or the URL.
    """
    try:
        yield
    except TimeoutError as exc:
        raise APITimeoutError() from exc

    # Only a certificate that fails verification is hopeless; another
    # SSL error (a handshake cut off) is a connection failure, retried
    # as the official SDKs do.
    except (
        ClientConnectorCertificateError,
        ServerFingerprintMismatch,
    ) as exc:
        raise APITLSError() from exc

    except TooManyRedirects as exc:
        raise APIRedirectError() from exc

    except ClientConnectionError as exc:
        # aiohttp reports a failing upload as a connection error, which
        # would be retried: re-raise the source's own error instead.
        if (source := _find_request_error(exc)) is not None:
            raise source from source.__cause__

        raise APIConnectionError() from exc

    except ClientResponseError as exc:
        raise APIResponseError(
            "Invalid HTTP response",
            status=exc.status or None,
        ) from exc

    # Last: certificate errors are ValueErrors too.
    except ValueError as exc:
        raise APIRequestError(f"Invalid request: {exc}") from exc


@contextmanager
def body_errors() -> Generator[None]:
    """Translate aiohttp exceptions raised while a response body is read.

    A body that ends before it is complete means the connection was lost,
    so it is reported, and retried, like any other connection failure.

    Yields:
        Control to the wrapped block.

    Raises:
        APIConnectionError: If the body ends before it is complete.
        APITransportError: On the failures ``transport_errors`` maps.
        APIResponseError: If the response is not valid HTTP.
    """
    try:
        with transport_errors():
            yield
    except ClientPayloadError as exc:
        raise APIConnectionError(
            "Connection lost before the response body was complete"
        ) from exc
