"""Base classes for clients and provider clients."""

from __future__ import annotations

import warnings
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

from aiollm.client.binding import ResourceBinding
from aiollm.client.validators import (
    check_same_provider,
    provider_of,
    resolve_token,
)
from aiollm.enums import Provider
from aiollm.exceptions import SessionMismatchWarning
from aiollm.methods.protocols import IAPIMethod, IStreamMethod
from aiollm.response.stream import Stream
from aiollm.session.base import BaseSession
from aiollm.session.protocols import ISession, ISessionFactory, Timeout
from aiollm.session.retry.policy import RetryPolicy

if TYPE_CHECKING:
    from types import TracebackType

    from aiollm.http.protocols import HeadersType

_LIBRARY: Final = str(Path(__file__).resolve().parents[1])
"""The package directory, skipped so a warning points at the caller."""

_PARTS: Final = (
    "provider",
    "session_factory",
    "api_key_env",
    "auth_headers",
)
"""Class attributes every provider client sets."""


class BaseClient[SessionT: ISession]:
    """Sends methods through a session; a provider client subclasses it.

    The client owns its session: ``close()`` and leaving ``async with``
    close it. ``SessionT`` keeps the session's own type, so a client
    built on ``AiohttpSession`` exposes its settings, such as ``loop``.

    Attributes:
        session: The session that sends the requests.

    Example:
        >>> async with OpenAIClient("sk-...") as client:
        ...     models = await client.send_request(ListModels())
    """

    def __init__(self, session: SessionT) -> None:
        """Wrap a session.

        Args:
            session: The session that sends the requests.
        """
        self.session: SessionT = session

    async def __aenter__(self) -> Self:
        """Open the session.

        Returns:
            The client itself.
        """
        await self.session.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the session's context, which closes it.

        Delegated as ``__aenter__`` is, so a session with its own exit
        logic gets the exception that ended the block.

        Args:
            exc_type: Type of the exception raised in the block, if any.
            exc_value: The exception raised in the block, if any.
            traceback: Its traceback, if any.
        """
        await self.session.__aexit__(exc_type, exc_value, traceback)

    async def close(self) -> None:
        """Close the session and its connections."""
        await self.session.close()

    async def send_request[T](
        self,
        method: IAPIMethod[T],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> T:
        """Send a method and return its parsed response.

        Args:
            method: The method to send.
            headers: Headers for this request only.
            timeout: Seconds the whole request may take; ``None`` for
                the session's timeout.
            retry_policy: Retry policy of this request; ``None`` for the
                session's.

        Returns:
            The parsed response, typed by the method's generic argument.
        """
        return await self.session.send_request(
            method,
            headers=headers,
            timeout=timeout,
            retry_policy=retry_policy,
        )

    def stream[T](
        self,
        method: IStreamMethod[T],
        *,
        headers: HeadersType | None = None,
        timeout: Timeout | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> AbstractAsyncContextManager[Stream[T]]:
        """Send a stream method; use the result with ``async with``.

        Args:
            method: The stream method to send.
            headers: Headers for this request only.
            timeout: Longest silence between two chunks, in seconds;
                ``None`` for the session's stream timeout.
            retry_policy: Retry policy of opening the stream; ``None``
                for the session's.

        Returns:
            A context manager that yields the stream and closes it on
            exit.
        """
        return self.session.stream(
            method,
            headers=headers,
            timeout=timeout,
            retry_policy=retry_policy,
        )


class BaseProviderClient(BaseClient[ISession]):
    """What a provider's client shares: its session, from a token.

    A subclass declares its parts and its shared resources, each with
    ``bind_resource``: the resource class and the adapter it runs on.
    The client builds every adapter before its session, so bad adapter
    options fail without one, and every resource right after it. The
    parts name one provider, checked when the subclass is defined, for
    every adapter it declares.

    The bindings are found among the class attributes of the client and
    its bases, by MRO; a subclass replaces one under the same name or
    drops it with any other value. On an instance the attribute is the
    resource; on the class it stays the ``ResourceBinding``, though
    type checkers see the resource (see ``bind_resource``).

    The client takes a token, a ready ``session=`` or, without either,
    the token in ``api_key_env``; ``session_kwargs`` configure the
    session built from a token. A given session is checked by
    ``warn_about_session``.

    Attributes:
        provider: The provider of the client.
        session_factory: Builds the session from a token; its error
            resolver is the provider's.
        api_key_env: Environment variable read without a token or a
            session, such as ``OPENAI_API_KEY``.
        auth_headers: Headers the provider takes the key in, such as
            ``Authorization``.

    Example:
        >>> class ExampleClient(BaseProviderClient):
        ...     provider = PROVIDER
        ...     session_factory = session_factory
        ...     api_key_env = "EXAMPLE_API_KEY"
        ...     auth_headers = frozenset({AUTHORIZATION})
        ...
        ...     generation = bind_resource(
        ...         GenerationResource[Response, ResponseStreamEvent],
        ...         ExampleGenerationAdapter,
        ...     )
    """

    provider: ClassVar[Provider]
    session_factory: ClassVar[ISessionFactory[ISession, Any, ...]]
    api_key_env: ClassVar[str]
    auth_headers: ClassVar[frozenset[str]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check that the parts of the client name one provider.

        Args:
            **kwargs: Class keyword arguments.

        Raises:
            TypeError: If a part is missing, an adapter class declares
                no provider, or the client, the session factory's error
                resolver and an adapter name different providers.
        """
        super().__init_subclass__(**kwargs)

        missing = [name for name in _PARTS if not hasattr(cls, name)]
        if missing:
            message = f"{cls.__name__} must set {', '.join(missing)}"
            raise TypeError(message)

        # Keyed by attribute name, unique in a class, so no adapter can
        # drop out of the check.
        adapters = {
            f"{name} adapter {binding.adapter_type.__name__}": provider_of(
                binding.adapter_type, f"{name} adapter"
            )
            for name, binding in cls._resource_bindings().items()
        }
        check_same_provider(
            client=cls.provider,
            error_resolver=provider_of(
                cls.session_factory.error_resolver, "error resolver"
            ),
            **adapters,
        )

    @classmethod
    def _resource_bindings(cls) -> dict[str, ResourceBinding[Any, Any]]:
        """Find the shared resources of the client, inherited ones too.

        A subclass replaces an inherited binding with another one under
        the same name, or drops it with any other value.

        Returns:
            The bindings, by attribute name, in declaration order.
        """
        bindings: dict[str, ResourceBinding[Any, Any]] = {}

        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if isinstance(value, ResourceBinding):
                    bindings[name] = value
                else:
                    bindings.pop(name, None)

        return bindings

    def __init__(
        self,
        token: str | None = None,
        *,
        session: ISession | None = None,
        **session_kwargs: Any,
    ) -> None:
        """Create the client from a token or a ready session.

        A subclass types ``session_kwargs`` by its session's options
        and sets the options its adapter factories read before calling
        this. The adapters are built first, the resources once the
        session exists.

        Args:
            token: API key; the session is built from it by
                ``session_factory``. Without a token and a session, the
                key in ``api_key_env`` is used.
            session: A ready session, for custom settings; it is checked
                with ``warn_about_session``.
            **session_kwargs: Further arguments of the session built from
                the token, such as ``timeout``.

        Raises:
            ValueError: If both ``token`` and ``session`` are given, or
                neither is and ``api_key_env`` is unset or empty, or
                ``session_kwargs`` are given with a ``session``.
            TypeError: If an adapter factory returns another class than
                its binding declares.
        """
        bindings = self._resource_bindings()
        adapters = {
            name: binding.build_adapter(self)
            for name, binding in bindings.items()
        }
        token = resolve_token(token, session, self.api_key_env)

        if session is None:
            assert token is not None  # noqa: S101 - checked above
            session = self.session_factory(token, **session_kwargs)
        else:
            if session_kwargs:
                names = ", ".join(sorted(session_kwargs))
                message = (
                    "session_kwargs configure the session built from a "
                    f"token; pass them to the given session instead: {names}"
                )
                raise ValueError(message)
            self.warn_about_session(session)

        super().__init__(session)

        for name, binding in bindings.items():
            resource = binding.build_resource(self.session, adapters[name])
            setattr(self, name, resource)

    @classmethod
    def warn_about_session(cls, session: ISession) -> None:
        """Warn about a given session that does not fit the provider.

        Only ``BaseSession`` subclasses are checked; other sessions are
        trusted as they are. The session's ``api`` is not checked:
        gateways, proxies and hosted deployments (such as Azure) serve a
        provider's API at another ``base_url`` on purpose.

        Args:
            session: The session given to the client.

        Warns:
            SessionMismatchWarning: If the session's error resolver is
                not the provider's (a subclass of it is), or the session
                sends the key in a header the provider does not take. The
                warning points at the code that built the client.
        """
        if not isinstance(session, BaseSession):
            return

        expected = type(cls.session_factory.error_resolver)
        given = session.error_resolver

        if not isinstance(given, expected):
            warnings.warn(
                f"{cls.__name__} is using {type(given).__name__}. "
                f"Consider using {expected.__name__} for provider-specific "
                "error handling.",
                SessionMismatchWarning,
                skip_file_prefixes=(_LIBRARY,),
            )

        accepted = {header.lower() for header in cls.auth_headers}
        if session.auth.header_name.lower() not in accepted:
            headers = " or ".join(sorted(cls.auth_headers))
            warnings.warn(
                f"{cls.__name__} sends the key in "
                f"{session.auth.header_name}, but {cls.provider.value} "
                f"takes it in {headers}.",
                SessionMismatchWarning,
                skip_file_prefixes=(_LIBRARY,),
            )
