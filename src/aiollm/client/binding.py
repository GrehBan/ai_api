"""Bindings of shared resources to the adapters of a provider client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from aiollm.session.protocols import ISession


class ResourceBinding[AdapterT, ResourceT]:
    """How a provider client builds one of its shared resources.

    ``bind_resource`` makes it; ``BaseProviderClient`` finds it among
    the class attributes, checks the adapter's provider when the class
    is defined, builds the adapter before the session and the resource
    after it.

    Attributes:
        resource_type: Builds the resource from a session and an
            adapter, such as ``GenerationResource[Response, Event]``.
        adapter_type: The adapter class; its provider is checked.
        adapter_factory: Builds the adapter from the client, or
            ``None`` to call ``adapter_type()`` without arguments.
    """

    __slots__ = ("adapter_factory", "adapter_type", "resource_type")

    def __init__(
        self,
        resource_type: Callable[[ISession, AdapterT], ResourceT],
        adapter_type: type[AdapterT],
        *,
        adapter_factory: Callable[[Any], AdapterT] | None = None,
    ) -> None:
        """Keep the parts of the binding.

        Args:
            resource_type: Builds the resource from a session and an
                adapter.
            adapter_type: The adapter class.
            adapter_factory: Builds the adapter from the client; without
                it, ``adapter_type()`` is called.
        """
        self.resource_type = resource_type
        self.adapter_type = adapter_type
        self.adapter_factory = adapter_factory

    def __repr__(self) -> str:
        """Name the adapter class of the binding.

        Returns:
            Such as ``ResourceBinding(OpenAIGenerationAdapter)``.
        """
        return f"{type(self).__name__}({self.adapter_type.__name__})"

    def build_adapter(self, client: object) -> AdapterT:
        """Build the adapter for a client whose session is not made yet.

        Args:
            client: The provider client being created.

        Returns:
            The adapter.

        Raises:
            TypeError: If the factory returns something other than an
                ``adapter_type`` of its provider: only that class had
                its provider checked, and a subclass may name another.
        """
        if self.adapter_factory is None:
            return self.adapter_type()

        adapter = self.adapter_factory(client)
        expected = getattr(self.adapter_type, "provider", None)

        if not isinstance(adapter, self.adapter_type) or (
            getattr(adapter, "provider", None) is not expected
        ):
            message = (
                f"The adapter factory returned {type(adapter).__name__}, "
                f"not {self.adapter_type.__name__} of its provider"
            )
            raise TypeError(message)

        return adapter

    def build_resource(
        self, session: ISession, adapter: AdapterT
    ) -> ResourceT:
        """Build the resource on the client's session.

        Args:
            session: The client's session.
            adapter: The adapter from ``build_adapter``.

        Returns:
            The resource.
        """
        return self.resource_type(session, adapter)


def bind_resource[AdapterT, ResourceT](
    resource_type: Callable[[ISession, AdapterT], ResourceT],
    adapter_type: type[AdapterT],
    *,
    adapter_factory: Callable[[Any], AdapterT] | None = None,
) -> ResourceT:
    """Declare a shared resource of a provider client.

    Assigned to a class attribute of a ``BaseProviderClient`` subclass,
    it makes that attribute of every client the resource, built on the
    client's session. Like ``dataclasses.field``, it returns a
    ``ResourceBinding`` typed as the resource, so type checkers see the
    attribute as the resource; on the class itself it stays the binding.

    Args:
        resource_type: Builds the resource from a session and an
            adapter, such as ``GenerationResource[Response, Event]``.
        adapter_type: The adapter class; its provider must be the
            client's, checked when the client class is defined.
        adapter_factory: Builds the adapter from the client, before the
            session exists (so bad options fail without one), such as a
            method of the client that reads its own options; without
            it, ``adapter_type()`` is called.

    Returns:
        The binding, typed as the resource.

    Example:
        >>> class ExampleClient(BaseProviderClient):
        ...     generation = bind_resource(
        ...         GenerationResource[Response, ResponseStreamEvent],
        ...         ExampleGenerationAdapter,
        ...     )
    """
    binding = ResourceBinding(
        resource_type, adapter_type, adapter_factory=adapter_factory
    )
    return cast("ResourceT", binding)
