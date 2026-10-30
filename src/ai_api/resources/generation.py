"""The shared resource of text generation."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from ai_api.adapters.protocols import ICompositeAdapter
from ai_api.resources.base import BaseResource
from ai_api.resources.usecases import AdapterCompositeUseCase
from ai_api.response.stream import Stream
from ai_api.session.protocols import ISession
from ai_api.types.requests.generation import GenerationRequest
from ai_api.types.responses.events import GenerationEvent
from ai_api.types.responses.generation import GenerationResponse


class GenerationResource[RawT, RawItemT, SessionT: ISession = ISession](
    BaseResource[SessionT],
):
    """Generates text with any provider through that provider's adapter.

    The class is the same for every provider: a provider's client builds
    it with its own adapter, so calling code reads the same for all of
    them and only the type of ``raw`` differs. ``RawT`` is the provider's
    response, ``RawItemT`` one item of its stream.

    Example:
        >>> generation = GenerationResource(session, ResponsesAdapter())
        >>> request = GenerationRequest(model="gpt-5", input="Hi")
        >>> response = await generation.create(request)
        >>> async with generation.stream(request) as events:
        ...     async for event in events:
        ...         ...
    """

    def __init__(
        self,
        session: SessionT,
        adapter: ICompositeAdapter[
            GenerationRequest,
            RawT,
            GenerationResponse[RawT],
            RawItemT,
            GenerationEvent[RawItemT],
        ],
    ) -> None:
        """Keep the session and build the use case on the adapter.

        Args:
            session: The session that sends the requests.
            adapter: Converts the shared request, the provider's response
                and its stream items.
        """
        super().__init__(session)
        self._generate = AdapterCompositeUseCase(session, adapter)

    async def create(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse[RawT]:
        """Generate the whole response at once.

        Args:
            request: What to generate.

        Returns:
            The generated text with the provider's own response.
        """
        return await self._generate(request)

    def stream(
        self,
        request: GenerationRequest,
    ) -> AbstractAsyncContextManager[Stream[GenerationEvent[RawItemT]]]:
        """Open a stream of generation events; use with ``async with``.

        Args:
            request: What to generate.

        Returns:
            A context manager that yields the events and closes the
            converted and the provider's stream on exit.
        """
        return self._generate.stream(request)
