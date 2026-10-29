"""Binding of sessions to one event loop."""

import asyncio


def bind_loop(
    bound: asyncio.AbstractEventLoop | None,
    owner: str,
) -> asyncio.AbstractEventLoop:
    """Return the loop a session is bound to, binding the running one.

    Transport clients and their connections belong to one event loop;
    used from another, they fail with unclear errors such as ``Event loop
    is closed``. A session can therefore be created anywhere (for
    example, at module level) and is bound to the loop of its first use.

    Args:
        bound: The loop the session is bound to; ``None`` if it is not
            bound yet.
        owner: Name of the session class, for error messages.

    Returns:
        ``bound``, or the running loop if ``bound`` is ``None``; store it
        as the session's loop.

    Raises:
        RuntimeError: If no loop is running, or the running loop is not
            ``bound``.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError(
            f"{owner} can only be used inside a coroutine"
        ) from None

    if bound is not None and running is not bound:
        raise RuntimeError(
            f"{owner} is bound to another event loop; create one session "
            "per loop"
        )

    return running
