from threading import Event

_shutdown_event = Event()


def set_shutdown() -> None:
    """Request shutdown from signal handler or external caller."""
    _shutdown_event.set()


def is_shutdown() -> bool:
    """Return True if shutdown was requested."""
    return _shutdown_event.is_set()


def wait(timeout: float | None = None) -> bool:
    """Block until shutdown requested or timeout. Returns True if set."""
    return _shutdown_event.wait(timeout)
