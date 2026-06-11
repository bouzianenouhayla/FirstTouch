import threading
import time
from collections.abc import Callable
from typing import Any


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the circuit is open."""


class CircuitBreaker:
    """Thread-safe circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED.

    Tracks consecutive failures per instance so each agent fails independently.
    """

    _CLOSED = "closed"
    _OPEN = "open"
    _HALF_OPEN = "half_open"

    def __init__(self, fail_max: int = 5, reset_timeout: float = 60.0) -> None:
        """Args:
        fail_max: Consecutive failures before the circuit opens.
        reset_timeout: Seconds before attempting recovery (OPEN → HALF_OPEN).
        """
        self._fail_max = fail_max
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._state = self._CLOSED
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute fn, tripping the breaker on consecutive failures.

        Args:
            fn: Callable to protect.
            *args, **kwargs: Forwarded to fn.

        Returns:
            Return value of fn.

        Raises:
            CircuitBreakerOpen: Circuit is open and reset_timeout has not elapsed.
            Any exception raised by fn (also recorded as a failure).
        """
        with self._lock:
            if self._state == self._OPEN:
                elapsed = time.perf_counter() - (self._opened_at or 0)
                if elapsed >= self._reset_timeout:
                    self._state = self._HALF_OPEN
                else:
                    remaining = self._reset_timeout - elapsed
                    raise CircuitBreakerOpen(
                        f"Circuit open — retrying in {remaining:.0f}s"
                    )

        try:
            result = fn(*args, **kwargs)
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self._fail_max:
                    self._state = self._OPEN
                    self._opened_at = time.perf_counter()
            raise
        else:
            with self._lock:
                self._failures = 0
                self._state = self._CLOSED
            return result
