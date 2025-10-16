"""
Thin, typed client surface for accessing ComfyUI HTTP endpoints and the websocket.

This module defines the public protocol, DTOs, and exceptions that callers and
tests depend on. Implementations are intentionally minimal at this stage; network
behavior will be added in subsequent commits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Protocol, TypedDict


__all__ = [
    "ComfyUIClientProtocol",
    "ComfyUILocalClient",
    "SystemStats",
    "WsEvent",
    "ComfyUIClientError",
    "ComfyUIClientTimeout",
    "ComfyUIClientWsClosed",
]


class WsEvent(TypedDict, total=False):
    """Typed view of websocket events emitted by ComfyUI.

    Fields:
    - type: Event type string (e.g., "status", "progress", "executed").
    - data: Opaque payload; commonly includes keys like "prompt_id", "node", etc.
    """

    type: str
    data: Dict[str, Any]


@dataclass
class SystemStats:
    """Subset of system statistics reported by ComfyUI `/system_stats`.

    Known fields are optional; unknown fields should be ignored by parsers. When
    present, RAM-related metrics may also appear (e.g., `ram_total`, `ram_free`).

    Attributes:
        vram_free: Free VRAM reported by the backend, if available.
        torch_vram_free: Free VRAM according to torch, if available.
        ram_total: Total system RAM (bytes) when reported.
        ram_free: Free system RAM (bytes) when reported.
    """

    vram_free: Optional[float] = None
    torch_vram_free: Optional[float] = None
    ram_total: Optional[float] = None
    ram_free: Optional[float] = None


class ComfyUIClientError(Exception):
    """Base error for all ComfyUI client failures.

    Args:
        message: Human-friendly error message.
        route: Route path (e.g., "/prompt").
        method: HTTP method (e.g., "GET", "POST").
        status: Optional HTTP status code or websocket close code.
        code: Optional machine-readable error code (e.g., "timeout").
        body_snippet: Optional diagnostic snippet from a response payload.
    """

    def __init__(
        self,
        message: str,
        *,
        route: Optional[str] = None,
        method: Optional[str] = None,
        status: Optional[int] = None,
        code: Optional[str] = None,
        body_snippet: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.route: Optional[str] = route
        self.method: Optional[str] = method
        self.status: Optional[int] = status
        self.code: Optional[str] = code
        self.body_snippet: Optional[str] = body_snippet


class ComfyUIClientTimeout(ComfyUIClientError):
    """Raised when an operation exceeds its allowed timeout."""


class ComfyUIClientWsClosed(ComfyUIClientError):
    """Raised when the websocket is closed or cannot be maintained."""


class ComfyUIClientProtocol(Protocol):
    """Protocol for ComfyUI clients.

    Callers and tests should depend on this interface rather than a concrete
    implementation. Methods are asynchronous and may raise subclasses of
    `ComfyUIClientError`.
    """

    async def submit_prompt(self, payload: Dict[str, Any]) -> str:
        """Submit a prompt to ComfyUI and return the resulting `prompt_id`.

        Args:
            payload: JSON-serializable payload for the `/prompt` endpoint.

        Returns:
            The non-empty `prompt_id` string returned by the server.
        """

    async def get_system_stats(self) -> SystemStats:
        """Fetch system statistics from `/system_stats`."""

    async def free(
        self, *, free_memory: bool = False, unload_models: bool = False
    ) -> None:
        """Invoke `/free` with the provided flags."""

    async def ws_connect(self, client_id: str) -> AsyncIterator[WsEvent]:
        """Connect to the websocket (`/ws?clientId=...`) and yield parsed events."""


class ComfyUILocalClient(ComfyUIClientProtocol):
    """Local HTTP/WebSocket client for a running ComfyUI instance.

    This class provides the concrete implementation for the protocol. At this
    stage it only declares the interface and stores constructor parameters; the
    network behavior will be implemented in subsequent commits.

    Args:
        base_url: Base HTTP URL for ComfyUI endpoints (e.g., `/prompt`).
        ws_url: Base WebSocket URL (e.g., `/ws`).
        session: Optional externally-managed aiohttp session for reuse.
        logger: Optional logger compatible with the worker's logging API.
        timeout: Default timeout in seconds for HTTP operations.
    """

    def __init__(
        self,
        base_url: str,
        ws_url: str,
        session: Optional["aiohttp.ClientSession"] = None,
        logger: Optional[Any] = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url: str = base_url
        self._ws_url: str = ws_url
        self._session = session  # type: ignore[assignment]
        self._logger = logger
        self._timeout: float = float(timeout)

    # Lifecycle methods may be implemented later; for now they act as no-ops.
    async def __aenter__(self) -> "ComfyUILocalClient":
        """Enter async context; no-op until internal session management is added."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Exit async context; no-op until internal session management is added."""
        return None

    # Protocol methods — to be implemented in subsequent commits.
    async def submit_prompt(self, payload: Dict[str, Any]) -> str:  # type: ignore[override]
        raise NotImplementedError("submit_prompt is not implemented yet")

    async def get_system_stats(self) -> SystemStats:  # type: ignore[override]
        raise NotImplementedError("get_system_stats is not implemented yet")

    async def free(self, *, free_memory: bool = False, unload_models: bool = False) -> None:  # type: ignore[override]
        raise NotImplementedError("free is not implemented yet")

    async def ws_connect(self, client_id: str) -> AsyncIterator[WsEvent]:  # type: ignore[override]
        raise NotImplementedError("ws_connect is not implemented yet")


# Deferred import used only for typing to avoid mandatory runtime dependency here.
try:  # pragma: no cover - typing aid only
    import aiohttp  # type: ignore
except Exception:  # pragma: no cover - typing aid only
    aiohttp = None  # type: ignore
