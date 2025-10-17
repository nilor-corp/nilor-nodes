"""
Thin, typed client surface for accessing ComfyUI HTTP endpoints and the websocket.

This module defines the public protocol, DTOs, and exceptions that callers and
tests depend on. Implementations are intentionally minimal at this stage; network
behavior will be added in subsequent commits.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional, Protocol, TypedDict, Tuple

from urllib.parse import quote, urlparse


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
        vram_total: Total VRAM (bytes) when reported.
        vram_free: Free VRAM reported by the backend, if available.
        torch_vram_free: Free VRAM according to torch, if available.
        ram_total: Total system RAM (bytes) when reported.
        ram_free: Free system RAM (bytes) when reported.
    """

    vram_total: Optional[float] = None
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

    async def probe(self) -> None:
        """Lightweight health probe for connectivity/parseability.

        Executes a GET `/system_stats` with a short timeout and no retries. Raises
        `ComfyUIClientError` subclasses on failure; returns `None` on success.
        """

    async def supports_hygiene(self) -> bool:
        """Return True if both `/system_stats` and `/free` are supported.

        Performs a one-time capability probe; caches results for the session.
        """


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
        self._owned_session: Optional[aiohttp.ClientSession] = None
        # Backoff defaults per plan (commit 3)
        self._retry_base_seconds: float = 0.25
        self._retry_multiplier: float = 2.0
        self._retry_jitter_seconds: float = 0.25
        self._retry_max_sleep_seconds: float = 4.0
        self._retry_max_attempts: int = 3
        # WebSocket reconnect policy (commit 4)
        self._ws_max_reconnect_attempts: int = 5
        self._ws_max_total_backoff_seconds: float = 30.0
        # Capability probe cache (None = unknown, True/False = probed)
        self._supports_system_stats: Optional[bool] = None
        self._supports_free: Optional[bool] = None
        self._capability_warning_emitted: bool = False

    # Lifecycle methods may be implemented later; for now they act as no-ops.
    async def __aenter__(self) -> "ComfyUILocalClient":
        """Enter async context; create an internal session when none provided."""
        if self._session is None and self._owned_session is None:
            self._owned_session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        """Exit async context; close internal session if owned by this client."""
        if self._owned_session is not None:
            try:
                await self._owned_session.close()
            finally:
                self._owned_session = None
        return None

    # Protocol methods — to be implemented in subsequent commits.
    async def submit_prompt(self, payload: Dict[str, Any]) -> str:  # type: ignore[override]
        route = "/prompt"
        method = "POST"
        url = self._join_http(route)
        try:
            data = await self._http_request_json(
                method,
                url,
                json_payload=payload,
                timeout_s=self._timeout,
                retry_idempotent=False,
            )
        except asyncio.TimeoutError as e:
            raise ComfyUIClientTimeout(
                f"Timeout while calling {method} {route}",
                route=route,
                method=method,
                code="timeout",
            ) from e
        except _MappedHttpError as e:
            raise e.to_public_error(route=route, method=method)
        except _MappedConnError as e:
            raise e.to_public_error(route=route, method=method)

        prompt_id = data.get("prompt_id") if isinstance(data, dict) else None
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            snippet = _safe_preview(data)
            raise ComfyUIClientError(
                "Invalid response payload: missing non-empty prompt_id",
                route=route,
                method=method,
                code="bad_json",
                body_snippet=snippet,
            )
        return prompt_id

    async def get_system_stats(self) -> SystemStats:  # type: ignore[override]
        route = "/system_stats"
        method = "GET"
        url = self._join_http(route)
        try:
            data = await self._http_request_json(
                method,
                url,
                json_payload=None,
                timeout_s=self._timeout,
                retry_idempotent=True,
            )
        except asyncio.TimeoutError as e:
            raise ComfyUIClientTimeout(
                f"Timeout while calling {method} {route}",
                route=route,
                method=method,
                code="timeout",
            ) from e
        except _MappedHttpError as e:
            raise e.to_public_error(route=route, method=method)
        except _MappedConnError as e:
            raise e.to_public_error(route=route, method=method)

        # Parse known fields, tolerate missing/unknown; try alternate shapes; fallback to torch
        vram_total = None
        vram_free = None
        torch_vram_free = None
        ram_total = None
        ram_free = None

        if isinstance(data, dict):
            vram_total = _coerce_optional_float(data.get("vram_total"))
            vram_free = _coerce_optional_float(data.get("vram_free"))
            torch_vram_free = _coerce_optional_float(data.get("torch_vram_free"))
            ram_total = _coerce_optional_float(data.get("ram_total"))
            ram_free = _coerce_optional_float(data.get("ram_free"))

            # Common alternates
            if vram_total is None:
                vram_total = _coerce_optional_float(data.get("total_vram"))
            if vram_free is None:
                vram_free = _coerce_optional_float(data.get("free_vram"))

            vram_obj = data.get("vram") if isinstance(data.get("vram"), dict) else None
            if vram_obj:
                if vram_total is None:
                    vram_total = _coerce_optional_float(vram_obj.get("total"))
                if vram_free is None:
                    vram_free = _coerce_optional_float(vram_obj.get("free"))

            ram_obj = data.get("ram") if isinstance(data.get("ram"), dict) else None
            if ram_obj:
                if ram_total is None:
                    ram_total = _coerce_optional_float(ram_obj.get("total"))
                if ram_free is None:
                    ram_free = _coerce_optional_float(ram_obj.get("free"))

            # devices[0] fallback (common in mock/alt servers)
            devices = (
                data.get("devices") if isinstance(data.get("devices"), list) else None
            )
            if devices and len(devices) > 0 and isinstance(devices[0], dict):
                dev0 = devices[0]
                if vram_total is None:
                    vram_total = _coerce_optional_float(dev0.get("vram_total"))
                if vram_free is None:
                    vram_free = _coerce_optional_float(dev0.get("vram_free"))
        # Note: we rely solely on server-reported values; no local torch fallback

        # Optional debug logging of reported stats
        if self._logger:
            try:
                self._logger.debug(
                    "ComfyUI /system_stats: vram_total=%s vram_free=%s torch_vram_free=%s ram_total=%s ram_free=%s",
                    vram_total,
                    vram_free,
                    torch_vram_free,
                    ram_total,
                    ram_free,
                )
            except Exception:
                pass
        return SystemStats(
            vram_total=vram_total,
            vram_free=vram_free,
            torch_vram_free=torch_vram_free,
            ram_total=ram_total,
            ram_free=ram_free,
        )

    async def free(self, *, free_memory: bool = False, unload_models: bool = False) -> None:  # type: ignore[override]
        route = "/free"
        method = "POST"
        url = self._join_http(route)
        body = {"free_memory": bool(free_memory), "unload_models": bool(unload_models)}
        try:
            await self._http_request_json(
                method,
                url,
                json_payload=body,
                timeout_s=self._timeout,
                retry_idempotent=True,  # idempotent when flags identical
            )
        except asyncio.TimeoutError as e:
            raise ComfyUIClientTimeout(
                f"Timeout while calling {method} {route}",
                route=route,
                method=method,
                code="timeout",
            ) from e
        except _MappedHttpError as e:
            raise e.to_public_error(route=route, method=method)
        except _MappedConnError as e:
            raise e.to_public_error(route=route, method=method)

    async def ws_connect(self, client_id: str) -> AsyncIterator[WsEvent]:  # type: ignore[override]
        url = _ws_url(self._ws_url, client_id)
        attempts = 0
        total_backoff = 0.0
        base = self._retry_base_seconds
        multiplier = self._retry_multiplier
        jitter = self._retry_jitter_seconds
        max_sleep = self._retry_max_sleep_seconds

        while True:
            try:
                # Allow large frames and set ping/pong defaults
                async with websockets.connect(
                    url,
                    max_size=None,
                    read_limit=64 * 1024 * 1024,
                    max_queue=4,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    # On successful connect, reset counters
                    attempts = 0
                    total_backoff = 0.0
                    if self._logger:
                        try:
                            self._logger.debug(
                                f"ComfyUI client: connected websocket {url}"
                            )
                        except Exception:
                            pass

                    while True:
                        message = await websocket.recv()
                        if isinstance(message, bytes):
                            try:
                                message = message.decode("utf-8", errors="replace")
                            except Exception:
                                yield WsEvent(type="binary", data={"length": len(message)})  # type: ignore[call-arg]
                                continue
                        try:
                            payload = json.loads(message)
                        except Exception:
                            yield WsEvent(type="text", data={"message": message})  # type: ignore[call-arg]
                            continue

                        if isinstance(payload, dict):
                            event_type = str(payload.get("type", "event"))
                            event_data = payload.get("data")
                            if not isinstance(event_data, dict):
                                event_data = {"raw": payload}
                            yield WsEvent(type=event_type, data=event_data)  # type: ignore[call-arg]
                        else:
                            yield WsEvent(type="event", data={"raw": payload})  # type: ignore[call-arg]

            except asyncio.CancelledError:
                # Allow clean shutdown by propagating cancellation
                raise
            except ws_exc.ConnectionClosedOK as e:
                raise ComfyUIClientWsClosed(
                    "WebSocket closed normally",
                    route="/ws",
                    method="GET",
                    status=getattr(e, "code", 1000),
                    code="ws_closed",
                    body_snippet=str(getattr(e, "reason", ""))[:256],
                ) from e
            except ws_exc.ConnectionClosedError as e:
                # Abnormal close: attempt bounded reconnect
                attempts += 1
                if attempts > self._ws_max_reconnect_attempts:
                    raise ComfyUIClientError(
                        "WebSocket reconnect attempts exhausted",
                        route="/ws",
                        method="GET",
                        code="ws_closed",
                        body_snippet=str(getattr(e, "reason", ""))[:256],
                    ) from e
                delay = _next_backoff(attempts - 1, base, multiplier, jitter, max_sleep)
                total_backoff += delay
                if total_backoff > self._ws_max_total_backoff_seconds:
                    raise ComfyUIClientError(
                        "WebSocket reconnect backoff budget exhausted",
                        route="/ws",
                        method="GET",
                        code="ws_closed",
                        body_snippet=str(getattr(e, "reason", ""))[:256],
                    ) from e
                await asyncio.sleep(delay)
                continue
            except ws_exc.InvalidStatus as e:
                raise ComfyUIClientError(
                    "WebSocket handshake failed",
                    route="/ws",
                    method="GET",
                    code="ws_handshake",
                ) from e
            except ws_exc.InvalidURI as e:
                raise ComfyUIClientError(
                    "Invalid WebSocket URI",
                    route="/ws",
                    method="GET",
                    code="invalid_ws_uri",
                ) from e
            except Exception as e:
                # Treat as connection error; bounded reconnect
                attempts += 1
                if attempts > self._ws_max_reconnect_attempts:
                    raise ComfyUIClientError(
                        "WebSocket reconnect attempts exhausted",
                        route="/ws",
                        method="GET",
                        code="connection_error",
                    ) from e
                delay = _next_backoff(attempts - 1, base, multiplier, jitter, max_sleep)
                total_backoff += delay
                if total_backoff > self._ws_max_total_backoff_seconds:
                    raise ComfyUIClientError(
                        "WebSocket reconnect backoff budget exhausted",
                        route="/ws",
                        method="GET",
                        code="connection_error",
                    ) from e
                await asyncio.sleep(delay)
                continue

    async def probe(self) -> None:  # type: ignore[override]
        route = "/system_stats"
        method = "GET"
        url = self._join_http(route)
        short_timeout = min(self._timeout, 3.0)
        try:
            # No retries: retry_idempotent=False
            await self._http_request_json(
                method,
                url,
                json_payload=None,
                timeout_s=short_timeout,
                retry_idempotent=False,
            )
        except asyncio.TimeoutError as e:
            raise ComfyUIClientTimeout(
                f"Timeout while calling {method} {route}",
                route=route,
                method=method,
                code="timeout",
            ) from e
        except _MappedHttpError as e:
            raise e.to_public_error(route=route, method=method)
        except _MappedConnError as e:
            raise e.to_public_error(route=route, method=method)

    async def supports_hygiene(self) -> bool:  # type: ignore[override]
        # If both probed, return cached decision
        if self._supports_system_stats is not None and self._supports_free is not None:
            return bool(self._supports_system_stats and self._supports_free)

        await self._probe_capabilities_once()
        supported = bool(
            (self._supports_system_stats is True) and (self._supports_free is True)
        )
        if not supported and not self._capability_warning_emitted and self._logger:
            try:
                self._logger.warning(
                    "ComfyUI client: hygiene disabled — missing /system_stats or /free support"
                )
            except Exception:
                pass
            self._capability_warning_emitted = True
        return supported

    async def _probe_capabilities_once(self) -> None:
        """Probe `/system_stats` and `/free` capabilities once and cache results.

        Only marks capabilities as False on definitive 404/405 responses. Transient
        failures leave the capability as None so a future call may retry.
        """
        short_timeout = min(self._timeout, 3.0)

        # Probe /system_stats support
        if self._supports_system_stats is None:
            route = "/system_stats"
            url = self._join_http(route)
            try:
                await self._http_request_json(
                    "GET",
                    url,
                    json_payload=None,
                    timeout_s=short_timeout,
                    retry_idempotent=False,
                )
                self._supports_system_stats = True
            except ComfyUIClientError as e:
                if getattr(e, "status", None) in (404, 405):
                    self._supports_system_stats = False

        # Probe /free support (no-op body)
        if self._supports_free is None:
            route = "/free"
            url = self._join_http(route)
            try:
                await self._http_request_json(
                    "POST",
                    url,
                    json_payload={"free_memory": False, "unload_models": False},
                    timeout_s=short_timeout,
                    retry_idempotent=False,
                )
                self._supports_free = True
            except ComfyUIClientError as e:
                if getattr(e, "status", None) in (404, 405):
                    self._supports_free = False


# Runtime dependency; imported here to avoid issues if module is scanned without execution
import aiohttp  # type: ignore
import websockets  # type: ignore
from websockets import exceptions as ws_exc  # type: ignore


# ---- Internal helpers (HTTP) ----


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _is_transient_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _safe_preview(data: Any, limit: int = 512) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False)
    except Exception:
        text = str(data)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


class _MappedHttpError(Exception):
    def __init__(self, *, status: Optional[int], body_snippet: Optional[str]) -> None:
        self.status = status
        self.body_snippet = body_snippet

    def to_public_error(self, *, route: str, method: str) -> ComfyUIClientError:
        return ComfyUIClientError(
            f"HTTP error while calling {method} {route}",
            route=route,
            method=method,
            status=self.status,
            code="http_error",
            body_snippet=self.body_snippet,
        )


class _MappedConnError(Exception):
    def __init__(self, *, code: str = "connection_error") -> None:
        self.code = code

    def to_public_error(self, *, route: str, method: str) -> ComfyUIClientError:
        return ComfyUIClientError(
            f"Connection error while calling {method} {route}",
            route=route,
            method=method,
            code=self.code,
        )


async def _read_limited_text(resp: aiohttp.ClientResponse, limit: int = 512) -> str:
    try:
        raw = await resp.read()
        # Truncate at byte level then decode safely
        raw = raw[:limit]
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _with_jitter(seconds: float, jitter: float) -> float:
    if jitter <= 0:
        return seconds
    return max(0.0, seconds + random.uniform(-jitter, jitter))


def _next_backoff(
    attempt_index: int,
    base: float,
    multiplier: float,
    jitter: float,
    max_sleep: float,
) -> float:
    # attempt_index is 0-based
    delay = base * (multiplier**attempt_index)
    delay = min(delay, max_sleep)
    return _with_jitter(delay, jitter)


def _should_retry(
    *,
    retry_idempotent: bool,
    exc: Optional[BaseException] = None,
    status: Optional[int] = None,
) -> bool:
    if not retry_idempotent:
        return False
    if isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientConnectionError)):
        return True
    if status is not None and _is_transient_status(status):
        return True
    return False


def _finalize_attempts(
    method: str,
    url: str,
    *,
    last_exc: Optional[BaseException],
    last_status: Optional[int],
    last_body_snippet: Optional[str],
) -> BaseException:
    if isinstance(last_exc, asyncio.TimeoutError):
        return ComfyUIClientTimeout(
            f"Timeout while calling {method} {url}",
            route=_route_from_url(url),
            method=method,
            code="timeout",
        )
    if isinstance(last_exc, aiohttp.ClientConnectionError):
        return _MappedConnError().to_public_error(
            route=_route_from_url(url), method=method
        )
    # Otherwise treat as HTTP error
    return _MappedHttpError(
        status=last_status, body_snippet=last_body_snippet
    ).to_public_error(route=_route_from_url(url), method=method)


def _route_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.path or "/"
    except Exception:
        return url


class _TempSession:
    """Context manager that yields an aiohttp session, reusing if provided."""

    def __init__(self, session: Optional[aiohttp.ClientSession]):
        self._provided = session
        self._owned: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> aiohttp.ClientSession:
        if self._provided is not None:
            return self._provided
        self._owned = aiohttp.ClientSession()
        return self._owned

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if self._owned is not None:
            await self._owned.close()


async def _json_or_text(resp: aiohttp.ClientResponse) -> Any:
    ctype = resp.headers.get("Content-Type", "").lower()
    text = await _read_limited_text(resp)  # limited read to use for both cases
    if "json" in ctype:
        try:
            return json.loads(text)
        except Exception:
            # Fallthrough to treat as bad JSON
            raise _MappedHttpError(status=resp.status, body_snippet=text)
    # Not JSON; return raw text
    return text


async def _raise_for_status_with_snippet(resp: aiohttp.ClientResponse) -> None:
    if 200 <= resp.status <= 299:
        return
    snippet = await _read_limited_text(resp)
    raise _MappedHttpError(status=resp.status, body_snippet=snippet)


async def _request_once(
    method: str,
    url: str,
    *,
    session: aiohttp.ClientSession,
    json_payload: Optional[Dict[str, Any]],
    timeout_s: float,
) -> Tuple[Optional[int], Optional[str], Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with session.request(
            method, url, json=json_payload, timeout=timeout
        ) as resp:
            status = resp.status
            await _raise_for_status_with_snippet(resp)
            # Success path: attempt to parse JSON body; if not JSON, return text
            try:
                data = await resp.json(content_type=None)
            except aiohttp.ContentTypeError:
                # Not JSON; use limited text
                data = await _read_limited_text(resp)
            return status, None, data
    except asyncio.TimeoutError:
        raise
    except aiohttp.ClientConnectionError as e:
        raise e
    except aiohttp.ClientPayloadError as e:
        # Map as payload error
        raise _MappedHttpError(status=None, body_snippet=str(e))


async def _http_request_core(
    method: str,
    url: str,
    *,
    session: aiohttp.ClientSession,
    json_payload: Optional[Dict[str, Any]],
    timeout_s: float,
    retry_idempotent: bool,
    base: float,
    multiplier: float,
    jitter: float,
    max_sleep: float,
    max_attempts: int,
) -> Any:
    last_exc: Optional[BaseException] = None
    last_status: Optional[int] = None
    last_body: Optional[str] = None

    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            status, body_snippet, data = await _request_once(
                method,
                url,
                session=session,
                json_payload=json_payload,
                timeout_s=timeout_s,
            )
            return data
        except asyncio.TimeoutError as e:
            last_exc = e
            if attempt < attempts - 1 and _should_retry(
                retry_idempotent=retry_idempotent, exc=e
            ):
                await asyncio.sleep(
                    _next_backoff(attempt, base, multiplier, jitter, max_sleep)
                )
                continue
            break
        except aiohttp.ClientConnectionError as e:
            last_exc = e
            if attempt < attempts - 1 and _should_retry(
                retry_idempotent=retry_idempotent, exc=e
            ):
                await asyncio.sleep(
                    _next_backoff(attempt, base, multiplier, jitter, max_sleep)
                )
                continue
            break
        except _MappedHttpError as e:
            last_exc = None
            last_status = e.status
            last_body = e.body_snippet
            if attempt < attempts - 1 and _should_retry(
                retry_idempotent=retry_idempotent, status=e.status
            ):
                await asyncio.sleep(
                    _next_backoff(attempt, base, multiplier, jitter, max_sleep)
                )
                continue
            break

    raise _finalize_attempts(
        method,
        url,
        last_exc=last_exc,
        last_status=last_status,
        last_body_snippet=last_body,
    )


async def _http_request_json(
    self: "ComfyUILocalClient",
    method: str,
    url: str,
    *,
    json_payload: Optional[Dict[str, Any]],
    timeout_s: float,
    retry_idempotent: bool,
) -> Any:
    async with _TempSession(self._session or self._owned_session) as session:
        return await _http_request_core(
            method,
            url,
            session=session,
            json_payload=json_payload,
            timeout_s=timeout_s,
            retry_idempotent=retry_idempotent,
            base=self._retry_base_seconds,
            multiplier=self._retry_multiplier,
            jitter=self._retry_jitter_seconds,
            max_sleep=self._retry_max_sleep_seconds,
            max_attempts=self._retry_max_attempts,
        )


def _join_http_base(base: str, route: str) -> str:
    if not route:
        return base
    return f"{base.rstrip('/')}{route}"


def _join_ws_base(base: str, route: str) -> str:
    if not route:
        return base
    return f"{base.rstrip('/')}{route}"


def _ensure_scheme(base: str, allowed: Tuple[str, ...]) -> None:
    parsed = urlparse(base)
    if not parsed.scheme or parsed.scheme.lower() not in allowed:
        allowed_str = ", ".join(allowed)
        raise ValueError(
            f"Base URL must start with one of [{allowed_str}]; got: {base!r}"
        )


def _validate_bases(http_base: str, ws_base: str) -> None:
    _ensure_scheme(http_base, ("http", "https"))
    _ensure_scheme(ws_base, ("ws", "wss"))


def _quote_client_id(client_id: str) -> str:
    return quote(client_id, safe="")


def _ws_url(base_ws: str, client_id: str) -> str:
    return f"{base_ws.rstrip('/')}/ws?clientId={_quote_client_id(client_id)}"


# Bind helper methods to class namespace (private) without exposing publicly
ComfyUILocalClient._http_request_json = _http_request_json  # type: ignore[attr-defined]
ComfyUILocalClient._join_http = lambda self, route: _join_http_base(self._base_url, route)  # type: ignore[attr-defined]
ComfyUILocalClient._join_ws = lambda self, route: _join_ws_base(self._ws_url, route)  # type: ignore[attr-defined]
