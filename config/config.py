"""
Typed configuration scaffolding for the Nilor-Nodes ComfyUI sidecar.

This module defines the dataclasses and public loader API contract. The actual
implementation of precedence, parsing, and validation is added in a subsequent
commit. For now, only type definitions and the public `Config.load` signature
are provided to enable incremental integration without behavior changes.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

try:  # json5 is declared in nilor-nodes/requirements.txt
    import json5  # type: ignore
except Exception as _e:  # pragma: no cover
    json5 = None  # lazy failure in loader


@dataclass(frozen=True)
class ComfyApiConfig:
    """Configuration for the ComfyUI client.

    Args:
        api_url: Base HTTP URL for the ComfyUI REST API (e.g., "http://127.0.0.1:8188").
        ws_url: Base WebSocket URL for ComfyUI events (e.g., "ws://127.0.0.1:8188").
        timeout_s: Request timeout in seconds for ComfyUI HTTP calls.
    """

    api_url: str
    ws_url: str
    timeout_s: int


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for the worker and its SQS integration.

    Args:
        sqs_endpoint_url: URL of the SQS-compatible endpoint (e.g., ElasticMQ).
        jobs_queue: Name of the queue from which to pull new jobs.
        status_queue: Name of the queue to which job status updates are published.
        poll_wait_s: Long poll wait time in seconds (expected to be within [0, 20]).
        max_messages: Max number of messages pulled per poll.
        aws_access_key_id: Access key id for the SQS client (non-secret default acceptable for local dev).
        aws_secret_access_key: Secret access key for the SQS client (must be overridden via environment for real deployments).
        aws_region: AWS region name used by the SQS client.
        worker_client_id: Stable identifier for routing websocket events to this worker.
    """

    sqs_endpoint_url: str
    jobs_queue: str
    status_queue: str
    poll_wait_s: int
    max_messages: int
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    worker_client_id: str


@dataclass(frozen=True)
class NilorNodesConfig:
    """Aggregate configuration for the Nilor-Nodes sidecar.

    Args:
        comfy: Configuration for the ComfyUI HTTP/WS client.
        worker: Configuration for SQS and worker identity.
    """

    comfy: ComfyApiConfig
    worker: WorkerConfig


class Config:
    """Public configuration loader API for the Nilor-Nodes sidecar.

    The implementation is intentionally deferred to a subsequent commit. This
    placeholder establishes the method signature and return type to facilitate
    incremental refactors in consumers.
    """

    @staticmethod
    def load(
        env: os._Environ[str] = os.environ, json5_path: Optional[str] = None
    ) -> NilorNodesConfig:
        """Load and return a typed configuration instance.

        Args:
            env: Mapping of environment variables used for overrides.
            json5_path: Optional path to a JSON5 file containing non-secret defaults.

        Returns:
            NilorNodesConfig: The fully parsed and validated configuration object.
        """

        # Read JSON5 file into overrides if provided
        file_overrides: Dict[str, object] = {}
        if json5_path:
            if json5 is None:
                raise RuntimeError(
                    "json5 module is required to load configuration from JSON5 file"
                )
            if os.path.exists(json5_path):
                try:
                    with open(json5_path, "r", encoding="utf-8") as f:
                        loaded = json5.load(f)
                    if not isinstance(loaded, dict):
                        raise ValueError(
                            "JSON5 configuration must be a top-level object"
                        )
                    file_overrides = {str(k): v for k, v in loaded.items()}
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to read JSON5 configuration from '{json5_path}': {type(exc).__name__}: {exc}"
                    )

        # Normalize env into NILOR_* keys with legacy compatibility
        env_values = _normalize_env_to_nilor(env)

        # Precedence: env > file
        effective: Dict[str, object] = dict(file_overrides)
        effective.update(env_values)

        # Parse
        comfy = _parse_comfy_config(effective)
        worker = _parse_worker_config(effective)

        # Validate
        _validate_comfy_config(comfy)
        _validate_worker_config(worker)

        return NilorNodesConfig(comfy=comfy, worker=worker)


# ---- Internal helpers ----


def _normalize_env_to_nilor(env: os._Environ[str]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in env.items():
        if key.startswith("NILOR_"):
            normalized[key] = value
    return normalized


def _parse_comfy_config(values: Dict[str, object]) -> ComfyApiConfig:
    api_url = str(values.get("NILOR_COMFYUI_API_URL", "")).strip()
    ws_url = str(values.get("NILOR_COMFYUI_WS_URL", "")).strip()
    timeout_raw = values.get("NILOR_COMFY_API_TIMEOUT_SECONDS", 30)
    try:
        timeout_s = int(timeout_raw)
    except Exception:
        raise ValueError(
            f"Invalid integer for NILOR_COMFY_API_TIMEOUT_SECONDS: {timeout_raw!r}"
        )
    if not api_url:
        raise ValueError("Missing NILOR_COMFYUI_API_URL (env or JSON5)")
    if not ws_url:
        raise ValueError("Missing NILOR_COMFYUI_WS_URL (env or JSON5)")
    return ComfyApiConfig(api_url=api_url, ws_url=ws_url, timeout_s=timeout_s)


def _parse_worker_config(values: Dict[str, object]) -> WorkerConfig:
    sqs_endpoint_url = str(values.get("NILOR_SQS_ENDPOINT_URL", "")).strip()
    jobs_queue = str(values.get("NILOR_SQS_JOBS_TO_PROCESS_QUEUE_NAME", "")).strip()
    status_queue = str(
        values.get("NILOR_SQS_JOB_STATUS_UPDATES_QUEUE_NAME", "")
    ).strip()

    poll_wait_raw = values.get("NILOR_SQS_POLL_WAIT_TIME", 10)
    max_messages_raw = values.get("NILOR_SQS_MAX_MESSAGES", 1)
    try:
        poll_wait_s = int(poll_wait_raw)
    except Exception:
        raise ValueError(
            f"Invalid integer for NILOR_SQS_POLL_WAIT_TIME: {poll_wait_raw!r}"
        )
    try:
        max_messages = int(max_messages_raw)
    except Exception:
        raise ValueError(
            f"Invalid integer for NILOR_SQS_MAX_MESSAGES: {max_messages_raw!r}"
        )

    aws_access_key_id = str(values.get("NILOR_AWS_ACCESS_KEY_ID", "")).strip()
    aws_secret_access_key = str(values.get("NILOR_AWS_SECRET_ACCESS_KEY", "")).strip()
    aws_region = str(values.get("NILOR_AWS_DEFAULT_REGION", "")).strip()

    worker_client_id = str(values.get("NILOR_WORKER_CLIENT_ID", "")).strip()
    if not worker_client_id:
        worker_client_id = _generate_worker_client_id()

    if not sqs_endpoint_url:
        raise ValueError("Missing NILOR_SQS_ENDPOINT_URL (env or JSON5)")
    if not jobs_queue:
        raise ValueError("Missing NILOR_SQS_JOBS_TO_PROCESS_QUEUE_NAME (env or JSON5)")
    if not status_queue:
        raise ValueError(
            "Missing NILOR_SQS_JOB_STATUS_UPDATES_QUEUE_NAME (env or JSON5)"
        )
    if not aws_access_key_id:
        raise ValueError("Missing NILOR_AWS_ACCESS_KEY_ID (set via env)")
    if not aws_secret_access_key:
        raise ValueError("Missing NILOR_AWS_SECRET_ACCESS_KEY (set via env)")
    if not aws_region:
        raise ValueError("Missing NILOR_AWS_DEFAULT_REGION (env or JSON5)")

    return WorkerConfig(
        sqs_endpoint_url=sqs_endpoint_url,
        jobs_queue=jobs_queue,
        status_queue=status_queue,
        poll_wait_s=poll_wait_s,
        max_messages=max_messages,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
        worker_client_id=worker_client_id,
    )


def _validate_comfy_config(cfg: ComfyApiConfig) -> None:
    _require_url_scheme(cfg.api_url, {"http", "https"}, "NILOR_COMFYUI_API_URL")
    _require_url_scheme(cfg.ws_url, {"ws", "wss"}, "NILOR_COMFYUI_WS_URL")
    if cfg.timeout_s <= 0:
        raise ValueError(
            f"NILOR_COMFY_API_TIMEOUT_SECONDS must be a positive integer; got {cfg.timeout_s}"
        )


def _validate_worker_config(cfg: WorkerConfig) -> None:
    if cfg.poll_wait_s < 0 or cfg.poll_wait_s > 20:
        raise ValueError(
            f"NILOR_SQS_POLL_WAIT_TIME must be within [0, 20]; got {cfg.poll_wait_s}"
        )
    if cfg.max_messages <= 0:
        raise ValueError(
            f"NILOR_SQS_MAX_MESSAGES must be a positive integer; got {cfg.max_messages}"
        )


def _require_url_scheme(url: str, allowed: set[str], key_name: str) -> None:
    parsed = urlparse(url)
    if not parsed.scheme or parsed.scheme.lower() not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(
            f"{key_name} must start with one of [{allowed_str}]; got: {url!r}"
        )


def _generate_worker_client_id() -> str:
    host = socket.gethostname().strip() or "worker"
    suffix = _random_base36_suffix(5)
    return f"nilor-worker-{host}-{suffix}"


def _random_base36_suffix(length: int = 5) -> str:
    import random

    n = random.getrandbits(32)
    base36 = _to_base36(n)
    return base36[-length:]


def _to_base36(n: int) -> str:
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    sign = "-" if n < 0 else ""
    n = abs(n)
    res = []
    while n:
        n, r = divmod(n, 36)
        res.append(digits[r])
    return sign + "".join(reversed(res))
