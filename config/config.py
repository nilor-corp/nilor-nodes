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


class BaseConfig:  # type: ignore
    @classmethod
    def get_instance(cls):
        # Minimal fallback: load JSON5 directly when BaseConfig is unavailable
        path = os.path.join(os.path.dirname(__file__), "config.json5")
        with open(path, "r", encoding="utf-8") as f:
            data = json5.load(f) if json5 is not None else {}
        return cls.from_dict(data)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ComfyApiConfig:
    """Configuration for the ComfyUI client.

    Args:
        api_url: Base HTTP URL for the ComfyUI REST API (e.g., "http://127.0.0.1:8188").
        ws_url: Base WebSocket URL for ComfyUI events (e.g., "ws://127.0.0.1:8188").
        timeout_s: Request timeout in seconds for ComfyUI HTTP calls.
        client_enabled: Feature flag to enable the thin client usage.
        retry_base_seconds: Base backoff seconds for idempotent retries.
        retry_multiplier: Exponential backoff multiplier.
        retry_jitter_seconds: Jitter range (±seconds) added to backoff.
        retry_max_sleep_seconds: Maximum sleep per backoff step.
        retry_max_attempts: Maximum retry attempts for idempotent routes.
        ws_max_reconnect_attempts: Maximum websocket reconnect attempts.
        ws_max_total_backoff_seconds: Cap on total backoff time during WS reconnects.
    """

    api_url: str
    ws_url: str
    timeout_s: int
    client_enabled: bool
    retry_base_seconds: float
    retry_multiplier: float
    retry_jitter_seconds: float
    retry_max_sleep_seconds: float
    retry_max_attempts: int
    ws_max_reconnect_attempts: int
    ws_max_total_backoff_seconds: float


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


@dataclass
class NilorNodesConfig(BaseConfig):
    """Aggregate configuration for the Nilor-Nodes sidecar.

    Args:
        comfy: Configuration for the ComfyUI HTTP/WS client.
        worker: Configuration for SQS and worker identity.
        allow_env_override: When true, environment variables may override file values.
    """

    comfy: ComfyApiConfig
    worker: WorkerConfig
    allow_env_override: bool
    sqs_enabled: bool

    @classmethod
    def _get_config_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), "config.json5")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, object]) -> "NilorNodesConfig":
        allow_env_override = bool(config_dict.get("allow_env_override", True))
        sqs_enabled = _coerce_bool(config_dict.get("NILOR_SQS_ENABLED", False))

        # Build nested from flat NILOR_* keys present in JSON5
        comfy_cfg = ComfyApiConfig(
            api_url=str(config_dict.get("NILOR_COMFYUI_API_URL", "")).strip(),
            ws_url=str(config_dict.get("NILOR_COMFYUI_WS_URL", "")).strip(),
            timeout_s=int(config_dict.get("NILOR_COMFY_API_TIMEOUT_SECONDS", 30)),
            client_enabled=_coerce_bool(
                config_dict.get("NILOR_COMFY_CLIENT_ENABLED", True)
            ),
            retry_base_seconds=float(
                config_dict.get("NILOR_COMFY_RETRY_BASE_SECONDS", 0.25)
            ),
            retry_multiplier=float(
                config_dict.get("NILOR_COMFY_RETRY_MULTIPLIER", 2.0)
            ),
            retry_jitter_seconds=float(
                config_dict.get("NILOR_COMFY_RETRY_JITTER_SECONDS", 0.25)
            ),
            retry_max_sleep_seconds=float(
                config_dict.get("NILOR_COMFY_RETRY_MAX_SLEEP_SECONDS", 4.0)
            ),
            retry_max_attempts=int(
                config_dict.get("NILOR_COMFY_RETRY_MAX_ATTEMPTS", 3)
            ),
            ws_max_reconnect_attempts=int(
                config_dict.get("NILOR_COMFY_WS_MAX_RECONNECT_ATTEMPTS", 5)
            ),
            ws_max_total_backoff_seconds=float(
                config_dict.get("NILOR_COMFY_WS_MAX_TOTAL_BACKOFF_SECONDS", 30.0)
            ),
        )

        worker_client_id = (
            str(config_dict.get("NILOR_WORKER_CLIENT_ID", "")).strip()
            or _generate_worker_client_id()
        )

        worker_cfg = WorkerConfig(
            sqs_endpoint_url=str(config_dict.get("NILOR_SQS_ENDPOINT_URL", "")).strip(),
            jobs_queue=str(
                config_dict.get("NILOR_SQS_JOBS_TO_PROCESS_QUEUE_NAME", "")
            ).strip(),
            status_queue=str(
                config_dict.get("NILOR_SQS_JOB_STATUS_UPDATES_QUEUE_NAME", "")
            ).strip(),
            poll_wait_s=int(config_dict.get("NILOR_SQS_POLL_WAIT_TIME", 10)),
            max_messages=int(config_dict.get("NILOR_SQS_MAX_MESSAGES", 1)),
            aws_access_key_id=str(
                config_dict.get("NILOR_AWS_ACCESS_KEY_ID", "")
            ).strip(),
            aws_secret_access_key=str(
                config_dict.get("NILOR_AWS_SECRET_ACCESS_KEY", "")
            ).strip(),
            aws_region=str(config_dict.get("NILOR_AWS_DEFAULT_REGION", "")).strip(),
            worker_client_id=worker_client_id,
        )

        cfg = cls(
            comfy=comfy_cfg,
            worker=worker_cfg,
            allow_env_override=allow_env_override,
            sqs_enabled=sqs_enabled,
        )
        _validate_comfy_config(cfg.comfy)
        _validate_worker_config(cfg.worker)
        return cfg


# ---- Internal helpers ----


def _apply_env_overrides(cfg: NilorNodesConfig) -> None:
    if not cfg.allow_env_override:
        return

    # Feature flags
    sqs_enabled = os.getenv("NILOR_SQS_ENABLED", cfg.sqs_enabled)
    cfg.sqs_enabled = _coerce_bool(sqs_enabled)

    # Comfy (rebuild frozen dataclass)
    comfy_api_url = os.getenv("NILOR_COMFYUI_API_URL", cfg.comfy.api_url)
    comfy_ws_url = os.getenv("NILOR_COMFYUI_WS_URL", cfg.comfy.ws_url)
    comfy_timeout_s = int(
        os.getenv("NILOR_COMFY_API_TIMEOUT_SECONDS", cfg.comfy.timeout_s)
    )
    comfy_client_enabled = _coerce_bool(
        os.getenv("NILOR_COMFY_CLIENT_ENABLED", cfg.comfy.client_enabled)
    )
    comfy_retry_base = float(
        os.getenv("NILOR_COMFY_RETRY_BASE_SECONDS", cfg.comfy.retry_base_seconds)
    )
    comfy_retry_multiplier = float(
        os.getenv("NILOR_COMFY_RETRY_MULTIPLIER", cfg.comfy.retry_multiplier)
    )
    comfy_retry_jitter = float(
        os.getenv("NILOR_COMFY_RETRY_JITTER_SECONDS", cfg.comfy.retry_jitter_seconds)
    )
    comfy_retry_max_sleep = float(
        os.getenv(
            "NILOR_COMFY_RETRY_MAX_SLEEP_SECONDS", cfg.comfy.retry_max_sleep_seconds
        )
    )
    comfy_retry_max_attempts = int(
        os.getenv("NILOR_COMFY_RETRY_MAX_ATTEMPTS", cfg.comfy.retry_max_attempts)
    )
    comfy_ws_max_reconnect = int(
        os.getenv(
            "NILOR_COMFY_WS_MAX_RECONNECT_ATTEMPTS",
            cfg.comfy.ws_max_reconnect_attempts,
        )
    )
    comfy_ws_max_total_backoff = float(
        os.getenv(
            "NILOR_COMFY_WS_MAX_TOTAL_BACKOFF_SECONDS",
            cfg.comfy.ws_max_total_backoff_seconds,
        )
    )
    cfg.comfy = ComfyApiConfig(
        api_url=str(comfy_api_url),
        ws_url=str(comfy_ws_url),
        timeout_s=comfy_timeout_s,
        client_enabled=comfy_client_enabled,
        retry_base_seconds=comfy_retry_base,
        retry_multiplier=comfy_retry_multiplier,
        retry_jitter_seconds=comfy_retry_jitter,
        retry_max_sleep_seconds=comfy_retry_max_sleep,
        retry_max_attempts=comfy_retry_max_attempts,
        ws_max_reconnect_attempts=comfy_ws_max_reconnect,
        ws_max_total_backoff_seconds=comfy_ws_max_total_backoff,
    )

    # Worker (rebuild frozen dataclass)
    worker_sqs_endpoint_url = os.getenv(
        "NILOR_SQS_ENDPOINT_URL", cfg.worker.sqs_endpoint_url
    )
    worker_jobs_queue = os.getenv(
        "NILOR_SQS_JOBS_TO_PROCESS_QUEUE_NAME", cfg.worker.jobs_queue
    )
    worker_status_queue = os.getenv(
        "NILOR_SQS_JOB_STATUS_UPDATES_QUEUE_NAME", cfg.worker.status_queue
    )
    worker_poll_wait_s = int(
        os.getenv("NILOR_SQS_POLL_WAIT_TIME", cfg.worker.poll_wait_s)
    )
    worker_max_messages = int(
        os.getenv("NILOR_SQS_MAX_MESSAGES", cfg.worker.max_messages)
    )
    worker_access_key = os.getenv(
        "NILOR_AWS_ACCESS_KEY_ID", cfg.worker.aws_access_key_id
    )
    worker_secret_key = os.getenv(
        "NILOR_AWS_SECRET_ACCESS_KEY", cfg.worker.aws_secret_access_key
    )
    worker_region = os.getenv("NILOR_AWS_DEFAULT_REGION", cfg.worker.aws_region)
    worker_client_id = os.getenv("NILOR_WORKER_CLIENT_ID", cfg.worker.worker_client_id)
    cfg.worker = WorkerConfig(
        sqs_endpoint_url=str(worker_sqs_endpoint_url),
        jobs_queue=str(worker_jobs_queue),
        status_queue=str(worker_status_queue),
        poll_wait_s=worker_poll_wait_s,
        max_messages=worker_max_messages,
        aws_access_key_id=str(worker_access_key),
        aws_secret_access_key=str(worker_secret_key),
        aws_region=str(worker_region),
        worker_client_id=str(worker_client_id),
    )

    # Re-validate after overrides
    _validate_comfy_config(cfg.comfy)
    _validate_worker_config(cfg.worker)


def _validate_comfy_config(cfg: ComfyApiConfig) -> None:
    _require_url_scheme(cfg.api_url, {"http", "https"}, "NILOR_COMFYUI_API_URL")
    _require_url_scheme(cfg.ws_url, {"ws", "wss"}, "NILOR_COMFYUI_WS_URL")
    if cfg.timeout_s <= 0:
        raise ValueError(
            f"NILOR_COMFY_API_TIMEOUT_SECONDS must be a positive integer; got {cfg.timeout_s}"
        )
    if (
        cfg.retry_base_seconds < 0
        or cfg.retry_multiplier <= 0
        or cfg.retry_max_sleep_seconds <= 0
    ):
        raise ValueError("Invalid retry backoff parameters in Comfy client config")
    if cfg.retry_max_attempts <= 0:
        raise ValueError("NILOR_COMFY_RETRY_MAX_ATTEMPTS must be a positive integer")
    if cfg.ws_max_reconnect_attempts < 0 or cfg.ws_max_total_backoff_seconds < 0:
        raise ValueError(
            "Invalid websocket reconnect parameters in Comfy client config"
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


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(text)


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


def load_nilor_nodes_config() -> NilorNodesConfig:
    """Convenience loader aligning with brain_rnd config pattern.

    - Loads environment variables via python-dotenv if available
    - Reads JSON5 defaults and applies env overrides when enabled
    - Returns a typed `NilorNodesConfig`
    """
    try:
        from dotenv import load_dotenv  # optional dependency present in sidecar

        load_dotenv()
    except Exception:
        pass
    # Use BaseConfig-backed singleton load
    if json5 is None:
        raise RuntimeError(
            "json5 module is required to load configuration from JSON5 file"
        )
    cfg = NilorNodesConfig.get_instance()  # type: ignore[attr-defined]
    _apply_env_overrides(cfg)
    return cfg
