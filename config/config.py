"""
Typed configuration scaffolding for the Nilor-Nodes ComfyUI sidecar.

This module defines the dataclasses and public loader API contract. The actual
implementation of precedence, parsing, and validation is added in a subsequent
commit. For now, only type definitions and the public `Config.load` signature
are provided to enable incremental integration without behavior changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


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

        Note:
            The actual precedence (env > JSON5), parsing, validation, and worker id
            generation will be implemented in the next commit.
        """

        raise NotImplementedError(
            "Config.load implementation is added in a subsequent commit."
        )
