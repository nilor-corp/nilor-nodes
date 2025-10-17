"""
Memory Hygiene (Guardian) — typed skeleton and public API.

This module provides a lightweight, dependency-injected component that can be
invoked between jobs to assess memory pressure and (optionally) remediate via
the ComfyUI server's `/free` endpoint. This commit introduces the types and
public API only; detailed policy and remediation logic will be implemented in
subsequent commits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Literal, Any

from .config.config import MemoryHygieneConfig
from .comfyui_client import ComfyUIClientProtocol, SystemStats


RemediationAction = Literal["none", "free", "unload", "both", "auto"]


@dataclass(frozen=True)
class RemediationResult:
    """Outcome of a hygiene check/remediation cycle.

    Attributes:
        before: Stats captured before any remediation attempt.
        after: Stats captured after remediation (if attempted), or None.
        action: Action that was selected/executed for this cycle.
        attempts: Number of remediation attempts performed.
        elapsed_seconds: Wall-clock duration of the cycle in seconds.
        reason: Optional decision rationale (e.g., threshold that triggered).
        success: True when targets were met or no action was needed.
    """

    before: SystemStats
    after: Optional[SystemStats]
    action: RemediationAction
    attempts: int
    elapsed_seconds: float
    reason: Optional[str]
    success: bool


class MemoryHygiene:
    """Memory Guardian component orchestrating checks and remediation between jobs."""

    def __init__(
        self,
        *,
        client: ComfyUIClientProtocol,
        cfg: MemoryHygieneConfig,
        logger: Optional[Any] = None,
    ) -> None:
        self._client = client
        self._cfg = cfg
        self._logger = logger

    async def check_and_remediate(self) -> RemediationResult:
        """Run a single hygiene cycle.

        This skeleton performs capability and enablement checks, then captures a
        baseline stats snapshot and returns without modification. Subsequent
        commits implement policy evaluation and remediation.
        """
        start_ts = time.monotonic()

        if not self._cfg.enabled:
            before = await self._get_stats_safe()
            return RemediationResult(
                before=before,
                after=None,
                action="none",
                attempts=0,
                elapsed_seconds=max(0.0, time.monotonic() - start_ts),
                reason="disabled",
                success=True,
            )

        try:
            supported = await self._client.supports_hygiene()
        except Exception:
            supported = False

        before = await self._get_stats_safe()

        if not supported:
            return RemediationResult(
                before=before,
                after=None,
                action="none",
                attempts=0,
                elapsed_seconds=max(0.0, time.monotonic() - start_ts),
                reason="capability_unsupported",
                success=True,
            )

        # Placeholder: policy decision and remediation will be implemented later
        return RemediationResult(
            before=before,
            after=None,
            action="none",
            attempts=0,
            elapsed_seconds=max(0.0, time.monotonic() - start_ts),
            reason="policy_not_implemented",
            success=True,
        )

    async def _get_stats_safe(self) -> SystemStats:
        try:
            return await self._client.get_system_stats()
        except Exception:
            # Return an empty struct; callers tolerate partial data
            return SystemStats()


__all__ = [
    "RemediationResult",
    "RemediationAction",
    "MemoryHygiene",
]
