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
import asyncio
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


@dataclass(frozen=True)
class DerivedStats:
    """Computed metrics used by the policy engine.

    Attributes:
        vram_total: Total VRAM (bytes) if known.
        vram_free: Free VRAM (bytes) if known.
        vram_used_pct: Percent VRAM used in [0, 100] when computable.
        ram_total: Total RAM (bytes) if known.
        ram_free: Free RAM (bytes) if known.
        ram_used_pct: Percent RAM used in [0, 100] when computable.
    """

    vram_total: Optional[float]
    vram_free: Optional[float]
    vram_used_pct: Optional[float]
    ram_total: Optional[float]
    ram_free: Optional[float]
    ram_used_pct: Optional[float]


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
        self._cooldown_until: float = 0.0
        # Hardened capability state: disable guardian for session after unsupported
        self._capability_disabled: bool = False
        self._unsupported_warned: bool = False

    async def check_and_remediate(self) -> RemediationResult:
        """Run a single hygiene cycle.

        This skeleton performs capability and enablement checks, then captures a
        baseline stats snapshot and returns without modification. Subsequent
        commits implement policy evaluation and remediation.
        """
        start_ts = time.monotonic()

        if not self._cfg.enabled:
            before, _ = await self._collect_metrics()
            try:
                if self._logger:
                    self._logger.debug(
                        "⚠️\u2009 Nilor-Nodes (memory_hygiene): disabled; skipping remediation. vram_free=%s ram_free=%s",
                        getattr(before, "vram_free", None),
                        getattr(before, "ram_free", None),
                    )
            except Exception:
                pass
            return RemediationResult(
                before=before,
                after=None,
                action="none",
                attempts=0,
                elapsed_seconds=max(0.0, time.monotonic() - start_ts),
                reason="disabled",
                success=True,
            )

        # Session-wide disable if previously detected unsupported endpoints
        if self._capability_disabled:
            return RemediationResult(
                before=await self._get_stats_safe(),
                after=None,
                action="none",
                attempts=0,
                elapsed_seconds=max(0.0, time.monotonic() - start_ts),
                reason="capability_unsupported",
                success=True,
            )

        try:
            supported = await self._client.supports_hygiene()
        except Exception:
            supported = False

        before, derived = await self._collect_metrics()

        if not supported:
            # Disable for the rest of the session and emit a single warning
            self._capability_disabled = True
            if not self._unsupported_warned:
                try:
                    if self._logger:
                        self._logger.warning(
                            "⚠️\u2009 Nilor-Nodes (memory_hygiene): unsupported endpoints; disabling for this session."
                        )
                except Exception:
                    pass
                self._unsupported_warned = True
            return RemediationResult(
                before=before,
                after=None,
                action="none",
                attempts=0,
                elapsed_seconds=max(0.0, time.monotonic() - start_ts),
                reason="capability_unsupported",
                success=True,
            )

        now = time.monotonic()
        if now < self._cooldown_until:
            try:
                if self._logger:
                    self._logger.debug(
                        "ℹ️\u2009 Nilor-Nodes (memory_hygiene): cooldown active for %.2fs; skipping.",
                        max(0.0, self._cooldown_until - now),
                    )
            except Exception:
                pass
            return RemediationResult(
                before=before,
                after=None,
                action="none",
                attempts=0,
                elapsed_seconds=max(0.0, now - start_ts),
                reason="cooldown_active",
                success=True,
            )

        pressure_reason = self._pressure_reason(derived)
        if pressure_reason is None:
            try:
                if self._logger:
                    self._logger.debug(
                        "ℹ️\u2009 Nilor-Nodes (memory_hygiene): no pressure; nothing to do."
                    )
            except Exception:
                pass
            return RemediationResult(
                before=before,
                after=None,
                action="none",
                attempts=0,
                elapsed_seconds=max(0.0, time.monotonic() - start_ts),
                reason="no_pressure",
                success=True,
            )

        action = self._choose_initial_action(self._cfg.action_policy)
        try:
            if self._logger:
                self._logger.info(
                    "✅ Nilor-Nodes (memory_hygiene): start cycle action=%s reason=%s vram_free=%s ram_free=%s",
                    action,
                    pressure_reason,
                    getattr(before, "vram_free", None),
                    getattr(before, "ram_free", None),
                )
        except Exception:
            pass
        after, attempts, final_action, outcome_reason, success = (
            await self._remediate_cycle(
                initial_action=action,
                initial_reason=pressure_reason,
                cycle_start=start_ts,
            )
        )
        # Set cooldown after any attempted cycle (success or not)
        self._cooldown_until = time.monotonic() + float(self._cfg.cooldown_seconds)
        try:
            if self._logger:
                self._logger.info(
                    "✅ Nilor-Nodes (memory_hygiene): end cycle action=%s attempts=%s success=%s reason=%s vram_free_before=%s vram_free_after=%s ram_free_before=%s ram_free_after=%s elapsed=%.2fs",
                    final_action,
                    attempts,
                    success,
                    outcome_reason,
                    getattr(before, "vram_free", None),
                    getattr(after, "vram_free", None) if after else None,
                    getattr(before, "ram_free", None),
                    getattr(after, "ram_free", None) if after else None,
                    max(0.0, time.monotonic() - start_ts),
                )
        except Exception:
            pass
        return RemediationResult(
            before=before,
            after=after,
            action=final_action,
            attempts=attempts,
            elapsed_seconds=max(0.0, time.monotonic() - start_ts),
            reason=outcome_reason,
            success=success,
        )

    async def _get_stats_safe(self) -> SystemStats:
        try:
            return await self._client.get_system_stats()
        except Exception:
            # Return an empty struct; callers tolerate partial data
            return SystemStats()

    async def _collect_metrics(self) -> tuple[SystemStats, DerivedStats]:
        base = await self._get_stats_safe()
        vram_used_pct = _compute_used_pct(base.vram_total, base.vram_free)
        ram_used_pct = _compute_used_pct(base.ram_total, base.ram_free)
        derived = DerivedStats(
            vram_total=base.vram_total,
            vram_free=base.vram_free,
            vram_used_pct=vram_used_pct,
            ram_total=base.ram_total,
            ram_free=base.ram_free,
            ram_used_pct=ram_used_pct,
        )
        return base, derived

    def _pressure_reason(self, d: DerivedStats) -> Optional[str]:
        if _gt_pct(d.vram_used_pct, self._cfg.vram_usage_pct_max):
            return f"vram_used_pct {d.vram_used_pct}% > {self._cfg.vram_usage_pct_max}%"
        if _lt_bytes(d.vram_free, self._cfg.vram_min_free_mb):
            return f"vram_free below {self._cfg.vram_min_free_mb}MB"
        if _gt_pct(d.ram_used_pct, self._cfg.ram_usage_pct_max):
            return f"ram_used_pct {d.ram_used_pct}% > {self._cfg.ram_usage_pct_max}%"
        if _lt_bytes(d.ram_free, self._cfg.ram_min_free_mb):
            return f"ram_free below {self._cfg.ram_min_free_mb}MB"
        return None

    def _choose_initial_action(self, policy: str) -> RemediationAction:
        return _initial_action_for(policy)

    async def _remediate_cycle(
        self,
        *,
        initial_action: RemediationAction,
        initial_reason: str,
        cycle_start: float,
    ) -> tuple[Optional[SystemStats], int, RemediationAction, str, bool]:
        attempts = 0
        action = initial_action
        escalated = False
        last_stats: Optional[SystemStats] = None

        max_retries = max(0, int(self._cfg.max_retries))
        sleep_between = max(0, int(self._cfg.sleep_between_attempts_seconds))
        max_cycle_s = max(0, int(self._cfg.max_cycle_duration_seconds))

        def time_budget_exhausted() -> bool:
            if max_cycle_s <= 0:
                return False
            return (time.monotonic() - cycle_start) >= max_cycle_s

        outcome_reason = initial_reason

        while True:
            # Execute remediation step
            free_flag, unload_flag = _flags_for_action(action)
            try:
                if self._logger:
                    try:
                        self._logger.debug(
                            "ℹ️\u2009 Nilor-Nodes (memory_hygiene): calling /free free_memory=%s unload_models=%s",
                            free_flag,
                            unload_flag,
                        )
                    except Exception:
                        pass
                await self._client.free(
                    free_memory=free_flag, unload_models=unload_flag
                )
            except Exception:
                # Continue even on errors; treat as unsuccessful attempt
                pass

            # Wait and re-measure
            if sleep_between > 0:
                try:
                    await asyncio.sleep(sleep_between)
                except Exception:
                    pass

            last, derived = await self._collect_metrics()
            last_stats = last
            if self._pressure_reason(derived) is None:
                outcome_reason = "targets_met"
                return last_stats, attempts + 1, action, outcome_reason, True

            attempts += 1
            if attempts > max_retries:
                outcome_reason = "max_retries_exhausted"
                return last_stats, attempts, action, outcome_reason, False

            if time_budget_exhausted():
                outcome_reason = "max_duration_reached"
                return last_stats, attempts, action, outcome_reason, False

            # Escalation logic for auto: free -> unload (once)
            if initial_action == "auto" and not escalated:
                action = "unload"
                escalated = True
            # For explicit free/unload/both, keep same action for subsequent attempts


def _compute_used_pct(total: Optional[float], free: Optional[float]) -> Optional[float]:
    try:
        if total is None or free is None:
            return None
        total_f = float(total)
        free_f = float(free)
        if total_f <= 0:
            return None
        used = max(0.0, min(1.0, (total_f - max(0.0, free_f)) / total_f))
        return round(used * 100.0, 2)
    except Exception:
        return None


def _mb_to_bytes(mb: Optional[int]) -> Optional[float]:
    if mb is None:
        return None
    try:
        return float(max(0, int(mb))) * 1024.0 * 1024.0
    except Exception:
        return None


def _gt_pct(value: Optional[float], threshold_pct: Optional[int]) -> bool:
    if value is None or threshold_pct is None:
        return False
    try:
        return float(value) > float(threshold_pct)
    except Exception:
        return False


def _lt_bytes(value_bytes: Optional[float], threshold_mb: Optional[int]) -> bool:
    if value_bytes is None or threshold_mb is None:
        return False
    thr = _mb_to_bytes(threshold_mb)
    if thr is None:
        return False
    try:
        return float(value_bytes) < float(thr)
    except Exception:
        return False


def _normalize_policy(policy: str) -> str:
    try:
        return str(policy).strip().lower()
    except Exception:
        return "auto"


def _action_literal(policy: str) -> RemediationAction:
    p = _normalize_policy(policy)
    if p in ("free", "unload", "both"):
        return p  # type: ignore[return-value]
    return "auto"


def _initial_action_for(policy: str) -> RemediationAction:
    lit = _action_literal(policy)
    if lit == "auto":
        return "free"
    return lit


def _flags_for_action(action: RemediationAction) -> tuple[bool, bool]:
    if action == "free":
        return True, False
    if action == "unload":
        return False, True
    if action == "both":
        return True, True
    # auto is staged; when executing a step, treat like free unless escalated update chooses unload
    return True, False


__all__ = [
    "RemediationResult",
    "RemediationAction",
    "MemoryHygiene",
    "DerivedStats",
]
