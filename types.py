"""Shared types and enums for the Nilor-Nodes sidecar.

This module intentionally contains minimal placeholders to support the
configuration loader and future extensions without introducing unnecessary
complexity at this stage.
"""

from __future__ import annotations

from enum import Enum


class ConfigSource(Enum):
    """Represents the origin of a configuration value."""

    ENV = "env"
    JSON5 = "json5"
