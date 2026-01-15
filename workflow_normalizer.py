"""
Workflow normalization helpers.

Goal: accept ComfyUI "prompt" (API workflow graph) authored on a different OS and
rewrite OS-sensitive path formatting (primarily path separators) so that the
local ComfyUI instance can resolve model filenames correctly.

This is intentionally conservative: we only rewrite strings that look like file
paths / model names (e.g. end in ".safetensors") and we avoid touching URLs and
free-form prompt text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

__all__ = [
    "PathRemap",
    "normalize_comfyui_prompt_for_current_os",
]


@dataclass(frozen=True)
class PathRemap:
    """Prefix remap for absolute paths across OSes.

    Example:
        PathRemap(from_prefix="D:\\ComfyUI\\models", to_prefix="/mnt/models")

    Matching is performed in a canonicalized form (both prefixes and candidate
    paths have backslashes converted to forward slashes). After remapping, path
    separators are normalized for the current OS.
    """

    from_prefix: str
    to_prefix: str


_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

# Common file extensions encountered in ComfyUI prompts (models + media).
_PATH_EXTS = {
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".onnx",
    ".json",
    ".json5",
    ".yaml",
    ".yml",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".exr",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".wav",
    ".mp3",
    ".flac",
}


def _canonicalize_for_prefix_match(path: str) -> str:
    # Use forward slashes for prefix matching regardless of host OS.
    return path.replace("\\", "/")


def _normalize_separators_for_current_os(path: str) -> str:
    # ComfyUI often uses OS-native separators in its model name registry.
    # We normalize to the current OS so lookup keys match.
    if os.name == "nt":
        return path.replace("/", "\\")
    return path.replace("\\", "/")


def _looks_like_path_value(s: str) -> bool:
    s_stripped = s.strip()
    if not s_stripped:
        return False

    # Don't touch URLs (presigned uploads, http inputs, etc.)
    if _URL_RE.match(s_stripped):
        return False

    # Don't touch placeholder tokens used by the system.
    if s_stripped.startswith("<") and s_stripped.endswith(">"):
        return False

    # Avoid common sentinel.
    if s_stripped == "None":
        return False

    lower = s_stripped.lower()
    if any(lower.endswith(ext) for ext in _PATH_EXTS):
        return True

    # Absolute Windows paths even without extensions.
    if _WIN_DRIVE_RE.match(s_stripped):
        return True

    # Relative paths with explicit prefixes.
    if s_stripped.startswith(("./", "../", "~/", "~\\")):
        return True

    return False


def _apply_prefix_remaps(path: str, remaps: Iterable[PathRemap]) -> str:
    if not remaps:
        return path

    cand = _canonicalize_for_prefix_match(path)
    for r in remaps:
        frm = _canonicalize_for_prefix_match(str(r.from_prefix))
        if cand.startswith(frm):
            to = str(r.to_prefix)
            # Replace using canonical representation then return in that form;
            # the caller will normalize separators for current OS afterwards.
            replaced = to + cand[len(frm) :]
            return replaced
    return path


def normalize_comfyui_prompt_for_current_os(
    prompt: Any, *, path_remaps: Iterable[PathRemap] | None = None
) -> Tuple[Any, int]:
    """Normalize a ComfyUI API `prompt` graph for the current OS.

    Args:
        prompt: The value of the `/prompt` payload's `prompt` key. Typically a
            dict mapping node ids to `{class_type, inputs, ...}`.
        path_remaps: Optional prefix remaps applied before separator
            normalization (useful for absolute paths).

    Returns:
        (normalized_prompt, num_rewritten_strings)
    """

    remaps: List[PathRemap] = list(path_remaps or [])
    rewritten = 0

    def walk(x: Any) -> Any:
        nonlocal rewritten
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, tuple):
            return tuple(walk(v) for v in x)
        if isinstance(x, str) and _looks_like_path_value(x):
            y = _apply_prefix_remaps(x, remaps)
            y = _normalize_separators_for_current_os(y)
            if y != x:
                rewritten += 1
            return y
        return x

    return walk(prompt), rewritten

