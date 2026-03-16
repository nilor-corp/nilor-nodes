"""RifeStreamVFI: Memory-efficient, chunked RIFE frame interpolation from a URL.

Downloads a video from a presigned URL and runs RIFE N× frame interpolation in
temporal chunks to avoid OOM on large-resolution canvases.

Peak RAM per chunk (float32, chunk_size=64) vs full-video approach:
  OLED 3840×2160  (482f):  ~18 GB/chunk   vs ~137 GB total — use 404S or Linux
  LED_Wall 6144×1952 (482f): ~24 GB/chunk vs ~200 GB+ total — use 404S
  Projection_Centre 5840×1072 (482f): ~14 GB/chunk vs ~108 GB total

Chunking strategy:
  - Overlap of 1 frame at chunk boundaries to ensure seamless interpolation.
  - The first frame of each non-first chunk is the same as the last frame of
    the previous chunk; its RIFE-duplicated counterpart in the output is
    dropped during concatenation to avoid a double frame.

Output frame count for multiplier=2, N input frames:
  2N - 1  (identical to non-chunked RIFE VFI)
"""

from __future__ import annotations

import gc
import io
import typing

import imageio.v2 as imageio
import numpy as np
import requests
import torch
from PIL import Image

from .logger import logger

_DEFAULT_CHUNK_SIZE = 64
_CATEGORY = "Nilor Nodes 👺/Streaming"


def _get_rife_vfi_cls() -> type:
    """Return the RIFE VFI node class from ComfyUI's registered node mappings.

    Deferred import so this module can be loaded at ComfyUI startup without
    depending on the load order of ComfyUI-Frame-Interpolation.

    Returns:
        The RIFE_VFI class.

    Raises:
        RuntimeError: If ComfyUI nodes haven't been loaded or RIFE VFI is absent.
    """
    try:
        import nodes as comfy_nodes  # ComfyUI top-level nodes registry
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import ComfyUI 'nodes' module — is this running inside ComfyUI?"
        ) from exc

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("RIFE VFI")
    if cls is None:
        raise RuntimeError(
            "RIFE VFI node is not registered in ComfyUI NODE_CLASS_MAPPINGS. "
            "Ensure ComfyUI-Frame-Interpolation is installed and loaded."
        )
    return cls


class RifeStreamVFI:
    """Download a video from a presigned URL and run RIFE N× interpolation in chunks.

    Replaces the MediaStreamInput → RIFE VFI node pair for large canvases. Frames
    are loaded as uint8 numpy arrays first (~3× cheaper than float32), then
    converted to float32 one chunk at a time during inference.

    Args:
        presigned_download_url: Presigned URL to the input video.
        input_name: Logical name used in log messages (mirrors MediaStreamInput).
        ckpt_name: RIFE model checkpoint filename.
        multiplier: Frame multiplier (2 = 2× frame rate, i.e. 2N-1 output frames).
        chunk_size: Frames per chunk including 1-frame boundary overlap.
        fast_mode: RIFE fast mode flag.
        ensemble: RIFE ensemble flag (improves quality at minor cost).
        scale_factor: Spatial scale for RIFE optical-flow computation.
        dtype: Inference precision ("float32", "float16", "bfloat16").
        batch_size: GPU batch size per RIFE forward pass.
        clear_cache_after_n_frames: CUDA cache-clear cadence (pairs processed).
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        """Declare ComfyUI inputs for this node."""
        return {
            "required": {
                "presigned_download_url": (
                    "STRING",
                    {"multiline": True, "default": "<auto-filled by system>"},
                ),
                "input_name": (
                    "STRING",
                    {"default": "input_video", "multiline": False},
                ),
                "ckpt_name": (
                    ["rife47.pth", "rife49.pth"],
                    {"default": "rife47.pth"},
                ),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 8}),
                "chunk_size": (
                    "INT",
                    {
                        "default": _DEFAULT_CHUNK_SIZE,
                        "min": 4,
                        "max": 512,
                        "tooltip": (
                            "Frames per processing chunk (1-frame overlap at boundaries). "
                            "Lower = less peak RAM. 64 keeps 4K canvases under ~20 GB."
                        ),
                    },
                ),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                "dtype": (
                    ["float32", "float16", "bfloat16"],
                    {"default": "float32"},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "GPU batch size per RIFE forward pass.",
                    },
                ),
                "clear_cache_after_n_frames": (
                    "INT",
                    {"default": 10, "min": 1, "max": 1000},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "interpolate_chunked"
    CATEGORY = _CATEGORY

    def interpolate_chunked(
        self,
        presigned_download_url: str,
        input_name: str = "input_video",
        ckpt_name: str = "rife47.pth",
        multiplier: int = 2,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        fast_mode: bool = True,
        ensemble: bool = True,
        scale_factor: float = 1.0,
        dtype: str = "float32",
        batch_size: int = 1,
        clear_cache_after_n_frames: int = 10,
    ) -> tuple:
        """Download a video and run chunked RIFE interpolation.

        Args:
            presigned_download_url: URL to download the source video.
            input_name: Logical name for log messages.
            ckpt_name: RIFE checkpoint filename.
            multiplier: Interpolation multiplier (2 = 2× frame rate).
            chunk_size: Frames per chunk (1-frame overlap at boundaries).
            fast_mode: RIFE fast mode toggle.
            ensemble: RIFE ensemble mode toggle.
            scale_factor: Spatial scale for RIFE flow field.
            dtype: Inference precision.
            batch_size: GPU batch size per RIFE call.
            clear_cache_after_n_frames: How often to clear CUDA cache.

        Returns:
            Tuple of one float32 IMAGE tensor shaped [N_out, H, W, 3].
        """
        logger.info(
            "ℹ️\u2009 Nilor-Nodes (RifeStreamVFI): [%s] Starting. "
            "chunk_size=%d multiplier=%d ckpt=%s",
            input_name,
            chunk_size,
            multiplier,
            ckpt_name,
        )

        response = requests.get(presigned_download_url, timeout=300)
        response.raise_for_status()
        video_bytes = response.content

        # Decode all frames into uint8 numpy arrays. uint8 uses ~4× less RAM than
        # float32 and covers the full frame count cheaply before chunk processing.
        raw_frames: list[np.ndarray] = []
        with imageio.get_reader(io.BytesIO(video_bytes), format="mp4") as reader:
            for frame in reader:
                raw_frames.append(np.asarray(Image.fromarray(frame).convert("RGB")))

        del video_bytes

        total_frames = len(raw_frames)
        if total_frames < 2:
            raise ValueError(
                f"🛑\u2009 Nilor-Nodes (RifeStreamVFI): [{input_name}] "
                f"Video has {total_frames} frame(s); need ≥ 2."
            )

        h, w, _ = raw_frames[0].shape
        stride = max(1, chunk_size - 1)
        n_chunks = max(1, (total_frames - 1 + stride - 1) // stride)
        logger.info(
            "ℹ️\u2009 Nilor-Nodes (RifeStreamVFI): [%s] %d frames at %dx%d — "
            "%d chunk(s) of %d (stride %d).",
            input_name,
            total_frames,
            w,
            h,
            n_chunks,
            chunk_size,
            stride,
        )

        rife_cls = _get_rife_vfi_cls()
        rife_node = rife_cls()

        output_chunks: list[torch.Tensor] = []
        chunk_idx = 0
        chunk_start = 0

        while chunk_start < total_frames - 1:
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk_raw = raw_frames[chunk_start:chunk_end]
            n_chunk = len(chunk_raw)

            logger.info(
                "ℹ️\u2009 Nilor-Nodes (RifeStreamVFI): [%s] Chunk %d/%d — "
                "input frames [%d..%d] (%d frames)",
                input_name,
                chunk_idx + 1,
                n_chunks,
                chunk_start,
                chunk_end - 1,
                n_chunk,
            )

            # Build float32 tensor for this chunk: [N, H, W, C]
            chunk_tensor = torch.empty(n_chunk, h, w, 3, dtype=torch.float32)
            for i, frame_np in enumerate(chunk_raw):
                chunk_tensor[i] = torch.from_numpy(frame_np.astype(np.float32) / 255.0)

            (interpolated,) = rife_node.vfi(
                ckpt_name=ckpt_name,
                frames=chunk_tensor,
                multiplier=multiplier,
                fast_mode=fast_mode,
                ensemble=ensemble,
                scale_factor=scale_factor,
                dtype=dtype,
                torch_compile=False,
                batch_size=batch_size,
                clear_cache_after_n_frames=clear_cache_after_n_frames,
            )

            # Skip the first output frame of non-first chunks: it duplicates the
            # last frame of the previous chunk's output (the 1-frame overlap).
            skip = 1 if chunk_idx > 0 else 0
            output_chunks.append(interpolated[skip:].cpu())

            del chunk_tensor, interpolated
            gc.collect()

            chunk_start += stride
            chunk_idx += 1

        total_out = sum(t.shape[0] for t in output_chunks)
        logger.info(
            "✅ Nilor-Nodes (RifeStreamVFI): [%s] Complete — "
            "%d input frames → %d output frames across %d chunk(s).",
            input_name,
            total_frames,
            total_out,
            chunk_idx,
        )

        out = torch.cat(output_chunks, dim=0)
        return (out,)


# ---------------------------------------------------------------------------
# ComfyUI Node Mappings
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: typing.Dict[str, type] = {
    "RifeStreamVFI": RifeStreamVFI,
}

NODE_DISPLAY_NAME_MAPPINGS: typing.Dict[str, str] = {
    "RifeStreamVFI": "👺 RIFE Stream VFI (Chunked, from URL)",
}
