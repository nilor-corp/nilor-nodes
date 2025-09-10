import threading

# --- Nilor-Nodes Custom Node Registration and Startup ---
# This file is executed when ComfyUI starts and discovers this custom node directory.
# It's responsible for:
# 1. Starting background services (like the SQS worker and a FastAPI server).
# 2. Registering the custom nodes with ComfyUI so they appear in the menu.

# --- Background Services ---

# 1. Start the SQS Worker Consumer
import asyncio
from .worker_consumer import consume_jobs


def start_consumer_loop():
    """Synchronous wrapper to run the asyncio event loop for the consumer."""
    asyncio.run(consume_jobs())


consumer_thread = threading.Thread(target=start_consumer_loop, daemon=True)
consumer_thread.start()
print("✅ Nilor-Nodes: SQS worker consumer thread started.")

# --- Node Registration ---
from .nilornodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .media_stream import (
    NODE_CLASS_MAPPINGS as ms_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ms_NODE_DISPLAY_NAME_MAPPINGS,
)
from .user_input import (
    NODE_CLASS_MAPPINGS as ui_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ui_NODE_DISPLAY_NAME_MAPPINGS,
)
from .controllers import (
    NODE_CLASS_MAPPINGS as ctrl_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ctrl_NODE_DISPLAY_NAME_MAPPINGS,
)
from .model_loaders import (
    NilorModelLoader_Diffusion,
    NilorModelLoader_Lora,
    NilorModelLoader_Clip,
    NilorModelLoader_TextEncoder,
    NilorModelLoader_VAE,
)

NODE_CLASS_MAPPINGS = {
    "NilorModelLoader_Diffusion": NilorModelLoader_Diffusion,
    "NilorModelLoader_Lora": NilorModelLoader_Lora,
    "NilorModelLoader_Clip": NilorModelLoader_Clip,
    "NilorModelLoader_TextEncoder": NilorModelLoader_TextEncoder,
    "NilorModelLoader_VAE": NilorModelLoader_VAE,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NilorModelLoader_Diffusion": "👺 Nilor Get Diffusion Model",
    "NilorModelLoader_Lora": "👺 Nilor Get Lora Model",
    "NilorModelLoader_Clip": "👺 Nilor Get Clip Model",
    "NilorModelLoader_TextEncoder": "👺 Nilor Get Text Encoder Model",
    "NilorModelLoader_VAE": "👺 Nilor Get VAE Model",
}

NODE_CLASS_MAPPINGS.update(ms_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ms_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ui_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ui_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ctrl_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ctrl_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
print("✅ Nilor-Nodes: All custom nodes registered.")
