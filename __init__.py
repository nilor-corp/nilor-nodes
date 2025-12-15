import os
import threading
import asyncio
import logging
from dotenv import load_dotenv

# --- Nilor-Nodes Custom Node Registration and Startup ---
# This file is executed when ComfyUI starts and discovers this custom node directory.
# It's responsible for:
# 1. Starting background services (like the SQS worker and a FastAPI server).
# 2. Registering the custom nodes with ComfyUI so they appear in the menu.


# --- Load Environment Variables ---
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file
dotenv_path = os.path.join(current_dir, ".env")
# Load the .env file, overriding any pre-existing process env for these keys
load_dotenv(dotenv_path=dotenv_path, override=True)


# --- Package Logger (applied early for all nilor-nodes modules) ---
from .logger import configure_from_env, logger

configure_from_env()


# --- Background Services ---


def start_consumer_loop():
    """Synchronous wrapper to run the asyncio event loop for the consumer."""
    from .worker_consumer import consume_jobs

    asyncio.run(consume_jobs())


from .config.config import load_nilor_nodes_config

cfg = load_nilor_nodes_config()

# Start the SQS Worker Consumer (controlled by NILOR_SQS_ENABLED)
if cfg.sqs_enabled:
    consumer_thread = threading.Thread(target=start_consumer_loop, daemon=True)
    consumer_thread.start()
    print(
        "✅ Nilor-Nodes: SQS worker consumer thread started (NILOR_SQS_ENABLED=true)."
    )
else:
    print(
        "⚠️ Nilor-Nodes: SQS worker consumer functionality is disabled (NILOR_SQS_ENABLED=false)."
    )


# --- Node Registration ---
from .nilornodes import (
    NODE_CLASS_MAPPINGS as base_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as base_NODE_DISPLAY_NAME_MAPPINGS,
)
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

NODE_CLASS_MAPPINGS = dict(base_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(base_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ms_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ms_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ui_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ui_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ctrl_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ctrl_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
print("✅ Nilor-Nodes: All custom nodes registered.")
