import threading

# --- Nilor-Nodes Custom Node Registration and Startup ---
# This file is executed when ComfyUI starts and discovers this custom node directory.
# It's responsible for:
# 1. Starting background services (like the SQS worker and a FastAPI server).
# 2. Registering the custom nodes with ComfyUI so they appear in the menu.

# --- Background Services ---

# 1. Start the SQS Worker Consumer
from .worker_consumer import consume_jobs
consumer_thread = threading.Thread(target=consume_jobs, daemon=True)
consumer_thread.start()
print("✅ Nilor-Nodes: SQS worker consumer thread started.")

# --- Node Registration ---
from .nilornodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .media_stream import NODE_CLASS_MAPPINGS as ms_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ms_NODE_DISPLAY_NAME_MAPPINGS
from .user_input import NODE_CLASS_MAPPINGS as ui_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ui_NODE_DISPLAY_NAME_MAPPINGS
from .controllers import NODE_CLASS_MAPPINGS as ctrl_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ctrl_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(ms_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ms_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ui_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ui_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ctrl_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ctrl_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("✅ Nilor-Nodes: All custom nodes registered.")
