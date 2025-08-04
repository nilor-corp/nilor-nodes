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

# 2. Start the Image Stream FastAPI Server
from .image_stream import start_server as start_image_stream_server
start_image_stream_server()
print("✅ Nilor-Nodes: Image stream server started.")


# --- Node Registration ---
from .nilornodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .image_stream import NODE_CLASS_MAPPINGS as is_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as is_NODE_DISPLAY_NAME_MAPPINGS
from .media_stream import NODE_CLASS_MAPPINGS as ms_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ms_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(is_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(is_NODE_DISPLAY_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(ms_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ms_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
print("✅ Nilor-Nodes: All custom nodes registered.")
