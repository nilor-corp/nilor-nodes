import threading
import os
import uvicorn
import atexit
from fastapi import FastAPI, UploadFile, HTTPException, Security
from fastapi.security import APIKeyHeader
import asyncio
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import aiofiles.os as aio_os
import folder_paths
from dotenv import load_dotenv

load_dotenv()

# --- Server & Security Configuration ---
API_KEY = os.getenv("IMAGE_STREAM_API_KEY", "your-super-secret-key")
API_KEY_NAME = "Authorization"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
STREAM_JOBS_ROOT = Path(folder_paths.get_input_directory()) / "stream_jobs"
STREAM_JOBS_ROOT.mkdir(parents=True, exist_ok=True)

# --- FastAPI & Server Management ---
app = FastAPI()
server_thread: threading.Thread = None
uvicorn_server: uvicorn.Server = None

# --- Hardened Global State ---
# This lock is crucial for thread-safe access between the FastAPI server thread and the ComfyUI main thread.
job_state_lock = threading.Lock()

# Key: prompt_id (str). Value: Dictionary containing the asyncio loop and a list of event objects.
# This is necessary for thread-safe communication between the server thread and the ComfyUI thread.
job_waiters: dict[str, dict[str, any]] = {}

# Stores prompt_ids that have received a trigger. Prevents race conditions.
triggered_jobs: set[str] = set()

# NEW: Stores prompt_ids that were cancelled. Nodes check this after waking up.
cancelled_jobs: set[str] = set()

# Key: tuple(prompt_id, node_id). Value: PyTorch Tensor. Caches the output to avoid re-loading.
node_output_cache: dict[tuple[str, str], torch.Tensor] = {}

def sanitize_id(node_id: str) -> str:
    """Prevents path traversal attacks."""
    return "".join(c for c in node_id if c.isalnum() or c in ('_', '-')).rstrip()

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Security dependency to validate the API key."""
    # The header should be "Bearer <key>". We extract just the key part.
    if " " not in api_key_header or not api_key_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    token = api_key_header.split(" ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/upload_image/{prompt_id}/{node_id}", dependencies=[Security(get_api_key)])
async def upload_image(prompt_id: str, node_id: str, image_file: UploadFile):
    s_prompt_id = sanitize_id(prompt_id)
    s_node_id = sanitize_id(node_id)
    
    job_dir = STREAM_JOBS_ROOT / s_prompt_id / s_node_id
    if not job_dir.resolve().is_relative_to(STREAM_JOBS_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Invalid path detected.")
        
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename as well
    s_filename = sanitize_id(Path(image_file.filename).stem) + Path(image_file.filename).suffix
    file_path = job_dir / s_filename
    with open(file_path, "wb") as f:
        f.write(await image_file.read())
    return {"status": "success", "path": str(file_path)}

@app.post("/trigger_workflow/{prompt_id}", dependencies=[Security(get_api_key)])
async def trigger_workflow(prompt_id: str):
    with job_state_lock:
        triggered_jobs.add(prompt_id)
        if prompt_id in job_waiters:
            waiter_info = job_waiters.pop(prompt_id)
            loop = waiter_info["loop"]
            for event in waiter_info["events"]:
                loop.call_soon_threadsafe(event.set)
    return {"status": f"Trigger signal sent for job {prompt_id}"}

@app.post("/cancel_workflow/{prompt_id}", dependencies=[Security(get_api_key)])
async def cancel_workflow(prompt_id: str):
    with job_state_lock:
        cancelled_jobs.add(prompt_id)
        # Also mark as triggered to unblock any waiters. They will check the cancel flag upon waking.
        triggered_jobs.add(prompt_id)
        if prompt_id in job_waiters:
            waiter_info = job_waiters.pop(prompt_id)
            loop = waiter_info["loop"]
            for event in waiter_info["events"]:
                loop.call_soon_threadsafe(event.set)

    # Asynchronously clean up disk resources for the cancelled job
    job_dir = STREAM_JOBS_ROOT / sanitize_id(prompt_id)
    if job_dir.exists():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: shutil.rmtree(job_dir, ignore_errors=True))
        
    return {"status": f"Cancel signal sent and resources cleaned for job {prompt_id}"}

class ImageStreamInput:
    FUNCTION = "await_and_load_images"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "latent/loaders"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_id": ("STRING", {"forceInput": True}),
                "node_id": ("STRING", {"default": "input_1"}),
                "timeout": ("INT", {"default": 60, "min": 1, "max": 3600}),
            }
        }

    async def await_and_load_images(self, prompt_id: str, node_id: str, timeout: int):
        worker_key = (prompt_id, node_id)

        if worker_key in node_output_cache:
            return (node_output_cache[worker_key],)

        event = asyncio.Event()
        must_wait = False
        
        try:
            with job_state_lock:
                if prompt_id in triggered_jobs:
                    pass  # Proceed directly to loading
                else:
                    loop = asyncio.get_running_loop()
                    waiter_info = job_waiters.setdefault(prompt_id, {"loop": loop, "events": []})
                    waiter_info["events"].append(event)
                    must_wait = True

            if must_wait:
                print(f"--- [ImageStreamInput] Node '{node_id}' is waiting for trigger on prompt_id: {prompt_id} ---")
                await asyncio.wait_for(event.wait(), timeout=float(timeout))
            
            # CRITICAL: Check for cancellation immediately after waking up.
            with job_state_lock:
                if prompt_id in cancelled_jobs:
                    raise Exception(f"Workflow {prompt_id} was cancelled by an external request.")

            # Load images from disk and convert to the correct tensor format.
            image_dir = STREAM_JOBS_ROOT / sanitize_id(prompt_id) / sanitize_id(node_id)
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Input directory not found for job {prompt_id}, node {node_id}")

            image_paths = sorted([p for p in image_dir.iterdir() if p.is_file()])
            if not image_paths:
                raise FileNotFoundError(f"No images found in input directory for job {prompt_id}, node {node_id}")

            images = []
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(torch.from_numpy(img_array)[None,])
            
            loaded_images_tensor = torch.cat(images, dim=0)

            node_output_cache[worker_key] = loaded_images_tensor
            return (loaded_images_tensor,)

        except asyncio.TimeoutError:
            raise Exception(f"ImageStreamInput timed out for prompt_id: {prompt_id}")
        finally:
            # Guaranteed cleanup to prevent this specific event from leaking.
            with job_state_lock:
                if prompt_id in job_waiters and event in job_waiters[prompt_id]["events"]:
                    job_waiters[prompt_id]["events"].remove(event)
                    if not job_waiters[prompt_id]["events"]:
                        del job_waiters[prompt_id]

@app.get("/")
async def root():
    return {"status": "ok"}

def start_server():
    """Starts the Uvicorn server in a separate thread."""
    global server_thread, uvicorn_server
    host = os.getenv("IMAGE_STREAM_HOST", "127.0.0.1")
    port = int(os.getenv("IMAGE_STREAM_PORT", 8189))
    
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    
    server_thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    server_thread.start()
    atexit.register(stop_server)
    print(f"--- Image Stream Server started at http://{host}:{port} ---")

def stop_server():
    """Signals the Uvicorn server to shut down."""
    if uvicorn_server:
        uvicorn_server.should_exit = True
    if server_thread:
        server_thread.join(timeout=5) # Wait for thread to terminate
        print("--- Image Stream Server stopped ---")

# This should be called once when the custom node is loaded by ComfyUI.
# A common pattern is to place it at the module level of the .py file.
start_server()

# Required for ComfyUI to recognize this file as a custom node
NODE_CLASS_MAPPINGS = {
    "ImageStreamInput": ImageStreamInput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Nilor ImageStreamInput": "👺 Image Stream Input",
} 