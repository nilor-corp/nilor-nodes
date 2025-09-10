import io
import torch
import base64
import numpy as np
from pkg_resources import parse_version
from PIL import Image
from huggingface_hub import list_repo_files, hf_hub_download
import folder_paths
from tempfile import TemporaryDirectory
import shutil
import os


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)


## Helper function equivalent to Mikey's pil2tensor
# def pil2tensor(self, image):
#    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)


def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def tensor2bytes(image: torch.Tensor) -> bytes:
    return tensor2pil(image).tobytes()


def pil2base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def get_hf_model_lists():
    """
    Fetches and categorizes model file paths from the Hugging Face Hub.

    Returns:
        tuple: A tuple containing five lists: diffusion_models, loras, clip,
               text_encoders, and vae.
    """
    repo_id = "nilor-corp/brain-models"

    try:
        repo_files = list_repo_files(repo_id)
    except Exception as e:
        print(f"NilorNodes: Error fetching model list from Hugging Face Hub: {e}")
        return [], [], [], [], []

    diffusion_models = [f for f in repo_files if f.startswith("diffusion/")]
    loras = [f for f in repo_files if f.startswith("loras/")]
    clip = [f for f in repo_files if f.startswith("clip/")]
    text_encoders = [f for f in repo_files if f.startswith("text_encoders/")]
    vae = [f for f in repo_files if f.startswith("vae/")]

    return diffusion_models, loras, clip, text_encoders, vae


def download_model_and_get_paths(model_name_full, comfy_folder_name):
    """
    Handles downloading a model from HF and placing it in the correct ComfyUI directory.

    Args:
        model_name_full (str): The full path of the model in the HF repo (e.g., "diffusion/model.safetensors").
        comfy_folder_name (str): The target ComfyUI folder name (e.g., "checkpoints").

    Returns:
        tuple: A tuple containing the relative model path for ComfyUI and the full local path.
    """
    # Strip the repo's root folder (e.g., "diffusion/") to get the relative path for the local destination
    relative_path = os.path.join(*model_name_full.split("/")[1:])

    # Get the correct ComfyUI base folder for this model type
    # We use get_folder_paths which returns a list, and we take the first one which is the primary models dir.
    comfy_base_dir = folder_paths.get_folder_paths(comfy_folder_name)[0]

    # Construct the full final local path
    local_path = os.path.join(comfy_base_dir, relative_path)

    # Download if it doesn't exist
    if not os.path.exists(local_path):
        print(f"NilorNodes: Downloading {model_name_full} to {local_path}...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download to a temporary cache and then move it to the correct final location
        with TemporaryDirectory() as tmpdir:
            try:
                temp_path = hf_hub_download(
                    repo_id="nilor-corp/brain-models",
                    filename=model_name_full,
                    cache_dir=tmpdir,
                    etag_timeout=100,
                )
                shutil.move(temp_path, local_path)
            except Exception as e:
                print(f"NilorNodes: Failed to download {model_name_full}. Error: {e}")
                return (None, None)

    return (
        os.path.basename(relative_path),
        local_path,
    )
