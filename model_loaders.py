import os
from huggingface_hub import hf_hub_download
import folder_paths
from .utils import get_hf_model_lists
import shutil
from tempfile import TemporaryDirectory
from tqdm.auto import tqdm
import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import get_session
import time
import comfy.utils
import comfy.sd
import torch

# Fetch model lists at startup
diffusion_models, loras, clip, text_encoders, vae = get_hf_model_lists()


# TODO: delete dummy model files if the actual model file is not downloaded and the workflow gets interrupted


def _get_model_path_and_download(model_name, model_type):
    """A helper function to centralize the download and path logic."""
    parts = model_name.split("/")
    if len(parts) > 1:
        relative_model_path = os.path.join(*parts[1:])
    else:
        relative_model_path = model_name
    model_type_folder = folder_paths.get_folder_paths(model_type)[0]
    model_path = os.path.join(model_type_folder, relative_model_path)

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            pass

    if os.path.getsize(model_path) == 0:
        print(f"NilorNodes: Downloading {model_name}...")
        try:
            url = hf_hub_url(repo_id="nilor-corp/brain-models", filename=model_name)
            hf_session = get_session()

            with hf_session.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))

                with tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=f"Downloading {os.path.basename(model_name)}",
                ) as pbar:
                    with open(model_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

            folder_paths.filename_list_cache.clear()
        except Exception as e:
            print(f"NilorNodes: Download failed: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise e

    return (
        relative_model_path,
        model_path,
    )


def _validate_model_input(model_name, model_type):
    parts = model_name.split("/")
    if len(parts) > 1:
        relative_model_path = os.path.join(*parts[1:])
    else:
        relative_model_path = model_name

    model_type_folder = folder_paths.get_folder_paths(model_type)[0]
    model_path = os.path.join(model_type_folder, relative_model_path)

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            pass
        folder_paths.filename_list_cache.clear()

    model_list = folder_paths.get_filename_list(model_type)
    if relative_model_path not in model_list:
        # Forcefully update the cache
        if model_type in folder_paths.filename_list_cache:
            folder_paths.filename_list_cache[model_type][0].append(relative_model_path)

    return True


class NilorModelLoader_Diffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (diffusion_models,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        return _validate_model_input(model_name, "diffusion_models")

    @classmethod
    def IS_CHANGED(s, model_name):
        _validate_model_input(model_name, "diffusion_models")
        s.RETURN_TYPES = (folder_paths.get_filename_list("diffusion_models"),)
        return time.time()

    RETURN_TYPES = (folder_paths.get_filename_list("diffusion_models"),)
    RETURN_NAMES = ("ckpt_name",)
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Diffusion Model"

    def get_path(self, model_name):
        relative_model_path, model_path = _get_model_path_and_download(
            model_name, "diffusion_models"
        )
        return (relative_model_path,)


class NilorModelLoader_Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (loras,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        return _validate_model_input(model_name, "loras")

    @classmethod
    def IS_CHANGED(s, model_name):
        _validate_model_input(model_name, "loras")
        s.RETURN_TYPES = (folder_paths.get_filename_list("loras"),)
        return time.time()

    RETURN_TYPES = (folder_paths.get_filename_list("loras"),)
    RETURN_NAMES = ("lora_name",)
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Lora Model"

    def get_path(self, model_name):
        relative_model_path, model_path = _get_model_path_and_download(
            model_name, "loras"
        )
        return (relative_model_path,)


class NilorModelLoader_Clip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (clip,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        return _validate_model_input(model_name, "clip")

    @classmethod
    def IS_CHANGED(s, model_name):
        _validate_model_input(model_name, "clip")
        s.RETURN_TYPES = (folder_paths.get_filename_list("clip"),)
        return time.time()

    RETURN_TYPES = (folder_paths.get_filename_list("clip"),)
    RETURN_NAMES = ("clip_name",)
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Clip Model"

    def get_path(self, model_name):
        relative_model_path, model_path = _get_model_path_and_download(
            model_name, "clip"
        )
        return (relative_model_path,)


class NilorModelLoader_TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (text_encoders,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        return _validate_model_input(model_name, "text_encoders")

    @classmethod
    def IS_CHANGED(s, model_name):
        _validate_model_input(model_name, "text_encoders")
        s.RETURN_TYPES = (folder_paths.get_filename_list("text_encoders"),)
        return time.time()

    RETURN_TYPES = (folder_paths.get_filename_list("text_encoders"),)
    RETURN_NAMES = ("clip_name",)
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Text Encoder Model"

    def get_path(self, model_name):
        relative_model_path, model_path = _get_model_path_and_download(
            model_name, "text_encoders"
        )
        return (relative_model_path,)


class NilorModelLoader_VAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (vae,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        return _validate_model_input(model_name, "vae")

    @classmethod
    def IS_CHANGED(s, model_name):
        _validate_model_input(model_name, "vae")
        s.RETURN_TYPES = (folder_paths.get_filename_list("vae"),)
        return time.time()

    RETURN_TYPES = (folder_paths.get_filename_list("vae"),)
    RETURN_NAMES = ("vae_name",)
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get VAE Model"

    def get_path(self, model_name):
        relative_model_path, model_path = _get_model_path_and_download(
            model_name, "vae"
        )
        return (relative_model_path,)
