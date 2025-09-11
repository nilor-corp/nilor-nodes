import os
from huggingface_hub import hf_hub_download
import folder_paths
from .utils import get_hf_model_lists
import shutil
from tempfile import TemporaryDirectory

# Fetch model lists at startup
diffusion_models, loras, clip, text_encoders, vae = get_hf_model_lists()


def _get_model_path_and_download(model_name, model_type):
    """A helper function to centralize the download and path logic."""
    relative_model_path = os.path.join(*model_name.split("/")[1:])
    model_type_folder = folder_paths.get_folder_paths(model_type)[0]
    model_path = os.path.join(model_type_folder, relative_model_path)

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            pass

    if os.path.getsize(model_path) == 0:
        print(f"NilorNodes: Downloading {model_name}...")
        with TemporaryDirectory() as tmpdir:
            try:
                temp_path = hf_hub_download(
                    repo_id="nilor-corp/brain-models",
                    filename=model_name,
                    cache_dir=tmpdir,
                )
                shutil.move(temp_path, model_path)
                folder_paths.filename_list_cache.clear()
            except Exception as e:
                print(f"NilorNodes: Download failed: {e}")
                os.remove(model_path)
                raise e
    return (
        relative_model_path,
        model_path,
    )


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
        # Pre-populate the folder_paths list to satisfy the validator
        for model in diffusion_models:
            relative_model_path = os.path.join(*model.split("/")[1:])
            if relative_model_path not in folder_paths.get_filename_list(
                "diffusion_models"
            ):
                folder_paths.get_filename_list("diffusion_models").append(
                    relative_model_path
                )
        return True

    RETURN_TYPES = (folder_paths.get_filename_list("diffusion_models"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Diffusion Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "diffusion_models")


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
        for model in loras:
            relative_model_path = os.path.join(*model.split("/")[1:])
            if relative_model_path not in folder_paths.get_filename_list("loras"):
                folder_paths.get_filename_list("loras").append(relative_model_path)
        return True

    RETURN_TYPES = (folder_paths.get_filename_list("loras"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Lora Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "loras")


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
        for model in clip:
            relative_model_path = os.path.join(*model.split("/")[1:])
            if relative_model_path not in folder_paths.get_filename_list("clip"):
                folder_paths.get_filename_list("clip").append(relative_model_path)
        return True

    RETURN_TYPES = (folder_paths.get_filename_list("clip"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Clip Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "clip")


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
        for model in text_encoders:
            relative_model_path = os.path.join(*model.split("/")[1:])
            if relative_model_path not in folder_paths.get_filename_list(
                "text_encoders"
            ):
                folder_paths.get_filename_list("text_encoders").append(
                    relative_model_path
                )
        return True

    RETURN_TYPES = (folder_paths.get_filename_list("text_encoders"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Text Encoder Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "text_encoders")


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
        for model in vae:
            relative_model_path = os.path.join(*model.split("/")[1:])
            if relative_model_path not in folder_paths.get_filename_list("vae"):
                folder_paths.get_filename_list("vae").append(relative_model_path)
        return True

    RETURN_TYPES = (folder_paths.get_filename_list("vae"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get VAE Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "vae")
