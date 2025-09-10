import os
from huggingface_hub import hf_hub_download
import folder_paths
from .utils import get_hf_model_lists
import shutil
from tempfile import TemporaryDirectory

# Fetch model lists at startup
diffusion_models, loras, clip, text_encoders, vae = get_hf_model_lists()


class NilorModelLoader_Diffusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (diffusion_models,),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Diffusion Model"

    def get_path(self, model_name):
        # model_name is the full repo path, e.g., "diffusion/WanVideo/model.safetensors"
        # We need the relative path for local use, e.g., "WanVideo/model.safetensors"
        relative_model_path = os.path.join(*model_name.split("/")[1:])

        model_type_folder = folder_paths.get_folder_paths("checkpoints")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            print(f"NilorNodes: Downloading {model_name}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with TemporaryDirectory() as tmpdir:
                temp_path = hf_hub_download(
                    repo_id="nilor-corp/brain-models",
                    filename=model_name,
                    cache_dir=tmpdir,
                )
                shutil.move(temp_path, model_path)
        return (
            relative_model_path,
            model_path,
        )


class NilorModelLoader_Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (loras,),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("loras"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Lora Model"

    def get_path(self, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("loras")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            print(f"NilorNodes: Downloading {model_name}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with TemporaryDirectory() as tmpdir:
                temp_path = hf_hub_download(
                    repo_id="nilor-corp/brain-models",
                    filename=model_name,
                    cache_dir=tmpdir,
                )
                shutil.move(temp_path, model_path)
        return (
            relative_model_path,
            model_path,
        )


class NilorModelLoader_Clip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (clip,),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("clip"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Clip Model"

    def get_path(self, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("clip")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            print(f"NilorNodes: Downloading {model_name}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with TemporaryDirectory() as tmpdir:
                temp_path = hf_hub_download(
                    repo_id="nilor-corp/brain-models",
                    filename=model_name,
                    cache_dir=tmpdir,
                )
                shutil.move(temp_path, model_path)
        return (
            relative_model_path,
            model_path,
        )


class NilorModelLoader_TextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (text_encoders,),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("text_encoders"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Text Encoder Model"

    def get_path(self, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("text_encoders")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            print(f"NilorNodes: Downloading {model_name}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with TemporaryDirectory() as tmpdir:
                temp_path = hf_hub_download(
                    repo_id="nilor-corp/brain-models",
                    filename=model_name,
                    cache_dir=tmpdir,
                )
                shutil.move(temp_path, model_path)
        return (
            relative_model_path,
            model_path,
        )


class NilorModelLoader_VAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (vae,),
            }
        }

    RETURN_TYPES = (folder_paths.get_filename_list("vae"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get VAE Model"

    def get_path(self, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("vae")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            print(f"NilorNodes: Downloading {model_name}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with TemporaryDirectory() as tmpdir:
                temp_path = hf_hub_download(
                    repo_id="nilor-corp/brain-models",
                    filename=model_name,
                    cache_dir=tmpdir,
                )
                shutil.move(temp_path, model_path)
        return (
            relative_model_path,
            model_path,
        )
