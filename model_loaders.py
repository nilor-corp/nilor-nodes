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


class NilorModelLoader_Diffusion:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (diffusion_models,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("diffusion_models")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "w") as f:
                pass

        model_list = folder_paths.get_filename_list("diffusion_models")
        if relative_model_path not in model_list:
            model_list.append(relative_model_path)

        return True

    RETURN_TYPES = (folder_paths.get_filename_list("diffusion_models"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Diffusion Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "diffusion_models")


class NilorModelLoader_Lora:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (loras,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("loras")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "w") as f:
                pass

        model_list = folder_paths.get_filename_list("loras")
        if relative_model_path not in model_list:
            model_list.append(relative_model_path)

        return True

    RETURN_TYPES = (folder_paths.get_filename_list("loras"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Lora Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "loras")


class NilorModelLoader_Clip:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (clip,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("clip")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "w") as f:
                pass

        model_list = folder_paths.get_filename_list("clip")
        if relative_model_path not in model_list:
            model_list.append(relative_model_path)

        return True

    RETURN_TYPES = (folder_paths.get_filename_list("clip"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Clip Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "clip")


class NilorModelLoader_TextEncoder:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (text_encoders,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("text_encoders")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "w") as f:
                pass

        model_list = folder_paths.get_filename_list("text_encoders")
        if relative_model_path not in model_list:
            model_list.append(relative_model_path)

        return True

    RETURN_TYPES = (folder_paths.get_filename_list("text_encoders"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get Text Encoder Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "text_encoders")


class NilorModelLoader_VAE:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (vae,),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        relative_model_path = os.path.join(*model_name.split("/")[1:])
        model_type_folder = folder_paths.get_folder_paths("vae")[0]
        model_path = os.path.join(model_type_folder, relative_model_path)

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "w") as f:
                pass

        model_list = folder_paths.get_filename_list("vae")
        if relative_model_path not in model_list:
            model_list.append(relative_model_path)

        return True

    RETURN_TYPES = (folder_paths.get_filename_list("vae"), "STRING")
    RETURN_NAMES = ("model_name", "model_path")
    FUNCTION = "get_path"
    CATEGORY = "NilorNodes/model_loaders"
    DISPLAY_NAME = "👺 Nilor Get VAE Model"

    def get_path(self, model_name):
        return _get_model_path_and_download(model_name, "vae")
