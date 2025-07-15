import os
import io
import math
import random
from scipy.interpolate import interp1d
from numpy import linspace
import numpy as np
from huggingface_hub import HfApi
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import OpenEXR
import Imath
import folder_paths
import torch
import builtins
from pathlib import Path
import cv2
from .utils import pil2tensor, tensor2pil

BIGMIN = -(2**53 - 1)
BIGMAX = 2**53 - 1

category = "Nilor Nodes 👺"
subcategories = {
    "generators": "/Generators",
    "utilities": "/Utilities",
    "io": "/IO",
}


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class NilorInterpolatedFloatList:  # Generate interpolated float values based on a number of sections
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "number_of_floats": ("INT", {"forceInput": False}),
                "number_of_sections": ("INT", {"forceInput": False}),
                "section_number": ("INT", {"forceInput": False}),
                "interpolation_type": (
                    ["slinear", "quadratic", "cubic"],
                    {},
                ),  # Type of interpolation to use
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("floats",)

    FUNCTION = "generate_float_list"
    CATEGORY = category + subcategories["generators"]

    @staticmethod
    def interpolate_values(start, end, num_points, interp_type):
        # Linear interpolation between start and end over num_points
        x = linspace(0, num_points - 1, num_points)
        y = linspace(start, end, num_points)

        f = interp1d(x, y, kind=interp_type)
        return f(x)

    def generate_float_list(
        self, number_of_floats, number_of_sections, section_number, interpolation_type
    ):
        # Initializes the array with zeros
        my_floats = [0.0] * number_of_floats
        # Calculate the length of each portion based on total frames and number of images
        portion_length = int((number_of_floats - 1) / (number_of_sections - 1))

        # Handling the first image (special case for the first segment)
        if section_number == 1:
            portion_values = self.interpolate_values(
                1, 0, portion_length, interpolation_type
            )
            my_floats[0:portion_length] = portion_values
        # Handling the last image (special case for the last segment)
        elif section_number == number_of_sections:
            portion_values = self.interpolate_values(
                0, 1, portion_length, interpolation_type
            )
            start_index = int((number_of_sections - 2) * portion_length)
            my_floats[start_index:] = portion_values
        # Handling middle images (general case for dual segments)
        else:
            portion_values = np.concatenate(
                [
                    self.interpolate_values(0, 1, portion_length, interpolation_type),
                    self.interpolate_values(1, 0, portion_length, interpolation_type),
                ]
            )
            start_index = int((section_number - 2) * portion_length)
            end_index = start_index + (2 * portion_length)
            my_floats[start_index:end_index] = portion_values
        # Returns the modified list of float values
        return (my_floats,)


class NilorOneMinusFloatList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "list_of_floats": ("FLOAT", {"input_is_list": True}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("floats",)

    FUNCTION = "one_minus_float_list"
    CATEGORY = category + subcategories["generators"]

    def one_minus_float_list(self, list_of_floats):
        return ([1 - x for x in list_of_floats],)


class NilorRemapFloatList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "list_of_floats": ("FLOAT", {"input_is_list": True}),
                "min_input": ("FLOAT", {"default": 0.0}),
                "max_input": ("FLOAT", {"default": 1.0}),
                "min_output": ("FLOAT", {"default": 0.0}),
                "max_output": ("FLOAT", {"default": 1.0}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("remapped_floats",)

    FUNCTION = "remap_float_list"
    CATEGORY = category + subcategories["generators"]

    def remap_float_list(
        self, list_of_floats, min_input, max_input, min_output, max_output
    ):
        # Avoid division by zero
        if max_input - min_input == 0:
            raise ValueError("max_input and min_input cannot be the same value.")

        scale = (max_output - min_output) / (max_input - min_input)
        return ([min_output + (x - min_input) * scale for x in list_of_floats],)


class NilorRemapFloatListAutoInput:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list_of_floats": ("FLOAT", {"input_is_list": True}),
                "min_output": ("FLOAT", {"default": 0.0}),
                "max_output": ("FLOAT", {"default": 1.0}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("remapped_list",)

    FUNCTION = "remap_float_list_auto_input"
    CATEGORY = category + subcategories["generators"]

    def remap_float_list_auto_input(self, list_of_floats, min_output, max_output):
        min_input = min(list_of_floats)
        max_input = max(list_of_floats)

        scale = (max_output - min_output) / (max_input - min_input)
        return ([min_output + (x - min_input) * scale for x in list_of_floats],)


class NilorInverseMapFloatList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list_of_floats": ("FLOAT", {"input_is_list": True}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("floats",)

    FUNCTION = "inverse_map_float_list"
    CATEGORY = category + subcategories["generators"]

    def inverse_map_float_list(self, list_of_floats):
        if not list_of_floats:
            raise ValueError("The input list_of_floats cannot be empty.")

        min_input = min(list_of_floats)
        max_input = max(list_of_floats)

        return ([min_input + max_input - x for x in list_of_floats],)


class NilorIntToListOfBools:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "number_of_images": ("INT", {"forceInput": False}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("booleans",)

    FUNCTION = "boolify"
    CATEGORY = category + subcategories["generators"]

    OUTPUT_IS_LIST = (True,)

    def boolify(self, number_of_images, max_images=10):
        # Initializes the array with zeros
        my_bools = [False] * max_images

        for i in range(max_images):
            # Set the boolean value to True if the index is less than the number of images
            my_bools[i] = i < number_of_images

        return (my_bools,)


class NilorListOfInts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min": ("INT", {"forceInput": False, "default": 0}),
                "max": ("INT", {"forceInput": False, "default": 9}),
                "shuffle": ("BOOLEAN", {"default": False}),  # Toggle to randomize order
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("ints",)
    FUNCTION = "int_list"
    CATEGORY = category + subcategories["generators"]
    OUTPUT_IS_LIST = (
        True,
    )  # Indicates that the output should be processed as a list of individual elements

    def int_list(self, min=1, max=10, shuffle=False):
        # Generate the list
        ints_list = list(range(min, max + 1))
        if shuffle:
            random.shuffle(ints_list)

        return (ints_list,)


class NilorCountImagesInDirectory:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/images"}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)

    FUNCTION = "count_images_in_directory"
    CATEGORY = category + subcategories["utilities"]

    INPUT_IS_LIST = False

    def count_images_in_directory(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.")

        list_dir = []
        list_dir = os.listdir(directory)
        count = 0
        for file in list_dir:
            if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                count += 1

        return [count]


class NilorSelectIndexFromList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_of_any": (
                    any,
                    {"forceInput": False},
                ),  # Marking as lazy if processing could be deferred
                "index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("any",)
    FUNCTION = "any_by_index"
    CATEGORY = category + subcategories["utilities"]
    INPUT_IS_LIST = True  # Treats input list as a whole, rather than processing each item separately
    OUTPUT_IS_LIST = (False,)  # Output is a single element, not a list

    def any_by_index(self, list_of_any, index=0):
        # The input is a tensor so we need to unpack one level
        if isinstance(list_of_any, list) and len(list_of_any) == 1:
            actual_list = list_of_any[0]
        else:
            actual_list = list_of_any

        # Handle index access safely
        if isinstance(index, list):
            index = index[0]

        # Ensure the index is within bounds
        if index < 0 or index >= len(actual_list):
            raise ValueError("Index is outside the bounds of the array.")

        # Returns the value at the given index
        return (actual_list[index],)


class NilorSaveEXRArbitrary:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "channels": (
                    any,
                ),  # This should match the 'any' type list from List of Any
                "filename_prefix": ("STRING", {"default": "output"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()

    FUNCTION = "save_exr_arbitrary"  # The execution function
    CATEGORY = category + subcategories["io"]

    # INPUT_IS_LIST = True
    OUTPUT_NODE = True

    def save_exr_arbitrary(
        self, channels=None, filename_prefix="output", prompt=None, extra_pnginfo=None
    ):

        print("Running save_exr_arbitrary")
        # print(f"channels: {channels}")
        # print(f"filename_prefix: {filename_prefix}")

        actual_channels = channels
        # actual_channels = channels[0]  # Unpack the channels list
        # filename_prefix = filename_prefix[0]  # Unpack the filename_prefix list

        # check if actual_channels is subscriptable
        try:
            actual_channels[0]
        except TypeError:
            print("actual_channels is not subscriptable")
            return

        # File path handling
        useabs = os.path.isabs(filename_prefix)
        if not useabs:
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path(
                    filename_prefix,
                    self.output_dir,
                    actual_channels[0].shape[-1],
                    actual_channels[0].shape[-2],
                )
            )

        # Determine if the input contains a batch
        is_batch = (
            len(actual_channels[0].shape) == 3
        )  # If batch, shape is [batch_size, height, width]
        if is_batch:
            batch_size = actual_channels[0].shape[0]
        else:
            batch_size = 1

        for i in range(batch_size):
            # Extract each image's channels
            if is_batch:
                image_channels = [
                    tensor[i] for tensor in actual_channels
                ]  # For batch, select i-th image
            else:
                image_channels = actual_channels  # For single image, use channels as is

            # Validate each tensor
            height, width = image_channels[0].shape[-2:]
            for tensor in image_channels:
                if tensor.shape[-2:] != (height, width):
                    raise ValueError("All input tensors must have the same dimensions")

            # Channel naming
            default_names = ["R", "G", "B", "A"] + [
                f"Channel{j}" for j in range(4, len(image_channels))
            ]

            # Prepare data for EXR writing
            exr_data = {}
            for j, tensor in enumerate(image_channels):
                exr_data[default_names[j]] = tensor.cpu().numpy()

            # Handle file naming and saving
            if useabs:
                writepath = filename_prefix
            else:
                file = f"{filename}_{counter:05}_.exr"
                writepath = os.path.join(full_output_folder, file)
                counter += 1

            # Write EXR file
            self.write_exr(writepath, exr_data)

        return filename_prefix

    def write_exr(self, writepath, exr_data):
        try:
            # Determine the height and width from one of the provided channels
            height, width = list(exr_data.values())[0].shape[:2]

            # Create the EXR file header with dynamic channel names
            header = OpenEXR.Header(width, height)
            header["channels"] = {
                name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                for name in exr_data.keys()
            }

            # Create the EXR file
            exr_file = OpenEXR.OutputFile(writepath, header)

            # Prepare the data for each channel
            channel_data = {
                name: data.astype(np.float32).tobytes()
                for name, data in exr_data.items()
            }

            # Write the channel data to the EXR file
            exr_file.writePixels(channel_data)
            exr_file.close()

            print(f"EXR file saved successfully to {writepath}")
        except Exception as e:
            print(f"Failed to write EXR file: {e}")


class NilorSaveVideoToHFDataset:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "nilor_save"}),
                "filenames": ("VHS_FILENAMES",),
                "hf_auth_token": ("STRING", {"default": "auth_token"}),
                "repository_id": ("STRING", {"default": "nilor_dataset"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video_to_hf_dataset"
    OUTPUT_NODE = True
    CATEGORY = category + subcategories["io"]

    def save_video_to_hf_dataset(
        self, filenames, hf_auth_token, repository_id, filename_prefix="nilor_save"
    ):
        files = filenames[1]
        results = list()
        for path in files:
            ext = path.split(".")[-1]
            name = f"{filename_prefix}.{ext}"
            api = HfApi(token=hf_auth_token)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=name,
                repo_id=repository_id,
                repo_type="dataset",
            )
            results.append(name)
        return {"ui": {"string_field": results}}


class NilorSaveImageToHFDataset:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "repository_id": ("STRING", {"default": "nilor_dataset"}),
                "hf_auth_token": ("STRING", {"default": "auth_token"}),
                "filename_prefix": ("STRING", {"default": "nilor_image"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_image_to_hf_dataset"
    OUTPUT_NODE = True
    CATEGORY = category + subcategories["io"]

    def save_image_to_hf_dataset(
        self,
        image,
        repository_id,
        hf_auth_token,
        filename_prefix="nilor_image",
        prompt=None,
        extra_pnginfo=None,
    ):
        # Save the image to the dataset
        metadata = PngInfo()
        metadata.add_text("workflow", "testing, this should be png data")
        results = list()
        for i, tensor in enumerate(image):
            data = 255.0 * tensor.cpu().numpy()
            img = Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG", pnginfo=metadata)
            img_byte_arr = img_byte_arr.getvalue()
            now = datetime.now()
            date_string = now.strftime("%Y-%m-%d-%H-%M-%S")
            image_name = f"{filename_prefix}_{i}_{date_string}.png"
            api = HfApi(token=hf_auth_token)
            api.upload_file(
                path_or_fileobj=img_byte_arr,
                path_in_repo=image_name,
                repo_id=repository_id,
                repo_type="dataset",
            )
            results.append(image_name)

        return {"ui": {"string_field": results}}


class NilorShuffleImageBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "shuffle_image_batch"
    CATEGORY = category + subcategories["utilities"]

    def _check_image_dimensions(self, images):
        if images.shape[0] == 0:
            raise ValueError("Input images tensor is empty.")

        # All images in the batch should have the same dimensions
        if len(images.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor (batch, channels, height, width), got shape {images.shape}"
            )

    def shuffle_image_batch(self, images: torch.Tensor, seed):
        self._check_image_dimensions(images)

        # Get the number of images in the batch
        num_images = images.shape[0]

        # Generate indices and shuffle them
        torch.manual_seed(seed)
        indices = torch.randperm(num_images)

        # Shuffle the images using the indices
        shuffled_images = images[indices]

        return (shuffled_images,)


class NilorRepeatTrimImageBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "count": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "repeat_trim_image_batch"
    CATEGORY = category + subcategories["utilities"]

    def _check_image_dimensions(self, images):
        if images.shape[0] == 0:
            raise ValueError("Input images tensor is empty.")

        # All images in the batch should have the same dimensions
        if len(images.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor (batch, channels, height, width), got shape {images.shape}"
            )

    def repeat_trim_image_batch(self, images: torch.Tensor, count):
        self._check_image_dimensions(images)

        batch_count = images.size(0)
        amount = math.ceil(count / batch_count)

        appended_tensors = (images.repeat(amount, 1, 1, 1),)
        batched_tensors = torch.cat(appended_tensors, dim=0)
        trimmed_tensors = batched_tensors[:count]

        return (trimmed_tensors,)


class NilorRepeatShuffleTrimImageBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                "count": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "repeat_shuffle_trim_image_batch"
    CATEGORY = category + subcategories["utilities"]

    def _check_image_dimensions(self, images):
        if images.shape[0] == 0:
            raise ValueError("Input images tensor is empty.")

        # All images in the batch should have the same dimensions
        if len(images.shape) != 4:
            raise ValueError(
                f"Expected 4D tensor (batch, channels, height, width), got shape {images.shape}"
            )

    def repeat_shuffle_trim_image_batch(self, images: torch.Tensor, seed, count):
        self._check_image_dimensions(images)

        torch.manual_seed(seed)

        batch_count = images.size(0)
        amount = math.ceil(count / batch_count)

        appended_tensors = []
        while len(appended_tensors) < count:
            indices = torch.randperm(batch_count)
            appended_tensors.append(images[indices])

        batched_tensors = torch.cat(appended_tensors, dim=0)
        trimmed_tensors = batched_tensors[:count]

        return (trimmed_tensors,)


class NilorOutputFilenameString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("STRING", {"default": "nilor"}),
                "project": ("STRING", {"default": "research"}),
                "section": ("STRING", {"default": "test-1"}),
                "name": ("STRING", {"default": "out-1"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "notify"
    CATEGORY = category + subcategories["utilities"]
    OUTPUT_NODE = True
    IS_CHANGED = True

    def get_time(self, format: str):
        now = datetime.now()
        return now.strftime(format)

    def notify(
        self, client, project, section, name, unique_id=None, extra_pnginfo=None
    ):
        time = self.get_time("%y%m%d-%H%M%S")

        client = client or "nilor"
        project = project or "research"
        section = section or "test-1"
        name = name or "out-1"

        text = f"{client}_{project}/{section}/{time}_{section}/{time}_{client}_{project}_{section}_{name}"

        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        # TODO: make this node's text string preview widget work
        return {"ui": {"text": text}, "result": (text,)}


class NilorNFractionsOfInt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "numerator": ("INT", {"default": 10}),
                "denominator": ("INT", {"default": 2}),
                "type": (["starts", "ends", "centres", "start + end"], {}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("fractions",)

    FUNCTION = "n_fractions_of_int"
    CATEGORY = category + subcategories["utilities"]
    OUTPUT_IS_LIST = (True,)

    def n_fractions_of_int(self, numerator, denominator, type):
        # the number of fractions to generate is the denominator
        if type == "starts":
            return ([i * numerator // denominator for i in range(denominator)],)
        elif type == "ends":
            return ([(i + 1) * numerator // denominator for i in range(denominator)],)
        elif type == "centres":
            return (
                [
                    (i * numerator + numerator // 2) // denominator
                    for i in range(denominator)
                ],
            )
        elif type == "start + end":
            return ([i * numerator // (denominator - 1) for i in range(denominator)],)
        else:
            raise ValueError(f"Unknown type: {type}")


class NilorCategorizeString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"default": ""}),
                "number_of_categories": ("INT", {"default": 2, "min": 1, "max": 10}),
                "category_0": ("STRING", {"default": "apple, red fruit"}),
                "category_1": ("STRING", {"default": "banana, yellow fruit"}),
            },
            "optional": {
                "category_2": ("STRING", {"default": ""}),
                "category_3": ("STRING", {"default": ""}),
                "category_4": ("STRING", {"default": ""}),
                "category_5": ("STRING", {"default": ""}),
                "category_6": ("STRING", {"default": ""}),
                "category_7": ("STRING", {"default": ""}),
                "category_8": ("STRING", {"default": ""}),
                "category_9": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("category_index",)
    FUNCTION = "categorize_string"
    CATEGORY = category + subcategories["utilities"]

    def categorize_string(
        self,
        input_string,
        number_of_categories,
        category_0,
        category_1,
        category_2="",
        category_3="",
        category_4="",
        category_5="",
        category_6="",
        category_7="",
        category_8="",
        category_9="",
    ):
        # Convert input string to lowercase for case-insensitive matching
        input_string = input_string.lower()

        # Create categories dictionary from inputs
        categories = {}
        all_categories = [
            category_0,
            category_1,
            category_2,
            category_3,
            category_4,
            category_5,
            category_6,
            category_7,
            category_8,
            category_9,
        ]

        # Only process the number of categories specified
        for i in range(number_of_categories):
            if all_categories[i]:  # Only add non-empty categories
                # Split the comma-separated string and clean up whitespace
                keywords = [k.strip().lower() for k in all_categories[i].split(",")]
                categories[i] = keywords

        # Check each category's keywords against the input string
        for index, keywords in categories.items():
            if builtins.any(keyword in input_string for keyword in keywords):
                return (index,)

        return (-1,)  # Default case if no matches found


class NilorRandomString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiline_text": (
                    "STRING",
                    {"default": "option1, option2, option3", "multiline": True},
                ),
                "max_options": ("INT", {"default": 3, "min": 1}),
                "delimiter": ("STRING", {"default": ","}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("chosen_string",)
    FUNCTION = "choose_random_string"
    CATEGORY = category + subcategories["utilities"]

    def choose_random_string(self, multiline_text, max_options, delimiter, seed):
        import random

        random.seed(seed)

        # If the delimiter is literally "\n", use the actual newline character.
        if delimiter == r"\n" or delimiter == "\\n":
            actual_delimiter = "\n"
        else:
            actual_delimiter = delimiter

        # Split the input text using the actual delimiter and remove any extra whitespace
        options = [
            item.strip()
            for item in multiline_text.split(actual_delimiter)
            if item.strip()
        ]
        if not options:
            raise ValueError("No valid choices provided.")

        # Limit to the first 'max_options' entries if there are more options
        if len(options) > max_options:
            options = options[:max_options]

        chosen = random.choice(options)
        return (chosen,)


class NilorLoadImageByIndex:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_directory": (
                    "STRING",
                    {"default": "", "placeholder": "Image Directory"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sort_mode": (
                    ["filename", "creation_time", "modification_time", "size"],
                    {"default": "filename"},
                ),
                "reverse_sort": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "filename", "filepath")
    FUNCTION = "load_image_by_index"
    CATEGORY = category + subcategories["io"]

    @classmethod
    def IS_CHANGED(s, image_directory, seed, sort_mode, reverse_sort):
        return seed

    def load_image_by_index(self, image_directory, seed, sort_mode, reverse_sort):
        if not os.path.exists(image_directory):
            raise FileNotFoundError(f"Image directory {image_directory} does not exist")

        # Get list of image files
        files = []
        for f in os.listdir(image_directory):
            file_path = os.path.join(image_directory, f)
            if os.path.isfile(file_path) and f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
            ):
                files.append(file_path)

        if not files:
            raise ValueError(f"No image files found in {image_directory}")

        # Sort files based on selected mode
        if sort_mode == "filename":
            files.sort()
        elif sort_mode == "creation_time":
            files.sort(key=lambda x: os.path.getctime(x))
        elif sort_mode == "modification_time":
            files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_mode == "size":
            files.sort(key=lambda x: os.path.getsize(x))

        # Apply reverse sort if requested
        if reverse_sort:
            files.reverse()

        # Get file at index (with wrapping)
        file_index = seed % len(files)
        selected_file = files[file_index]

        # Get filename
        filename = os.path.basename(selected_file)

        # Load image using PIL and convert to tensor using our helper function
        img = Image.open(selected_file)
        img_tensor = pil2tensor(img)

        return (img_tensor, filename, selected_file)


class NilorExtractFilenameFromPath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("name", "name_with_extension")
    FUNCTION = "extract_filename"
    CATEGORY = category + subcategories["utilities"]

    def extract_filename(self, filepath):
        # Ensure the input is a valid path
        if not filepath:
            raise ValueError("Filepath cannot be empty.")

        path = Path(filepath)

        # Extract filename with and without extension
        name = path.stem  # Filename without extension
        name_with_extension = path.name  # Filename with extension

        return (name, name_with_extension)


class NilorBlurAnalysis:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Input image batch as a 4D tensor.
                "block_size": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blur_analysis",)
    FUNCTION = "analyze_blur"
    CATEGORY = category + subcategories["utilities"]

    def analyze_blur(self, images, block_size):
        """
        Performs blur analysis on each image using OpenCV's Laplacian method.
        """
        # Ensure images is a 4D tensor.
        if images.dim() != 4:
            raise ValueError("Input images must be a 4D tensor (batch, channels/height, height/width, width/channels)")

        # Detect if using NCHW or NHWC.
        if images.shape[1] not in (1, 3):
            if images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2)
            else:
                raise ValueError("Cannot determine image format (expected channel to be 1 or 3).")

        output_images = []
        batch_size = images.shape[0]
        for i in range(batch_size):
            # Get the i-th image (in NCHW: [channels, height, width]).
            img_tensor = images[i].cpu()
            img_np = img_tensor.numpy()  # shape: (C, H, W)

            # Convert to grayscale.
            if img_np.shape[0] >= 3:
                gray = (0.299 * img_np[0] +
                        0.587 * img_np[1] +
                        0.114 * img_np[2])
            else:
                gray = np.squeeze(img_np, axis=0)  # shape: (H, W)

            # Scale from [0, 1] to [0, 255] and convert to uint8.
            gray = np.clip(gray * 255.0, 0, 255).astype(np.uint8)

            # Compute Laplacian using a 3x3 kernel.
            lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            abs_lap = np.absolute(lap)

            # Apply local averaging using cv2.blur with window size = (block_size, block_size).
            local_edge = cv2.blur(abs_lap, (block_size, block_size))

            # Normalize and invert the edge response.
            max_val = local_edge.max()
            if max_val > 0:
                norm_edge = local_edge / max_val
            else:
                norm_edge = local_edge
            blur_map = 1.0 - norm_edge

            # Scale back to 0-255 and convert to uint8.
            out_img = (blur_map * 255.0).astype(np.uint8)

            # Convert the single channel output to a 3-channel image.
            # This ensures downstream nodes (like MaskFromRGBCMYBW) that index into channels work properly.
            if out_img.ndim == 2:
                out_img = np.stack([out_img, out_img, out_img], axis=-1)  # shape becomes (H, W, 3)

            # Convert from PIL image (or numpy array) to tensor.
            # pil2tensor should create a tensor in a format that downstream nodes expect.
            output_images.append(pil2tensor(out_img))

        # ---
        # Fix 2: Use torch.stack to preserve the batch dimension.
        # If each output has shape, say, (H, W, 3), stacking them gives a tensor of shape (B, H, W, 3).
        return (torch.cat(output_images, dim=0),)

class NilorToSparseIndexMethod:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ints": ("INT", {"default": 0}), # Removed input_is_list=True
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sparse_method",)
    OUTPUT_IS_LIST = (False,)
    INPUT_IS_LIST = True # Crucial: Receive entire list in one go

    FUNCTION = "convert_to_sparse_index_method"
    CATEGORY = category + subcategories["utilities"]

    def convert_to_sparse_index_method(self, ints):
        # 'ints' will now be the full list of integers, e.g., [0, 32, 64, 96]
        indexes_str = ",".join(map(str, ints))
        
        print(f"NilorToSparseIndexMethod final output: {indexes_str}")
        print(f"Type of final output: {type(indexes_str)}")
        
        return (indexes_str,)


# Mapping class names to objects for potential export
NODE_CLASS_MAPPINGS = {
    "Nilor Interpolated Float List": NilorInterpolatedFloatList,
    "Nilor One Minus Float List": NilorOneMinusFloatList,
    "Nilor Remap Float List": NilorRemapFloatList,
    "Nilor Remap Float List Auto Input": NilorRemapFloatListAutoInput,
    "Nilor Inverse Map Float List": NilorInverseMapFloatList,
    "Nilor Int To List Of Bools": NilorIntToListOfBools,
    "Nilor List of Ints": NilorListOfInts,
    "Nilor Count Images In Directory": NilorCountImagesInDirectory,
    "Nilor Save Image To HF Dataset": NilorSaveImageToHFDataset,
    "Nilor Save Video To HF Dataset": NilorSaveVideoToHFDataset,
    "Nilor Select Index From List": NilorSelectIndexFromList,
    "Nilor Save EXR Arbitrary": NilorSaveEXRArbitrary,
    "Nilor Shuffle Image Batch": NilorShuffleImageBatch,
    "Nilor Repeat & Trim Image Batch": NilorRepeatTrimImageBatch,
    "Nilor Repeat, Shuffle, & Trim Image Batch": NilorRepeatShuffleTrimImageBatch,
    "Nilor Output Filename String": NilorOutputFilenameString,
    "Nilor n Fractions of Int": NilorNFractionsOfInt,
    "Nilor Categorize String": NilorCategorizeString,
    "Nilor Random String": NilorRandomString,
    "Nilor Extract Filename from Path": NilorExtractFilenameFromPath,
    "Nilor Load Image By Index": NilorLoadImageByIndex,
    "Nilor Blur Analysis": NilorBlurAnalysis,
    "Nilor To Sparse Index Method": NilorToSparseIndexMethod,
}

# Mapping nodes to human-readable names
NODE_DISPLAY_NAME_MAPPINGS = {
    "Nilor Interpolated Float List": "👺 Interpolated Float List",
    "Nilor One Minus Float List": "👺 One Minus Float List",
    "Nilor Remap Float List": "👺 Nilor Remap Float List",
    "Nilor Remap Float List Auto Input": "👺 Nilor Remap Float List Auto Input",
    "Nilor Inverse Map Float List": "👺 Nilor Inverse Map Float List",
    "Nilor Int To List Of Bools": "👺 Int To List Of Bools",
    "Nilor List of Ints": "👺 List of Ints",
    "Nilor Count Images In Directory": "👺 Count Images In Directory",
    "Nilor Save Image To HF Dataset": "👺 Save Image To HF Dataset",
    "Nilor Save Video To HF Dataset": "👺 Save Video To HF Dataset",
    "Nilor Select Index From List": "👺 Select Index From List",
    "Nilor Save EXR Arbitrary": "👺 Save EXR Arbitrary",
    "Nilor Shuffle Image Batch": "👺 Nilor Shuffle Image Batch",
    "Nilor Repeat & Trim Image Batch": "👺 Nilor Repeat & Trim Image Batch",
    "Nilor Repeat, Shuffle, & Trim Image Batch": "👺 Nilor Repeat, Shuffle, & Trim Image Batch",
    "Nilor Output Filename String": "👺 Nilor Output Filename String",
    "Nilor n Fractions of Int": "👺 Nilor n Fractions of Int",
    "Nilor Categorize String": "👺 Categorize String",
    "Nilor Random String": "👺 Random String",
    "Nilor Extract Filename from Path": "👺 Extract Filename from Path",
    "Nilor Load Image By Index": "👺 Load Image By Index",
    "Nilor Blur Analysis": "👺 Blur Analysis",
    "Nilor To Sparse Index Method": "👺 To Sparse Index Method",
}
