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

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

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




class NilorInterpolatedFloatList: # Generate interpolated float values based on a number of sections
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
                "interpolation_type": (["slinear","quadratic", "cubic"], {}),  # Type of interpolation to use
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

    def generate_float_list(self, number_of_floats, number_of_sections, section_number, interpolation_type):
        # Initializes the array with zeros
        my_floats = [0.0] * number_of_floats
        # Calculate the length of each portion based on total frames and number of images
        portion_length = int((number_of_floats - 1) / (number_of_sections - 1))

        # Handling the first image (special case for the first segment)
        if section_number == 1:
            portion_values = self.interpolate_values(1, 0, portion_length, interpolation_type)
            my_floats[0:portion_length] = portion_values
        # Handling the last image (special case for the last segment)
        elif section_number == number_of_sections:
            portion_values = self.interpolate_values(0, 1, portion_length, interpolation_type)
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
    OUTPUT_IS_LIST = (True,)  # Indicates that the output should be processed as a list of individual elements

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
                "list_of_any": (any, {"forceInput": False}),  # Marking as lazy if processing could be deferred
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
                "channels": (any,),  # This should match the 'any' type list from List of Any
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

    def save_exr_arbitrary(self, channels=None, filename_prefix="output", prompt=None, extra_pnginfo=None):

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
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, actual_channels[0].shape[-1], actual_channels[0].shape[-2])

        # Determine if the input contains a batch
        is_batch = len(actual_channels[0].shape) == 3  # If batch, shape is [batch_size, height, width]
        if is_batch:
            batch_size = actual_channels[0].shape[0]
        else:
            batch_size = 1

        for i in range(batch_size):
            # Extract each image's channels
            if is_batch:
                image_channels = [tensor[i] for tensor in actual_channels]  # For batch, select i-th image
            else:
                image_channels = actual_channels  # For single image, use channels as is

            # Validate each tensor
            height, width = image_channels[0].shape[-2:]
            for tensor in image_channels:
                if tensor.shape[-2:] != (height, width):
                    raise ValueError("All input tensors must have the same dimensions")

            # Channel naming
            default_names = ["R", "G", "B", "A"] + [f"Channel{j}" for j in range(4, len(image_channels))]
            
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
            header['channels'] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in exr_data.keys()}

            # Create the EXR file
            exr_file = OpenEXR.OutputFile(writepath, header)

            # Prepare the data for each channel
            channel_data = {name: data.astype(np.float32).tobytes() for name, data in exr_data.items()}

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "shuffle_image_batch"
    CATEGORY = "nilor-nodes"

    def _check_image_dimensions(self, images):
        if images.shape[0] == 0:
            raise ValueError("Input images tensor is empty.")
        
        # All images in the batch should have the same dimensions
        if len(images.shape) != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got shape {images.shape}")

    def shuffle_image_batch(self, images, seed):
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
                "count": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "repeat_trim_image_batch"
    CATEGORY = "nilor-nodes"
    
    def _check_image_dimensions(self, images):
        if images.shape[0] == 0:
            raise ValueError("Input images tensor is empty.")
        
        # All images in the batch should have the same dimensions
        if len(images.shape) != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got shape {images.shape}")

    def repeat_trim_image_batch(self, images: torch.Tensor, count):
        self._check_image_dimensions(images)

        batch_count = images.size(0)
        amount = math.ceil(count / batch_count)
        
        appended_tensors = images.repeat(amount, 1, 1, 1),
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
    CATEGORY = "nilor-nodes"
    OUTPUT_NODE = True
    IS_CHANGED = True

    def get_time(self, format: str):
        now = datetime.now()
        return now.strftime(format)

    def notify(self, client, project, section, name, unique_id=None, extra_pnginfo=None):
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

        #TODO: make this node's text string preview widget work
        return {"ui": {"text": text}, "result": (text,)}


# Mapping class names to objects for potential export
NODE_CLASS_MAPPINGS = {
    "Nilor Interpolated Float List": NilorInterpolatedFloatList,
    "Nilor Int To List Of Bools": NilorIntToListOfBools,
    "Nilor List of Ints": NilorListOfInts,
    "Nilor Count Images In Directory": NilorCountImagesInDirectory,
    "Nilor Save Image To HF Dataset": NilorSaveImageToHFDataset,
    "Nilor Save Video To HF Dataset": NilorSaveVideoToHFDataset,
    "Nilor Select Index From List": NilorSelectIndexFromList,
    "Nilor Save EXR Arbitrary": NilorSaveEXRArbitrary,
    "Nilor Shuffle Image Batch": NilorShuffleImageBatch,
    "Nilor Repeat & Trim Image Batch": NilorRepeatTrimImageBatch,
    "Nilor Output Filename String": NilorOutputFilenameString

}

# Mapping nodes to human-readable names
NODE_DISPLAY_NAME_MAPPINGS = {
    "Nilor Interpolated Float List": "👺 Interpolated Float List",
    "Nilor Int To List Of Bools": "👺 Int To List Of Bools",
    "Nilor List of Ints": "👺 List of Ints",
    "Nilor Count Images In Directory": "👺 Count Images In Directory",
    "Nilor Save Image To HF Dataset": "👺 Save Image To HF Dataset",
    "Nilor Save Video To HF Dataset": "👺 Save Video To HF Dataset",
    "Nilor Select Index From List": "👺 Select Index From List",
    "Nilor Save EXR Arbitrary": "👺 Save EXR Arbitrary",
    "Nilor Shuffle Image Batch": "👺 Nilor Shuffle Image Batch",
    "Nilor Repeat & Trim Image Batch": "👺 Nilor Repeat & Trim Image Batch",
    "Nilor Output Filename String": "👺 Nilor Output Filename String"
}
