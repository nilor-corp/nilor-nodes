import os
import math
import random
from scipy.interpolate import interp1d
from numpy import linspace
import numpy as np


class NilorFloats:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "number_of_frames": ("INT", {"forceInput": False}),
                "number_of_images": ("INT", {"forceInput": False}),
                "image_number": ("INT", {"forceInput": False}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("floats",)

    FUNCTION = "test"
    CATEGORY = "nilor-nodes"

    @staticmethod
    def interpolate_values(start, end, num_points):
        # Linear interpolation between start and end over num_points
        x = linspace(0, num_points - 1, num_points)
        y = linspace(start, end, num_points)
        f = interp1d(x, y, kind="cubic")
        return f(x)

    def test(self, number_of_frames, number_of_images, image_number):
        # Initializes the array with zeros
        my_floats = [0.0] * number_of_frames
        # Calculate the length of each portion based on total frames and number of images
        portion_length = int((number_of_frames - 1) / (number_of_images - 1))

        # Handling the first image (special case for the first segment)
        if image_number == 1:
            portion_values = NilorFloats.interpolate_values(1, 0, portion_length)
            my_floats[0:portion_length] = portion_values
        # Handling the last image (special case for the last segment)
        elif image_number == number_of_images:
            portion_values = NilorFloats.interpolate_values(0, 1, portion_length)
            start_index = int((number_of_images - 2) * portion_length)
            my_floats[start_index:] = portion_values
        # Handling middle images (general case for dual segments)
        else:
            portion_values = np.concatenate(
                [
                    NilorFloats.interpolate_values(0, 1, portion_length),
                    NilorFloats.interpolate_values(1, 0, portion_length),
                ]
            )
            start_index = int((image_number - 2) * portion_length)
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
    CATEGORY = "nilor-nodes"

    OUTPUT_IS_LIST = (True,)

    def boolify(self, number_of_images, max_images=10):
        # Initializes the array with zeros
        my_bools = [False] * max_images

        for i in range(max_images):
            # Set the boolean value to True if the index is less than the number of images
            my_bools[i] = i < number_of_images

        return (my_bools,)


class NilorBoolFromListOfBools:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "booleans": ("BOOLEAN", {"forceInput": False}),
                "index": ("INT", {"forceInput": False}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)

    FUNCTION = "bool_by_index"
    CATEGORY = "nilor-nodes"

    INPUT_IS_LIST = True

    def bool_by_index(self, booleans, index):
        actual_index = index[0] if isinstance(index, list) else index

        if actual_index < 0 or actual_index >= len(booleans):
            raise ValueError("Index is outside the bounds of the array.")

        # Returns the boolean value at the given index
        desired_bool = booleans[actual_index]
        return [desired_bool]

class NilorIntFromListOfInts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "ints": ("INT", {"forceInput": False}),
                "index": ("INT", {"forceInput": False}),
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)

    FUNCTION = "int_by_index"
    CATEGORY = "nilor-nodes"

    INPUT_IS_LIST = True

    def int_by_index(self, ints, index=0):
        actual_index = index[0] if isinstance(index, list) else index

        if actual_index < 0 or actual_index >= len(ints):
            raise ValueError("Index is outside the bounds of the array.")

        # Returns the int value at the given index
        desired_int = ints[actual_index]
        return [desired_int]

class NilorListOfInts:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Dictionary that defines input types for each field
        return {
            "required": {
                "min": ("INT", {"forceInput": False}, {"default": 0}),
                "max": ("INT", {"forceInput": False}, {"default": 9}),
                "shuffle": ("BOOLEAN", {"default": False}),  # Toggle to randomize order
                "run_trigger": ("INT", {"default": 0}),  # Dummy input for caching issue
            },
        }

    # Define return types and names for outputs of the node
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("ints",)

    FUNCTION = "int_list"
    CATEGORY = "nilor-nodes"

    OUTPUT_IS_LIST = (True,)

    def int_list(self, run_trigger, min=1, max=10, shuffle=False):
        if max < min:
            raise ValueError("Input maximum is less than input minimum.")

        # Create a list of sequential integers from min_value to max_value
        ints_list = list(range(min, max + 1))

        # Shuffle the list
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
    CATEGORY = "nilor-nodes"

    INPUT_IS_LIST = False

    def count_images_in_directory(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory} cannot be found.")

        list_dir = []
        list_dir = os.listdir(directory)
        count = 0
        for file in list_dir:
            if file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg'):
                count += 1
        
        return [count]

# Mapping class names to objects for potential export
NODE_CLASS_MAPPINGS = {
    "Nilor Floats": NilorFloats,
    "Nilor Int To List Of Bools": NilorIntToListOfBools,
    "Nilor Bool From List Of Bools": NilorBoolFromListOfBools,
    "Nilor Int From List Of Ints": NilorIntFromListOfInts,
    "Nilor List of Ints": NilorListOfInts,
    "Nilor Count Images In Directory": NilorCountImagesInDirectory,
}
# Mapping nodes to human-readable names
NODE_DISPLAY_NAME_MAPPINGS = {
    "FirstNode": "My First Node",
    "SecondNode": "My Second Node",
    "ThirdNode": "My Third Node",
}
