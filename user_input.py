category = "Nilor Nodes 👺"
subcategories = {
    "io": "/IO",
}

import random
from datetime import datetime

from .controllers import CONTROLLER_HOOK


class NilorUserInput_String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": (
                    "STRING",
                    {"default": "my_string_input", "multiline": False},
                ),
                "value": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", CONTROLLER_HOOK)
    RETURN_NAMES = ("string", "_controller_hook")
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value, None)


class NilorUserInput_Int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": (
                    "STRING",
                    {"default": "my_int_input", "multiline": False},
                ),
                "value": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT", CONTROLLER_HOOK)
    RETURN_NAMES = ("int", "_controller_hook")
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value, None)


class NilorUserInput_Seed:
    MAX_COMFYUI_SEED = 1125899906842624
    SEED_RANDOM_STATE = None

    @classmethod
    def _ensure_seed_random_state(cls):
        if cls.SEED_RANDOM_STATE is not None:
            return

        initial_random_state = random.getstate()
        random.seed(datetime.now().timestamp())
        cls.SEED_RANDOM_STATE = random.getstate()
        random.setstate(initial_random_state)

    @classmethod
    def generate_random_seed(cls):
        cls._ensure_seed_random_state()

        prev_random_state = random.getstate()
        random.setstate(cls.SEED_RANDOM_STATE)
        seed = random.randint(0, cls.MAX_COMFYUI_SEED)
        cls.SEED_RANDOM_STATE = random.getstate()
        random.setstate(prev_random_state)
        return seed

    @classmethod
    def resolve_seed(cls, value):
        if value in (None, 0, -1):
            return cls.generate_random_seed()
        try:
            return int(value) % (cls.MAX_COMFYUI_SEED + 1)
        except (TypeError, ValueError):
            return cls.generate_random_seed()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": (
                    "STRING",
                    {"default": "my_seed_input", "multiline": False},
                ),
                "value": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": cls.MAX_COMFYUI_SEED,
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT", CONTROLLER_HOOK)
    RETURN_NAMES = ("seed", "_controller_hook")
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    @classmethod
    def IS_CHANGED(
        cls, input_name, value, prompt=None, extra_pnginfo=None, unique_id=None
    ):
        # Force node re-execution while using randomize sentinel values.
        return cls.resolve_seed(value)

    def get_value(
        self, input_name, value, prompt=None, extra_pnginfo=None, unique_id=None
    ):
        value = self.resolve_seed(value)
        return (value, None)


class NilorUserInput_Float:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": (
                    "STRING",
                    {"default": "my_float_input", "multiline": False},
                ),
                "value": ("FLOAT", {"default": 0.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("FLOAT", CONTROLLER_HOOK)
    RETURN_NAMES = ("float", "_controller_hook")
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value, None)


class NilorUserInput_Boolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": (
                    "STRING",
                    {"default": "my_bool_input", "multiline": False},
                ),
                "value": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", CONTROLLER_HOOK)
    RETURN_NAMES = ("boolean", "_controller_hook")
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value, None)


NODE_CLASS_MAPPINGS = {
    "NilorUserInput_String": NilorUserInput_String,
    "NilorUserInput_Int": NilorUserInput_Int,
    "NilorUserInput_Seed": NilorUserInput_Seed,
    "NilorUserInput_Float": NilorUserInput_Float,
    "NilorUserInput_Boolean": NilorUserInput_Boolean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NilorUserInput_String": "👺 User Input (String)",
    "NilorUserInput_Int": "👺 User Input (Int)",
    "NilorUserInput_Seed": "👺 User Input (Seed)",
    "NilorUserInput_Float": "👺 User Input (Float)",
    "NilorUserInput_Boolean": "👺 User Input (Boolean)",
}
