category = "Nilor Nodes 👺"
subcategories = {
    "io": "/IO",
}

class NilorUserInput_String:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {"default": "my_string_input", "multiline": False}),
                "value": ("STRING", {"default": "", "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value,)

class NilorUserInput_Int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {"default": "my_int_input", "multiline": False}),
                "value": ("INT", {"default": 0}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value,)

class NilorUserInput_Float:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {"default": "my_float_input", "multiline": False}),
                "value": ("FLOAT", {"default": 0.0}),
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value,)

class NilorUserInput_Boolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_name": ("STRING", {"default": "my_bool_input", "multiline": False}),
                "value": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "get_value"
    CATEGORY = category + subcategories["io"]

    def get_value(self, input_name, value):
        return (value,)

NODE_CLASS_MAPPINGS = {
    "NilorUserInput_String": NilorUserInput_String,
    "NilorUserInput_Int": NilorUserInput_Int,
    "NilorUserInput_Float": NilorUserInput_Float,
    "NilorUserInput_Boolean": NilorUserInput_Boolean,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NilorUserInput_String": "👺 User Input (String)",
    "NilorUserInput_Int": "👺 User Input (Int)",
    "NilorUserInput_Float": "👺 User Input (Float)",
    "NilorUserInput_Boolean": "👺 User Input (Boolean)",
}
