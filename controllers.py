category = "Nilor Nodes 👺"
subcategories = {
    "io": "/IO",
}

# Unique hook type for preset wiring
PRESET_HOOK = "PRESET_HOOK"


class NilorPreset:
    @classmethod
    def INPUT_TYPES(cls):
        # Start with a single hook; dynamic inputs handled by companion JS
        optional_inputs = {"_preset_hook_1": (PRESET_HOOK,)}
        return {
            "required": {
                "preset_group_name": ("STRING", {"default": "resolutions", "multiline": False}),
            },
            "optional": optional_inputs,
        }

    # No outputs; declarative controller only
    RETURN_TYPES = tuple()
    RETURN_NAMES = tuple()
    FUNCTION = "do_nothing"
    CATEGORY = category + subcategories["io"]
    OUTPUT_NODE = True

    def do_nothing(self, **kwargs):
        # This node performs no computation; it exists for declarative wiring only
        return tuple()


NODE_CLASS_MAPPINGS = {
    "NilorPreset": NilorPreset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NilorPreset": "👺 UserInput Preset Controller",
}


