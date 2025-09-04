category = "Nilor Nodes 👺"
subcategories = {
    "io": "/IO",
}

# Unique hook type for controller wiring (used by both Preset and Group controllers)
CONTROLLER_HOOK = "CONTROLLER_HOOK"


class NilorPreset:
    """
    Declarative controller that binds a Brando preset group to a set of connected inputs.

    - preset_group_name: Semantic key used to look up choices and values in
      presets_config.json5 via PresetsService (drives dropdown + value application).
    - _preset_hook_*: Dynamic inputs that accept CONTROLLER_HOOK from NilorUserInput_* `_controller_hook` outputs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Start with a single hook; dynamic inputs handled by companion JS
        optional_inputs = {"_preset_hook_1": (CONTROLLER_HOOK,)}
        return {
            "required": {
                # Lookup key in presets_config.json5 (NOT a UI label)
                "preset_group_name": ("STRING", {"default": "my_preset", "multiline": False}),
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


class NilorGroup:
    """
    Declarative UI-grouper that clusters connected inputs together in the Brando UI.

    - group_label: Purely a visual label for the gr.Group that will contain the inputs.
      It does NOT look up presets or apply values.
    - _group_hook_*: Dynamic inputs that accept CONTROLLER_HOOK from NilorUserInput_* `_controller_hook` outputs.
      Reuses a shared controller hook so no additional output types are required on input nodes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Start with a single hook; dynamic inputs handled by companion JS
        optional_inputs = {"_group_hook_1": (CONTROLLER_HOOK,)}
        return {
            "required": {
                # UI label only (NOT used to look up presets)
                "group_label": ("STRING", {"default": "my_group", "multiline": False}),
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
        # Declarative only
        return tuple()


NODE_CLASS_MAPPINGS = {
    "NilorPreset": NilorPreset,
    "NilorGroup": NilorGroup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NilorPreset": "👺 User Input Preset Controller",
    "NilorGroup": "👺 User Input Group Controller",
}


