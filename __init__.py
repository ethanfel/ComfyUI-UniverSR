"""ComfyUI-UniverSR — vocoder-free audio super-resolution (8/12/16/24 kHz -> 48 kHz)."""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:  # surface import errors in the ComfyUI log without crashing startup
    print(f"[ComfyUI-UniverSR] Failed to load nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
