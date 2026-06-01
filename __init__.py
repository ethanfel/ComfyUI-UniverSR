"""ComfyUI-UniverSR — vocoder-free audio super-resolution (8/12/16/24 kHz -> 48 kHz)."""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS as _sr_nodes, NODE_DISPLAY_NAME_MAPPINGS as _sr_display
    NODE_CLASS_MAPPINGS.update(_sr_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(_sr_display)
except Exception as e:  # surface errors in the ComfyUI log without crashing startup
    print(f"[ComfyUI-UniverSR] Failed to load SR nodes: {e}")

try:
    from .nodes_video import NODE_CLASS_MAPPINGS as _vid_nodes, NODE_DISPLAY_NAME_MAPPINGS as _vid_display
    NODE_CLASS_MAPPINGS.update(_vid_nodes)
    NODE_DISPLAY_NAME_MAPPINGS.update(_vid_display)
except Exception as e:  # video nodes are optional (need ffmpeg/soundfile)
    print(f"[ComfyUI-UniverSR] Failed to load video nodes: {e}")

# Serves web/js/UniverSRVideo.js — the inline video preview + upload widget.
# (./web + web/js/ + ../../../scripts imports mirrors the FoleyTune layout exactly.)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
