"""ComfyUI-UniverSR nodes.

Two-node design (mirrors the ComfyUI-Flash-AudioSR pattern):
    UniverSRModelLoader  -> UNIVERSR_MODEL  (loads + caches weights, auto-downloads)
    UniverSRSampler      -> AUDIO, IMAGE    (runs the super-resolution)
"""

import torch

from . import universr_wrapper as usr

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except Exception:  # pragma: no cover
    HAS_COMFY = False


def _default_device() -> str:
    if HAS_COMFY:
        try:
            return "cuda" if mm.get_torch_device().type == "cuda" else "cpu"
        except Exception:
            pass
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
#  Model loader
# --------------------------------------------------------------------------- #
class UniverSRModelLoader:
    """Load a UniverSR checkpoint. Auto-downloads the presets on first use.

    Output: UNIVERSR_MODEL -> connect to UniverSR Super-Resolution.
    """

    DESCRIPTION = ("Load UniverSR (vocoder-free audio super-resolution, ICASSP 2026). "
                   "Presets auto-download to models/universr on first use.")
    CATEGORY = "audio/UniverSR"

    @classmethod
    def INPUT_TYPES(cls):
        choices = list(usr.HF_REPOS.keys()) + usr.list_local_models()
        return {
            "required": {
                "model": (choices, {
                    "default": choices[0],
                    "tooltip": "universr-audio = general (music/SFX/mixed, recommended); "
                               "universr-speech = voice only. Both download (~230 MB) on first use. "
                               "Local checkpoint folders in models/universr also appear here.",
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to load the model onto.",
                }),
            },
            "optional": {
                "local_path": ("STRING", {
                    "default": "",
                    "tooltip": "Override: a folder with config.yaml + pytorch_model.bin, "
                               "or a raw .pth/.ckpt file (uses config_path or the bundled config).",
                }),
                "config_path": ("STRING", {
                    "default": "",
                    "tooltip": "config.yaml for a raw checkpoint given in local_path. "
                               "Leave empty to use the bundled default config.",
                }),
            },
        }

    RETURN_TYPES = ("UNIVERSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"

    def load(self, model, device, local_path="", config_path=""):
        dev = _default_device() if device == "auto" else device
        if dev == "cuda" and not torch.cuda.is_available():
            print("[UniverSR] CUDA unavailable, falling back to CPU")
            dev = "cpu"
        model_obj, cache_key = usr.load_model(model, dev, local_path=local_path, config_path=config_path)
        return ({"model": model_obj, "device": dev, "cache_key": cache_key},)

    @classmethod
    def IS_CHANGED(cls, model, device, local_path="", config_path=""):
        return f"{model}:{device}:{local_path}:{config_path}"


# --------------------------------------------------------------------------- #
#  Sampler
# --------------------------------------------------------------------------- #
class UniverSRSampler:
    """Super-resolve audio to 48 kHz with UniverSR. Long clips are processed in
    overlapping chunks (click-free overlap-add) to stay within VRAM."""

    DESCRIPTION = ("Upscale low-bandwidth audio to 48 kHz with UniverSR. Pick input_sr to "
                   "match the effective bandwidth of your content (the model regenerates "
                   "everything above input_sr/2).")
    CATEGORY = "audio/UniverSR"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {}),
                "model": ("UNIVERSR_MODEL", {}),
                "input_sr": ([8000, 12000, 16000, 24000], {
                    "default": 8000,
                    "tooltip": "Effective input bandwidth (Hz). Content is treated as valid up to "
                               "input_sr/2 and regenerated above it. 8000 = genuine low-rate audio "
                               "(strongest, 8 kHz->48 kHz). 16000 = brighten muffled audio above 8 kHz.",
                }),
            },
            "optional": {
                "ode_method": (["midpoint", "euler", "rk4"], {
                    "default": "midpoint",
                    "tooltip": "ODE solver. euler (fastest) -> midpoint (balanced) -> rk4 (best).",
                }),
                "ode_steps": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Flow-matching integration steps. 4 is fast and validated; 4-10 is a good range.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 6.0, "step": 0.25,
                    "tooltip": "Classifier-free guidance. Speech 1.0-1.5, music 1.5-2.0, SFX ~1.5. "
                               "Higher = denser highs but less faithful. 0 disables CFG.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Noise seed for the flow-matching source. 0 = random each run.",
                }),
                "chunk_seconds": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Process long audio in chunks of this length (seconds) to avoid OOM. "
                               "0 = process the whole clip at once.",
                }),
                "overlap_seconds": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Crossfade overlap between chunks (seconds). Prevents seam clicks.",
                }),
                "blend": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Wet/dry mix. 1.0 = full super-resolution. Lower to keep more of the "
                               "original (useful when brightening already-48 kHz audio).",
                }),
                "unload_model": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Free the model from VRAM after this run.",
                }),
                "show_spectrogram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also output a before/after spectrogram comparison image.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "IMAGE")
    RETURN_NAMES = ("audio", "spectrogram")
    FUNCTION = "run"

    def run(self, audio, model, input_sr, ode_method="midpoint", ode_steps=4,
            guidance_scale=1.5, seed=0, chunk_seconds=10.0, overlap_seconds=0.5,
            blend=1.0, unload_model=False, show_spectrogram=True):

        model_obj = model["model"]
        waveform, sr = usr.comfy_audio_to_tensor(audio)
        dur = waveform.shape[-1] / max(sr, 1)
        print(f"[UniverSR] {tuple(waveform.shape)} @ {sr} Hz ({dur:.2f}s) -> 48 kHz | "
              f"input_sr={input_sr}, {ode_method}/{ode_steps}, cfg={guidance_scale}, blend={blend}")

        out, dry48 = usr.super_resolve(
            model_obj, waveform, sr, int(input_sr),
            ode_method=ode_method, ode_steps=int(ode_steps), guidance_scale=guidance_scale,
            seed=int(seed), chunk_seconds=float(chunk_seconds),
            overlap_seconds=float(overlap_seconds), blend=float(blend),
        )

        audio_out = usr.tensor_to_comfy_audio(out, usr.TARGET_SR)

        spec = torch.zeros(1, 64, 64, 3)
        if show_spectrogram:
            in_mono = dry48[0].mean(0).numpy()
            out_mono = out[0].mean(0).numpy()
            spec = usr.make_spectrogram_image(in_mono, out_mono, int(input_sr))

        if unload_model:
            usr.evict_model(model["cache_key"])

        print(f"[UniverSR] Done -> {out.shape[-1] / usr.TARGET_SR:.2f}s at 48 kHz")
        return (audio_out, spec)


NODE_CLASS_MAPPINGS = {
    "UniverSRModelLoader": UniverSRModelLoader,
    "UniverSRSampler": UniverSRSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniverSRModelLoader": "UniverSR Model Loader",
    "UniverSRSampler": "UniverSR Super-Resolution",
}
