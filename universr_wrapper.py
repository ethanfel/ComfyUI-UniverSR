"""Core wrapper for ComfyUI-UniverSR.

Bootstraps the `universr` package (prefers a pip-installed copy, falls back to
the vendored one under ./vendor), manages model loading/caching, and runs the
super-resolution itself with optional overlap-add chunking for long audio.

UniverSR (ICASSP 2026) is a vocoder-free audio super-resolution model that
regenerates high-frequency content in the complex-STFT domain via flow matching.
A single model handles 8 / 12 / 16 / 24 kHz effective bandwidth -> 48 kHz.

Key design note — why we resample ourselves instead of handing UniverSR a file:
    UniverSR's `enhance()` file path calls `torchaudio.load`, whose torchcodec
    backend is fragile across environments; its *tensor* path assumes the tensor
    is already at `input_sr`. ComfyUI audio arrives at an arbitrary real sample
    rate, so we do the band-limit ourselves: resample to 48 kHz, downsample each
    chunk to `input_sr` (pure DSP, no codec), and hand UniverSR a genuine
    low-rate tensor to super-resolve. This reproduces the exact training-time
    degradation and was validated in the FoleyTune BWE node.
"""

import os
import threading

import numpy as np
import torch
import torchaudio

# --------------------------------------------------------------------------- #
#  Optional ComfyUI integration (degrade gracefully outside ComfyUI / in tests)
# --------------------------------------------------------------------------- #
try:
    import comfy.model_management as mm
    import comfy.utils
    HAS_COMFY = True
except Exception:  # pragma: no cover - allows standalone import / pytest
    HAS_COMFY = False

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except Exception:  # pragma: no cover
    HAS_FOLDER_PATHS = False


TARGET_SR = 48_000
SUPPORTED_INPUT_SR = (8000, 12000, 16000, 24000)
# UniverSR.enhance() zero-pads anything shorter than this (≈0.68 s @ 48 kHz) before
# running the ODE, so chunks below it just waste compute — clamp to it.
MODEL_MIN_SAMPLES = 32_768
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_NODE_DIR, "vendor")
_BUNDLED_CONFIG = os.path.join(_NODE_DIR, "configs", "config.yaml")

# HuggingFace repos for the two released checkpoints.
HF_REPOS = {
    "universr-audio": "woongzip1/universr-audio",
    "universr-speech": "woongzip1/universr-speech",
}


# --------------------------------------------------------------------------- #
#  Package bootstrap
# --------------------------------------------------------------------------- #
def get_universr_cls():
    """Return the `UniverSR` class, preferring an installed copy over the vendored one."""
    try:
        from universr import UniverSR  # installed (e.g. via the FoleyTune node)
        return UniverSR
    except Exception:
        pass
    import sys
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    try:
        from universr import UniverSR  # vendored fallback
        return UniverSR
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import the 'universr' package (neither installed nor vendored). "
            "Try: pip install torchdiffeq  (the only dependency ComfyUI does not already ship).\n"
            f"Underlying error: {e}"
        )


# --------------------------------------------------------------------------- #
#  Model directory + cache
# --------------------------------------------------------------------------- #
def get_models_dir() -> str:
    if HAS_FOLDER_PATHS:
        base = folder_paths.models_dir
    else:
        base = os.path.join(_NODE_DIR, "..", "..", "models")
    return os.path.abspath(os.path.join(base, "universr"))


def list_local_models() -> list:
    """Subdirectories of models/universr that look like a UniverSR checkpoint dir."""
    root = get_models_dir()
    found = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.yaml")) \
                    and os.path.exists(os.path.join(d, "pytorch_model.bin")):
                if name not in HF_REPOS:
                    found.append(name)
    return found


_MODEL_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _download_preset(name: str) -> str:
    """Download a preset checkpoint into models/universr/<name> and return that dir."""
    from huggingface_hub import snapshot_download
    repo_id = HF_REPOS[name]
    target = os.path.join(get_models_dir(), name)
    have = os.path.exists(os.path.join(target, "config.yaml")) and \
        os.path.exists(os.path.join(target, "pytorch_model.bin"))
    if not have:
        os.makedirs(target, exist_ok=True)
        print(f"[UniverSR] Downloading {repo_id} -> {target} (~230 MB)...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            allow_patterns=["config.yaml", "pytorch_model.bin"],
        )
        print(f"[UniverSR] Downloaded {name}.")
    return target


def resolve_model_ref(model: str, local_path: str = "") -> tuple:
    """Resolve the loader inputs to (kind, path). kind in {'dir', 'ckpt'}.

    - local_path wins if set: a directory (config.yaml + pytorch_model.bin) -> 'dir';
      a .pth/.pt/.ckpt file -> 'ckpt' (loaded via from_local with a config).
    - a preset name ('universr-audio' / 'universr-speech') -> download -> 'dir'.
    - a local subdir name discovered under models/universr -> 'dir'.
    """
    local_path = (local_path or "").strip()
    if local_path:
        if os.path.isdir(local_path):
            return ("dir", local_path)
        if os.path.isfile(local_path):
            return ("ckpt", local_path)
        raise FileNotFoundError(f"local_path does not exist: {local_path}")

    if model in HF_REPOS:
        return ("dir", _download_preset(model))

    cand = os.path.join(get_models_dir(), model)
    if os.path.isdir(cand):
        return ("dir", cand)
    raise FileNotFoundError(
        f"Unknown model '{model}'. Use a preset {list(HF_REPOS)}, a local subdir of "
        f"{get_models_dir()}, or set local_path."
    )


def apply_tf32(enabled: bool):
    """Enable/disable TF32 for BOTH matmul and cuDNN convolutions on Ampere+ GPUs.

    ~1.15x when on. In our spectral A/B (centroid + HF energy) TF32 was tonally
    neutral, but it is NOT bit-exact (10 mantissa bits vs 23), so it's off by
    default. Off sets true fp32 — note PyTorch otherwise leaves cuDNN conv-TF32 ON
    by default, so we explicitly disable it here too. Global process setting."""
    try:
        torch.set_float32_matmul_precision("high" if enabled else "highest")  # matmul TF32
        torch.backends.cudnn.allow_tf32 = enabled                              # conv TF32
    except Exception:
        pass


def load_model(model: str, device: str, local_path: str = "", config_path: str = "",
               tf32: bool = False, compile_model: bool = False):
    """Load (and cache) a UniverSR model. Returns (model_obj, cache_key)."""
    apply_tf32(tf32)  # global; apply before the cache short-circuit so toggling takes effect
    kind, path = resolve_model_ref(model, local_path)
    cache_key = f"{kind}:{os.path.abspath(path)}:{device}:compile={bool(compile_model)}"

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            print(f"[UniverSR] Using cached model ({cache_key})")
            return _MODEL_CACHE[cache_key], cache_key

        UniverSR = get_universr_cls()
        if kind == "dir":
            print(f"[UniverSR] Loading from_pretrained({path}) on {device}")
            model_obj = UniverSR.from_pretrained(path, device=device)
        else:
            cfg = (config_path or "").strip() or _BUNDLED_CONFIG
            if not os.path.exists(cfg):
                raise FileNotFoundError(
                    f"config_path required for a raw checkpoint and not found: {cfg}"
                )
            print(f"[UniverSR] Loading from_local(ckpt={path}, config={cfg}) on {device}")
            model_obj = UniverSR.from_local(ckpt_path=path, config_path=cfg, device=device)

        model_obj.eval()

        # torch.compile the UNet (~2.1x measured). Static shapes only — the model's
        # adaptive-avg-pool can't trace dynamic shapes — so the sampler pads every
        # chunk to a fixed length (see _universr_compiled flag) to compile exactly once.
        compiled = False
        if compile_model and device == "cuda":
            try:
                import torch._dynamo as _dynamo
                _dynamo.config.cache_size_limit = max(getattr(_dynamo.config, "cache_size_limit", 8), 32)
                model_obj.model = torch.compile(model_obj.model, mode="default", dynamic=False)
                compiled = True
                print("[UniverSR] torch.compile enabled (first run compiles ~10-35s, then ~2x).")
            except Exception as e:
                print(f"[UniverSR] torch.compile unavailable, continuing eager: {e}")
        model_obj._universr_compiled = compiled

        n = sum(p.numel() for p in model_obj.parameters()) / 1e6
        print(f"[UniverSR] Ready - {n:.1f}M params on {device} (tf32={tf32}, compile={compiled})")
        _MODEL_CACHE[cache_key] = model_obj
        return model_obj, cache_key


def evict_model(cache_key: str):
    import gc
    with _CACHE_LOCK:
        _MODEL_CACHE.pop(cache_key, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[UniverSR] Model unloaded ({cache_key})")


# --------------------------------------------------------------------------- #
#  Audio helpers
# --------------------------------------------------------------------------- #
def comfy_audio_to_tensor(audio) -> tuple:
    """ComfyUI AUDIO (dict or legacy tuple) -> (waveform [B, C, T] float32 cpu, sr)."""
    if isinstance(audio, dict):
        waveform, sr = audio["waveform"], audio["sample_rate"]
    else:
        waveform, sr = audio
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()
    if waveform.dim() == 1:        # (T,)
        waveform = waveform[None, None, :]
    elif waveform.dim() == 2:      # (C, T)
        waveform = waveform[None, :, :]
    return waveform, int(sr)


def tensor_to_comfy_audio(waveform: torch.Tensor, sr: int) -> dict:
    if waveform.dim() == 1:
        waveform = waveform[None, None, :]
    elif waveform.dim() == 2:
        waveform = waveform[None, :, :]
    return {"waveform": waveform.detach().cpu().contiguous(), "sample_rate": int(sr)}


def _resample(x: torch.Tensor, orig: int, target: int) -> torch.Tensor:
    if orig == target:
        return x
    return torchaudio.functional.resample(x, orig, target)


def _fit(x: torch.Tensor, n: int) -> torch.Tensor:
    """Crop or zero-pad a 1-D tensor to exactly n samples."""
    if x.shape[-1] == n:
        return x
    if x.shape[-1] > n:
        return x[:n]
    return torch.nn.functional.pad(x, (0, n - x.shape[-1]))


def _crossfade_window(length: int, ov: int, first: bool, last: bool) -> torch.Tensor:
    """Linear fade-in/out over the overlap regions; flat 1.0 elsewhere.

    Combined with weight-sum normalisation this gives click-free overlap-add.
    """
    w = torch.ones(length)
    if ov > 0:
        f = min(ov, length)
        if not first:
            w[:f] = torch.minimum(w[:f], torch.linspace(0.0, 1.0, f))
        if not last:
            w[-f:] = torch.minimum(w[-f:], torch.linspace(1.0, 0.0, f))
    return w


# --------------------------------------------------------------------------- #
#  Inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _enhance_segment(model, seg48: torch.Tensor, input_sr: int,
                     ode_method: str, ode_steps: int, guidance_scale) -> torch.Tensor:
    """Super-resolve one 48 kHz mono segment. Returns 1-D tensor @48 kHz on CPU."""
    low = _resample(seg48.unsqueeze(0), TARGET_SR, input_sr).squeeze(0)  # genuine LR-rate signal
    cfg = float(guidance_scale) if (guidance_scale and guidance_scale > 0) else None
    out = model.enhance(
        low, input_sr=int(input_sr),
        ode_method=ode_method, ode_steps=int(ode_steps), guidance_scale=cfg,
    )
    return out.reshape(-1).float().cpu()


def _chunk_starts(total: int, chunk: int, hop: int) -> list:
    if chunk <= 0 or total <= chunk:
        return [0]
    starts = list(range(0, max(1, total - chunk) + 1, hop))
    if starts[-1] + chunk < total:
        starts.append(total - chunk)
    return starts


@torch.no_grad()
def _enhance_channel(model, ch48: torch.Tensor, input_sr, ode_method, ode_steps,
                     guidance_scale, chunk: int, ov: int, pbar, pad_to: int = 0) -> torch.Tensor:
    T = ch48.shape[-1]

    def seg_enhance(seg: torch.Tensor) -> torch.Tensor:
        # pad_to>0 (compiled model) → zero-pad to a fixed length so the UNet always
        # sees one input shape (compile once), then crop the result back.
        L = seg.shape[-1]
        if pad_to and pad_to > L:
            seg = _fit(seg, pad_to)
        return _fit(_enhance_segment(model, seg, input_sr, ode_method, ode_steps, guidance_scale), L)

    if chunk <= 0 or T <= chunk:
        if pbar is not None:
            pbar.update(1)
        return _fit(seg_enhance(ch48), T)

    hop = max(1, chunk - ov)
    starts = _chunk_starts(T, chunk, hop)
    out = torch.zeros(T)
    wsum = torch.zeros(T)
    for i, s in enumerate(starts):
        if HAS_COMFY:
            mm.throw_exception_if_processing_interrupted()
        e = min(s + chunk, T)
        enh = seg_enhance(ch48[s:e])
        w = _crossfade_window(e - s, ov, first=(i == 0), last=(e >= T))
        out[s:e] += enh * w
        wsum[s:e] += w
        if pbar is not None:
            pbar.update(1)
    return out / torch.clamp(wsum, min=1e-8)


@torch.no_grad()
def super_resolve(model, waveform: torch.Tensor, sr: int, input_sr: int,
                  ode_method: str = "midpoint", ode_steps: int = 4,
                  guidance_scale=1.5, seed: int = 0,
                  chunk_seconds: float = 10.0, overlap_seconds: float = 0.5,
                  blend: float = 1.0):
    """Run UniverSR over a [B, C, T] waveform. Returns (out [B, C, T48], dry48 [B, C, T48])."""
    if int(input_sr) not in SUPPORTED_INPUT_SR:
        raise ValueError(f"input_sr must be one of {SUPPORTED_INPUT_SR}, got {input_sr}")

    waveform = waveform.float().cpu()
    if waveform.dim() != 3:
        raise ValueError(f"Expected a [B, C, T] waveform, got shape {tuple(waveform.shape)}")
    B, C, _ = waveform.shape
    dry48 = _resample(waveform, sr, TARGET_SR)  # [B, C, T48]
    T48 = dry48.shape[-1]
    if T48 == 0:  # empty input — nothing to do
        empty = torch.zeros(B, C, 0)
        return empty, empty

    chunk = int(round(chunk_seconds * TARGET_SR)) if (chunk_seconds and chunk_seconds > 0) else 0
    if 0 < chunk < MODEL_MIN_SAMPLES:
        print(f"[UniverSR] chunk_seconds too small; raising to the model floor "
              f"({MODEL_MIN_SAMPLES / TARGET_SR:.2f}s).")
        chunk = MODEL_MIN_SAMPLES

    # A torch.compile'd model needs a fixed input shape, so force chunking and pad
    # every chunk to `chunk` samples (compile once, reuse). Without compile, pad_to=0.
    compiled = getattr(model, "_universr_compiled", False)
    if compiled and chunk <= 0:
        chunk = int(round(10.0 * TARGET_SR))
        print("[UniverSR] compile: forcing 10.0s chunks for fixed input shapes.")
    pad_to = chunk if compiled else 0

    ov = max(0, min(int(round(overlap_seconds * TARGET_SR)), chunk - 1)) if chunk > 0 else 0
    n_per_ch = len(_chunk_starts(T48, chunk, max(1, chunk - ov))) if chunk > 0 else 1

    pbar = comfy.utils.ProgressBar(B * C * n_per_ch) if HAS_COMFY else None

    # Isolate the global RNG: snapshot, seed, run, restore. Without this the model's
    # torch.randn_like noise would advance (and a fixed seed would freeze) the global
    # generator that downstream ComfyUI nodes rely on. seed=0 → fresh OS entropy.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    actual_seed = int(seed) if (seed and int(seed) != 0) else int.from_bytes(os.urandom(8), "little")
    try:
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)
        wet = torch.zeros(B, C, T48)
        for b in range(B):
            for c in range(C):
                wet[b, c] = _fit(
                    _enhance_channel(model, dry48[b, c], input_sr, ode_method, ode_steps,
                                     guidance_scale, chunk, ov, pbar, pad_to=pad_to),
                    T48,
                )
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    blend = float(blend)
    out = wet if blend >= 1.0 else (1.0 - blend) * dry48 + blend * wet
    return out.clamp(-1.0, 1.0), dry48


# --------------------------------------------------------------------------- #
#  Spectrogram comparison (optional IMAGE output)
# --------------------------------------------------------------------------- #
def _stft_db(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    win = torch.hann_window(1024)
    spec = torch.stft(t, n_fft=1024, hop_length=512, window=win, return_complex=True)
    db = 20.0 * torch.log10(spec.abs().clamp(min=1e-5))
    db = db - db.max()
    return db.numpy()


def make_spectrogram_image(input48_mono: np.ndarray, output48_mono: np.ndarray,
                           input_sr: int) -> torch.Tensor:
    """Before/after spectrogram comparison -> IMAGE tensor [1, H, W, 3] in [0, 1].

    Left panel is the band-limited input (content valid up to input_sr/2); right
    panel is the 48 kHz output. The dashed line marks the LR Nyquist, so the
    regenerated high-frequency band is the energy above it on the right.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Visualise the band-limit the model actually saw, not the raw container.
        lr = torch.from_numpy(np.ascontiguousarray(input48_mono)).float()[None]
        lr = _resample(_resample(lr, TARGET_SR, int(input_sr)), int(input_sr), TARGET_SR).squeeze(0).numpy()
        n = min(len(lr), len(output48_mono), int(8.0 * TARGET_SR))
        lr, hr = lr[:n], output48_mono[:n]
        nyq = int(input_sr) / 2.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), facecolor="#0d0f16")
        for ax, sig, title, cmap in (
            (axes[0], lr, f"Input  (<= {int(input_sr)//1000} kHz)", "magma"),
            (axes[1], hr, "UniverSR output  (48 kHz)", "viridis"),
        ):
            db = _stft_db(sig)
            ax.imshow(db, origin="lower", aspect="auto", cmap=cmap,
                      extent=[0, n / TARGET_SR, 0, TARGET_SR / 2], vmin=-80, vmax=0)
            ax.axhline(nyq, color="w", ls="--", lw=0.8, alpha=0.6)
            ax.set_title(title, color="#cfe0ff", fontsize=10)
            ax.set_xlabel("Time (s)", color="#7a93bd", fontsize=8)
            ax.set_ylabel("Hz", color="#7a93bd", fontsize=8)
            ax.tick_params(colors="#5a6e90", labelsize=7)
            ax.set_facecolor("#0d0f16")
        fig.tight_layout()

        fig.canvas.draw()
        # np.asarray(buffer_rgba()) yields (H, W, 4) at the real pixel size — robust to HiDPI.
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
        plt.close(fig)
        return torch.from_numpy(np.ascontiguousarray(img))[None]
    except Exception as e:  # matplotlib missing / headless edge cases
        print(f"[UniverSR] Spectrogram render skipped: {e}")
        return torch.zeros(1, 64, 64, 3)
