"""Video helper nodes for ComfyUI-UniverSR.

Modelled directly on the HunyuanVideo-FoleyTune video loader/combiner (same
upload widget, drag-drop, and inline preview via web/js/UniverSRVideo.js), but
the loader outputs the video's **audio** alongside a video reference instead of
visual features — so you can super-resolve the audio and remux it back:

    UniverSR Load Video Audio  -> UNIVERSR_VIDEO + AUDIO   (ffmpeg audio extract + preview)
    UniverSR Video Combiner    -> STRING (output path)     (ffmpeg mux, no video re-encode)

ffmpeg must be on PATH. Audio is read through a WAV pipe with soundfile, avoiding
torchaudio's fragile torchcodec backend (same reasoning as the SR node).
"""

import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile

import torch

try:
    import folder_paths
    HAS_FOLDER_PATHS = True
except Exception:  # pragma: no cover
    HAS_FOLDER_PATHS = False

VIDEO_EXTENSIONS = {"webm", "mp4", "mkv", "gif", "mov", "avi", "flv", "wmv", "m4v", "mpg", "mpeg", "ts"}


# --------------------------------------------------------------------------- #
#  ffmpeg helpers
# --------------------------------------------------------------------------- #
def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (e.g. `apt install ffmpeg`, "
            "`brew install ffmpeg`, or `conda install -c conda-forge ffmpeg`) to use the video nodes."
        )
    return exe


def _trim_args(start_time: float, duration: float) -> list:
    args = []
    if start_time and start_time > 0:
        args += ["-ss", f"{float(start_time):.6f}"]
    if duration and duration > 0:
        args += ["-t", f"{float(duration):.6f}"]
    return args


def _extract_audio(path: str, start_time: float = 0.0, duration: float = 0.0):
    """Extract a video's audio track -> (waveform [1, C, L] float32, sample_rate).

    Native sample rate / channel count, no resampling. Accurate (post-input) seek.
    """
    import soundfile as sf
    cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error",
           "-i", str(path), *_trim_args(start_time, duration),
           "-vn", "-f", "wav", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr.decode('utf-8', 'replace').strip()}")
    if not result.stdout:
        raise RuntimeError(f"No audio stream found in: {path}")
    wav_np, sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)  # [L, C]
    wav = torch.from_numpy(wav_np).T.unsqueeze(0).contiguous()  # [1, C, L]
    return wav, int(sr)


def _write_temp_wav(audio: dict) -> str:
    """Write a ComfyUI AUDIO dict to a temp WAV, return the path."""
    import soundfile as sf
    wav = audio["waveform"]
    if wav.dim() == 3:
        wav = wav[0]            # [C, L]
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)  # [1, L]
    wav_np = wav.detach().cpu().float().numpy().T  # [L, C]
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, wav_np, int(audio["sample_rate"]))
    return tmp


def _temp_preview_symlink(path: str) -> str:
    """Link `path` into ComfyUI's temp/ dir for an inline preview; return the temp filename."""
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".mp4"
    name = f"universr_preview_{hashlib.md5(path.encode()).hexdigest()[:8]}{ext}"
    dst = os.path.join(temp_dir, name)
    if os.path.islink(dst) or os.path.exists(dst):
        try:
            os.unlink(dst)
        except OSError:
            pass
    try:
        os.symlink(os.path.abspath(path), dst)
    except OSError:
        shutil.copy(path, dst)  # filesystems without symlink support
    return name


def _list_input_videos() -> list:
    if not HAS_FOLDER_PATHS:
        return []
    try:
        in_dir = folder_paths.get_input_directory()
        return sorted(
            f for f in os.listdir(in_dir)
            if os.path.isfile(os.path.join(in_dir, f))
            and f.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS
        )
    except Exception:
        return []


def _load_video_audio(video_path: str, start_time: float, duration: float) -> dict:
    """Shared loader body: extract audio + build the (video, audio) result and preview."""
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    waveform, sr = _extract_audio(video_path, start_time, duration)
    dur = waveform.shape[-1] / max(sr, 1)
    print(f"[UniverSR] Loaded audio from {os.path.basename(video_path)}: "
          f"{waveform.shape[1]}ch @ {sr} Hz ({dur:.2f}s)")

    audio = {"waveform": waveform, "sample_rate": sr}
    info = {"video_path": os.path.abspath(video_path), "start_time": float(start_time),
            "duration": float(duration), "source_sr": sr, "source_channels": int(waveform.shape[1])}

    temp_name = _temp_preview_symlink(video_path)
    ext = (os.path.splitext(video_path)[1] or ".mp4").lstrip(".")
    return {"ui": {"gifs": [{"filename": temp_name, "subfolder": "", "type": "temp",
                             "format": f"video/{ext}"}]},
            "result": (info, audio)}


# --------------------------------------------------------------------------- #
#  Load Video Audio  (mirrors FoleyTuneVideoLoaderUpload; outputs video + audio)
# --------------------------------------------------------------------------- #
class UniverSRLoadVideoAudio:
    """Upload/select a video, extract its audio track, and keep a reference to remux later.

    Outputs the video reference (-> UniverSR Video Combiner) and the AUDIO
    (-> UniverSR Super-Resolution). The video previews inline in the node.
    """

    DESCRIPTION = "Load a video: outputs its audio (to super-resolve) and a reference (to remux)."
    CATEGORY = "audio/UniverSR"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (_list_input_videos(), {"video_upload": True}),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360000.0, "step": 0.1,
                               "tooltip": "Trim start in seconds."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360000.0, "step": 0.1,
                             "tooltip": "Trim length in seconds (0 = to end)."}),
            },
        }

    RETURN_TYPES = ("UNIVERSR_VIDEO", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "load"
    OUTPUT_NODE = True

    def load(self, video, start_time=0.0, duration=0.0):
        video_path = folder_paths.get_annotated_filepath(video)
        return _load_video_audio(video_path, start_time, duration)

    @classmethod
    def IS_CHANGED(cls, video, start_time=0.0, duration=0.0):
        try:
            p = folder_paths.get_annotated_filepath(video)
            m = os.path.getmtime(p) if os.path.isfile(p) else 0
        except Exception:
            m = 0
        return f"{video}:{start_time}:{duration}:{m}"

    @classmethod
    def VALIDATE_INPUTS(cls, video, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


# --------------------------------------------------------------------------- #
#  Load Video Audio (Path)  (mirrors FoleyTuneVideoLoader; outputs video + audio)
# --------------------------------------------------------------------------- #
class UniverSRLoadVideoAudioPath:
    """Same as UniverSR Load Video Audio, but takes an absolute file path instead of
    an upload — handy for files outside ComfyUI's input/ folder. Previews after running."""

    DESCRIPTION = "Load a video by file path: outputs its audio (to super-resolve) and a reference (to remux)."
    CATEGORY = "audio/UniverSR"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "placeholder": "/path/to/video.mp4"}),
            },
            "optional": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360000.0, "step": 0.1,
                               "tooltip": "Trim start in seconds."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360000.0, "step": 0.1,
                             "tooltip": "Trim length in seconds (0 = to end)."}),
            },
        }

    RETURN_TYPES = ("UNIVERSR_VIDEO", "AUDIO")
    RETURN_NAMES = ("video", "audio")
    FUNCTION = "load"
    OUTPUT_NODE = True

    def load(self, video_path, start_time=0.0, duration=0.0):
        return _load_video_audio((video_path or "").strip(), start_time, duration)

    @classmethod
    def IS_CHANGED(cls, video_path, start_time=0.0, duration=0.0):
        p = (video_path or "").strip()
        m = os.path.getmtime(p) if p and os.path.isfile(p) else 0
        return f"{video_path}:{start_time}:{duration}:{m}"


# --------------------------------------------------------------------------- #
#  Video Combiner  (mirrors FoleyTuneVideoCombiner)
# --------------------------------------------------------------------------- #
class UniverSRVideoCombiner:
    """Mux audio onto the source video (no video re-encode) and save the result."""

    DESCRIPTION = "Remux the enhanced audio onto the original video with ffmpeg (video stream copied)."
    CATEGORY = "audio/UniverSR"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("UNIVERSR_VIDEO",),
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "UniverSR"}),
            },
            "optional": {
                "audio_codec": (["aac", "flac", "pcm_s16le", "libopus", "libmp3lame"], {
                    "default": "aac",
                    "tooltip": "Codec for the muxed audio track. aac is the safe default for MP4.",
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save to the ComfyUI output/ folder (else temp/).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "combine"
    OUTPUT_NODE = True

    def combine(self, video, audio, filename_prefix="UniverSR", audio_codec="aac", save_output=True):
        source_video = video["video_path"]
        if not os.path.isfile(source_video):
            raise FileNotFoundError(f"Source video not found: {source_video}")
        src_ext = os.path.splitext(source_video)[1] or ".mp4"

        if HAS_FOLDER_PATHS:
            out_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
            out_type = "output" if save_output else "temp"
            full_folder, filename, _, _, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        else:  # standalone fallback
            out_dir = os.path.abspath("universr_output")
            out_type = "output"
            full_folder, filename = out_dir, filename_prefix
        os.makedirs(full_folder, exist_ok=True)

        # Auto-increment counter (VHS-style), scoped to this prefix.
        max_counter = 0
        matcher = re.compile(rf"{re.escape(filename)}_(\d+)\..+", re.IGNORECASE)
        for f in os.listdir(full_folder):
            m = matcher.fullmatch(f)
            if m:
                max_counter = max(max_counter, int(m.group(1)))
        out_name = f"{filename}_{max_counter + 1:05}{src_ext}"
        out_path = os.path.join(full_folder, out_name)

        # Align the video to the same trim window the audio was extracted with.
        start_time = float(video.get("start_time", 0.0) or 0.0)
        duration = float(video.get("duration", 0.0) or 0.0)

        tmp_wav = _write_temp_wav(audio)
        try:
            cmd = [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                   *_trim_args(start_time, duration), "-i", str(source_video),
                   "-i", tmp_wav,
                   "-c:v", "copy", "-c:a", audio_codec,
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-shortest", str(out_path)]
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg mux failed:\n{result.stderr.decode('utf-8', 'replace').strip()}")
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

        print(f"[UniverSR] Muxed enhanced audio -> {out_path}")

        ui = {}
        if HAS_FOLDER_PATHS:
            subfolder = os.path.relpath(full_folder, out_dir)
            if subfolder == ".":
                subfolder = ""
            ui = {"gifs": [{"filename": out_name, "subfolder": subfolder, "type": out_type,
                            "format": f"video/{src_ext.lstrip('.')}"}]}
        return {"ui": ui, "result": (str(out_path),)}


NODE_CLASS_MAPPINGS = {
    "UniverSRLoadVideoAudio": UniverSRLoadVideoAudio,
    "UniverSRLoadVideoAudioPath": UniverSRLoadVideoAudioPath,
    "UniverSRVideoCombiner": UniverSRVideoCombiner,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniverSRLoadVideoAudio": "UniverSR Load Video Audio",
    "UniverSRLoadVideoAudioPath": "UniverSR Load Video Audio (Path)",
    "UniverSRVideoCombiner": "UniverSR Video Combiner",
}
