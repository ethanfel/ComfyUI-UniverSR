"""Tests for auto input_sr bandwidth detection (detect_input_sr).

Runnable with pytest, or standalone:  python tests/test_auto_input_sr.py
Only needs torch (no ComfyUI). Loads universr_wrapper by path.
"""
import importlib.util
import os

import torch

_ND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("_usr", os.path.join(_ND, "universr_wrapper.py"))
usr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(usr)

SR = 48000


def _broadband(seconds=3.0, sr=SR):
    torch.manual_seed(0)
    return (torch.randn(int(sr * seconds)) * 0.3)


def _brickwall(x, sr, cut):
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(x.shape[-1], 1 / sr)
    X[f > cut] = 0
    return torch.fft.irfft(X, n=x.shape[-1])


def test_sharp_cutoffs_map_to_expected_input_sr():
    base = _broadband()
    for cut, expected in [(4000, 8000), (6000, 12000), (8000, 16000), (12000, 24000)]:
        x = _brickwall(base, SR, cut).reshape(1, 1, -1)
        isr, info = usr.detect_input_sr(x, SR)
        assert info["confident"], f"cut={cut}: expected a confident cliff, got {info}"
        assert isr == expected, f"cut={cut}: got input_sr={isr} ({info['cutoff_hz']:.0f} Hz)"


def test_full_band_falls_back_to_24000():
    x = _broadband().reshape(1, 1, -1)
    isr, info = usr.detect_input_sr(x, SR)
    assert not info["confident"]
    assert isr == 24000, info


def test_genuine_low_rate_file_maps_to_its_rate():
    # A real 8 kHz file: sr=8000, content fills its 4 kHz band -> input_sr 8000.
    torch.manual_seed(1)
    x = (torch.randn(8000 * 3) * 0.3).reshape(1, 1, -1)
    isr, info = usr.detect_input_sr(x, 8000)
    assert isr == 8000, info


def test_silent_or_tiny_falls_back():
    isr, info = usr.detect_input_sr(torch.zeros(1, 1, 1000), SR)
    assert isr == 24000 and not info["confident"]


def test_stereo_is_mono_mixed():
    base = _broadband()
    x = _brickwall(base, SR, 8000)
    stereo = torch.stack([x, x], 0).reshape(1, 2, -1)
    isr, _ = usr.detect_input_sr(stereo, SR)
    assert isr == 16000


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
