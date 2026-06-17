# Design: `auto` input_sr mode

Date: 2026-06-17

## Problem

`input_sr` is the most important and most confusing knob on the Sampler: it tells UniverSR the
**effective bandwidth** of the content (everything above `input_sr/2` is regenerated). Picking it wrong
either wastes the model on an empty band (Nyquist set too high → model treats empty band as valid and
won't regenerate it) or needlessly discards real content (Nyquist too low). An `auto` mode should read
the audio's spectrum and choose for the user.

## Feasibility (validated empirically)

- Naive "highest bin above a threshold" **fails** — it catches resampler skirts / leakage.
- A **cliff/edge detector with a confidence score** works: on broadband audio with a sharp cutoff it
  recovers 4/6/8/12 kHz to within ~60 Hz with a 50–60 dB drop; on full-band/soft-rolloff it reports a
  small drop and abstains.

## Algorithm — `detect_input_sr(waveform, sr) -> (input_sr:int, info:dict)`

1. Mono-mix (mean over batch + channels); analyze at the **native** sample rate.
2. Average magnitude spectrum over time (n_fft=4096, Hann, hop n_fft//4), smoothed (avg-pool ~9 bins),
   converted to dB relative to peak.
3. Find the **steepest negative gradient** (cliff). Confidence = `median(level just below) -
   median(level just above)` in dB.
4. **Effective cutoff** = cliff frequency if a confident cliff (drop ≥ `CONF_DB` ≈ 25) exists below
   ~0.95·(sr/2); **else sr/2** (content fills its band → no edge). This single rule covers:
   - 48 kHz file band-limited to 8 kHz → cliff 8 kHz → 16000.
   - 48 kHz full-band → no cliff → 24 kHz → 24000 (the "unsure" fallback, emerges naturally).
   - Genuine 8 kHz file → content to its 4 kHz Nyquist, no cliff → sr/2 = 4 kHz → 8000.
5. **Map:** largest supported Nyquist ≤ cutoff (+~300 Hz snap), `input_sr = Nyquist × 2`, clamped to
   [8000, 24000]. Rounding **down** is deliberate (avoids the empty-band failure).

## UX

- Dropdown becomes `["auto", "8000", "12000", "16000", "24000"]`, default `"auto"`.
- Resolved in the Sampler `run()`: if `input_sr == "auto"`, call `detect_input_sr`, log the decision,
  pass the concrete int to `super_resolve` (which keeps taking a plain int).
- Console log: `auto: cutoff 8.0 kHz (drop 53 dB) -> input_sr=16000` or
  `auto: no clear cutoff -> input_sr=24000`.

## Edge cases

- Stereo / batch → mono-mix for analysis (one `input_sr` applies to the whole batch).
- Very long clips → one STFT over the whole signal; cost negligible vs the ODE.
- Empty/too-short audio → fall back to 24000.

## Testing

Unit-test `detect_input_sr` on synthetic signals:
- brick-wall broadband at 4/6/8/12 kHz → 8000/12000/16000/24000,
- full-band broadband → 24000,
- a "genuine 8 kHz" clip (sr=8000, full band) → 8000.

## Out of scope (YAGNI)

No new output ports, no sensitivity slider, no per-chunk re-detection.
