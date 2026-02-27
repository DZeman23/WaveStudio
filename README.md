# WaveStudio
A lightweight, Python-based graphical wavetable editor designed to draw, generate, manipulate, and export custom 2048-sample single-cycle waveforms. Built specifically as a companion tool for custom synthesizer engines (like CML-1), this studio outputs clean 16-bit RAW audio files ready for direct engine ingestion.


# 🎛️ CML-1 Wavetable Studio — Wave Workstation

A desktop wavetable editor built in Python with tkinter, designed as a companion tool for the **COBOL Synth (CML-1)**. Draw, sculpt, and export 2048-sample wavetables directly to your synth engine — no DAW required.

---

## ✨ Features

- **Freehand Drawing** — Pencil, Line, and Selection tools for precise waveform sculpting
- **Additive Synthesis** — Blend your drawn wave with classic waveforms (Sine, Sawtooth, Square, Triangle, Trumpet, Flute) using the Macro Blender
- **Spectral Merge (FFT)** — Load a second wavetable and blend low/mid/high frequency bands between the two
- **Waveform Processing** — Normalize, Smooth, Invert, Reverse, Remove DC offset, Wavefold
- **Seamless Loop** — Smart 64-sample crossfade to eliminate clicks at loop points
- **Soft Analogue Character** — Subtle harmonic saturation for a warmer feel
- **Live Preview Playback** — Audition your waveform at any note (C2–C8) via `sounddevice`
- **Import / Export** — Load `.wav` or `.raw` files; export as 16-bit `.raw` or `.npy` backup
- **Full Undo/Redo** — 60-step history

---

## 🖥️ Requirements

```
Python 3.8+
numpy
tkinter (included with most Python installs)
sounddevice (optional — required for audio preview)
```

Install dependencies:

```bash
pip install numpy sounddevice
```

> If `sounddevice` is not installed, the app will still run — playback will simply be disabled.

---

## 🚀 Getting Started

```bash
python WaveWorkStation.py
```

---

## ⚠️ Important: Update File Paths

Before running the app, you **must update the two hardcoded file paths** in `WaveWorkStation.py` to match your own system. Find these lines near the top of the `__init__` method:

```python
self.default_source = r"C:\Users\USER\Desktop\WaveSources\FineWine.raw"
self.default_output = r"C:\Users\USER\Desktop\Sine Generator\Output1.raw"
```

Replace them with your own paths:

- **`default_source`** — The `.raw` wavetable file loaded by *File → Load FineWine.raw (default)*
- **`default_output`** — Where *File → Export directly to Output1.raw* and the **Export RAW → Engine** button will save your wavetable

Example:
```python
self.default_source = r"C:\Users\YourName\Wavetables\MyWave.raw"
self.default_output = r"C:\Users\YourName\SynthEngine\Output1.raw"
```

On macOS/Linux, use forward slashes:
```python
self.default_source = "/home/yourname/wavetables/MyWave.raw"
self.default_output = "/home/yourname/synth/Output1.raw"
```

---

## 📁 File Format

Wavetables are exported as **16-bit signed little-endian `.raw`** files at **2048 samples**. This is the format expected by the CML-1 / COBOL Synth engine.

---

## 🎹 Workflow Overview

1. **Draw** a waveform on the canvas using the Pencil or Line tool
2. **Shape** it with tools like Smooth, Wavefold, or the Additive Synth blender
3. **Preview** playback at your chosen note (C2–C8)
4. **Export** directly to your synth engine output path with one click

---

## 📄 License

MIT — free to use, modify, and distribute.
