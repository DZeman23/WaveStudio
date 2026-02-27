import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

class CMLWavetableEditor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CML-1 Wavetable Studio • COBOL Synth Companion")
        self.root.geometry("1200x700")
        self.root.minsize(800, 500)
        self.root.configure(bg="#0a0a0a")

        self.NUM_SAMPLES = 2048
        self.CANVAS_W = 1160
        self.CANVAS_H = 460
        self.HISTORY_LIMIT = 60

        self.waveform = np.zeros(self.NUM_SAMPLES, dtype=np.float32)
        self.wave_b = None
        self.history = []
        self.redo_stack = []
        self.tool = "pencil"
        self.selection = None
        self.last_idx = None
        self.last_val = None
        self.line_start = None
        self.temp_line = None

        self.default_source = r"C:\Users\USER\Desktop\WaveSources\FineWine.raw"
        self.default_output = r"C:\Users\USER\Desktop\Sine Generator\Output1.raw"

        self.status_var = tk.StringVar(value="Ready • Seamless Loop now uses smart 64-sample crossfade (no clicks!)")
        self.preview_note = tk.StringVar(value="C4")

        self.setup_ui()
        self.generate_sine()

    def setup_ui(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Wavetable (.wav or .raw)...", command=self.load_file, font=('Helvetica', 10, 'bold'))
        file_menu.add_command(label="Load FineWine.raw (default)", command=self.load_default_raw)
        file_menu.add_separator()
        file_menu.add_command(label="Spectral Merge (FFT Bands)...", command=self.open_spectral_merge, font=('Helvetica', 10, 'bold'))
        file_menu.add_separator()
        file_menu.add_command(label="Export 16-bit RAW...", command=self.export_raw_primary, font=('Helvetica', 10, 'bold'))
        file_menu.add_command(label="Export directly to Output1.raw", command=self.export_to_engine_default)
        file_menu.add_command(label="Export .npy (backup)", command=self.export_npy)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo  Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Redo  Ctrl+Y", command=self.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Invert", command=self.invert)
        edit_menu.add_command(label="Reverse", command=self.reverse)
        edit_menu.add_command(label="Remove DC", command=self.remove_dc)
        edit_menu.add_command(label="Wavefold", command=self.wavefold)
        edit_menu.add_command(label="Make Seamless Loop (crossfade)", command=self.make_seamless_loop)

        # TOOLBAR
        tb = tk.Frame(self.root, bg="#1a1a1a", height=58)
        tb.pack(fill="x", pady=10, padx=15)

        tk.Label(tb, text="Tool:", bg="#1a1a1a", fg="#ddd", font=("Segoe UI", 11)).pack(side="left", padx=(0,10))
        for t, txt in [("pencil","✏ Pencil"), ("line","📏 Line"), ("select","▭ Select")]:
            btn = tk.Button(tb, text=txt, width=11,
                            command=lambda tool=t: self.set_tool(tool),
                            bg="#00cc77" if t=="pencil" else "#333333", fg="white", relief="flat")
            btn.pack(side="left", padx=4)
            if t == "pencil": self.active_btn = btn

        tk.Label(tb, text="   ", bg="#1a1a1a").pack(side="left")

        tk.Button(tb, text="Normalize", command=self.normalize, bg="#333333", fg="#fff", width=12).pack(side="left", padx=4)
        tk.Button(tb, text="Smooth", command=self.smooth, bg="#333333", fg="#fff", width=12).pack(side="left", padx=4)
        tk.Button(tb, text="Soft Analogue", command=self.apply_analogue_character, bg="#333333", fg="#fff", width=13).pack(side="left", padx=4)
        tk.Button(tb, text="Seamless Loop", command=self.make_seamless_loop,
                  bg="#00aaff", fg="white", width=13, font=("Segoe UI", 10, "bold")).pack(side="left", padx=4)
        tk.Button(tb, text="Additive Synth", command=self.open_harmonic_dialog, bg="#7733ff", fg="#fff", font=("Segoe UI", 9, "bold"), width=14).pack(side="left", padx=4)

        # Preview Note Selector
        tk.Label(tb, text="  Note:", bg="#1a1a1a", fg="#ddd", font=("Segoe UI", 10)).pack(side="left", padx=(20,5))
        self.note_combo = ttk.Combobox(tb, textvariable=self.preview_note, width=7, state="readonly")
        self.note_combo.pack(side="left", padx=4)
        self.populate_preview_notes()

        # RIGHT SIDE BUTTONS
        tk.Button(tb, text="▶ Play", command=self.play, bg="#ff3366", fg="white", font=("Segoe UI", 11, "bold"), width=12).pack(side="right", padx=8)
        tk.Button(tb, text="■ Stop", command=self.stop_playback, bg="#555555", fg="white", font=("Segoe UI", 11), width=8).pack(side="right", padx=5)
        tk.Button(tb, text="Export RAW → Engine", command=self.export_to_engine_default,
                  bg="#00cc66", fg="white", font=("Segoe UI", 11, "bold"), width=22).pack(side="right", padx=8)

        # CANVAS
        cf = tk.Frame(self.root, bg="#111111", bd=6, relief="sunken")
        cf.pack(pady=12, padx=15, fill="both", expand=True)
        self.canvas = tk.Canvas(cf, bg="#050505", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.canvas.bind("<Motion>", self.mouse_move)
        self.canvas.bind("<Configure>", self.on_resize)

        tk.Label(self.root, textvariable=self.status_var, bg="#111111", fg="#aaa", font=("Consolas", 10), anchor="w", padx=20).pack(fill="x", pady=6)

        self.redraw()

    def on_resize(self, event):
        if event.width > 10 and event.height > 10:
            if event.width != self.CANVAS_W or event.height != self.CANVAS_H:
                self.CANVAS_W = event.width
                self.CANVAS_H = event.height
                self.redraw()

    def set_tool(self, tool):
        self.tool = tool
        self.active_btn.config(bg="#00cc77" if tool == "pencil" else "#333333")

    def save_history(self):
        self.history.append(self.waveform.copy())
        if len(self.history) > self.HISTORY_LIMIT: self.history.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            self.redo_stack.append(self.waveform.copy())
            self.waveform = self.history.pop()
            self.redraw()

    def redo(self):
        if self.redo_stack:
            self.history.append(self.waveform.copy())
            self.waveform = self.redo_stack.pop()
            self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        for i in range(9):
            y = i * self.CANVAS_H // 8
            self.canvas.create_line(0, y, self.CANVAS_W, y, fill="#333333" if i != 4 else "#666666")
        for i in range(1, 8):
            x = i * self.CANVAS_W // 8
            self.canvas.create_line(x, 0, x, self.CANVAS_H, fill="#222222")
        self.canvas.create_line(0, self.CANVAS_H//2, self.CANVAS_W, self.CANVAS_H//2, fill="#00ffcc", width=2)

        points = []
        for i in range(self.NUM_SAMPLES):
            x = i * self.CANVAS_W / self.NUM_SAMPLES
            y = (1.0 - self.waveform[i]) * self.CANVAS_H / 2
            points.extend([x, y])
        self.canvas.create_line(points, fill="#00ff99", width=2.8)

        if self.selection:
            x1 = self.selection[0] * self.CANVAS_W / self.NUM_SAMPLES
            x2 = self.selection[1] * self.CANVAS_W / self.NUM_SAMPLES
            self.canvas.create_rectangle(x1, 0, x2, self.CANVAS_H, outline="#ffff00", width=2, dash=(6,4))

    def get_coords(self, event):
        x = max(0, min(event.x, self.CANVAS_W - 1))
        y = max(0, min(event.y, self.CANVAS_H - 1))
        idx = int((x / self.CANVAS_W) * self.NUM_SAMPLES)
        val = np.clip(1.0 - 2.0 * (y / self.CANVAS_H), -1.0, 1.0)
        return idx, val

    def start_draw(self, event):
        self.save_history()
        idx, val = self.get_coords(event)
        if self.tool == "pencil":
            self.last_idx = idx
            self.last_val = val
            self.waveform[idx] = val
        elif self.tool == "line":
            self.line_start = (idx, val)
        elif self.tool == "select":
            self.selection = [idx, idx]
        self.redraw()

    def draw(self, event):
        idx, val = self.get_coords(event)
        if self.tool == "pencil" and self.last_idx is not None:
            if idx != self.last_idx:
                step = 1 if idx > self.last_idx else -1
                for i in range(self.last_idx, idx + step, step):
                    frac = (i - self.last_idx) / (idx - self.last_idx) if idx != self.last_idx else 0.0
                    self.waveform[i] = np.clip(self.last_val + frac * (val - self.last_val), -1.0, 1.0)
            self.last_idx = idx
            self.last_val = val
            self.redraw()
        elif self.tool == "line" and self.line_start:
            if self.temp_line: self.canvas.delete(self.temp_line)
            x1 = self.line_start[0] * self.CANVAS_W / self.NUM_SAMPLES
            y1 = (1.0 - self.line_start[1]) * self.CANVAS_H / 2
            x2 = idx * self.CANVAS_W / self.NUM_SAMPLES
            y2 = (1.0 - val) * self.CANVAS_H / 2
            self.temp_line = self.canvas.create_line(x1, y1, x2, y2, fill="#ffff00", width=3, dash=(4, 2))
        elif self.tool == "select" and self.selection:
            self.selection[1] = idx
            self.redraw()

    def stop_draw(self, event):
        if self.tool == "line" and self.line_start:
            idx, val = self.get_coords(event)
            i1, v1 = self.line_start
            min_i, max_i = min(i1, idx), max(i1, idx)
            for i in range(min_i, max_i + 1):
                frac = (i - i1) / (idx - i1) if idx != i1 else 0.0
                self.waveform[i] = np.clip(v1 + frac * (val - v1), -1.0, 1.0)
        if self.temp_line:
            self.canvas.delete(self.temp_line)
            self.temp_line = None
        self.line_start = None
        self.last_idx = None
        self.redraw()

    def mouse_move(self, event):
        idx, val = self.get_coords(event)
        sel = f" | Sel {self.selection[0]}-{self.selection[1]}" if self.selection else ""
        self.status_var.set(f"Sample {idx:4d}   Value {val:+.4f}{sel}")

    def generate_sine(self): self._generate(lambda t: np.sin(t))
    def generate_square(self): self._generate(lambda t: np.sign(np.sin(t)))
    def generate_saw(self): self.save_history(); self.waveform = np.linspace(1.0, -1.0, self.NUM_SAMPLES, endpoint=False); self.redraw()
    def generate_triangle(self): self.save_history(); s = np.linspace(1.0, -1.0, self.NUM_SAMPLES, endpoint=False); self.waveform = 2.0 * np.abs(s) - 1.0; self.redraw()
    def generate_noise(self): self.save_history(); self.waveform = np.random.uniform(-1.0, 1.0, self.NUM_SAMPLES); self.redraw()

    def _generate(self, func):
        self.save_history()
        t = np.linspace(0, 2 * np.pi, self.NUM_SAMPLES, endpoint=False)
        self.waveform = func(t)
        self.redraw()

    def normalize(self):
        self.save_history()
        mx = np.max(np.abs(self.waveform))
        if mx > 1e-8: self.waveform /= mx
        self.redraw()

    def smooth(self):
        self.save_history()
        self.waveform = np.convolve(self.waveform, np.ones(5)/5, mode='same')
        self.redraw()

    def invert(self):
        self.save_history()
        self.waveform = -self.waveform
        self.redraw()

    def reverse(self):
        self.save_history()
        self.waveform = np.flip(self.waveform)
        self.redraw()

    def remove_dc(self):
        self.save_history()
        self.waveform -= np.mean(self.waveform)
        self.redraw()

    def wavefold(self):
        self.save_history()
        self.waveform = np.sin(self.waveform * np.pi * 2)
        self.redraw()

    def apply_analogue_character(self):
        self.save_history()
        self.waveform = np.tanh(self.waveform * 1.8) * 0.98
        self.redraw()

    def make_seamless_loop(self):
        self.save_history()
        N = 64
        if self.NUM_SAMPLES < 2 * N:
            N = self.NUM_SAMPLES // 4

        fade_in = np.linspace(0.0, 1.0, N)
        fade_out = np.linspace(1.0, 0.0, N)

        self.waveform[:N] = self.waveform[:N] * fade_in
        self.waveform[-N:] = self.waveform[-N:] * fade_out

        self.waveform[0] = 0.0
        self.waveform[-1] = 0.0

        self.redraw()
        self.status_var.set(f"✅ Seamless loop: Endpoints smoothly tapered to absolute 0.0 over {N} samples!")

    def populate_preview_notes(self):
        notes = []
        for octave in range(1, 8):
            for n in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']:
                notes.append(f"{n}{octave}")
        self.note_combo['values'] = notes
        self.preview_note.set("C4")

    def note_to_freq(self, note):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        try:
            pitch = note[:-1]
            octave = int(note[-1])
            semitone = note_names.index(pitch)
            midi = (octave + 1) * 12 + semitone
            return 440.0 * (2 ** ((midi - 69) / 12.0))
        except:
            return 261.63

    def play(self):
        if not HAS_AUDIO:
            messagebox.showinfo("Audio Preview", "pip install sounddevice for playback")
            return
        try:
            sample_rate = 44100.0
            freq = self.note_to_freq(self.preview_note.get())
            duration = 2.8
            num_samples = int(sample_rate * duration)

            phase_inc = self.NUM_SAMPLES * freq / sample_rate
            phase = (np.cumsum(np.full(num_samples, phase_inc)) % self.NUM_SAMPLES)

            signal = np.interp(phase, np.arange(self.NUM_SAMPLES), self.waveform)

            fade = min(2048, num_samples // 10)
            if fade > 0:
                signal[:fade] *= np.linspace(0, 1, fade)
                signal[-fade:] *= np.linspace(1, 0, fade)

            sd.play((signal * 0.78).astype(np.float32), int(sample_rate))
            self.status_var.set(f"▶ Playing {self.preview_note.get()} • {freq:.1f} Hz")
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def stop_playback(self):
        if HAS_AUDIO:
            try:
                sd.stop()
            except:
                pass
        self.status_var.set("⏹ Playback stopped")

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Load Wavetable",
            filetypes=[("Wavetable files", "*.wav *.raw"), ("WAV files", "*.wav"), ("16-bit RAW", "*.raw")]
        )
        if not path: return
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            self.load_wav(path)
        elif ext == ".raw":
            self.load_raw(path)
        else:
            messagebox.showwarning("Unsupported", "Only .wav and .raw files are supported.")

    def load_wav(self, path):
        try:
            import wave
            with wave.open(path, 'r') as wf:
                raw = wf.readframes(wf.getnframes())
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if wf.getnchannels() > 1:
                    data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
            if len(data) != self.NUM_SAMPLES:
                old_x = np.linspace(0, 1, len(data))
                new_x = np.linspace(0, 1, self.NUM_SAMPLES)
                data = np.interp(new_x, old_x, data)
            self.save_history()
            self.waveform = np.clip(data, -1.0, 1.0)
            self.redraw()
            messagebox.showinfo("Loaded", f"{os.path.basename(path)} → ready for CML-1")
        except Exception as e:
            messagebox.showerror("Load Failed", str(e))

    def load_raw(self, path):
        try:
            data = np.fromfile(path, dtype=np.int16)
            if len(data) != self.NUM_SAMPLES:
                if len(data) > self.NUM_SAMPLES:
                    data = data[:self.NUM_SAMPLES]
                else:
                    data = np.pad(data, (0, self.NUM_SAMPLES - len(data)), mode='wrap')
            self.save_history()
            self.waveform = np.clip(data.astype(np.float32) / 32768.0, -1.0, 1.0)
            self.redraw()
            messagebox.showinfo("Loaded", f"{os.path.basename(path)} → perfect for your engine")
        except Exception as e:
            messagebox.showerror("Load Failed", str(e))

    def load_default_raw(self):
        if os.path.exists(self.default_source):
            self.load_raw(self.default_source)
        else:
            messagebox.showwarning("Not Found", f"{self.default_source}\nCreate it first.")

    def _export_raw(self, path):
        data = (self.waveform * 32767.0).astype(np.int16)
        data.tofile(path)
        messagebox.showinfo("Success", f"16-bit RAW saved\n{path}\nReady for CML-1!")

    def export_raw_primary(self):
        path = filedialog.asksaveasfilename(defaultextension=".raw", initialfile="MyWavetable.raw")
        if path: self._export_raw(path)

    def export_to_engine_default(self):
        if messagebox.askyesno("Overwrite?", f"Write to:\n{self.default_output}"):
            self._export_raw(self.default_output)

    def export_npy(self):
        path = filedialog.asksaveasfilename(defaultextension=".npy")
        if path:
            np.save(path, self.waveform)
            messagebox.showinfo("Saved", f"Saved {path}")

    def open_spectral_merge(self):
        if np.max(np.abs(self.waveform)) < 0.01:
            messagebox.showwarning("Empty Wave", "Draw or load a wave first (Wave A)")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Spectral Merge – FFT Band Blending")
        dlg.geometry("960x680")
        dlg.configure(bg="#1a1a1a")
        dlg.grab_set()

        self.wave_b = self.waveform.copy()

        tk.Label(dlg, text="Wave A (Current)", bg="#1a1a1a", fg="#00ff99", font=("Segoe UI", 12, "bold")).pack(pady=(15,5))
        tk.Label(dlg, text="Wave B (Load below)", bg="#1a1a1a", fg="#ffcc00", font=("Segoe UI", 12, "bold")).pack(pady=5)

        load_frame = tk.Frame(dlg, bg="#1a1a1a")
        load_frame.pack(pady=8)
        tk.Button(load_frame, text="Load Wave B (.wav or .raw)", command=lambda: self._load_for_merge(dlg),
                  bg="#333333", fg="#fff", width=28).pack(side="left", padx=10)

        slider_frame = tk.Frame(dlg, bg="#1a1a1a")
        slider_frame.pack(pady=20, padx=40, fill="x")

        self.low_var = tk.DoubleVar(value=0)
        self.mid_var = tk.DoubleVar(value=50)
        self.high_var = tk.DoubleVar(value=0)

        for label, var, color in [("LOW end blend from B (%)", self.low_var, "#00ddff"),
                                  ("MID blend from B (%)", self.mid_var, "#ffaa00"),
                                  ("HIGH end blend from B (%)", self.high_var, "#ff44aa")]:
            f = tk.Frame(slider_frame, bg="#1a1a1a")
            f.pack(fill="x", pady=12)
            tk.Label(f, text=label, bg="#1a1a1a", fg=color, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            tk.Scale(f, from_=0, to=100, orient="horizontal", variable=var, bg="#333333", fg=color,
                     length=720, resolution=1).pack()

        btn_frame = tk.Frame(dlg, bg="#1a1a1a")
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="PREVIEW MERGE", command=lambda: self._preview_merge(dlg),
                  bg="#00cc66", fg="white", font=("Segoe UI", 11, "bold"), width=18).pack(side="left", padx=15)
        tk.Button(btn_frame, text="APPLY TO EDITOR", command=lambda: self._apply_merge(dlg),
                  bg="#ff3366", fg="white", font=("Segoe UI", 11, "bold"), width=18).pack(side="left", padx=15)
        tk.Button(btn_frame, text="Cancel", command=dlg.destroy, bg="#444444", fg="#fff", width=12).pack(side="left", padx=15)

        self.merge_status = tk.StringVar(value="Load Wave B then adjust sliders → Preview → Apply")
        tk.Label(dlg, textvariable=self.merge_status, bg="#1a1a1a", fg="#aaaaaa").pack(pady=10)

    def _load_for_merge(self, dlg):
        path = filedialog.askopenfilename(filetypes=[("Wavetable", "*.wav *.raw")])
        if not path: return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".wav":
                import wave
                with wave.open(path, 'r') as wf:
                    raw = wf.readframes(wf.getnframes())
                    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if wf.getnchannels() > 1:
                        data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
                if len(data) != self.NUM_SAMPLES:
                    data = np.interp(np.linspace(0,1,self.NUM_SAMPLES), np.linspace(0,1,len(data)), data)
            else:
                data = np.fromfile(path, dtype=np.int16).astype(np.float32) / 32768.0
                if len(data) != self.NUM_SAMPLES:
                    data = np.pad(data[:self.NUM_SAMPLES], (0, max(0, self.NUM_SAMPLES - len(data))), mode='wrap')
            self.wave_b = np.clip(data, -1.0, 1.0)
            self.merge_status.set(f"Wave B loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _preview_merge(self, dlg):
        if self.wave_b is None:
            messagebox.showwarning("No Wave B", "Load Wave B first")
            return
        merged = self._spectral_blend(self.waveform, self.wave_b,
                                      self.low_var.get(), self.mid_var.get(), self.high_var.get())
        self.save_history()
        self.waveform = merged.copy()
        self.redraw()
        self.merge_status.set("Preview shown in main editor (Undo if you don't like it)")

    def _apply_merge(self, dlg):
        if self.wave_b is None:
            messagebox.showwarning("No Wave B", "Load Wave B first")
            return
        self.save_history()
        self.waveform = self._spectral_blend(self.waveform, self.wave_b,
                                             self.low_var.get(), self.mid_var.get(), self.high_var.get())
        self.redraw()
        dlg.destroy()
        messagebox.showinfo("Merged!", "Spectral merge applied – perfect hybrid wavetable for your COBOL engine")

    def _spectral_blend(self, a, b, low_p, mid_p, high_p):
        fft_a = np.fft.rfft(a)
        fft_b = np.fft.rfft(b)
        n = len(fft_a)

        low_cut = 48
        mid_cut = 192

        merged_fft = fft_a.copy()

        p_low = low_p / 100.0
        p_mid = mid_p / 100.0
        p_high = high_p / 100.0

        merged_fft[0:low_cut] = fft_a[0:low_cut] * (1 - p_low) + fft_b[0:low_cut] * p_low
        merged_fft[low_cut:mid_cut] = fft_a[low_cut:mid_cut] * (1 - p_mid) + fft_b[low_cut:mid_cut] * p_mid
        merged_fft[mid_cut:] = fft_a[mid_cut:] * (1 - p_high) + fft_b[mid_cut:] * p_high

        merged_wave = np.fft.irfft(merged_fft, n=self.NUM_SAMPLES)
        merged_wave = np.clip(merged_wave.real, -1.0, 1.0)

        mx = np.max(np.abs(merged_wave))
        if mx > 1e-8:
            merged_wave /= mx

        return merged_wave.astype(np.float32)

    # ==================== FIXED ADDITIVE SYNTH DIALOG ====================
    def open_harmonic_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Wavetable Macro Blender – Mix Sound Characters")
        dlg.geometry("820x680")
        dlg.configure(bg="#1a1a1a")
        dlg.grab_set()

        original_wave = self.waveform.copy()

        presets = {
            "Sine":     [1.0] + [0.0]*15,
            "Sawtooth": [((-1.0)**i) / (i + 1) for i in range(16)],
            "Square":   [1.0 / (i + 1) if (i + 1) % 2 == 1 else 0.0 for i in range(16)],
            "Triangle": [((-1.0)**(i // 2)) / ((i + 1)**2) if (i + 1) % 2 == 1 else 0.0 for i in range(16)],
            "Trumpet":  [1.00, 0.82, 0.68, 0.58, 0.52, 0.45, 0.38, 0.32, 0.27, 0.22, 0.18, 0.14, 0.11, 0.09, 0.07, 0.05],
            "Flute":    [1.00, 0.28, 0.15, 0.09, 0.06, 0.04, 0.025, 0.015, 0.01, 0.008, 0.005, 0.003, 0.002, 0.001, 0.001, 0.0],
        }

        sound_names = ["Original (Canvas)"] + list(presets.keys())
        blend_vars = {name: tk.DoubleVar(value=100.0 if name == "Original (Canvas)" else 0.0) for name in sound_names}

        tk.Label(dlg, text="Wavetable Macro Blender", bg="#1a1a1a", fg="#00ff99", font=("Segoe UI", 16, "bold")).pack(pady=(20,5))
        tk.Label(dlg, text="Mix your drawn wave with mathematical acoustic presets", 
                 bg="#1a1a1a", fg="#aaaaaa", font=("Segoe UI", 10)).pack(pady=(0,25))

        slider_container = tk.Frame(dlg, bg="#1a1a1a")
        slider_container.pack(fill="x", padx=50, pady=10)

        colors = ["#ffffff", "#00ffcc", "#ffaa00", "#ff3366", "#7733ff", "#00cc77", "#ffcc33"]
        
        for idx, name in enumerate(sound_names):
            f = tk.Frame(slider_container, bg="#1a1a1a")
            f.pack(fill="x", pady=12)
            tk.Label(f, text=name, width=15, anchor="w", bg="#1a1a1a", 
                     fg=colors[idx], font=("Segoe UI", 11, "bold")).pack(side="left")
            tk.Scale(f, from_=0, to=100, orient="horizontal", variable=blend_vars[name],
                     bg="#333333", fg="#ddd", length=480, resolution=1).pack(side="left", fill="x", expand=True, padx=(25,0))
            tk.Label(f, text="%", bg="#1a1a1a", fg="#aaa", font=("Segoe UI", 9)).pack(side="left", padx=8)

        def synthesize():
            self.save_history()
            t = np.linspace(0, 2 * np.pi, self.NUM_SAMPLES, endpoint=False)
            w = np.zeros(self.NUM_SAMPLES, dtype=np.float32)

            weights = np.array([blend_vars[name].get() for name in sound_names]) / 100.0
            total_w = np.sum(weights)

            if total_w < 1e-6:
                w = original_wave.copy()
            else:
                weights /= total_w

                # 1. Time-domain Original (Canvas)
                if weights[0] > 1e-6:
                    w += original_wave * weights[0]

                # 2. Harmonic presets
                preset_list = list(presets.keys())
                for h in range(16):
                    amp_h = 0.0
                    for s_idx, name in enumerate(preset_list):
                        amp_h += weights[s_idx + 1] * presets[name][h]
                    if abs(amp_h) > 0.001:
                        w += amp_h * np.sin((h + 1) * t)

            # Normalize
            mx = np.max(np.abs(w))
            if mx > 1e-8:
                w /= mx

            self.waveform = w.astype(np.float32)
            self.redraw()
            dlg.destroy()
            messagebox.showinfo("Done!", "Hybrid wavetable synthesized!\n(Original + chosen harmonics blended)")

        tk.Button(dlg, text="SYNTHESIZE → 2048-SAMPLE WAVETABLE", command=synthesize,
                  bg="#00cc66", fg="white", height=2, font=("Segoe UI", 12, "bold")).pack(pady=30, padx=80, fill="x")

        tk.Label(dlg, text="Tip: Draw a jagged shape → set Original to 70% + Flute 30% for breathy character.", 
                 bg="#1a1a1a", fg="#666666", font=("Segoe UI", 9)).pack(pady=(0,20))

    # =====================================================================

if __name__ == "__main__":
    app = CMLWavetableEditor()
    app.root.mainloop()