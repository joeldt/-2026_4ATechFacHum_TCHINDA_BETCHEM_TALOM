from __future__ import annotations

import argparse
import queue
import time
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from .acquisition import SensorSource, create_source
from .biofeedback import BiofeedbackEngine, BiofeedbackState
from .config import AppConfig, load_config
from .processing import SignalFeatures, SignalProcessor


class AcquisitionThread(threading.Thread):
    def __init__(self, source: SensorSource, processor: SignalProcessor, engine: BiofeedbackEngine) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.processor = processor
        self.engine = engine
        self.output: queue.Queue[tuple[SignalFeatures, BiofeedbackState]] = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            frame = self.source.read()
            features = self.processor.update(frame)
            state = self.engine.update(features)
            try:
                self.output.put_nowait((features, state))
            except queue.Full:
                _ = self.output.get_nowait()
                self.output.put_nowait((features, state))

    def stop(self) -> None:
        self.stop_event.set()
        self.source.close()


class FeatherMindApp:
    def __init__(self, root: tk.Tk, config: AppConfig, mode: str) -> None:
        self.root = root
        self.config = config
        self.mode = mode
        self.processor = SignalProcessor(config.sampling_rate_hz)
        self.engine = BiofeedbackEngine(config.thresholds)
        self.source = create_source(mode, config)
        self.worker = AcquisitionThread(self.source, self.processor, self.engine)
        self.last_alarm_at = 0.0
        self.last_motion_alarm_at = 0.0

        root.title("FeatherMind")
        root.geometry("980x620")
        root.minsize(820, 520)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.status = tk.StringVar(value="initialisation")
        self.respiration = tk.StringVar(value="-- bpm")
        self.heart = tk.StringVar(value="-- bpm")
        self.movement = tk.StringVar(value="--")
        self.accelerometer = tk.StringVar(value="--")
        self.stability = tk.StringVar(value="--")
        self.score = tk.StringVar(value="-- %")
        self.respiration_quality = tk.StringVar(value="--")
        self.ppg_quality = tk.StringVar(value="--")
        self.alarm = tk.StringVar(value="active" if config.alarm.enabled else "desactivee")
        self.motion_alarm = tk.StringVar(value="active" if config.motion_alarm.enabled else "desactivee")
        self.score_progress: ttk.Progressbar | None = None

        self._build_ui()
        self.worker.start()
        self.root.after(80, self.refresh)

    def _build_ui(self) -> None:
        self._configure_style()
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        self.root.configure(bg="#ece7df")
        self.canvas = tk.Canvas(self.root, bg="#f5f1ea", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        panel_canvas = tk.Canvas(self.root, bg="#fbfaf7", highlightthickness=0, width=360)
        panel_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=panel_canvas.yview)
        panel_canvas.configure(yscrollcommand=panel_scrollbar.set)
        panel_canvas.grid(row=0, column=1, sticky="nsew")
        panel_scrollbar.grid(row=0, column=2, sticky="ns")

        panel = ttk.Frame(panel_canvas, padding=18, style="Panel.TFrame")
        panel_window = panel_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda event: panel_canvas.configure(scrollregion=panel_canvas.bbox("all")))
        panel_canvas.bind("<Configure>", lambda event: panel_canvas.itemconfigure(panel_window, width=event.width))
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="FeatherMind", style="Title.TLabel").grid(sticky="w")
        ttk.Label(panel, text=f"Mode: {self.mode}", style="Subtle.TLabel").grid(sticky="w", pady=(0, 18))

        self._section(panel, "Etat global")
        self._metric(panel, "Etat", self.status)
        self._metric(panel, "Score relaxation", self.score)
        self.score_progress = ttk.Progressbar(panel, maximum=100, mode="determinate", style="Score.Horizontal.TProgressbar")
        self.score_progress.grid(sticky="ew", pady=(2, 12))

        self._section(panel, "Signaux")
        self._metric(panel, "Respiration", self.respiration)
        self._metric(panel, "Qualite respiration", self.respiration_quality)
        self._metric(panel, "PPG / coeur", self.heart)
        self._metric(panel, "Qualite PPG", self.ppg_quality)
        self._metric(panel, "Accelerometre X/Y/Z", self.accelerometer)
        self._metric(panel, "Mouvement", self.movement)
        self._metric(panel, "Stabilite", self.stability)

        self._section(panel, "Alertes")
        self._metric(panel, "Alarme reveil", self.alarm)
        self._metric(panel, "Alarme mouvement", self.motion_alarm)

        ttk.Separator(panel).grid(sticky="ew", pady=16)
        ttk.Label(
            panel,
            text=(
                "Objectif: respiration lente et reguliere, peu de mouvement, "
                "signal PPG stable. Le score privilegie la respiration, puis "
                "l'immobilite, puis le PPG. La plume descend quand le score augmente."
            ),
            wraplength=300,
            justify="left",
            style="Subtle.TLabel",
        ).grid(sticky="w")

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.configure("Panel.TFrame", background="#fbfaf7")
        style.configure("Title.TLabel", background="#fbfaf7", foreground="#1f2422", font=("Segoe UI", 22, "bold"))
        style.configure("Section.TLabel", background="#fbfaf7", foreground="#6a5c4d", font=("Segoe UI", 10, "bold"))
        style.configure("MetricName.TLabel", background="#fbfaf7", foreground="#6c6c66", font=("Segoe UI", 9, "bold"))
        style.configure("MetricValue.TLabel", background="#fbfaf7", foreground="#1f2422", font=("Segoe UI", 14))
        style.configure("Subtle.TLabel", background="#fbfaf7", foreground="#6c6c66", font=("Segoe UI", 10))
        style.configure("Score.Horizontal.TProgressbar", thickness=9)

    def _section(self, parent: ttk.Frame, label: str) -> None:
        ttk.Label(parent, text=label.upper(), style="Section.TLabel").grid(sticky="w", pady=(12, 3))

    def _metric(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label, style="MetricName.TLabel").grid(sticky="w", pady=(5, 0))
        ttk.Label(parent, textvariable=variable, style="MetricValue.TLabel").grid(sticky="w")

    def refresh(self) -> None:
        latest: tuple[SignalFeatures, BiofeedbackState] | None = None
        while not self.worker.output.empty():
            latest = self.worker.output.get_nowait()
        if latest:
            features, state = latest
            self.status.set(state.label)
            self.respiration.set(format_bpm(features.respiration_bpm))
            self.respiration_quality.set(
                f"amp {features.respiration_amplitude:.4f} / reg {features.respiration_regularity * 100:.0f} %"
            )
            self.heart.set(format_bpm(features.heart_bpm))
            self.ppg_quality.set(ppg_quality_label(features.ppg_quality))
            self.movement.set(f"{features.movement_rms:.3f} rms")
            self.accelerometer.set(
                f"{features.acc_x_level:.4f} / {features.acc_y_level:.4f} / {features.acc_z_level:.4f}"
            )
            self.stability.set(f"{state.movement_label} ({state.movement_score * 100:.0f} %)")
            self.score.set(f"{state.score * 100:.0f} %")
            if self.score_progress is not None:
                self.score_progress["value"] = state.score * 100
            self._maybe_alarm(state)
            self._maybe_motion_alarm(features)
            self.draw(features, state)
        self.root.after(80, self.refresh)

    def _maybe_alarm(self, state: BiofeedbackState) -> None:
        alarm = self.config.alarm
        if not alarm.enabled or state.score < alarm.trigger_score:
            return

        now = time.time()
        if now - self.last_alarm_at < alarm.cooldown_seconds:
            return

        self.last_alarm_at = now
        self.alarm.set("declenchee")
        threading.Thread(target=play_alarm, args=(alarm.frequency_hz, alarm.duration_ms), daemon=True).start()
        self.root.after(1200, lambda: self.alarm.set("active"))

    def _maybe_motion_alarm(self, features: SignalFeatures) -> None:
        alarm = self.config.motion_alarm
        if not alarm.enabled or features.movement_rms < alarm.trigger_rms:
            return

        now = time.time()
        if now - self.last_motion_alarm_at < alarm.cooldown_seconds:
            return

        self.last_motion_alarm_at = now
        self.motion_alarm.set("mouvement brusque")
        threading.Thread(target=play_motion_alarm, args=(alarm.frequency_hz, alarm.duration_ms), daemon=True).start()
        self.root.after(1200, lambda: self.motion_alarm.set("active"))

    def draw(self, features: SignalFeatures, state: BiofeedbackState) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())

        self.canvas.create_rectangle(0, 0, width, height, fill="#f5f1ea", outline="")
        top_y = height * 0.18
        bottom_y = height * 0.76
        alarm_y = height * (0.18 + 0.58 * self.config.alarm.trigger_score)
        self.canvas.create_text(width * 0.08, top_y, anchor="w", text="eveil", fill="#8b8174", font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(width * 0.08, bottom_y - 16, anchor="w", text="plume tombee", fill="#8b8174", font=("Segoe UI", 10, "bold"))
        self.canvas.create_line(width * 0.18, bottom_y, width * 0.82, bottom_y, fill="#c9b9a4", width=2)
        self.canvas.create_line(width * 0.22, alarm_y, width * 0.78, alarm_y, fill="#c88468", width=1, dash=(5, 5))
        self.canvas.create_text(width * 0.79, alarm_y, anchor="w", text="reveil", fill="#a55f4f", font=("Segoe UI", 9, "bold"))

        feather_x = width * 0.50
        feather_y = height * (0.18 + 0.58 * state.feather_y)
        self._draw_feather(feather_x, feather_y, state.score)

        self._draw_series("pzt", 30, height - 184, width - 60, 44, "#4b8f8c")
        self._draw_series("ppg", 30, height - 130, width - 60, 44, "#b65050")
        self._draw_series("acc_magnitude", 30, height - 76, width - 60, 44, "#5e6fb5")

        self.canvas.create_text(34, height - 200, anchor="w", text="PZT", fill="#4b8f8c", font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(34, height - 146, anchor="w", text="PPG", fill="#b65050", font=("Segoe UI", 10, "bold"))
        self.canvas.create_text(34, height - 92, anchor="w", text="ACC", fill="#5e6fb5", font=("Segoe UI", 10, "bold"))

    def _draw_feather(self, x: float, y: float, score: float) -> None:
        color = "#507f73" if score > 0.65 else "#9b7a4a" if score > 0.4 else "#9b5656"
        self.canvas.create_line(x, y - 90, x, y + 92, fill=color, width=4, smooth=True)
        for i in range(9):
            offset = i * 18
            self.canvas.create_line(x, y - 72 + offset, x - 110 + i * 7, y - 112 + offset, fill=color, width=3, smooth=True)
            self.canvas.create_line(x, y - 72 + offset, x + 110 - i * 7, y - 112 + offset, fill=color, width=3, smooth=True)
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="")

    def _draw_series(self, key: str, x: int, y: int, width: int, height: int, color: str) -> None:
        values = self.processor.recent_series(key)
        if len(values) < 2:
            return
        lo, hi = min(values), max(values)
        span = max(1e-6, hi - lo)
        step = width / max(1, len(values) - 1)
        points: list[float] = []
        for index, value in enumerate(values):
            px = x + index * step
            py = y + height - ((value - lo) / span) * height
            points.extend([px, py])
        self.canvas.create_rectangle(x, y, x + width, y + height, outline="#ddd2c2")
        self.canvas.create_line(*points, fill=color, width=2, smooth=True)

    def close(self) -> None:
        self.worker.stop()
        self.root.destroy()


def format_bpm(value: float | None) -> str:
    return "-- bpm" if value is None else f"{value:.1f} bpm"


def ppg_quality_label(value: float) -> str:
    if value > 0.75:
        return f"bon ({value * 100:.0f} %)"
    if value > 0.45:
        return f"moyen ({value * 100:.0f} %)"
    if value > 0.15:
        return f"faible ({value * 100:.0f} %)"
    return "instable"


def play_alarm(frequency_hz: int, duration_ms: int) -> None:
    try:
        import winsound

        winsound.Beep(frequency_hz, duration_ms)
        winsound.Beep(max(300, frequency_hz - 220), max(150, duration_ms // 2))
    except Exception:
        print("\a", end="", flush=True)


def play_motion_alarm(frequency_hz: int, duration_ms: int) -> None:
    try:
        import winsound

        winsound.Beep(frequency_hz, duration_ms)
        winsound.Beep(frequency_hz, duration_ms)
        winsound.Beep(max(300, frequency_hz - 330), max(120, duration_ms // 2))
    except Exception:
        print("\a\a", end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FeatherMind biofeedback app")
    parser.add_argument("--mode", choices=["simulate", "demo", "plux", "bitalino"], default="simulate")
    parser.add_argument("--config", default="config.example.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = tk.Tk()
    try:
        FeatherMindApp(root, config, args.mode)
        root.mainloop()
    except RuntimeError as exc:
        messagebox.showerror("FeatherMind", str(exc))
        root.destroy()


if __name__ == "__main__":
    main()
