from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .acquisition import SensorFrame


@dataclass(frozen=True)
class SignalFeatures:
    respiration_bpm: float | None
    respiration_amplitude: float
    respiration_regularity: float
    heart_bpm: float | None
    ppg_quality: float
    movement_rms: float
    acc_x_level: float
    acc_y_level: float
    acc_z_level: float
    acc_magnitude: float
    pzt_level: float
    ppg_level: float


class SignalProcessor:
    def __init__(self, sampling_rate_hz: int, window_seconds: float = 12.0) -> None:
        self.sampling_rate_hz = sampling_rate_hz
        self.max_samples = int(sampling_rate_hz * window_seconds)
        self.timestamp: deque[float] = deque(maxlen=self.max_samples)
        self.acc_x: deque[float] = deque(maxlen=self.max_samples)
        self.acc_y: deque[float] = deque(maxlen=self.max_samples)
        self.acc_z: deque[float] = deque(maxlen=self.max_samples)
        self.acc_magnitude: deque[float] = deque(maxlen=self.max_samples)
        self.ppg: deque[float] = deque(maxlen=self.max_samples)
        self.pzt: deque[float] = deque(maxlen=self.max_samples)

    def update(self, frame: SensorFrame) -> SignalFeatures:
        self.timestamp.extend(frame.timestamp.tolist())
        self.acc_x.extend(frame.acc_x.tolist())
        self.acc_y.extend(frame.acc_y.tolist())
        self.acc_z.extend(frame.acc_z.tolist())
        magnitude = np.sqrt(frame.acc_x * frame.acc_x + frame.acc_y * frame.acc_y + frame.acc_z * frame.acc_z)
        self.acc_magnitude.extend(magnitude.tolist())
        self.ppg.extend(frame.ppg.tolist())
        self.pzt.extend(frame.pzt.tolist())
        return self.features()

    def features(self) -> SignalFeatures:
        pzt = np.asarray(self.pzt, dtype=float)
        ppg = np.asarray(self.ppg, dtype=float)
        acc_x = np.asarray(self.acc_x, dtype=float)
        acc_y = np.asarray(self.acc_y, dtype=float)
        acc_z = np.asarray(self.acc_z, dtype=float)

        movement_rms = self._movement_rms(acc_x, acc_y, acc_z)
        respiration_bpm = self._estimate_rate_bpm(pzt, min_bpm=4, max_bpm=25)
        respiration_amplitude = signal_amplitude(pzt)
        respiration_regularity = self._regularity_score(pzt, min_bpm=4, max_bpm=25)
        heart_bpm = self._estimate_rate_bpm(ppg, min_bpm=40, max_bpm=150)
        ppg_quality = self._ppg_quality(ppg, heart_bpm)

        return SignalFeatures(
            respiration_bpm=respiration_bpm,
            respiration_amplitude=respiration_amplitude,
            respiration_regularity=respiration_regularity,
            heart_bpm=heart_bpm,
            ppg_quality=ppg_quality,
            movement_rms=movement_rms,
            acc_x_level=float(acc_x[-1]) if len(acc_x) else 0.0,
            acc_y_level=float(acc_y[-1]) if len(acc_y) else 0.0,
            acc_z_level=float(acc_z[-1]) if len(acc_z) else 0.0,
            acc_magnitude=float(np.sqrt(acc_x[-1] ** 2 + acc_y[-1] ** 2 + acc_z[-1] ** 2)) if len(acc_x) else 0.0,
            pzt_level=float(pzt[-1]) if len(pzt) else 0.0,
            ppg_level=float(ppg[-1]) if len(ppg) else 0.0,
        )

    def recent_series(self, key: str, max_points: int = 300) -> list[float]:
        values = getattr(self, key)
        if len(values) <= max_points:
            return list(values)
        return list(values)[-max_points:]

    def _movement_rms(self, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> float:
        if len(acc_x) < 3:
            return 0.0
        mag = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)
        dynamic = mag - moving_average(mag, max(3, self.sampling_rate_hz // 2))
        return float(np.sqrt(np.mean(dynamic * dynamic)))

    def _estimate_rate_bpm(
        self, signal: np.ndarray, min_bpm: float, max_bpm: float
    ) -> float | None:
        fs = self.sampling_rate_hz
        min_samples = int(fs * 6)
        if len(signal) < min_samples:
            return None

        centered = signal - moving_average(signal, max(3, fs))
        centered = centered - np.mean(centered)
        std = np.std(centered)
        if std < 1e-4:
            return None

        min_distance = int(fs * 60.0 / max_bpm)
        threshold = 0.35 * std
        peaks = find_peaks(centered, min_distance=min_distance, threshold=threshold)
        if len(peaks) < 2:
            return None

        intervals = np.diff(peaks) / fs
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            return None

        bpm = 60.0 / float(np.median(intervals))
        if min_bpm <= bpm <= max_bpm:
            return bpm
        return None

    def _regularity_score(self, signal: np.ndarray, min_bpm: float, max_bpm: float) -> float:
        fs = self.sampling_rate_hz
        if len(signal) < int(fs * 6):
            return 0.0
        centered = signal - moving_average(signal, max(3, fs))
        centered = centered - np.mean(centered)
        std = np.std(centered)
        if std < 1e-4:
            return 0.0
        peaks = find_peaks(
            centered,
            min_distance=int(fs * 60.0 / max_bpm),
            threshold=0.35 * std,
        )
        if len(peaks) < 3:
            return 0.0
        intervals = np.diff(peaks) / fs
        intervals = intervals[intervals > 0]
        if len(intervals) < 2:
            return 0.0
        variation = float(np.std(intervals) / max(1e-6, np.mean(intervals)))
        return clamp(1.0 - (variation / 0.35), 0.0, 1.0)

    def _ppg_quality(self, signal: np.ndarray, heart_bpm: float | None) -> float:
        if len(signal) < int(self.sampling_rate_hz * 4):
            return 0.0
        amplitude = signal_amplitude(signal)
        amplitude_score = clamp(amplitude / 0.004, 0.0, 1.0)
        bpm_score = 1.0 if heart_bpm is not None else 0.0
        return (0.65 * amplitude_score) + (0.35 * bpm_score)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def signal_amplitude(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def find_peaks(values: np.ndarray, min_distance: int, threshold: float) -> np.ndarray:
    if len(values) < 3:
        return np.array([], dtype=int)

    candidates = np.where(
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
        & (values[1:-1] > threshold)
    )[0] + 1
    if len(candidates) == 0:
        return np.array([], dtype=int)

    selected: list[int] = []
    for idx in candidates:
        if not selected or idx - selected[-1] >= min_distance:
            selected.append(int(idx))
        elif values[idx] > values[selected[-1]]:
            selected[-1] = int(idx)
    return np.asarray(selected, dtype=int)
