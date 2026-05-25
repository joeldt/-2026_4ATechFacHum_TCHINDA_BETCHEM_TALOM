from __future__ import annotations

from dataclasses import dataclass

from .config import Thresholds
from .processing import SignalFeatures


@dataclass(frozen=True)
class BiofeedbackState:
    score: float
    feather_y: float
    label: str
    respiration_score: float
    heart_score: float
    movement_score: float
    movement_label: str


class BiofeedbackEngine:
    def __init__(self, thresholds: Thresholds) -> None:
        self.thresholds = thresholds
        self.score = 0.35

    def update(self, features: SignalFeatures) -> BiofeedbackState:
        movement_score = self._movement_score(features)
        respiration_score = self._respiration_score(features)
        heart_score = self._heart_score(features)
        target = self._target_score(respiration_score, movement_score, heart_score)
        self.score = clamp((0.92 * self.score) + (0.08 * target), 0.0, 1.0)
        feather_y = self.score

        if movement_score < 0.25:
            label = "mouvement fort"
        elif self.score > 0.72:
            label = "calme stable"
        elif self.score > 0.45:
            label = "transition"
        else:
            label = "trop actif"

        return BiofeedbackState(
            score=self.score,
            feather_y=feather_y,
            label=label,
            respiration_score=respiration_score,
            heart_score=heart_score,
            movement_score=movement_score,
            movement_label=movement_label(movement_score),
        )

    def _target_score(self, respiration: float, movement: float, heart: float) -> float:
        return clamp((0.50 * respiration) + (0.35 * movement) + (0.15 * heart), 0.0, 1.0)

    def _respiration_score(self, features: SignalFeatures) -> float:
        rate_score = 0.35
        if features.respiration_bpm is not None:
            lo = self.thresholds.respiration_min_bpm
            hi = self.thresholds.respiration_max_bpm
            center = (lo + hi) / 2.0
            half_width = max(1.0, (hi - lo) / 2.0)
            rate_score = 1.0 - min(1.0, abs(features.respiration_bpm - center) / half_width)

        amplitude_score = clamp(features.respiration_amplitude / 0.004, 0.0, 1.0)
        return clamp((0.60 * rate_score) + (0.25 * features.respiration_regularity) + (0.15 * amplitude_score), 0.0, 1.0)

    def _heart_score(self, features: SignalFeatures) -> float:
        heart = 0.35
        if features.heart_bpm is not None:
            if self.thresholds.heart_min_bpm <= features.heart_bpm <= self.thresholds.heart_max_bpm:
                heart = 0.75
            else:
                heart = 0.25
        return clamp((0.65 * features.ppg_quality) + (0.35 * heart), 0.0, 1.0)

    def _movement_score(self, features: SignalFeatures) -> float:
        return scale_inverse(
            features.movement_rms,
            low=self.thresholds.movement_calm_rms,
            high=self.thresholds.movement_high_rms,
        )


def scale_inverse(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp(1.0 - ((value - low) / (high - low)), 0.0, 1.0)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def movement_label(score: float) -> str:
    if score > 0.75:
        return "immobile"
    if score > 0.45:
        return "leger mouvement"
    if score > 0.20:
        return "agite"
    return "mouvement fort"
