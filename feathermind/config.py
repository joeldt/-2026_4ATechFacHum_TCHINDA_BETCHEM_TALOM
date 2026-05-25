from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChannelMap:
    acc_x: int
    acc_y: int
    acc_z: int
    ppg: int
    pzt: int

    @property
    def acquisition_channels(self) -> list[int]:
        return sorted({self.acc_x, self.acc_y, self.acc_z, self.ppg, self.pzt})


@dataclass(frozen=True)
class Thresholds:
    respiration_min_bpm: float
    respiration_max_bpm: float
    heart_min_bpm: float
    heart_max_bpm: float
    movement_calm_rms: float
    movement_high_rms: float


@dataclass(frozen=True)
class AlarmConfig:
    enabled: bool
    trigger_score: float
    cooldown_seconds: float
    frequency_hz: int
    duration_ms: int


@dataclass(frozen=True)
class MotionAlarmConfig:
    enabled: bool
    trigger_rms: float
    cooldown_seconds: float
    frequency_hz: int
    duration_ms: int


@dataclass(frozen=True)
class AppConfig:
    mac_address: str
    sampling_rate_hz: int
    read_chunk_samples: int
    channels: ChannelMap
    thresholds: Thresholds
    alarm: AlarmConfig
    motion_alarm: MotionAlarmConfig
    simulation_scenario: str
    log_directory: Path


def load_config(path: str | Path = "config.example.json") -> AppConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> AppConfig:
    alarm = raw.get("alarm", {})
    motion_alarm = raw.get("motion_alarm", {})
    return AppConfig(
        mac_address=str(raw["mac_address"]),
        sampling_rate_hz=int(raw["sampling_rate_hz"]),
        read_chunk_samples=int(raw["read_chunk_samples"]),
        channels=ChannelMap(**raw["channels"]),
        thresholds=Thresholds(**raw["thresholds"]),
        alarm=AlarmConfig(
            enabled=bool(alarm.get("enabled", True)),
            trigger_score=float(alarm.get("trigger_score", 0.82)),
            cooldown_seconds=float(alarm.get("cooldown_seconds", 8.0)),
            frequency_hz=int(alarm.get("frequency_hz", 880)),
            duration_ms=int(alarm.get("duration_ms", 700)),
        ),
        motion_alarm=MotionAlarmConfig(
            enabled=bool(motion_alarm.get("enabled", True)),
            trigger_rms=float(motion_alarm.get("trigger_rms", 0.00045)),
            cooldown_seconds=float(motion_alarm.get("cooldown_seconds", 3.0)),
            frequency_hz=int(motion_alarm.get("frequency_hz", 1320)),
            duration_ms=int(motion_alarm.get("duration_ms", 450)),
        ),
        simulation_scenario=str(raw.get("simulation_scenario", "default")),
        log_directory=Path(raw.get("log_directory", "data")),
    )
