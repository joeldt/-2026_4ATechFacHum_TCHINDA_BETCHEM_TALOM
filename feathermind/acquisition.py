from __future__ import annotations

from dataclasses import dataclass
import math
import platform
from pathlib import Path
import queue
import random
import sys
import threading
import time
from typing import Protocol

import numpy as np

from .config import AppConfig


@dataclass
class SensorFrame:
    timestamp: np.ndarray
    acc_x: np.ndarray
    acc_y: np.ndarray
    acc_z: np.ndarray
    ppg: np.ndarray
    pzt: np.ndarray


class SensorSource(Protocol):
    def read(self) -> SensorFrame:
        ...

    def close(self) -> None:
        ...


class SimulatedSource:
    def __init__(self, config: AppConfig, scenario: str | None = None) -> None:
        self.config = config
        self.scenario = scenario or config.simulation_scenario
        self.sample_index = 0
        self.started_at = time.time()
        self.next_read_at = self.started_at

    def read(self) -> SensorFrame:
        fs = self.config.sampling_rate_hz
        n = self.config.read_chunk_samples
        now = time.time()
        if now < self.next_read_at:
            time.sleep(self.next_read_at - now)
        self.next_read_at = max(time.time(), self.next_read_at) + (n / fs)

        idx = np.arange(self.sample_index, self.sample_index + n)
        t = idx / fs
        self.sample_index += n

        elapsed = time.time() - self.started_at
        scenario = demo_profile(elapsed) if self.scenario == "demo" else default_profile(elapsed)
        respiration_bpm = scenario["respiration_bpm"]
        heart_bpm = scenario["heart_bpm"]
        movement = scenario["movement"]
        ppg_noise = scenario["ppg_noise"]
        pzt_noise = scenario["pzt_noise"]
        shock = bool(scenario["shock"])

        pzt = 0.5 + 0.35 * np.sin(2 * math.pi * (respiration_bpm / 60.0) * t)
        pzt += np.random.normal(0, pzt_noise, size=n)

        ppg = 0.5 + 0.16 * np.sin(2 * math.pi * (heart_bpm / 60.0) * t)
        ppg += 0.05 * np.sin(2 * math.pi * (2 * heart_bpm / 60.0) * t)
        ppg += np.random.normal(0, ppg_noise, size=n)

        shake = np.random.normal(0, movement, size=(3, n))
        if shock or random.random() < scenario["random_shock_probability"]:
            shock_width = min(n, max(1, n // 3))
            shock_start = random.randint(0, max(0, n - shock_width))
            shake[:, shock_start : shock_start + shock_width] += np.random.normal(0, 0.003, size=(3, shock_width))

        return SensorFrame(
            timestamp=t,
            acc_x=0.5 + shake[0],
            acc_y=0.5 + shake[1],
            acc_z=0.7 + shake[2],
            ppg=ppg,
            pzt=pzt,
        )

    def close(self) -> None:
        return None


def default_profile(elapsed: float) -> dict[str, float | bool]:
    stress = 1.0 if int(elapsed // 25) % 2 == 1 else 0.0
    return {
        "respiration_bpm": 7.0 + 8.0 * stress,
        "heart_bpm": 64.0 + 20.0 * stress,
        "movement": 0.015 + 0.08 * stress,
        "ppg_noise": 0.012,
        "pzt_noise": 0.015,
        "shock": False,
        "random_shock_probability": 0.02 + 0.08 * stress,
    }


def demo_profile(elapsed: float) -> dict[str, float | bool]:
    phase = elapsed % 80.0
    if phase < 12.0:
        return {
            "respiration_bpm": 16.0,
            "heart_bpm": 86.0,
            "movement": 0.00009,
            "ppg_noise": 0.025,
            "pzt_noise": 0.025,
            "shock": False,
            "random_shock_probability": 0.0,
        }
    if phase < 30.0:
        progress = (phase - 12.0) / 18.0
        return {
            "respiration_bpm": 16.0 - (8.5 * progress),
            "heart_bpm": 86.0 - (18.0 * progress),
            "movement": 0.00009 - (0.00007 * progress),
            "ppg_noise": 0.025 - (0.012 * progress),
            "pzt_noise": 0.025 - (0.010 * progress),
            "shock": False,
            "random_shock_probability": 0.0,
        }
    if phase < 48.0:
        return {
            "respiration_bpm": 6.8,
            "heart_bpm": 62.0,
            "movement": 0.000006,
            "ppg_noise": 0.008,
            "pzt_noise": 0.008,
            "shock": False,
            "random_shock_probability": 0.0,
        }
    if phase < 53.0:
        return {
            "respiration_bpm": 7.0,
            "heart_bpm": 70.0,
            "movement": 0.000010,
            "ppg_noise": 0.010,
            "pzt_noise": 0.010,
            "shock": 48.0 <= phase < 48.6,
            "random_shock_probability": 0.0,
        }
    if phase < 65.0:
        return {
            "respiration_bpm": 14.0,
            "heart_bpm": 82.0,
            "movement": 0.000055,
            "ppg_noise": 0.020,
            "pzt_noise": 0.018,
            "shock": False,
            "random_shock_probability": 0.0,
        }
    return {
        "respiration_bpm": 8.0,
        "heart_bpm": 66.0,
        "movement": 0.000018,
        "ppg_noise": 0.010,
        "pzt_noise": 0.010,
        "shock": False,
        "random_shock_probability": 0.0,
    }


class BitalinoSource:
    def __init__(self, config: AppConfig) -> None:
        try:
            from bitalino import BITalino
        except ImportError as exc:
            raise RuntimeError(
                "Le paquet 'bitalino' n'est pas installe. Lance: python -m pip install -r requirements.txt"
            ) from exc

        self.config = config
        self.channels = config.channels.acquisition_channels
        self.channel_to_column = {
            channel: analog_index for analog_index, channel in enumerate(self.channels)
        }
        self.device = BITalino(config.mac_address)
        self.device.start(config.sampling_rate_hz, self.channels)
        self.sample_index = 0

    def read(self) -> SensorFrame:
        n = self.config.read_chunk_samples
        raw = np.asarray(self.device.read(n), dtype=float)
        analog = raw[:, 5:]
        fs = self.config.sampling_rate_hz
        timestamp = np.arange(self.sample_index, self.sample_index + len(raw)) / fs
        self.sample_index += len(raw)

        def channel(name: str) -> np.ndarray:
            channel_id = getattr(self.config.channels, name)
            column = self.channel_to_column[channel_id]
            return analog[:, column] / 1023.0

        return SensorFrame(
            timestamp=timestamp,
            acc_x=channel("acc_x"),
            acc_y=channel("acc_y"),
            acc_z=channel("acc_z"),
            ppg=channel("ppg"),
            pzt=channel("pzt"),
        )

    def close(self) -> None:
        try:
            self.device.stop()
        finally:
            self.device.close()


class PluxDevice:
    def __new__(cls, address: str, output: queue.Queue[tuple[int, list[float]]], stop_event: threading.Event):
        plux = import_plux()

        class _Device(plux.SignalsDev):
            def __init__(self, device_address: str) -> None:
                plux.SignalsDev.__init__(device_address)
                self.output = output
                self.stop_event = stop_event

            def onRawFrame(self, nSeq, data):  # noqa: N802 - PLUX API callback name
                row = (int(nSeq), [float(value) for value in data])
                try:
                    self.output.put_nowait(row)
                except queue.Full:
                    _ = self.output.get_nowait()
                    self.output.put_nowait(row)
                return self.stop_event.is_set()

        last_error: RuntimeError | None = None
        for attempt in range(1, 4):
            try:
                return _Device(address)
            except RuntimeError as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5)

        raise RuntimeError(
            f"Carte PLUX introuvable a l'adresse {address} apres 3 tentatives. "
            "Verifie que la carte est allumee, chargee, appairee en Bluetooth dans Windows, "
            "qu'aucun autre logiciel comme OpenSignals n'est connecte dessus, "
            "puis eteins/rallume la carte si elle vient d'etre utilisee par un autre script."
        ) from last_error


class PluxSource:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.channels = config.channels.acquisition_channels
        self.port_to_column = {port: index for index, port in enumerate(self.channels)}
        self.rows: queue.Queue[tuple[int, list[float]]] = queue.Queue(maxsize=config.sampling_rate_hz * 5)
        self.stop_event = threading.Event()
        self.device = PluxDevice(config.mac_address, self.rows, self.stop_event)
        self.device.start(config.sampling_rate_hz, self.channels, 16)
        self.loop_thread = threading.Thread(target=self.device.loop, daemon=True)
        self.loop_thread.start()
        self.sample_index = 0

    def read(self) -> SensorFrame:
        n = self.config.read_chunk_samples
        rows: list[list[float]] = []
        for _ in range(n):
            try:
                _, data = self.rows.get(timeout=3.0)
            except queue.Empty as exc:
                raise RuntimeError("Aucune donnee recue depuis la carte PLUX.") from exc
            rows.append(data)

        analog = np.asarray(rows, dtype=float)
        fs = self.config.sampling_rate_hz
        timestamp = np.arange(self.sample_index, self.sample_index + len(analog)) / fs
        self.sample_index += len(analog)

        def channel(name: str) -> np.ndarray:
            port = getattr(self.config.channels, name)
            column = self.port_to_column[port]
            return analog[:, column] / 65535.0

        return SensorFrame(
            timestamp=timestamp,
            acc_x=channel("acc_x"),
            acc_y=channel("acc_y"),
            acc_z=channel("acc_z"),
            ppg=channel("ppg"),
            pzt=channel("pzt"),
        )

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.device.stop()
        finally:
            self.device.close()
        if self.loop_thread.is_alive():
            self.loop_thread.join(timeout=1.0)


def import_plux():
    os_dic = {
        "Darwin": f"MacOS/Intel{''.join(platform.python_version().split('.')[:2])}",
        "Linux": "Linux64",
        "Windows": f"Win{platform.architecture()[0][:2]}_{''.join(platform.python_version().split('.')[:2])}",
    }
    system = platform.system()
    api_folder = os_dic.get(system)
    if api_folder is None:
        raise RuntimeError(f"Systeme non supporte par la PLUX API: {system}")

    candidates = [
        Path.cwd() / "PLUX-API-Python3",
        Path(__file__).resolve().parent / "PLUX-API-Python3",
        Path(__file__).resolve().parents[1] / "PLUX-API-Python3",
        Path.home() / "Downloads" / "PLUX-API-Python3",
    ]

    for root in candidates:
        api_path = root / api_folder
        if api_path.exists():
            api_path_text = str(api_path)
            if api_path_text not in sys.path:
                sys.path.append(api_path_text)
            break

    try:
        import plux
    except ImportError as exc:
        searched = ", ".join(str(path / api_folder) for path in candidates)
        raise RuntimeError(
            "Impossible d'importer l'API PLUX. Place le dossier PLUX-API-Python3 "
            f"dans le projet, dans feathermind ou dans Downloads. Chemins testes: {searched}"
        ) from exc
    return plux


def create_source(mode: str, config: AppConfig) -> SensorSource:
    if mode == "simulate":
        return SimulatedSource(config)
    if mode == "demo":
        return SimulatedSource(config, scenario="demo")
    if mode == "plux":
        return PluxSource(config)
    if mode == "bitalino":
        return BitalinoSource(config)
    raise ValueError(f"Mode inconnu: {mode}")
