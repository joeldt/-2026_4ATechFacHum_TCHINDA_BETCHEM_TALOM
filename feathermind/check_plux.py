from __future__ import annotations

import argparse
import queue
import threading

from .acquisition import PluxDevice, import_plux
from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verifie l'import et la connexion PLUX")
    parser.add_argument("--config", default="config.local.json")
    parser.add_argument("--address", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    address = args.address or config.mac_address

    plux = import_plux()
    print(f"Import PLUX OK: {getattr(plux, '__file__', 'module natif')}")
    print(f"Tentative de connexion: {address}")

    stop_event = threading.Event()
    rows: queue.Queue[tuple[int, list[float]]] = queue.Queue(maxsize=10)
    device = PluxDevice(address, rows, stop_event)
    try:
        device.close()
    except RuntimeError:
        pass
    print("Connexion PLUX OK.")


if __name__ == "__main__":
    main()
