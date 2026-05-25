from __future__ import annotations

import argparse
import csv
import time

from .acquisition import create_source
from .config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enregistre des donnees brutes FeatherMind")
    parser.add_argument("--mode", choices=["simulate", "demo", "plux", "bitalino"], default="plux")
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.log_directory.mkdir(parents=True, exist_ok=True)
    output = args.output or str(config.log_directory / "raw_capture.csv")

    source = create_source(args.mode, config)
    deadline = time.time() + args.seconds

    try:
        with open(output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "acc_x", "acc_y", "acc_z", "ppg", "pzt"])
            while time.time() < deadline:
                frame = source.read()
                for row in zip(frame.timestamp, frame.acc_x, frame.acc_y, frame.acc_z, frame.ppg, frame.pzt):
                    writer.writerow([f"{float(value):.6f}" for value in row])
    finally:
        source.close()

    print(f"Capture terminee: {output}")


if __name__ == "__main__":
    main()
