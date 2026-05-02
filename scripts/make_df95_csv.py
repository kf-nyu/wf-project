#!/usr/bin/env python3
"""
Build df95_metadata.csv and a small debug packet CSV from raw DF95 trace files.

Paths default to the repository containing this script (portable: Ubuntu, Mac, any clone).
Override with --data-dir / --out-dir if needed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DF95 metadata and debug CSVs.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "df95" / "undefended",
        help="Directory of raw trace files (default: <repo>/data/df95/undefended)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT / "processed",
        help="Output directory (default: <repo>/processed)",
    )
    p.add_argument(
        "--debug-traces",
        type=int,
        default=5,
        help="Number of traces to include in the packet-level debug CSV (0 to skip).",
    )
    return p.parse_args()


def parse_trace_filename(path: Path) -> tuple[int, int]:
    # Example filename: 38-271
    name = path.name
    site_str, trace_str = name.split("-", 1)
    return int(site_str), int(trace_str)


def count_packets(path: Path) -> int:
    """Count non-empty lines (binary, fast; matches UTF-8 for this dataset)."""
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_metadata_csv(base_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = out_dir / "df95_metadata.csv"
    if not base_dir.is_dir():
        raise SystemExit(f"Data directory not found: {base_dir}")

    files = sorted([p for p in base_dir.iterdir() if p.is_file()])
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "site_id", "trace_id", "num_packets"])
        for path in files:
            site_id, trace_id = parse_trace_filename(path)
            num_packets = count_packets(path)
            writer.writerow([str(path.resolve()), site_id, trace_id, num_packets])

    print(f"Wrote metadata CSV: {metadata_csv}")
    print(f"Total trace files: {len(files)}")
    return metadata_csv


def build_debug_packet_csv(
    base_dir: Path, out_dir: Path, max_traces: int
) -> Path | None:
    if max_traces <= 0:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_csv = out_dir / "df95_debug_packets.csv"
    files = sorted([p for p in base_dir.iterdir() if p.is_file()])[:max_traces]
    with debug_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "site_id",
                "trace_id",
                "packet_index",
                "timestamp",
                "signed_packet_size",
                "direction",
            ]
        )
        for path in files:
            site_id, trace_id = parse_trace_filename(path)
            with path.open("r", encoding="utf-8") as trace_file:
                for packet_index, line in enumerate(trace_file):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    timestamp = float(parts[0])
                    signed_packet_size = int(parts[1])
                    direction = 1 if signed_packet_size > 0 else -1
                    writer.writerow(
                        [
                            site_id,
                            trace_id,
                            packet_index,
                            timestamp,
                            signed_packet_size,
                            direction,
                        ]
                    )
    print(f"Wrote debug packet CSV: {debug_csv}")
    print(f"Included first {len(files)} trace files")
    return debug_csv


def main() -> None:
    args = parse_args()
    base_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    build_metadata_csv(base_dir, out_dir)
    build_debug_packet_csv(base_dir, out_dir, args.debug_traces)


if __name__ == "__main__":
    main()
