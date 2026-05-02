#!/usr/bin/env python3
"""
Build awf_large_cw_metadata.csv from an extracted awf_large_cw tree (AWF_Large_CW style).

Layout (after ``tar -xzf AWF_Large_CW.tar.gz -C data``)::

  data/awf_large_cw/<domain.com>/<trace_id>   # trace_id = numeric string, no extension

Each trace line: ``<timestamp>\\t<signed_packet_size>`` (whitespace separation is also accepted).

site_id: contiguous integers 0..K-1 in **sorted** domain name order (stable, reproducible).

Override paths with --data-dir / --out-path. Use --max-sites / --max-traces-per-site to sample
a subset when the full corpus is huge.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def count_packets(path: Path) -> int:
    """Count non-empty lines (fast binary scan)."""
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AWF large closed-world metadata CSV.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "awf_large_cw",
        help="Extracted awf_large_cw root (default: <repo>/data/awf_large_cw)",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=_PROJECT_ROOT / "processed" / "awf_large_cw_metadata.csv",
        help="Output CSV path (default: <repo>/processed/awf_large_cw_metadata.csv)",
    )
    p.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="If set, only use the first N domain folders after sorting by name (alphabetical).",
    )
    p.add_argument(
        "--max-traces-per-site",
        type=int,
        default=None,
        help="If set, cap traces per site (lowest trace_id numerically first).",
    )
    p.add_argument(
        "--debug-traces",
        type=int,
        default=0,
        help="If >0, also write a packet-level debug CSV with this many traces (total, across sites).",
    )
    return p.parse_args()


def _site_dirs(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        raise SystemExit(f"Data directory not found: {data_root}")
    dirs = [p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(dirs, key=lambda p: p.name)


def _trace_files(site_dir: Path) -> list[Path]:
    files: list[tuple[int, Path]] = []
    for p in site_dir.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        if not p.name.isdigit():
            continue
        files.append((int(p.name), p))
    files.sort(key=lambda t: t[0])
    return [t[1] for t in files]


def build_metadata_csv(
    data_root: Path,
    out_path: Path,
    max_sites: int | None,
    max_traces_per_site: int | None,
) -> int:
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_root = data_root.expanduser().resolve()

    all_sites = _site_dirs(data_root)
    if max_sites is not None:
        all_sites = all_sites[: max(0, max_sites)]
    if not all_sites:
        raise SystemExit(f"No site directories under {data_root}")

    k = len(all_sites)
    site_to_id = {p.name: i for i, p in enumerate(all_sites)}
    n_rows = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filepath", "site_id", "site_name", "trace_id", "num_packets"]
        )
        for site_dir in all_sites:
            site_name = site_dir.name
            sid = site_to_id[site_name]
            traces = _trace_files(site_dir)
            if max_traces_per_site is not None:
                traces = traces[: max(0, max_traces_per_site)]
            for path in traces:
                tid = int(path.name)
                n_pkt = count_packets(path)
                writer.writerow(
                    [str(path.resolve()), sid, site_name, tid, n_pkt]
                )
                n_rows += 1

    return n_rows, k


def build_debug_packet_csv(
    data_root: Path,
    out_path: Path,
    max_sites: int | None,
    max_traces_per_site: int | None,
    max_traces: int,
) -> Path | None:
    if max_traces <= 0:
        return None
    data_root = data_root.expanduser().resolve()
    all_sites = _site_dirs(data_root)
    if max_sites is not None:
        all_sites = all_sites[: max(0, max_sites)]
    site_to_id = {p.name: i for i, p in enumerate(all_sites)}

    out_debug = out_path.parent / (out_path.stem + "_debug_packets.csv")
    written_traces = 0
    with out_debug.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "site_id",
                "site_name",
                "trace_id",
                "packet_index",
                "timestamp",
                "signed_packet_size",
                "direction",
            ]
        )
        for site_dir in all_sites:
            site_name = site_dir.name
            sid = site_to_id[site_name]
            traces = _trace_files(site_dir)
            if max_traces_per_site is not None:
                traces = traces[: max(0, max_traces_per_site)]
            for path in traces:
                if written_traces >= max_traces:
                    print(f"Wrote debug packet CSV: {out_debug} (traces {written_traces})")
                    return out_debug
                tid = int(path.name)
                with path.open("r", encoding="utf-8", errors="replace") as trace_file:
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
                                sid,
                                site_name,
                                tid,
                                packet_index,
                                timestamp,
                                signed_packet_size,
                                direction,
                            ]
                        )
                written_traces += 1
    print(f"Wrote debug packet CSV: {out_debug} (traces {written_traces})")
    return out_debug


def main() -> None:
    args = parse_args()
    data_root = args.data_dir.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()

    n, k = build_metadata_csv(
        data_root,
        out_path,
        args.max_sites,
        args.max_traces_per_site,
    )
    print(f"Wrote metadata CSV: {out_path}")
    print(f"Sites (num_classes if using all listed): {k}")
    print(f"Total trace files listed: {n}")
    if args.debug_traces > 0:
        build_debug_packet_csv(
            data_root,
            out_path,
            args.max_sites,
            args.max_traces_per_site,
            args.debug_traces,
        )


if __name__ == "__main__":
    main()
