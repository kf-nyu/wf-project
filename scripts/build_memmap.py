#!/usr/bin/env python3
"""
Build train/val/test memmap .npy files from traces listed in metadata CSV (one-time, slow).
Matches stratified split + optional subset behavior of train_baseline.py.

  .venv/bin/python3 scripts/build_memmap.py
  .venv/bin/python3 scripts/build_memmap.py --subset-sites 0,1,2,3,4 --max-per-class 200

Outputs under --out-dir (default <repo>/processed/memmap/):
  memmap_manifest.json, train_X.npy, train_y.npy, val_*.npy, test_*.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from df95_dataset import (  # noqa: E402
    DEFAULT_METADATA,
    build_subset,
    load_metadata_rows,
    pad_truncate,
    parse_sites,
    stratified_trace_indices,
    trace_path_to_directions,
)


def fill_split(
    paths: list[str],
    labels: list[int],
    seq_len: int,
    x_mm: np.memmap,
    y_mm: np.memmap,
    pad_value: float,
) -> None:
    from pathlib import Path as P

    n = len(paths)
    t0 = time.perf_counter()
    for i in range(n):
        seq = trace_path_to_directions(P(paths[i]))
        seq = pad_truncate(seq, seq_len, pad_value)
        x_mm[i, :] = np.asarray(seq, dtype=np.float32)
        y_mm[i] = np.int32(labels[i])
        if (i + 1) % 5000 == 0 or i + 1 == n:
            dt = time.perf_counter() - t0
            print(f"  ... {i + 1}/{n} rows ({dt:.1f}s elapsed)")
    x_mm.flush()
    y_mm.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="Build DF95 memmap arrays for fast training")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--out-dir", type=Path, default=_PROJECT_ROOT / "processed" / "memmap")
    p.add_argument("--seq-len", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42, help="Must match train_baseline --seed")
    p.add_argument("--subset-sites", type=str, default=None)
    p.add_argument("--max-per-class", type=int, default=None)
    args = p.parse_args()

    metadata = args.metadata.expanduser().resolve()
    if not metadata.is_file():
        raise SystemExit(f"Metadata not found: {metadata}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, site_ids = load_metadata_rows(metadata)
    sites = parse_sites(args.subset_sites)
    if sites is not None:
        paths, site_ids, num_classes = build_subset(
            paths, site_ids, sites, args.max_per_class, args.seed
        )
        label_mode = f"subset remapped 0..{num_classes - 1}"
    else:
        if args.max_per_class is not None:
            allowed = sorted(set(site_ids))
            paths, site_ids, num_classes = build_subset(
                paths, site_ids, allowed, args.max_per_class, args.seed
            )
            label_mode = f"capped max {args.max_per_class}/class, remapped to {num_classes} classes"
        else:
            num_classes = max(site_ids) + 1
            label_mode = f"full {num_classes} classes, site_id as label"

    tr_i, va_i, te_i = stratified_trace_indices(site_ids, seed=args.seed)
    splits: dict[str, tuple[list[int], str, str]] = {
        "train": (tr_i, "train_X.npy", "train_y.npy"),
        "val": (va_i, "val_X.npy", "val_y.npy"),
        "test": (te_i, "test_X.npy", "test_y.npy"),
    }

    print(f"out_dir: {out_dir}")
    print(f"seq_len: {args.seq_len}  num_classes: {num_classes}  label_mode: {label_mode}")
    for name, (indices, x_name, y_name) in splits.items():
        n = len(indices)
        print(f"writing {name}: {n} samples -> {x_name}, {y_name}")
        x_path = out_dir / x_name
        y_path = out_dir / y_name
        x_mm: np.memmap = open_memmap(
            x_path, mode="w+", dtype=np.float32, shape=(n, args.seq_len)
        )
        y_mm: np.memmap = open_memmap(
            y_path, mode="w+", dtype=np.int32, shape=(n,)
        )
        p_subset = [paths[i] for i in indices]
        y_subset = [site_ids[i] for i in indices]
        fill_split(p_subset, y_subset, args.seq_len, x_mm, y_mm, 0.0)
        del x_mm
        del y_mm

    manifest = {
        "version": 1,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "num_classes": num_classes,
        "label_mode": label_mode,
        "metadata": str(metadata),
        "subset_sites": args.subset_sites,
        "max_per_class": args.max_per_class,
        "splits": {
            "train": {"n": len(tr_i), "x": "train_X.npy", "y": "train_y.npy"},
            "val": {"n": len(va_i), "x": "val_X.npy", "y": "val_y.npy"},
            "test": {"n": len(te_i), "x": "test_X.npy", "y": "test_y.npy"},
        },
    }
    man_path = out_dir / "memmap_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {man_path}")


if __name__ == "__main__":
    main()
