#!/usr/bin/env python3
"""
Verify site_id labels in df95_metadata.csv for CNN training.

- Counts unique labels, min/max, and whether IDs are contiguous (e.g. 0..94).
- No PyTorch required (stdlib only).

Usage:
  python3 scripts/verify_df95_labels.py
  python3 scripts/verify_df95_labels.py --metadata /path/to/df95_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_METADATA = _PROJECT_ROOT / "processed" / "df95_metadata.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DF95 site_id label set.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Path to df95_metadata.csv (default: {DEFAULT_METADATA})",
    )
    args = parser.parse_args()
    path: Path = args.metadata.expanduser().resolve()

    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    site_ids: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "site_id" not in reader.fieldnames:
            raise SystemExit("CSV must have a site_id column.")
        for row in reader:
            site_ids.append(int(row["site_id"]))

    n_rows = len(site_ids)
    counts = Counter(site_ids)
    unique = sorted(counts.keys())
    lo, hi = unique[0], unique[-1]
    n_classes = len(unique)
    span = hi - lo + 1
    present = set(unique)
    missing_in_span = [i for i in range(lo, hi + 1) if i not in present]
    contiguous_in_span = len(missing_in_span) == 0
    is_zero_based_contiguous = lo == 0 and hi == n_classes - 1 and contiguous_in_span

    print(f"metadata: {path}")
    print(f"rows (traces): {n_rows}")
    print(f"unique site_id (labels): {n_classes}")
    print(f"min site_id: {lo}  max site_id: {hi}")
    print(f"integer span [min..max]: {span} values")
    print(f"contiguous within [min..max]: {contiguous_in_span}")
    if missing_in_span:
        print(f"  missing IDs in span: {missing_in_span[:30]}{' ...' if len(missing_in_span) > 30 else ''}")
    print(f"fits 0..C-1 with C={n_classes} (no gaps): {is_zero_based_contiguous}")

    print()
    if is_zero_based_contiguous:
        print(f"Recommendation: num_classes = {n_classes}; use raw site_id as class index (no remap).")
    elif contiguous_in_span:
        print(
            f"Recommendation: labels are contiguous from {lo} to {hi} but not zero-based. "
            f"Remap y -> y - {lo} or set num_classes = {span} and shift targets in the dataset."
        )
    else:
        print(
            "Recommendation: non-contiguous label IDs. "
            f"Use num_classes = {n_classes} and remap site_id -> 0..{n_classes - 1} (e.g. sorted unique map)."
        )

    per_class = sorted(counts.values())
    print()
    print("traces per site_id: min", per_class[0], "max", per_class[-1], "mean", f"{n_rows / n_classes:.2f}")


if __name__ == "__main__":
    main()
