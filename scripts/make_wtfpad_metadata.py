#!/usr/bin/env python3
"""
Build a paired metadata CSV for WTF-PAD official closed-world zips
(unzipped to data/wtfpad/ — see downloads/README.md).

Each row: same basename in closed-world-original/ and closed-world-protected/,
label = site id from the filename (e.g. 12-26 -> site_id 12). Closed world has
100 sites (0..99) when the full set is present.

Output: processed/wtfpad_paired.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_ORIG = _PROJECT_ROOT / "data" / "wtfpad" / "closed-world-original"
_DEFAULT_PROT = _PROJECT_ROOT / "data" / "wtfpad" / "closed-world-protected"
_DEFAULT_OUT = _PROJECT_ROOT / "processed" / "wtfpad_paired.csv"


def parse_basename(name: str) -> tuple[int, int]:
    """WTF name: <site_id>-<trace_id> (single hyphen, site and trace are decimals)."""
    if "-" not in name:
        raise ValueError(f"bad trace name: {name!r}")
    a, b = name.split("-", 1)
    return int(a), int(b)


def main() -> None:
    p = argparse.ArgumentParser(description="Paired WTF-PAD original vs protected metadata CSV")
    p.add_argument("--original-dir", type=Path, default=_DEFAULT_ORIG)
    p.add_argument("--protected-dir", type=Path, default=_DEFAULT_PROT)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()

    orig = args.original_dir.expanduser().resolve()
    prot = args.protected_dir.expanduser().resolve()
    if not orig.is_dir() or not prot.is_dir():
        raise SystemExit(
            f"Need both directories (unzip releases into data/wtfpad/). "
            f"Missing: {orig} and/or {prot}"
        )

    orig_files = {f.name: f for f in orig.iterdir() if f.is_file() and not f.name.startswith(".")}
    prot_files = {f.name: f for f in prot.iterdir() if f.is_file() and not f.name.startswith(".")}

    common = sorted(set(orig_files) & set(prot_files))
    missing_o = set(orig_files) - set(prot_files)
    missing_p = set(prot_files) - set(orig_files)
    if missing_o or missing_p:
        print(
            f"Warning: {len(missing_o)} only in original, {len(missing_p)} only in protected",
            flush=True,
        )

    if not common:
        raise SystemExit("No paired trace filenames between original and protected.")

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    max_site = -1
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["basename", "path_original", "path_protected", "site_id", "trace_id"]
        )
        for name in common:
            site_id, trace_id = parse_basename(name)
            max_site = max(max_site, site_id)
            w.writerow(
                [
                    name,
                    str(orig_files[name].resolve()),
                    str(prot_files[name].resolve()),
                    site_id,
                    trace_id,
                ]
            )

    n_class = max_site + 1
    print(f"Wrote: {out}")
    print(f"Paired rows: {len(common)}  site_id in [0, {max_site}]  -> num_classes = {n_class}")


if __name__ == "__main__":
    main()
