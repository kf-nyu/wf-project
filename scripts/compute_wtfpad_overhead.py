#!/usr/bin/env python3
"""
WTF-PAD paired overhead (bandwidth + trace-duration proxies).

For each row in processed/wtfpad_paired.csv, reads matching original and protected
trace files and computes:
  - Bandwidth proxy B: sum of |signed cell size| over all lines (2nd column).
  - Duration proxy T: t_last - t_first (1st column, seconds).

Per-pair relative overheads: (B_p - B_o) / B_o, (T_p - T_o) / T_o
(skips if B_o == 0 or T_o == 0).

This matches the definitions discussed in memo.md (not browser page-load time).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_META = _PROJECT_ROOT / "processed" / "wtfpad_paired.csv"
_DEFAULT_OUT_JSON = _PROJECT_ROOT / "results" / "wtfpad" / "overhead_summary.json"


def trace_metrics(path: Path) -> tuple[int, int, float] | None:
    """
    Returns (sum_abs_size, n_lines, duration_sec) or None if no usable lines.
    """
    sum_abs = 0
    first_t: float | None = None
    last_t: float | None = None
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            t = float(parts[0])
            s = int(parts[1])
            n += 1
            if first_t is None:
                first_t = t
            last_t = t
            sum_abs += abs(s)
    if n == 0 or first_t is None or last_t is None:
        return None
    duration = last_t - first_t
    return sum_abs, n, float(duration)


def main() -> None:
    p = argparse.ArgumentParser(description="WTF-PAD paired bandwidth and duration overheads")
    p.add_argument("--metadata", type=Path, default=_DEFAULT_META)
    p.add_argument(
        "--out-json",
        type=Path,
        default=_DEFAULT_OUT_JSON,
        help="Write summary JSON (default: results/wtfpad/overhead_summary.json)",
    )
    p.add_argument(
        "--out-per-trace",
        type=Path,
        default=None,
        help="Optional CSV of per-trace B_o, B_p, T_o, T_p, h_bw, h_lat",
    )
    args = p.parse_args()

    meta = args.metadata.expanduser().resolve()
    if not meta.is_file():
        print(f"Missing {meta}; run scripts/make_wtfpad_metadata.py first.", file=sys.stderr)
        raise SystemExit(1)

    rel_bw: list[float] = []
    rel_lat: list[float] = []
    rel_nlines: list[float] = []
    rows_out: list[dict[str, object]] = []
    skipped = 0

    with meta.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_o = Path(row["path_original"])
            p_p = Path(row["path_protected"])
            m_o = trace_metrics(p_o)
            m_p = trace_metrics(p_p)
            if m_o is None or m_p is None:
                skipped += 1
                continue
            b_o, n_o, t_o = m_o
            b_p, n_p, t_p = m_p
            if b_o <= 0 or t_o <= 0:
                skipped += 1
                continue
            hb = (b_p - b_o) / b_o
            ht = (t_p - t_o) / t_o
            hn = (n_p - n_o) / n_o if n_o > 0 else float("nan")
            rel_bw.append(hb)
            rel_lat.append(ht)
            if n_o > 0:
                rel_nlines.append(hn)
            rows_out.append(
                {
                    "basename": row.get("basename", ""),
                    "B_orig": b_o,
                    "B_prot": b_p,
                    "T_orig_s": t_o,
                    "T_prot_s": t_p,
                    "h_bw": hb,
                    "h_lat": ht,
                    "h_nlines": hn,
                    "n_lines_o": n_o,
                    "n_lines_p": n_p,
                }
            )

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    def safe_median(xs: list[float]) -> float:
        return float(statistics.median(xs)) if xs else float("nan")

    summary = {
        "n_pairs": len(rel_bw),
        "skipped": skipped,
        "bandwidth_overhead": {
            "definition": "For each trace, B = sum_i |signed_size_i| (2nd column). h_bw = (B_prot - B_orig) / B_orig.",
            "mean_B_orig": mean([r["B_orig"] for r in rows_out]) if rows_out else None,
            "mean_B_prot": mean([r["B_prot"] for r in rows_out]) if rows_out else None,
            "mean_h_bw": mean(rel_bw),
            "median_h_bw": safe_median(rel_bw),
            "min_h_bw": min(rel_bw) if rel_bw else None,
            "max_h_bw": max(rel_bw) if rel_bw else None,
        },
        "latency_overhead": {
            "definition": "T = t_last - t_first (seconds, from 1st column). h_lat = (T_prot - T_orig) / T_orig. Trace-duration proxy, not browser PLT.",
            "mean_T_orig_s": mean([r["T_orig_s"] for r in rows_out]) if rows_out else None,
            "mean_T_prot_s": mean([r["T_prot_s"] for r in rows_out]) if rows_out else None,
            "mean_h_lat": mean(rel_lat),
            "median_h_lat": safe_median(rel_lat),
            "min_h_lat": min(rel_lat) if rel_lat else None,
            "max_h_lat": max(rel_lat) if rel_lat else None,
            "note": "If all h_lat are 0, this release has identical first/last timestamps on paired files (padding in-band); compare Juarez et al. bandwidth-focused defense.",
        },
        "line_count_overhead": {
            "definition": "h_nlines = (n_lines_prot - n_lines_orig) / n_lines_orig. Captures more Tor cells emitted even when T_last-T_first is unchanged.",
            "mean_h_nlines": mean(rel_nlines) if rel_nlines else None,
            "median_h_nlines": safe_median(rel_nlines) if rel_nlines else None,
        },
    }

    out_json = args.out_json.expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Wrote: {out_json}")

    if args.out_per_trace and rows_out:
        pcsv = args.out_per_trace.expanduser().resolve()
        pcsv.parent.mkdir(parents=True, exist_ok=True)
        with pcsv.open("w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(
                cf,
                fieldnames=[
                    "basename",
                    "B_orig",
                    "B_prot",
                    "T_orig_s",
                    "T_prot_s",
                    "h_bw",
                    "h_lat",
                    "h_nlines",
                    "n_lines_o",
                    "n_lines_p",
                ],
            )
            w.writeheader()
            for r in rows_out:
                w.writerow(r)  # type: ignore[arg-type]
        print(f"Wrote: {pcsv}")

    # Human-readable
    print()
    print("WTF-PAD paired overhead (", summary["n_pairs"], "traces, skipped=", skipped, ")", sep="")
    b = summary["bandwidth_overhead"]
    t = summary["latency_overhead"]
    print("Bandwidth proxy:  mean h_bw =", f"{b['mean_h_bw']:.4%}" if b["mean_h_bw"] is not None else "n/a")
    print("                    median h_bw =", f"{b['median_h_bw']:.4%}" if b["median_h_bw"] is not None else "n/a")
    print("Duration proxy:   mean h_lat =", f"{t['mean_h_lat']:.4%}" if t["mean_h_lat"] is not None else "n/a")
    print("                    median h_lat =", f"{t['median_h_lat']:.4%}" if t["median_h_lat"] is not None else "n/a")
    nlines = summary.get("line_count_overhead", {})
    if nlines.get("mean_h_nlines") is not None:
        print("Line count:        mean h_nlines =", f"{nlines['mean_h_nlines']:.4%}")
        print("                    median h_nlines =", f"{nlines['median_h_nlines']:.4%}")
    if t.get("max_h_lat") == 0 and t.get("min_h_lat") == 0:
        print(
            "Note: h_lat is 0 for all pairs (identical t_first and t_last on these files);"
            " use h_bw and/or h_nlines, or a different end-to-end timing definition if you collect your own data.",
        )


if __name__ == "__main__":
    main()
