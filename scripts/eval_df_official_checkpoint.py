#!/usr/bin/env python3
"""
Evaluate a ``train_df_official.py`` checkpoint on the **test** split: top-1 accuracy,
macro precision / recall / F1, and optional confusion matrix save.

Data loading must match the training run (same ``--metadata``, ``--memmap-dir`` /
``--no-memmap``, ``--subset-sites``, ``--max-per-class``, ``--seed``, ``--seq-len``).

Preset A/B/C example (adjust paths and tags to your machine):

  .venv/bin/python3 scripts/eval_df_official_checkpoint.py \\
    --checkpoint results/df_ccs2018/checkpoints/<awf_preset_a>/df95_df_ccs2018_best.pt \\
    --metadata processed/<awf>_k50_meta.csv \\
    --memmap-dir processed/memmap_<awf>_k50/

  # repeat with B/C checkpoints and matching metadata + memmap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from df95_dataset import (  # noqa: E402
    DEFAULT_MEMMAP_DIR,
    DEFAULT_METADATA,
    DF95MemmapDataset,
    DF95SequenceDataset,
    build_subset,
    load_metadata_rows,
    parse_sites,
    stratified_trace_indices,
)
from df_ccs2018_model import DFNetCCS2018  # noqa: E402
from train_baseline import _manifest_matches_run, evaluate_with_macro_metrics, pick_device  # noqa: E402

_DEFAULT_CKPT = _PROJECT_ROOT / "results" / "df_ccs2018" / "checkpoints"


def main() -> None:
    p = argparse.ArgumentParser(
        description="DFNet CCS'18 checkpoint — test metrics + optional confusion matrix (AWF / DF95 pipeline)"
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--memmap-dir", type=Path, default=DEFAULT_MEMMAP_DIR)
    p.add_argument("--no-memmap", action="store_true")
    p.add_argument("--subset-sites", type=str, default=None)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=5000, help="Only used when memmap omitted or raw-mode rebuild")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--confusion-npy",
        type=Path,
        default=None,
        help="Save K×K integer confusion counts (rows=true, cols=predicted)",
    )
    args = p.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    metadata = args.metadata.expanduser().resolve()
    if not metadata.is_file():
        raise SystemExit(f"Metadata not found: {metadata}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()

    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_classes = int(st["num_classes"])
    ckpt_seq_len = int(st.get("seq_len", args.seq_len))

    paths, site_ids = load_metadata_rows(metadata)
    sites = parse_sites(args.subset_sites)
    if sites is not None:
        paths, site_ids, num_classes = build_subset(
            paths, site_ids, sites, args.max_per_class, args.seed
        )
    else:
        if args.max_per_class is not None:
            allowed = sorted(set(site_ids))
            paths, site_ids, num_classes = build_subset(
                paths, site_ids, allowed, args.max_per_class, args.seed
            )
        else:
            num_classes = max(site_ids) + 1

    if num_classes != ckpt_classes:
        raise SystemExit(
            f"Metadata implies num_classes={num_classes} but checkpoint has {ckpt_classes} "
            "(wrong --metadata / --subset-sites / --max-per-class?)"
        )

    memmap_dir = args.memmap_dir.expanduser().resolve()
    manifest_path = memmap_dir / "memmap_manifest.json"
    use_memmap = (not args.no_memmap) and manifest_path.is_file()

    if use_memmap:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches_run(
            m,
            ckpt_seq_len,
            args.seed,
            num_classes,
            args.subset_sites,
            args.max_per_class,
        ):
            raise SystemExit(
                "memmap manifest does not match this eval run (seq_len, seed, num_classes, subset). "
                "Use the same memmap build as training, or pass matching flags."
            )
        sp = m["splits"]
        test_ds = DF95MemmapDataset(memmap_dir / sp["test"]["x"], memmap_dir / sp["test"]["y"])
        data_note = f"memmap ({memmap_dir.name})"
    else:
        if not args.no_memmap and not manifest_path.is_file():
            print(
                f"Note: no {manifest_path.name} — building test loader from raw traces (slow).",
                flush=True,
            )
        _, _, te_i = stratified_trace_indices(site_ids, seed=args.seed)
        test_ds = DF95SequenceDataset(
            [paths[i] for i in te_i],
            [site_ids[i] for i in te_i],
            seq_len=ckpt_seq_len,
        )
        data_note = "raw trace files"

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = DFNetCCS2018(num_classes=num_classes, seq_len=ckpt_seq_len).to(device)
    model.load_state_dict(st["model_state_dict"])

    loss, acc, macro_p, macro_r, macro_f1, conf = evaluate_with_macro_metrics(
        model, test_loader, device, num_classes
    )

    print(f"device: {device}")
    print(f"checkpoint: {ckpt_path}")
    print(f"data: {data_note}")
    print(f"test samples: {len(test_ds)}  num_classes: {num_classes}")
    print(f"test loss (mean CE): {loss:.4f}  top-1 accuracy: {acc:.4f}")
    print(f"macro precision: {macro_p:.4f}  macro recall: {macro_r:.4f}  macro F1: {macro_f1:.4f}")

    off = conf.sum().item() - conf.diag().sum().item()
    print(f"confusion off-diagonal count: {off} / {conf.sum().item()}")

    if args.confusion_npy is not None:
        out = args.confusion_npy.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, conf.cpu().numpy().astype(np.int64))
        print(f"wrote confusion matrix: {out}  shape {tuple(conf.shape)}")


if __name__ == "__main__":
    main()
