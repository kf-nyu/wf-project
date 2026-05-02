#!/usr/bin/env python3
"""
Train the **official (CCS'18) DF architecture** (PyTorch `DFNetCCS2018` in `df_ccs2018_model.py`)
on the same DF95 memmap / raw pipeline as `train_baseline.py`.

Separates results from the compact `DF95CNN` baseline: default checkpoint directory is
`results/df_ccs2018/checkpoints/`, not `results/checkpoints/`.

Paper-style defaults (from `src/ClosedWorld_DF_NoDef.py` in the df repo):
  Adamax lr=0.002, batch 128, 30 epochs (we still use early stopping via --patience).

From repo root:
  .venv/bin/python3 scripts/train_df_official.py
  .venv/bin/python3 scripts/train_df_official.py --checkpoint-tag df95_primary
  .venv/bin/python3 scripts/train_df_official.py --epochs 50 --patience 0   # no early stopping
  .venv/bin/python3 scripts/train_df_official.py --no-memmap
  .venv/bin/python3 scripts/train_df_official.py --subset-sites 0,1,2,3,4 --max-per-class 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

_SCRIPT_DIR   = Path(__file__).resolve().parent
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
from train_baseline import (  # noqa: E402
    _lr_for_epoch,
    _manifest_matches_run,
    _set_optimizer_lr,
    evaluate,
    pick_device,
    train_one_epoch,
)
from torch.utils.data import DataLoader  # noqa: E402

_DEFAULT_CKPT = _PROJECT_ROOT / "results" / "df_ccs2018" / "checkpoints"


def main() -> None:
    p = argparse.ArgumentParser(
        description="DF95 with CCS'18 official DF (DFNet) — separate results from train_baseline.py"
    )
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument(
        "--checkpoint-dir",
        type    = Path,
        default = _DEFAULT_CKPT,
        help    = "Directory that will hold df95_df_ccs2018_best.pt "
        "(or <parent>/<tag>/ when --checkpoint-tag is set).",
    )
    p.add_argument(
        "--checkpoint-tag",
        type    = str,
        default = None,
        help    = "Subdirectory under --checkpoint-dir for this run "
        "(e.g. awf_mini_k50_n100). Avoids overwriting DF95 or another AWF experiment.",
    )
    p.add_argument("--seq-len", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128, help="ClosedWorld_DF_NoDef.py uses 128")
    p.add_argument("--epochs", type=int, default=30, help="Paper script uses 30; early stop with --patience")
    p.add_argument(
        "--patience",
        type    = int,
        default = 8,
        help    = "Stop if val top-1 does not improve for this many epochs; 0 = no early stopping (use all --epochs).",
    )
    p.add_argument(
        "--lr",
        type    = float,
        default = 0.002,
        help    = "Default 0.002 matches Keras Adamax in ClosedWorld_DF_NoDef.py (after warmup if any).",
    )
    p.add_argument(
        "--warmup-epochs",
        type    = int,
        default = 0,
        help    = "Linearly ramp LR from 0 to --lr over this many epochs (0 = off, paper has none).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--subset-sites",
        type    = str,
        default = None,
        help    = "Comma-separated site_ids (e.g. 0,1,2). Omit to use every site_id in metadata.",
    )
    p.add_argument(
        "--max-per-class",
        type    = int,
        default = None,
        help    = "Max traces per site after subset filter (for quick / overfit runs).",
    )
    p.add_argument(
        "--memmap-dir",
        type    = Path,
        default = DEFAULT_MEMMAP_DIR,
        help    = "Dir with memmap_manifest.json and *.npy from scripts/build_memmap.py",
    )
    p.add_argument(
        "--no-memmap",
        action  = "store_true",
        help    = "Read raw trace files per step; ignore memmap in --memmap-dir.",
    )
    p.add_argument(
        "--adamax-eps",
        type    = float,
        default = 1e-8,
        help    = "eps for torch.optim.Adamax (Keras default 1e-8).",
    )
    args = p.parse_args()
    ck_root = args.checkpoint_dir.expanduser().resolve()
    args.checkpoint_dir = ck_root / args.checkpoint_tag if args.checkpoint_tag else ck_root

    if args.patience < 0:
        raise SystemExit("--patience must be >= 0 (use 0 to disable early stopping)")
    if args.warmup_epochs < 0:
        raise SystemExit("--warmup-epochs must be >= 0")
    if args.warmup_epochs > args.epochs:
        print(
            f"Warning: --warmup-epochs ({args.warmup_epochs}) > --epochs ({args.epochs}); "
            f"full --lr is only reached after epoch {args.warmup_epochs} (not within this run).",
            flush = True,
        )

    metadata = args.metadata.expanduser().resolve()
    if not metadata.is_file():
        raise SystemExit(f"Metadata not found: {metadata}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()

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

    memmap_dir    = args.memmap_dir.expanduser().resolve()
    manifest_path = memmap_dir / "memmap_manifest.json"
    use_memmap    = (not args.no_memmap) and manifest_path.is_file()

    if use_memmap:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches_run(
            m,
            args.seq_len,
            args.seed,
            num_classes,
            args.subset_sites,
            args.max_per_class,
        ):
            raise SystemExit(
                "memmap manifest does not match this run (seq_len, seed, num_classes, subset). "
                "Re-run scripts/build_memmap.py with the same options, or use --no-memmap."
            )
        sp = m["splits"]
        train_ds  = DF95MemmapDataset(memmap_dir / sp["train"]["x"], memmap_dir / sp["train"]["y"])
        val_ds    = DF95MemmapDataset(memmap_dir / sp["val"]["x"], memmap_dir / sp["val"]["y"])
        test_ds   = DF95MemmapDataset(memmap_dir / sp["test"]["x"], memmap_dir / sp["test"]["y"])
        data_mode = f"memmap ({memmap_dir.name})"
    else:
        if not args.no_memmap and not manifest_path.is_file():
            print(
                f"Note: no {manifest_path.name} under {memmap_dir} — using raw trace files (slower). "
                "Run: .venv/bin/python3 scripts/build_memmap.py",
                flush=True,
            )
        tr_i, va_i, te_i = stratified_trace_indices(site_ids, seed=args.seed)
        train_ds = DF95SequenceDataset(
            [paths[i] for i in tr_i], [site_ids[i] for i in tr_i], seq_len=args.seq_len
        )
        val_ds   = DF95SequenceDataset(
            [paths[i] for i in va_i], [site_ids[i] for i in va_i], seq_len=args.seq_len
        )
        test_ds  = DF95SequenceDataset(
            [paths[i] for i in te_i], [site_ids[i] for i in te_i], seq_len=args.seq_len
        )
        data_mode = "raw trace files"

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    model = DFNetCCS2018(num_classes=num_classes, seq_len=args.seq_len).to(device)
    # Keras: Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer = torch.optim.Adamax(
        model.parameters(),
        lr    = args.lr,
        betas = (0.9, 0.999),
        eps   = args.adamax_eps,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = args.checkpoint_dir / "df95_df_ccs2018_best.pt"

    best_val   = -1.0
    best_epoch = -1
    stale      = 0

    print(f"device: {device}")
    print(f"data: {data_mode}")
    print(f"model: DFNetCCS2018 (CCS'18 / Model_NoDef.py port)")
    print(f"labels: {label_mode}")
    print(f"num_classes: {num_classes}")
    print(f"train / val / test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print(f"checkpoint: {ckpt_path}")
    if args.warmup_epochs > 0:
        print(f"lr schedule: linear warmup 0 -> {args.lr} over {args.warmup_epochs} epoch(s)")
    if args.patience == 0:
        print("early stopping: off (patience=0)")

    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        cur_lr = _lr_for_epoch(args.lr, epoch, args.warmup_epochs)
        _set_optimizer_lr(optimizer, cur_lr)
        t_ep = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        lr_part = f"  lr {cur_lr:.2e}" if args.warmup_epochs > 0 else ""
        print(
            f"epoch {epoch:3d}{lr_part}  train loss {tr_loss:.4f} acc {tr_acc:.4f}  "
            f"val loss {va_loss:.4f} top1 {va_acc:.4f}  ({time.perf_counter() - t_ep:.1f}s)"
        )

        if va_acc > best_val + 1e-6:
            best_val   = va_acc
            best_epoch = epoch
            stale      = 0
            torch.save(
                {
                    "epoch"            : epoch,
                    "model_state_dict" : model.state_dict(),
                    "val_top1"         : va_acc,
                    "num_classes"      : num_classes,
                    "seq_len"          : args.seq_len,
                    "model_name"       : "DFNetCCS2018",
                    "args"             : vars(args),
                },
                ckpt_path,
            )
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(
                    f"early stop: no val improvement for {args.patience} epochs "
                    f"(best epoch {best_epoch}, val top1 {best_val:.4f})"
                )
                break

    print(f"total train wall time: {time.perf_counter() - t0:.1f}s")

    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(
            f"best checkpoint (epoch {state['epoch']}) -> test loss {te_loss:.4f} top1 {te_acc:.4f}"
        )
    else:
        print("no checkpoint saved")


if __name__ == "__main__":
    main()
