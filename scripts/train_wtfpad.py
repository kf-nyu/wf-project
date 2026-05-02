#!/usr/bin/env python3
"""
Train on WTF-PAD **closed-world-original** (100 classes, site 0..99), save checkpoint, then
you can run ``eval_wtfpad_defense.py`` on **closed-world-protected** (defense generalization:
same site label, no retrain on defended data).

Paired metadata: run ``scripts/make_wtfpad_metadata.py`` first → ``processed/wtfpad_paired.csv``.

  .venv/bin/python3 scripts/make_wtfpad_metadata.py
  .venv/bin/python3 scripts/train_wtfpad.py --arch dfnet
  .venv/bin/python3 scripts/eval_wtfpad_defense.py --checkpoint results/wtfpad/wtf100_dfnet_best.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from df95_dataset     import DF95SequenceDataset, stratified_trace_indices  # noqa: E402
from df_ccs2018_model import DFNetCCS2018  # noqa: E402
from train_baseline   import (  # noqa: E402
    DF95CNN,
    evaluate,
    pick_device,
    train_one_epoch,
    _lr_for_epoch,
    _set_optimizer_lr,
)

_DEFAULT_META     = _PROJECT_ROOT / "processed" / "wtfpad_paired.csv"
_DEFAULT_CKPT_DIR = _PROJECT_ROOT / "results" / "wtfpad"


def load_wtfpad_paired(
    path: Path,
) -> tuple[list[str], list[str], list[int]]:
    orig:  list[str] = []
    prot:  list[str] = []
    sites: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orig.append(row["path_original"])
            prot.append(row["path_protected"])
            sites.append(int(row["site_id"]))
    return orig, prot, sites


def build_model(arch: str, num_classes: int, seq_len: int) -> torch.nn.Module:
    if arch == "dfnet":
        return DFNetCCS2018(num_classes=num_classes, seq_len=seq_len)
    if arch == "compact":
        return DF95CNN(num_classes=num_classes)
    raise SystemExit(f"Unknown --arch: {arch}")


def main() -> None:
    p = argparse.ArgumentParser(description="WTF-PAD 100-class train on original traces")
    p.add_argument("--metadata", type=Path, default=_DEFAULT_META)
    p.add_argument(
        "--checkpoint-dir",
        type    = Path,
        default = _DEFAULT_CKPT_DIR,
        help    = "Default: results/wtfpad/",
    )
    p.add_argument(
        "--arch",
        choices = ("dfnet", "compact"),
        default = "dfnet",
        help    = "dfnet = DFNetCCS2018; compact = small DF95CNN",
    )
    p.add_argument("--seq-len", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.002, help="Adamax lr for dfnet; use 1e-3 for compact if needed")
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    metadata = args.metadata.expanduser().resolve()
    if not metadata.is_file():
        raise SystemExit(f"Run make_wtfpad_metadata.py first. Missing: {metadata}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()

    path_orig, _path_prot, site_ids = load_wtfpad_paired(metadata)
    num_classes = max(site_ids) + 1
    tr_i, va_i, te_i = stratified_trace_indices(site_ids, seed=args.seed)

    def ds(indices: list[int]) -> DF95SequenceDataset:
        return DF95SequenceDataset(
            [path_orig[i] for i in indices],
            [site_ids[i] for i in indices],
            seq_len=args.seq_len,
        )

    train_loader = DataLoader(
        ds(tr_i), batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )
    val_loader = DataLoader(
        ds(va_i), batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        ds(te_i), batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    model = build_model(args.arch, num_classes, args.seq_len).to(device)
    if args.arch == "dfnet":
        optimizer = torch.optim.Adamax(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr if args.lr != 0.002 else 1e-3)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name       = f"wtf100_{args.arch}_best.pt"
    ckpt_path  = args.checkpoint_dir / name

    best_val   = -1.0
    best_epoch = -1
    stale      = 0
    t0         = time.perf_counter()

    print(f"device: {device}")
    print(f"WTF-PAD original | num_classes: {num_classes} | arch: {args.arch}")
    print(f"train/val/test: {len(tr_i)} / {len(va_i)} / {len(te_i)}")
    print(f"checkpoint: {ckpt_path}")

    for epoch in range(1, args.epochs + 1):
        cur_lr = _lr_for_epoch(args.lr, epoch, args.warmup_epochs)
        if args.warmup_epochs:
            _set_optimizer_lr(optimizer, cur_lr)
        tr_l, tr_a = train_one_epoch(model, train_loader, optimizer, device)
        va_l, va_a = evaluate(model, val_loader, device)
        lr_note = f"  lr {cur_lr:.2e}" if args.warmup_epochs else ""
        print(
            f"epoch {epoch:3d}{lr_note}  train {tr_l:.4f} acc {tr_a:.4f}  val {va_l:.4f} top1 {va_a:.4f}"
        )
        if va_a > best_val + 1e-6:
            best_val, best_epoch, stale = va_a, epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_top1": va_a,
                    "num_classes": num_classes,
                    "seq_len": args.seq_len,
                    "arch": args.arch,
                    "metadata": str(metadata),
                    "seed": args.seed,
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop (best epoch {best_epoch}, val {best_val:.4f})")
                break

    print(f"wall time: {time.perf_counter() - t0:.1f}s")
    if ckpt_path.is_file():
        s = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(s["model_state_dict"])
        te_l, te_a = evaluate(model, test_loader, device)
        print(f"best ckpt (epoch {s['epoch']}) -> test on **original** split: {te_l:.4f} top1 {te_a:.4f}")
        print("Next: eval on **protected** traces without retraining:")
        print(f"  .venv/bin/python3 scripts/eval_wtfpad_defense.py --checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
