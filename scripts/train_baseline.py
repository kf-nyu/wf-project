#!/usr/bin/env python3
"""
Train a small 1D-CNN baseline on DF95 (undefended traces).
For the full CCS'18 DF (Model_NoDef / `DFNet`) use ``scripts/train_df_official.py`` and ``df_ccs2018_model.py`` (separate ``results/df_ccs2018/`` checkpoints).

- num_classes inferred from metadata in full mode (DF95 ⇒ 95; AWF ⇒ number of distinct site_id rows)
- Optional --subset-sites + --max-per-class for sanity / overfit (labels remapped 0..K-1)
- Early stopping on validation top-1 accuracy; saves best checkpoint

Run from repo root with project venv:
  .venv/bin/python3 scripts/build_memmap.py
  .venv/bin/python3 scripts/train_baseline.py
  .venv/bin/python3 scripts/train_baseline.py --no-memmap
  .venv/bin/python3 scripts/train_baseline.py --subset-sites 0,1,2,3,4 --max-per-class 200 --epochs 30 --patience 5
  .venv/bin/python3 scripts/train_baseline.py --lr 1e-4
  .venv/bin/python3 scripts/train_baseline.py --lr 1e-3 --warmup-epochs 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
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


def _manifest_matches_run(
    m: dict,
    seq_len: int,
    seed: int,
    num_classes: int,
    subset_sites: str | None,
    max_per_class: int | None,
) -> bool:
    if m.get("seq_len") != seq_len or m.get("seed") != seed:
        return False
    if m.get("num_classes") != num_classes:
        return False
    ms = m.get("subset_sites")
    ss = subset_sites
    if (ms in (None, "")) != (ss in (None, "")):
        return False
    if ms and ss and ms != ss:
        return False
    if m.get("max_per_class") != max_per_class:
        return False
    return True


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DF95CNN(nn.Module):
    """Compact Conv1d stack + linear head (Deep Fingerprinting–style, not a full DF replica)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=8, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).squeeze(-1)
        return self.head(h)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1
    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate_with_macro_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float, float, float, torch.Tensor]:
    """Multi-class CE loss (sample-weighted), top-1 acc, macro P/R/F1; confusion[row=true,col=pred]."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        total_loss += loss.item()
        total_correct += (pred == y).sum().item()
        total_samples += y.size(0)

        y_cpu = y.cpu()
        pred_cpu = pred.cpu()
        for true_label, pred_label in zip(y_cpu.tolist(), pred_cpu.tolist()):
            conf[true_label, pred_label] += 1

    if total_samples == 0:
        z = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        return 0.0, 0.0, 0.0, 0.0, 0.0, z

    tp = conf.diag().to(torch.float64)
    predicted = conf.sum(dim=0).to(torch.float64)
    actual = conf.sum(dim=1).to(torch.float64)

    precision_per_class = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall_per_class = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
    f1_per_class = torch.where(
        (precision_per_class + recall_per_class) > 0,
        2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class),
        torch.zeros_like(tp),
    )

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()
    return avg_loss, acc, macro_precision, macro_recall, macro_f1, conf


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def _lr_for_epoch(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    """
    Linear warmup: epoch 1..W use (epoch/W)*base_lr; after W use base_lr.
    epoch is 1-based; warmup disabled when warmup_epochs <= 0.
    """
    if warmup_epochs <= 0:
        return base_lr
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    return base_lr


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def main() -> None:
    p = argparse.ArgumentParser(description="DF95 closed-world CNN baseline")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--checkpoint-dir", type=Path, default=_PROJECT_ROOT / "results" / "checkpoints")
    p.add_argument("--seq-len", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8, help="Stop if val top-1 does not improve for this many epochs")
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (after warmup, if any).",
    )
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linearly ramp LR from 0 to --lr over this many epochs (0 = off).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--subset-sites",
        type=str,
        default=None,
        help="Comma-separated site_ids (e.g. 0,1,2). Omit for full 95-class training.",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Max traces per site after subset filter (for quick / overfit runs).",
    )
    p.add_argument(
        "--memmap-dir",
        type=Path,
        default=DEFAULT_MEMMAP_DIR,
        help="Dir with memmap_manifest.json and *.npy from scripts/build_memmap.py (faster training).",
    )
    p.add_argument(
        "--no-memmap",
        action="store_true",
        help="Read raw trace files per step; ignore memmap in --memmap-dir.",
    )
    args = p.parse_args()
    if args.warmup_epochs < 0:
        raise SystemExit("--warmup-epochs must be >= 0")
    if args.warmup_epochs > args.epochs:
        print(
            f"Warning: --warmup-epochs ({args.warmup_epochs}) > --epochs ({args.epochs}); "
            f"full --lr is only reached after epoch {args.warmup_epochs} (not within this run).",
            flush=True,
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

    memmap_dir = args.memmap_dir.expanduser().resolve()
    manifest_path = memmap_dir / "memmap_manifest.json"
    use_memmap = (not args.no_memmap) and manifest_path.is_file()

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
        train_ds = DF95MemmapDataset(memmap_dir / sp["train"]["x"], memmap_dir / sp["train"]["y"])
        val_ds = DF95MemmapDataset(memmap_dir / sp["val"]["x"], memmap_dir / sp["val"]["y"])
        test_ds = DF95MemmapDataset(memmap_dir / sp["test"]["x"], memmap_dir / sp["test"]["y"])
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
        val_ds = DF95SequenceDataset(
            [paths[i] for i in va_i], [site_ids[i] for i in va_i], seq_len=args.seq_len
        )
        test_ds = DF95SequenceDataset(
            [paths[i] for i in te_i], [site_ids[i] for i in te_i], seq_len=args.seq_len
        )
        data_mode = "raw trace files"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    model = DF95CNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoint_dir / "df95_baseline_best.pt"

    best_val = -1.0
    best_epoch = -1
    stale = 0

    print(f"device: {device}")
    print(f"data: {data_mode}")
    print(f"labels: {label_mode}")
    print(f"num_classes: {num_classes}")
    print(f"train / val / test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print(f"checkpoint: {ckpt_path}")
    if args.warmup_epochs > 0:
        print(f"lr schedule: linear warmup 0 -> {args.lr} over {args.warmup_epochs} epoch(s)")

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
            best_val = va_acc
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_top1": va_acc,
                    "num_classes": num_classes,
                    "seq_len": args.seq_len,
                    "args": vars(args),
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs (best epoch {best_epoch}, val top1 {best_val:.4f})")
                break

    print(f"total train wall time: {time.perf_counter() - t0:.1f}s")

    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"best checkpoint (epoch {state['epoch']}) -> test loss {te_loss:.4f} top1 {te_acc:.4f}")
    else:
        print("no checkpoint saved")


if __name__ == "__main__":
    main()
