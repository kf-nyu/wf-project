#!/usr/bin/env python3
"""
Defense generalization on WTF-PAD: load a checkpoint trained on **closed-world-original**,
evaluate on **closed-world-protected** for the same held-out **test** indices (matched by
basename / site_id). No retraining on defended traffic.

Requires: ``processed/wtfpad_paired.csv`` and the same ``--seed`` as training (default 42).

  .venv/bin/python3 scripts/eval_wtfpad_defense.py \\
    --checkpoint results/wtfpad/wtf100_dfnet_best.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from df95_dataset import DF95SequenceDataset, stratified_trace_indices  # noqa: E402
from df_ccs2018_model import DFNetCCS2018  # noqa: E402
from train_baseline import DF95CNN, evaluate_with_macro_metrics, pick_device  # noqa: E402

_DEFAULT_META = _PROJECT_ROOT / "processed" / "wtfpad_paired.csv"


def load_wtfpad_paired(
    path: Path,
) -> tuple[list[str], list[str], list[int]]:
    orig: list[str] = []
    prot: list[str] = []
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
    raise SystemExit(f"Unknown arch: {arch}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    p.add_argument("--metadata", type=Path, default=_DEFAULT_META)
    p.add_argument("--seq-len", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42, help="Must match training split")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument(
        "--traffic",
        choices=("protected", "original"),
        default="protected",
        help="Evaluate matched test indices on protected or original traces",
    )
    p.add_argument(
        "--arch",
        choices=("dfnet", "compact", "auto"),
        default="auto",
        help="Override if checkpoint has no 'arch' key (default: use ckpt)",
    )
    args = p.parse_args()

    metadata = args.metadata.expanduser().resolve()
    ckpt_path = args.checkpoint.expanduser().resolve()
    if not metadata.is_file():
        raise SystemExit(f"Missing metadata: {metadata}")
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()

    path_orig, path_prot, site_ids = load_wtfpad_paired(metadata)
    tr_i, va_i, te_i = stratified_trace_indices(site_ids, seed=args.seed)
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = int(st.get("num_classes", max(site_ids) + 1))
    seq_len = int(st.get("seq_len", args.seq_len))
    arch = st.get("arch", None) if args.arch == "auto" else args.arch
    if not arch:
        raise SystemExit("Checkpoint missing 'arch'; pass --arch dfnet or compact")
    if arch not in ("dfnet", "compact"):
        raise SystemExit(f"Bad arch: {arch}")

    eval_paths = path_prot if args.traffic == "protected" else path_orig
    test_ds = DF95SequenceDataset(
        [eval_paths[i] for i in te_i],
        [site_ids[i] for i in te_i],
        seq_len=seq_len,
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    model = build_model(arch, num_classes, seq_len).to(device)
    model.load_state_dict(st["model_state_dict"])

    loss, acc, macro_precision, macro_recall, macro_f1, _ = evaluate_with_macro_metrics(
        model, loader, device, num_classes
    )
    n_test = len(te_i)
    print(f"device: {device}")
    print(f"checkpoint: {ckpt_path}")
    print(f"traffic: WTF-PAD {args.traffic} (matched basenames, test split n={n_test})")
    print(f"test loss: {loss:.4f}  top-1 accuracy: {acc:.4f}")
    print(
        "macro precision: "
        f"{macro_precision:.4f}  macro recall: {macro_recall:.4f}  macro F1: {macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
