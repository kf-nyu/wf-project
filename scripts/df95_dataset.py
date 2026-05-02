"""
Load DF95 raw traces for a 1D-CNN (Deep Fingerprinting–style).

- Metadata CSV lists filepath + site_id (label); direction is read from each trace file
  (sign of the signed packet size column).
- Each sample is a float tensor of shape (1, seq_len) for Conv1d (batch, channels, length).
"""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python3"

try:
    import torch
except ModuleNotFoundError as exc:
    hint = (
        "PyTorch is not installed for the Python you are running.\n\n"
        f"  Interpreter: {sys.executable}\n\n"
        "Use the project virtualenv (where torch is installed), for example:\n\n"
        f"  {_VENV_PYTHON} {_SCRIPT_DIR / 'df95_dataset.py'}\n\n"
        "Or from the repo root:\n\n"
        "  source .venv/bin/activate\n"
        "  python scripts/df95_dataset.py\n\n"
        "If .venv is missing or broken, create it and install deps:\n\n"
        f"  cd {_PROJECT_ROOT}\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/python3 -m pip install -r requirements.txt\n"
    )
    raise SystemExit(hint) from exc

import numpy as np
from torch.utils.data import DataLoader, Dataset

DEFAULT_METADATA = _PROJECT_ROOT / "processed" / "df95_metadata.csv"
DEFAULT_MEMMAP_DIR = _PROJECT_ROOT / "processed" / "memmap"


def parse_sites(s: str | None) -> list[int] | None:
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_subset(
    paths: list[str],
    site_ids: list[int],
    allowed: list[int],
    max_per_class: int | None,
    seed: int,
) -> tuple[list[str], list[int], int]:
    """
    Keep only traces from allowed sites; optionally cap per site.
    Labels remapped to 0..K-1 in sorted(allowed) order.
    """
    order = sorted(allowed)
    remap = {old: i for i, old in enumerate(order)}
    k = len(order)
    rng = random.Random(seed)
    by_site: dict[int, list[str]] = defaultdict(list)
    for p, s in zip(paths, site_ids, strict=True):
        if s in remap:
            by_site[s].append(p)
    new_paths: list[str] = []
    new_y: list[int] = []
    for s in order:
        ps = by_site[s][:]
        rng.shuffle(ps)
        if max_per_class is not None:
            ps = ps[:max_per_class]
        label = remap[s]
        for p in ps:
            new_paths.append(p)
            new_y.append(label)
    if not new_paths:
        raise ValueError("Subset is empty: check subset sites and data paths.")
    return new_paths, new_y, k


def trace_path_to_directions(path: Path) -> list[float]:
    """Read one trace file: each line is timestamp + signed size (whitespace-separated)."""
    directions: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            size = int(parts[1])
            if size > 0:
                directions.append(1.0)
            elif size < 0:
                directions.append(-1.0)
            else:
                directions.append(0.0)
    return directions


def pad_truncate(seq: list[float], seq_len: int, pad_value: float = 0.0) -> list[float]:
    if len(seq) >= seq_len:
        return seq[:seq_len]
    return seq + [pad_value] * (seq_len - len(seq))


def load_metadata_rows(metadata_csv: Path) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    site_ids: list[int] = []
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["filepath"])
            site_ids.append(int(row["site_id"]))
    return paths, site_ids


def stratified_trace_indices(
    site_ids: list[int],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """
    Per-class split of trace indices so each site contributes to train/val/test.
    Closed-world multiclass: keeps class balance similar across splits.
    """
    by_site: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(site_ids):
        by_site[s].append(i)
    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for site in sorted(by_site.keys()):
        idx = by_site[site][:]
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_idx.append(idx[0])
            continue
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_train = max(1, n_train)
        n_val = max(0, n_val)
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
        if n_train < 1:
            n_train = 1
        n_test = n - n_train - n_val
        if n_test < 1:
            n_val = max(0, n_val - 1)
            n_test = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_val
        i = 0
        train_idx.extend(idx[i : i + n_train])
        i += n_train
        val_idx.extend(idx[i : i + n_val])
        i += n_val
        test_idx.extend(idx[i:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


class DF95SequenceDataset(Dataset):
    """
    PyTorch Dataset: reads trace files on demand.
    __getitem__ returns (x, y) with x.shape == (1, seq_len), y == site_id in 0..94.
    """

    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        seq_len: int = 5000,
        pad_value: float = 0.0,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("paths and labels length mismatch")
        self.paths = paths
        self.labels = labels
        self.seq_len = seq_len
        self.pad_value = pad_value

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.paths[i])
        label = int(self.labels[i])
        seq = trace_path_to_directions(path)
        seq = pad_truncate(seq, self.seq_len, self.pad_value)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class DF95MemmapDataset(Dataset):
    """
    Precomputed float32 [N, seq_len] and int32/64 labels on disk, loaded with mmap.
    __getitem__ returns (x, y) with x.shape (1, seq_len) for Conv1d.
    """

    def __init__(
        self,
        x_path: Path,
        y_path: Path,
    ) -> None:
        x_path = Path(x_path)
        y_path = Path(y_path)
        self.X: np.memmap | np.ndarray = np.load(x_path, mmap_mode="r")
        self.y: np.memmap | np.ndarray = np.load(y_path, mmap_mode="r")
        if self.X.ndim != 2 or self.y.ndim != 1:
            raise ValueError("Memmap X must be 2D and y must be 1D")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("Memmap X and y length mismatch")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # copy row: safe for dataloader and MPS
        x = torch.from_numpy(np.asarray(self.X[i], dtype=np.float32).copy()).view(1, -1)
        y = torch.tensor(int(self.y[i]), dtype=torch.long)
        return x, y


def build_dataloaders(
    metadata_csv: Path = DEFAULT_METADATA,
    seq_len: int = 5000,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    paths, site_ids = load_metadata_rows(metadata_csv)
    tr_i, va_i, te_i = stratified_trace_indices(site_ids, seed=seed)
    train_ds = DF95SequenceDataset(
        [paths[i] for i in tr_i], [site_ids[i] for i in tr_i], seq_len=seq_len
    )
    val_ds = DF95SequenceDataset(
        [paths[i] for i in va_i], [site_ids[i] for i in va_i], seq_len=seq_len
    )
    test_ds = DF95SequenceDataset(
        [paths[i] for i in te_i], [site_ids[i] for i in te_i], seq_len=seq_len
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


if __name__ == "__main__":
    meta = DEFAULT_METADATA
    if not meta.is_file():
        raise SystemExit(f"Missing metadata CSV: {meta}")
    train_loader, _, _ = build_dataloaders(metadata_csv=meta, batch_size=4, seq_len=5000)
    xb, yb = next(iter(train_loader))
    print("batch x:", xb.shape, xb.dtype)  # (4, 1, 5000)
    print("batch y:", yb.shape, yb.dtype)  # (4,) int64
    print("label range:", int(yb.min()), int(yb.max()))
