# wf-project

Evaluation of Tor website fingerprinting defenses against modern machine learning attacks, with a primary focus on `WTF-PAD` under `defense generalization`.

**Repository:** [github.com/kf-nyu/wf-project](https://github.com/kf-nyu/wf-project)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

Raw datasets are large and are not committed; see `downloads/README.md` for acquisition notes. Local metadata and memmaps under `processed/` are gitignored patterns (regenerate with the scripts below).

## Overview

This repository contains code and project materials for studying whether a strong modern website fingerprinting attacker still succeeds against a classic Tor defense. The current mainline experiment:

- trains a `Deep Fingerprinting`-style CNN on undefended traffic
- evaluates `WTF-PAD` without retraining on defended traffic
- reports attack effectiveness and trace-level overhead together

The project distinguishes clearly between:

- `DF95` for a strong undefended attacker baseline
- paired `WTF-PAD` traces for defense-generalization evaluation

These are different closed worlds and should not be merged into one headline accuracy claim.

## Main Scripts

- `scripts/make_df95_csv.py` — build DF95 metadata
- `scripts/build_memmap.py` — optional preprocessing for faster DF95 training
- `scripts/train_df_official.py` — train the headline DFNet-style DF95 baseline
- `scripts/eval_df_official_checkpoint.py` — evaluate a saved DF95 checkpoint (accuracy / macro P/R/F1, optional confusion export)
- `scripts/make_wtfpad_metadata.py` — build paired metadata for WTF-PAD original/protected traces
- `scripts/train_wtfpad.py` — train on WTF-PAD original traces only
- `scripts/eval_wtfpad_defense.py` — evaluate matched original/protected test traces
- `scripts/compute_wtfpad_overhead.py` — compute paired overhead summaries
- `scripts/make_awf_large_metadata.py` — build metadata for AWF large closed-world experiments

## Core Workflow

### 1. Train the undefended DF95 baseline

```bash
.venv/bin/python3 scripts/make_df95_csv.py
.venv/bin/python3 scripts/build_memmap.py
.venv/bin/python3 scripts/train_df_official.py
```

### 2. Prepare WTF-PAD paired metadata

```bash
.venv/bin/python3 scripts/make_wtfpad_metadata.py
```

### 3. Train on original WTF-PAD traces

```bash
.venv/bin/python3 scripts/train_wtfpad.py --arch dfnet
```

### 4. Evaluate defense generalization

Protected traces:

```bash
.venv/bin/python3 scripts/eval_wtfpad_defense.py \
  --checkpoint results/wtfpad/wtf100_dfnet_best.pt \
  --traffic protected
```

Original matched test split:

```bash
.venv/bin/python3 scripts/eval_wtfpad_defense.py \
  --checkpoint results/wtfpad/wtf100_dfnet_best.pt \
  --traffic original
```

### 5. Compute overhead

```bash
.venv/bin/python3 scripts/compute_wtfpad_overhead.py \
  --out-per-trace results/wtfpad/overhead_per_trace.csv
```

## Metrics

Current reporting includes:

- `top-1 accuracy`
- `macro precision`
- `macro recall`
- `macro F1`
- paired trace-level overhead:
  - byte-sum proxy
  - line-count proxy
  - trace-span duration proxy

Note: the trace-span duration proxy is not the same as browser page-load time.

## Datasets

### DF95

Used for the strong undefended attacker baseline. This repository works from raw trace files and builds local metadata from them.

### WTF-PAD paired release

Used for the defense experiment. The important property is that original and protected traces are aligned, enabling evaluation on the same held-out test IDs without retraining on defended data.

### AWF Large Closed World

Not part of the current mainline paper results, but supported as a larger follow-up benchmark for additional validation.

After training with `scripts/train_df_official.py` (and `--checkpoint-tag` per preset), recompute accuracy, macro precision/recall/F1, and optional confusion output **without** retraining:

```bash
.venv/bin/python3 scripts/eval_df_official_checkpoint.py \
  --checkpoint <path/to/best.pt> \
  --metadata <same as training> \
  --memmap-dir <same as training> \
  [--confusion-npy processed/awf_conf_K.npy]
```

See `downloads/README.md` for more dataset-specific notes.

## Notes

- Hardware, filesystem performance, and preprocessing choices materially affect training throughput on large trace corpora.
- LaTeX or course write-up trees are excluded from this repository by default (see `.gitignore`, e.g. `paper/`, `proposal/`). This repo tracks the reproducible experiment code and dataset notes.
