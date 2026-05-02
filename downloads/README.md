# Local dataset downloads

## Sirinam et al. (Deep Fingerprinting) — 95-class parallel NoDef / WTFPAD / Walkie-Talkie (pickles)

The [official `deep-fingerprinting/df` repo](https://github.com/deep-fingerprinting/df) points to a [Google Drive folder](https://drive.google.com/drive/folders/1kxjqlaWWJbMW56E3kbXlttqcKJFkeCy6?usp=sharing) with **closed-world** `X_train/valid/test_{NoDef|WTFPAD|WalkieTalkie}.pkl` and matching `y_*.pkl` files. That is the **standard** 95-class benchmark with **aligned** undefended and defended (WTF-PAD) splits for reproducing the CCS’18 paper — **not** the same on-disk format as the **raw** DF95 trace files under `data/df95/undefended/`. This project’s mainline pipeline uses **raw** traces +, for defense, the **WTF** zips below (a **different** closed world: 100 classes). See `memo.md` → **“DF95: parallel … re-check”**.

---

## WTF-PAD (closed-world traces, Juarez et al. release)

From [github.com/wtfpad/wtfpad releases](https://github.com/wtfpad/wtfpad/releases):

| Asset | Role |
|--------|------|
| `closed-world-original.zip` (~22 MB) | Undefended Tor traces |
| `closed-world-protected.zip` (~35 MB) | **Same** pages with **WTF-PAD** padding applied (paired by filename) |

**Local copies:** `downloads/closed-world-original.zip`, `downloads/closed-world-protected.zip`.

**Unpacked in this repo (mirrors upstream `data/` layout):**

- `data/wtfpad/closed-world-original/` — **3,933** trace files  
- `data/wtfpad/closed-world-protected/` — **3,933** trace files (names align across folders for **defense-generalization** eval)

**Re-unpack (from repo root):**

```bash
mkdir -p data/wtfpad
unzip -o downloads/closed-world-original.zip -d data/wtfpad
unzip -o downloads/closed-world-protected.zip -d data/wtfpad
```

Trace files use the same **`site-trace` style names** as in the WTF-PAD paper bundle (e.g. `12-26`). This corpus is **not** DF95 (different world / crawl); use it for **WTF-PAD–specific** results and cite the **paired** original vs. protected files.

**Pipeline in this repo (defense generalization, paper-style *same test IDs, protected inputs*):**

```bash
.venv/bin/python3 scripts/make_wtfpad_metadata.py
.venv/bin/python3 scripts/train_wtfpad.py --arch dfnet
.venv/bin/python3 scripts/eval_wtfpad_defense.py --checkpoint results/wtfpad/wtf100_dfnet_best.pt
```

- `make_wtfpad_metadata.py` → `processed/wtfpad_paired.csv` (3,933 paired rows, **100** classes).  
- `train_wtfpad.py` trains only on **original** paths; `eval_wtfpad_defense.py` runs the **same** split on **protected** paths without retraining.

**Bandwidth / duration / line-count overheads (paired traces):**

```bash
.venv/bin/python3 scripts/compute_wtfpad_overhead.py --out-per-trace results/wtfpad/overhead_per_trace.csv
```

Writes `results/wtfpad/overhead_summary.json` (aggregate) and optional per-trace CSV. On this release, **trace-duration** \((t_{\mathrm{last}}-t_{\mathrm{first}})\) is **0%** for all pairs (shared endpoints); **byte-sum** and **line-count** overheads are non-zero. See `memo.md` → **WTF-PAD official closed-world**.

---

## `AWF_Large_CW.tar.gz`

Large **closed-world** website fingerprinting trace bundle (per-domain folders, one trace per file; tab-separated `timestamp` and signed size, same family as the DF95 raw format).

- **Original location (this machine):** `downloads/AWF_Large_CW.tar.gz` (~**101.5 GiB** compressed; **`gzip -l`** reports ~**297 GiB** uncompressed; not committed to git). Copy or read from another volume (e.g. a USB stick) if needed; extraction needs **~300+ GiB** free on the **destination** drive.
- **Inside the archive:** `awf_large_cw/<domain.example.com>/<trace_id>` (numeric filename, no extension).

### Use with this repository

1. **Extract** to a volume with enough space and a filesystem that handles **many** small files (on Mac, **APFS** on a **~1 TB** external SSD has worked; **exFAT** can mis-report errors at large scale). You do not need a wrapper folder; you can extract directly to the drive root:

   ```bash
   tar -xzf "/path/to/AWF_Large_CW.tar.gz" -C "/Volumes/Extreme SSD"
   ```

   This yields `.../awf_large_cw/<domain>/...` (or use `-C` to any empty parent you prefer). In-repo `data/awf_large_cw/` is only for a **local** extract that fits; large extracts are **gitignored** there.

2. **Build metadata CSV** (absolute paths, like `df95_metadata.csv`):

   **Full inventory (example — this project’s completed run on external SSD):**

   - **Extract path:** `/Volumes/Extreme SSD/awf_large_cw/`
   - **Result:** **`1141`** site folders (labels **`site_id` `0..1140`** when all are used), **`4,469,403`** trace files listed.
   - **Command:**

   ```bash
   .venv/bin/python3 scripts/make_awf_large_metadata.py \
     --data-dir "/Volumes/Extreme SSD/awf_large_cw" \
     --out-path "/Users/kfunaki/Projects/wf-project/processed/awf_large_cw_metadata.csv"
   ```

   **Default (if you keep data under the repo):**

   ```bash
   .venv/bin/python3 scripts/make_awf_large_metadata.py
   ```

   For a **small dry run** (first *N* domains alphabetically, up to *K* traces each):

   ```bash
   .venv/bin/python3 scripts/make_awf_large_metadata.py --max-sites 20 --max-traces-per-site 100
   ```

3. **Output:** `processed/awf_large_cw_metadata.csv` (override with `--out-path`). Large; typically **gitignored**—regenerate on each machine. **Mount the SSD** before training if paths point there.

The AWF tree is **not** the same label space as **DF95** (95 `site_id`s). A separate dataloader and training run are required; do not mix label heads without a clear protocol. More context: **`memo.md`** (section *AWF Large closed-world*).

