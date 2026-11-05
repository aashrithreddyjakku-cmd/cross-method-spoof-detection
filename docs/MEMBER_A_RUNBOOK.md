# Member A runbook: from downloads to the CP1 hand-off (Windows, Git Bash)

## 0. Environment (once)
```bash
py -3.12 -m venv ~/.venvs/spoofdet          # keep the venv OUT of OneDrive
source ~/.venvs/spoofdet/Scripts/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt && pip install -e .
export SPOOFDET_DATA=/c/spoofdet-data       # raw audio + features OUT of OneDrive (about 40 GB)
export PYTHONIOENCODING=utf-8
pytest -q tests                              # 21 passed (14 shared + 7 data tests)
bash scripts/smoke_test.sh                   # SMOKE TEST PASSED
bash scripts/run_data_pipeline.sh --dummy    # DATA PIPELINE DONE (whole package A on dummy data)
```

## 1. Download (A.3)
**What we actually use: the small subset** (about 450 MB, about 15 min, no full download needed):
```bash
python scripts/download_subset.py --size small --dry-run   # 4,180 LA + 600 ITW files
python scripts/download_subset.py --size small             # resumable; same seed -> same files
bash scripts/run_data_pipeline.sh --subset                 # whole package A, about 5 min
```
Sizes: `tiny` (about 1.6 k + 600 files), `small` (default), `medium` (about 10.5 k + 3 k files). Full data, only if a machine can take it:
```bash
mkdir -p /c/spoofdet-data/raw/asvspoof2019 && cd /c/spoofdet-data
curl -L -C - -o LA.zip "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"     # 7.1 GB, resumable
unzip -q LA.zip -d raw/asvspoof2019                                                        # -> raw/asvspoof2019/LA/...
# In-the-Wild: download from https://deepfake-demo.aisec.fraunhofer.de/in_the_wild and unzip under raw/in_the_wild/
#   or: pip install datasets && python scripts/export_itw_hf.py
```
Read and accept each licence on the download page yourself, then write the date and licence into docs/DATA_CARD.md.

## 2. Build everything (A.4 to A.11)
```bash
cd <repo>
WORKERS=6 bash scripts/run_data_pipeline.sh       # about 12 h on a laptop (mostly the 153 k-file mel64_w400 extraction)
```
Or step by step: `verify_raw.py` → `build_manifest.py` → `manifest_checks.py` → `make_splits.py` → `check_splits.py` →
`union_splits.py` → `extract_features.py --feat mel64_w400` → `check_features.py --feat mel64_w400` → `bench_loader.py` →
ablation stores → `data_report.py` → `shortcut_baseline.py` → `checksums.py`.
Tip: push the splits (step A.6) as soon as `check_splits.py` passes, before the features finish.

## 3. Hand-off (A.12)
```bash
git checkout -b a/data
git add splits/ docs/ report/sections/data.md results/tables/data_* results/figures/data_* results/tables/shortcut_auc.csv results/figures/shortcut_*
git commit -m "a: real manifest checks, splits v1, data card"
python scripts/package_kaggle.py --user <kaggle-user>    # -> /c/spoofdet-data-v1 (hard links, no extra disk)
kaggle datasets create -p /c/spoofdet-data-v1 --dir-mode tar
```
Then fill the TODO cells in docs/DATA_CARD.md and report/sections/data.md with the measured numbers.
Once B has pushed a score file, **never regenerate the splits**.
