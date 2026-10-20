#!/usr/bin/env bash
# Member A: the whole data package in order (Git Bash / Linux / macOS). Each step is safe to re-run.
#   bash scripts/run_data_pipeline.sh            # real data (expects data/raw/... downloaded, see docs/DATA_CARD.md)
#   bash scripts/run_data_pipeline.sh --dummy    # dummy data in a temp dir (Day 0-1 check)
#   bash scripts/run_data_pipeline.sh --subset   # real data, small subset (python scripts/download_subset.py first)
#   WORKERS=6 SKIP_ABLATION=1 bash scripts/run_data_pipeline.sh
set -e
WORKERS=${WORKERS:-$(( $(python -c "import os; print(os.cpu_count() or 2)") - 1 ))}
FLAG=""
if [ "$1" == "--dummy" ]; then
  T=$(mktemp -d); export SPOOFDET_DATA=$T/data SPOOFDET_SPLITS=$T/splits SPOOFDET_RESULTS=$T/results
  FLAG="--dummy"; python scripts/make_dummy_data.py
elif [ "$1" == "--subset" ]; then
  FLAG="--subset"                       # data from scripts/download_subset.py (official counts do not apply)
fi
echo "== A.3 verify raw";        python scripts/verify_raw.py $FLAG
echo "== A.4 manifest";          python scripts/build_manifest.py
python scripts/manifest_checks.py $FLAG
echo "== A.6 splits";            python scripts/make_splits.py
python scripts/check_splits.py
python scripts/union_splits.py
echo "== A.7 default store";     python scripts/extract_features.py --feat mel64_w400 --workers "$WORKERS"
python scripts/check_features.py --feat mel64_w400
echo "== A.8 loader";            python scripts/bench_loader.py --split "${SPOOFDET_SPLITS:-splits}/P1_lomo/A10/train.csv" --plot
if [ -z "$SKIP_ABLATION" ]; then
  echo "== A.9 ablation stores"
  for f in mel40_w400 mel80_w400 mel128_w400 mel64_w200 mel64_w800 mel64_w400_trim; do
    python scripts/extract_features.py --feat $f --manifest "${SPOOFDET_SPLITS:-splits}/_union_P1.csv" --workers "$WORKERS"
    python scripts/check_features.py --feat $f --manifest "${SPOOFDET_SPLITS:-splits}/_union_P1.csv"
  done
fi
echo "== A.10 data report";      python scripts/data_report.py
python scripts/shortcut_baseline.py --n "${SHORTCUT_N:-2000}"
echo "== A.11 checksums";        python scripts/checksums.py
echo "DATA PIPELINE DONE ${T:+(temp dir: $T)}"
