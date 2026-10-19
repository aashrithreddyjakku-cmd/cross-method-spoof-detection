#!/usr/bin/env bash
# End-to-end test on DUMMY data (~2 min on CPU). Everything is written to a temporary folder, not your real data.
set -e
T=$(mktemp -d)
export SPOOFDET_DATA=$T/data SPOOFDET_SPLITS=$T/splits SPOOFDET_CKPT=$T/ckpt SPOOFDET_RESULTS=$T/results
python scripts/make_dummy_data.py
python scripts/build_manifest.py
python scripts/make_splits.py > /dev/null
python scripts/extract_features.py --feat mel64_w400
python -m spoofdet.train.run_experiment configs/experiments/_template.yaml exp_id=smoke model.name=lcnn train.epochs=2 train.batch_size=16 train.num_workers=0 protocol=P1_lofo/neural_vocoder
python -m spoofdet.eval.evaluate --all --boot 20
python -m spoofdet.eval.aggregate
echo "SMOKE TEST PASSED (temp dir: $T)"
