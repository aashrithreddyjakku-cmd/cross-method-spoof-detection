# Cross-Method Generalizable Synthetic Speech Detection Using CNN-Based Mel-Spectrogram Analysis

CNNs on log-mel spectrograms, evaluated under leave-one-method-out / leave-one-family-out and cross-dataset
(ASVspoof 2019/2021 -> In-the-Wild) protocols. Metrics: EER, AUC-ROC, accuracy, confusion matrices, per-method EER.
Explainability: Grad-CAM + mel-band occlusion. Demo: Gradio web app.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q tests                                            # unit tests
bash scripts/smoke_test.sh                                 # full pipeline on dummy data (~2 min, CPU)
```
## Pipeline
```bash
python scripts/build_manifest.py                           # A  raw data -> data/manifests/master.csv
python scripts/make_splits.py && python scripts/check_splits.py   # A  -> splits/
python scripts/extract_features.py --feat mel64_w400       # A  -> data/features/mel64_w400
python -m spoofdet.train.run_experiment configs/experiments/<exp>.yaml   # B  train + score
python -m spoofdet.eval.evaluate --all && python -m spoofdet.eval.aggregate  # C  tables
python -m spoofdet.app.demo --ckpt checkpoints/<exp>/best.pt             # C  demo
```
Ownership: A = spoofdet/data, scripts for data, splits/ ; B = spoofdet/models, spoofdet/train, configs/experiments ;
C = spoofdet/eval, spoofdet/app, spoofdet/infer.py, report/. Shared (change only by agreement): spoofdet/common, configs/features, configs/attack_taxonomy.yaml.
Conventions: label 1 = spoof, score = P(spoof), 16 kHz, 4 s clips.
