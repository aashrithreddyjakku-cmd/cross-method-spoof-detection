# Cross-Method Generalizable Synthetic Speech Detection Using CNN-Based Mel-Spectrogram Analysis

CNNs on log-mel spectrograms, evaluated under leave-one-method-out / leave-one-family-out and cross-dataset
(ASVspoof 2019 -> In-the-Wild) protocols. Metrics: EER, AUC-ROC, accuracy, confusion matrices, per-method EER.
Explainability: Grad-CAM + mel-band occlusion. Demo: Gradio web app.

## 1. Install (Python 3.10-3.12)
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q tests                                            # all tests must pass
bash scripts/smoke_test.sh                                 # whole pipeline on fake data, writes to a temp folder
```
Keep data outside any cloud-synced folder: `export SPOOFDET_DATA=/c/spoofdet-data` (Windows PowerShell: `$env:SPOOFDET_DATA="C:\spoofdet-data"`).

## 2. Data (Member A)
```bash
python scripts/download_subset.py --size small             # 4,180 ASVspoof 2019 LA + 600 In-the-Wild clips, about 450 MB
bash scripts/run_data_pipeline.sh --subset                 # manifest, splits and checks, 7 feature stores, data report
```
For the full corpora, download them as in `docs/MEMBER_A_RUNBOOK.md` and run `bash scripts/run_data_pipeline.sh`.
The private Kaggle dataset `yrk2005/spoofdet-data-v1` holds the features and manifests. Raw audio is never shared (licence).

## 3. Experiments (Member B)
```bash
python scripts/sanity_checks.py                            # overfit, shuffled labels, determinism, learning rate, speed
python scripts/run_programme.py --dry-run                  # list the planned runs
python scripts/run_programme.py --parallel 3 --threads 4   # resumable: finished runs are skipped
```
Single run: `python -m spoofdet.train.run_experiment configs/experiments/<exp>.yaml`. GPU: `notebooks/kaggle_runner.md`.

## 4. Evaluation and explanations (Member C)
```bash
python scripts/run_programme.py --finish                   # score checks, metrics, tables, figures, XAI, experiment log
```
Tables are in `results/tables/`, figures in `results/figures/`, the run list in `docs/EXPERIMENT_LOG.md`.

## 5. Demo
```bash
python scripts/prepare_demo.py                             # picks clips from the test scores and checks live inference
python -m spoofdet.app.demo --ckpt checkpoints/<exp>/best.pt
```
On Windows, double-click `start_demo.bat`. See `docs/DEMO_RUNBOOK.md`.

## Conventions
Label 1 = spoof, score = P(spoof) (higher = more likely fake), 16 kHz mono, 4 s clips. Contracts: `docs/CONTRACT_CHANGES.md`.

Ownership: A = spoofdet/data, data scripts, splits/ ; B = spoofdet/models, spoofdet/train, configs/experiments ;
C = spoofdet/eval, spoofdet/app, spoofdet/infer.py, report/. Shared (change only by agreement): spoofdet/common,
configs/features, configs/attack_taxonomy.yaml.

## Datasets and licences
ASVspoof 2019 (https://datashare.ed.ac.uk/handle/10283/3336) and In-the-Wild (https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)
remain under their own licences and are not part of this repository.
