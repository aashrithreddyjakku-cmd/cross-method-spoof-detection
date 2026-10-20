# Data card  (Member A owns)

> Status: **v1, built on 19 Oct 2026 from a SMALL SUBSET of the real corpora** (the team laptops cannot hold or process the
> full 15 GB / 153 k files). All numbers below are measured on that subset. To scale up, re-run with `--size medium` or the full
> download; the pipeline is identical.

## Datasets used
| name | version / date downloaded | source URL | licence | size used | files used (of total) | speakers | notes |
|------|---------------------------|------------|---------|-----------|-----------------------|----------|-------|
| ASVspoof 2019 LA | LA.zip (7.64 GB), subset fetched 2026-10-19 | https://datashare.ed.ac.uk/handle/10283/3336 | as stated on the DataShare page (research use; audio not redistributed) | 253 MB | 4,180 (of 121,461) | 107 (all) | FLAC, 16 kHz; all 19 attacks present |
| In-the-Wild | release_in_the_wild.zip (8.16 GB, HF mirror `mueller91/In-The-Wild`), subset fetched 2026-10-19 | https://huggingface.co/datasets/mueller91/In-The-Wild (original: https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | CC BY-SA 4.0 (HF card) | 197 MB | 600 (of about 31.8 k) | 54 (of 58) | WAV 16 kHz; test only |
| ASVspoof 2021 LA (stretch) | not used | https://zenodo.org/records/4837263 | (not used) | (none) | (none) | (none) | (not used) |

### How the subset was made (`scripts/download_subset.py --size small --seed 1`)
The zips were **not** downloaded in full. The script read each archive's index over HTTP range requests and fetched only the
selected members. Within each official partition, bona fide utterances and the spoofs of **every** attack were sampled
round-robin across speakers (seeded), so every speaker and every attack is present:

| partition | bona fide | spoof | per attack | speakers | attacks |
|---|---|---|---|---|---|
| LA train | 600 | 720 | 120 | 20 | A01–A06 |
| LA dev | 300 | 360 | 60 | 20 | A01–A06 |
| LA eval | 900 | 1,300 | 100 | 67 | A07–A19 |
| In-the-Wild | 300 | 300 | (unknown) | 54 | (unknown) |

The subset protocol files hold only the selected lines. The original full protocols are kept in
`raw/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/full/` and `raw/in_the_wild/release_in_the_wild/meta_full.csv.txt`.
Measured: every file was readable (0 NaN durations), all files are 16 kHz, and no LA spoof has an `unknown` family
(`results/tables/data_manifest_checks.txt`).

Local layout: raw audio and features live **outside OneDrive** in `SPOOFDET_DATA=C:\spoofdet-data` (raw 450 MB, all 7 feature
stores 1.7 GB). Only the code, `splits/` and the small result tables sit in the repository.

## Preprocessing (what happened to the audio)
- Audio is read with `soundfile`; stereo is averaged to mono, and any rate other than 16 kHz is resampled to 16 kHz (librosa).
- Every clip is fixed to **exactly 4.0 s (64,000 samples)**: longer clips keep the first 4 s, shorter clips are **tiled** (repeated).
  Measured mean durations: LA 3.31 s (bona fide 3.43, spoof 3.23); ITW 5.51 s (bona fide 4.68, spoof 6.34).
- Log-mel: n_fft 512, window 400 (25 ms), hop 160 (10 ms), 64 mel bands, 20–8000 Hz, power 2, `power_to_db(ref=1, amin=1e-10,
  top_db=80)`, so each clip's dynamic range is clipped to 80 dB below its own maximum. 4 s → shape **(64, 401)**.
- Stored as **raw dB** in float16: a memory-mapped `feats.npy` (N, n_mels, 401), plus `index.csv` and `feat_config.yaml`.
  Normalisation (`none` / `utt` default / `utt_band` / `global`) and SpecAugment are applied at load time in `SpoofDataset`.
- Stores built and integrity-checked (`scripts/check_features.py`): `mel64_w400` = all 4,780 utterances (LA + ITW). The six
  ablation stores (mel40/80/128_w400, mel64_w200, mel64_w800, mel64_w400_trim) hold the 4,180 LA utterances of the P1 union.
  None had a `failed.csv`, a non-finite value or a blank clip.
- `mel64_w400_trim` removes leading/trailing silence (`librosa.effects.trim(top_db=30)`) before the 4 s crop, so many clips
  become shorter and are then tiled. This is a known side-effect of the variant.

## Protocols (P0, P1_all, P1_lomo, P1_lofo): how built, seeds, caps, speaker counts
Exact commands (seed 1 everywhere):
```bash
export SPOOFDET_DATA=/c/spoofdet-data
python scripts/download_subset.py --size small --seed 1
python scripts/build_manifest.py
python scripts/make_splits.py --seed 1 --frac 0.60 0.15 0.25 --cap 1500 300 600 --cap-bona 0 0 0
python scripts/check_splits.py          # ALL SPLIT CHECKS PASSED
python scripts/union_splits.py
```
(`bash scripts/run_data_pipeline.sh --subset` runs all of package A in one go.)

- **P0_official:** train = LA train (1,320), val = LA dev (660), test = LA eval (2,200; A07–A19 unseen); `test_itw.csv` = 600 ITW clips.
- **P1 speaker pools:** the "known" pool is the train+dev speakers (40, attacks A01–A06), and the "eval" pool is the eval speakers
  (67, attacks A07–A19). Each pool was shuffled with seed 1 and split 60/15/25 %. **Measured: 64 / 16 / 27 speakers.**
- **P1_all:** train 2,396 (1,090 bona fide / 1,306 spoof), val 699 (271 / 428), test 1,085 (439 / 646).
- **Caps** (1,500 / 300 / 600 spoofs per method per group) were not binding on this subset, because the download already balances
  methods. All bona fide utterances were kept.
- **P1_lomo/<M>** (17 folds) and **P1_lofo/<family>** (4 folds) drop the held-out method/family from train **and** val.
  `test_unseen` = the 439 test bona fide + the held-out spoofs; `test_seen` = the same bona fide + the other methods.
- `scripts/check_splits.py` enforces: speaker-disjoint train/val/test_seen/test_unseen; no utterance overlap; both classes in every
  file; held-out group absent from train/val/test_seen; test_unseen spoofs = exactly the held-out group; identical bona fide in
  seen/unseen; each fold = P1_all minus the held-out rows; P0 train/dev/eval speaker-disjoint.
- **Small-test warning (measured).** In `test_unseen`, the A01–A06 folds have 54–80 spoofs, the A07–A18 folds only **24–25**,
  and the LOFO folds 25 (griffin_lim), 152, 208 and 261. Per-method EERs on 25 spoofs move in steps of 4 %, so C must report
  bootstrap CIs. LOFO and P0 are the reliable comparisons on this subset. `--size medium` gives about 2.5× more.
- Attack families: `configs/attack_taxonomy.yaml`, verified against the database paper in `docs/ATTACKS.md`.

## Known issues / biases
- **Silence shortcut (measured).** Single-cue AUCs (`scripts/shortcut_baseline.py --n 300`, `results/tables/shortcut_auc.csv`):

  | corpus | duration | leading silence | trailing silence |
  |---|---|---|---|
  | LA (train) | 0.51 | **0.29** | 0.53 |
  | In-the-Wild | 0.58 | 0.50 | 0.59 |

  In LA, bona fide clips start with more silence (mean 0.98 s vs 0.60 s), so the leading-silence length alone separates the
  classes (AUC 0.29, i.e. 0.71 when inverted). This confirms Müller et al. (2021) on our data. A CNN can learn it, and it does
  not transfer to ITW (0.50). This is the evidence behind the `mel64_w400_trim` ablation and the SA3 confound study: **tell B
  and C**. ITW duration and trailing silence (0.58–0.59) are weaker cues.
- **Attack ↔ speaker-pool confound.** A01–A06 spoof only the 40 known-pool speakers, and A07–A19 spoof only the 67 eval-pool
  speakers. The P1 split separates speakers within each pool but cannot break the link between "which attack" and "which pool".
- **Subset size.** Results come from 4,180 LA utterances (3.4 % of LA). Absolute EERs are noisier than published full-data numbers
  and must be read with their confidence intervals.
- **Class balance.** The subset is about 43 % bona fide (the full LA train is about 10 %), so class weights matter less here.
- **Tiling.** Clips shorter than 4 s are repeated, which creates periodic joins that do not exist in the real audio.
- **In-the-Wild** labels come from the dataset authors. It contains noise, music and compression, and the subset covers 54 of 58
  speakers. It is test-only and is never used for training, validation or thresholds.
- English only. The results do not cover other languages.

## Checksums of released artefacts
`docs/checksums.txt` (sha256, from `python scripts/checksums.py`). Key values:
- `manifests/master.csv`: `c81a9d9900c08135b0cbf226c1b6b333466c6b9ef2c928d917c710d56c916505`
- `splits/split_summary.csv`: `1bafe4df4035e019433f5ae969232a30b0ed05663651bd89bc33a999465aa2fc`
- `features/mel64_w400/feats.npy`: `1ce6cd19e08353bf8f2d639e69b513b58568b76331cba7054e05ba07edf95af8`

## Sharing
Private Kaggle dataset `<user>/spoofdet-data-v1` (`python scripts/package_kaggle.py --user <user>` → upload; about 1.7 GB). It
contains `manifests/` and `features/` only. **No raw audio** (licence). In notebooks set
`SPOOFDET_DATA=/kaggle/input/spoofdet-data-v1`. Teammates can instead rebuild the identical data locally in about 15 minutes
with `download_subset.py` + `run_data_pipeline.sh --subset` (same seed → same files and the same checksums for master.csv/splits).
Splits are versioned in git (commit: fill in after the PR).
