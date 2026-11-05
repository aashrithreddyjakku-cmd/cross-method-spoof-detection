# Data and protocols  (Member A; methods in past tense, limitations in present tense)

> All numbers were measured on the seeded subset built on 19 Oct 2026 (`scripts/download_subset.py --size small`).
> Sources: `results/tables/data_*.csv`, `splits/split_summary.csv`, `results/tables/shortcut_auc.csv`.

## Datasets
Two public corpora were used (Table D1). ASVspoof 2019 Logical Access (LA) (Wang et al., 2020) served as the training and in-domain
test corpus. It contains studio-quality read English speech derived from VCTK, bona fide and spoofed by 19 text-to-speech (TTS)
and voice-conversion (VC) systems (A01–A19), in speaker-disjoint train, development and evaluation partitions. In-the-Wild (Müller
et al., 2022) served only as a cross-dataset test set. It contains real-world deepfakes and bona fide recordings of 58 public
figures collected from the internet.

Because of limited compute, a balanced subset of both corpora was used. Inside each official partition, bona fide
utterances and the spoofs of every attack were sampled evenly across speakers with a fixed seed, so all 107 LA speakers and
all 19 attacks were retained. The subset was fetched directly from the published archives with HTTP range requests.

**Table D1.** Subset used in this study, as measured in the manifest (`data/manifests/master.csv`).

| corpus | partition | bona fide | spoof | speakers | attacks |
|---|---|---|---|---|---|
| ASVspoof 2019 LA | train | 600 | 720 | 20 | A01–A06 (120 each) |
| ASVspoof 2019 LA | dev | 300 | 360 | 20 | A01–A06 (60 each) |
| ASVspoof 2019 LA | eval | 900 | 1,300 | 67 | A07–A19 (100 each) |
| In-the-Wild | (test only) | 300 | 300 | 54 | unknown |

Attacks A16 and A19 use the same algorithms as A04 and A06 (Wang et al., 2020). They were therefore mapped to a canonical
method identifier, which gave 17 distinct methods. Each method was assigned to one of four families according to the stage that
finally generates the waveform: neural vocoder (A01, A08, A10, A12, A15), source-filter vocoder (A02, A03, A05, A07, A09, A14, A18),
waveform modification (A04, A06, A13, A17) and Griffin-Lim (A11). The assignment was checked row by row against Table 1 of the
database paper (docs/ATTACKS.md).

## Preprocessing
All audio was converted to mono at 16 kHz and fixed to 4.0 s. Longer clips were cropped to their first 4 s, and shorter clips
were tiled. Log-mel spectrograms were computed with a 512-point FFT, a 25 ms window, a 10 ms hop and 64 mel bands between 20
and 8000 Hz. They were converted to decibels and clipped 80 dB below each clip's maximum, which gave 64 × 401 arrays. The arrays
were stored once in float16 memory-mapped files. Normalisation (per-utterance standardisation by default, with per-band and global
variants) and SpecAugment were applied at load time, so one feature store served all experiments. Six further stores, each
changing one factor, were extracted for the ablation study: 40, 80 or 128 mel bands; 12.5 ms or 50 ms windows; and silence
trimming before cropping.

## Evaluation protocols
Protocol P0 reproduced the official partitions: training on LA train, early stopping on LA dev, and testing on LA eval and on all
of In-the-Wild. Protocol P1 was designed for the controlled cross-method study (Figure D1, from Part I §5.4). The 107 LA speakers
were first divided into the pool used by attacks A01–A06 (40 speakers) and the pool used by A07–A19 (67 speakers). Within each
pool, speakers were shuffled with a fixed seed and assigned 60/15/25 % to training, validation and test, giving 64/16/27
speakers. Speakers were thus divided into training, validation and test groups within each speaker pool, so no speaker was shared
between groups. To balance methods and bound computation, at most 1,500, 300 and 600 spoofed utterances per method were kept in
the three groups; all bona fide utterances were retained (the caps were not binding on the subset). Leave-one-method-out (LOMO) folds removed one method, and
leave-one-family-out (LOFO) folds removed one family, from both training and validation. Each fold was tested on two files
that share the same test speakers and bona fide utterances: test_seen (the remaining methods) and test_unseen (only the held-out
method or family). All rules were verified automatically by `scripts/check_splits.py`, and split sizes are listed in
`splits/split_summary.csv`. Each test_unseen file contained the same 439 bona fide utterances and between 24
(single unseen methods A07–A18) and 261 (source-filter family) held-out spoofs.

## Data quality and bias
Clip durations and leading and trailing silence were analysed as potential shortcuts (Müller et al., 2021). Table D2 reports the
area under the ROC curve obtained when each cue alone was used to separate bona fide from spoofed speech.

**Table D2.** Single-cue AUC (up to 300 utterances per class; LA train partition and In-the-Wild).

| corpus | duration | leading silence | trailing silence |
|---|---|---|---|
| ASVspoof 2019 LA | 0.51 | 0.29 | 0.53 |
| In-the-Wild | 0.58 | 0.50 | 0.59 |

Leading silence alone separated the LA classes (AUC 0.29; bona fide clips began with 0.98 s of silence on average, spoofed
clips with 0.60 s), which confirmed the shortcut reported by Müller et al. (2021). The same cue carried no information on
In-the-Wild (AUC 0.50), so a detector that relies on it is expected to fail across datasets. This motivated the
silence-trimmed feature variant (mel64_w400_trim) in the ablation study. ⟨Add the trim-ablation result from Member B/C.⟩

The link between attack identity and speaker pool remains a limitation: methods A01–A06 occur only with one set of speakers and
A07–A19 only with another, so part of a held-out method's difficulty may reflect its speakers rather than its synthesis algorithm.
The subset contains 3.4 % of LA, so unseen-method test sets are small (about 25 spoofs for most single methods) and per-method
EERs carry wide confidence intervals. Tiling of short clips introduces periodic joins that are absent from the original audio. The family taxonomy is our own design
choice; three systems (A07, A13, A18) are borderline. Both corpora are English only, and In-the-Wild labels are inherited from the
dataset authors and may contain noise.

## Ethics and licences
The corpora were used under their research licences. Raw audio was not redistributed: only derived log-mel features and manifests
were shared, within the team, through a private dataset. In-the-Wild contains the voices of real public figures, so clips are not
played publicly, and the demo uses only the user's own uploads.
