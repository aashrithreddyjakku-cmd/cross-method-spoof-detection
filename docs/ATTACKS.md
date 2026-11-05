# Attack taxonomy: verification record (task A.5, Member A)

**Source checked:** Wang et al. (2020), *ASVspoof 2019: A large-scale public database of synthesized, converted and
replayed speech*, arXiv:1911.01601, Table 1 "Summary of LA spoofing systems" and §3.1. The check was made on 19 Oct 2026.

**Result:** `configs/attack_taxonomy.yaml` was **kept unchanged** (v1.0). Every family assignment agrees with the
*waveform generator* column of Table 1. The borderline cases below are documented so that they can be defended in the report.

## Grouping rule
An attack's family is the stage that **finally produces the waveform** (the "Waveform generator" column of Table 1), not
the acoustic model or the input type (text or speech). We chose this rule because a CNN on log-mel spectrograms mostly sees
artefacts of the generator: spectral smoothing, missing phase detail, concatenation joins.

## Table 1 vs our families

| ID | Type | Input | Conversion / acoustic model | Waveform generator (Table 1) | method_id | Our family | Agrees |
|----|------|-------|-----------------------------|------------------------------|-----------|------------|--------|
| A01 | TTS | text | AR RNN + VAE | WaveNet | A01 | neural_vocoder | yes |
| A02 | TTS | text | AR RNN + VAE | WORLD | A02 | source_filter_vocoder | yes |
| A03 | TTS | text | feed-forward NN (Merlin) | WORLD | A03 | source_filter_vocoder | yes |
| A04 | TTS | text | CART (unit selection, MaryTTS) | waveform concatenation | A04 | waveform_modification | yes |
| A05 | VC | human speech | VAE | WORLD | A05 | source_filter_vocoder | yes |
| A06 | VC | human speech | GMM-UBM | spectral filtering + OLA | A06 | waveform_modification | yes |
| A07 | TTS | text | RNN | WORLD + GAN post-filter | A07 | source_filter_vocoder | yes (borderline, see 1) |
| A08 | TTS | text | AR RNN | neural source-filter | A08 | neural_vocoder | yes |
| A09 | TTS | text | RNN | Vocaine | A09 | source_filter_vocoder | yes |
| A10 | TTS | text | Tacotron 2 style + d-vector | WaveRNN | A10 | neural_vocoder | yes |
| A11 | TTS | text | as A10 | Griffin-Lim | A11 | griffin_lim | yes |
| A12 | TTS | text | RNN | WaveNet | A12 | neural_vocoder | yes |
| A13 | TTS→VC | TTS speech | moment matching NN | waveform filtering | A13 | waveform_modification | yes (borderline, see 2) |
| A14 | TTS→VC | TTS speech | ASR bottleneck + RNN | STRAIGHT | A14 | source_filter_vocoder | yes |
| A15 | TTS→VC | TTS speech | ASR bottleneck + RNN | WaveNet | A15 | neural_vocoder | yes |
| A16 | TTS | text | same algorithm as A04 | waveform concatenation | **A04** | waveform_modification | yes |
| A17 | VC | human speech | VAE (as A05) | waveform filtering (direct waveform modification) | A17 | waveform_modification | yes |
| A18 | VC | human speech | i-vector/PLDA linear | MFCC vocoder (residual + PSOLA) | A18 | source_filter_vocoder | yes (borderline, see 3) |
| A19 | VC | human speech | same algorithm as A06 | spectral filtering + OLA | **A06** | waveform_modification | yes |

The paper states that "A04 and A16 use same waveform concatenation TTS algorithm, and A06 and A19 use same VC algorithm".
This is why `canonical: {A16: A04, A19: A06}` exists. Without it, a LOMO fold that holds out A04 would still train on A16 and
leak the method. There are 17 distinct methods: neural_vocoder 5, source_filter_vocoder 7, waveform_modification 4, griffin_lim 1.

## Borderline cases (and why we kept them)
1. **A07 (WORLD + WaveCycleGAN2 post-filter).** A neural network post-processes the waveform, so A07 could be called
   `neural_vocoder`. We kept it in `source_filter_vocoder` because the waveform itself is produced by WORLD. The GAN only
   filters it. If A07's unseen EER in LOFO looks very different from the rest of its family, this choice is the first thing to check.
2. **A13 (TTS source + moment-matching VC + waveform filtering).** This is a hybrid, but its final stage filters an existing
   (TTS-generated) waveform, the same principle as A17. So it is `waveform_modification`.
3. **A18 (MFCC vocoder).** The paper says its vocoder is "similar to the transfer-function-based approach in A06": it modifies a
   residual excitation (PSOLA) and passes it through an all-pole synthesis filter. That is a source-filter synthesis from
   converted MFCCs, so we kept `source_filter_vocoder`. A grouping with A06 (`waveform_modification`) would also be defensible.
   This is stated as a threat to validity.

The paper's own x-vector clustering (Fig. 4) groups attacks by *speaker/source* similarity (for example, the VC systems that
use human speech cluster together). It does not group them by generator. So the families are our design choice and are not
a property of the data.

## What artefact each family tends to leave (for the XAI discussion, Member C)
- **neural_vocoder** (WaveNet, WaveRNN, NSF): very natural spectra but imperfect fine structure. Expect noise-like high-frequency
  energy, slightly smeared harmonics above about 4 kHz, and weak or unnatural energy in unvoiced and silent frames.
- **source_filter_vocoder** (WORLD, STRAIGHT, Vocaine, MFCC vocoder): spectral envelopes are over-smoothed (statistical
  averaging), formants are less sharp and harmonics are "buzzy" with a synthetic periodic excitation. Mid and high bands
  between harmonics look too regular.
- **waveform_modification** (unit concatenation, spectral or waveform filtering of real speech): mostly real speech, so the
  spectra are close to bona fide (A04/A16 overlap bona fide in the paper's Fig. 3). Cues are local: discontinuities at unit joins
  and filtering or OLA artefacts at specific frequencies. This family is usually the hardest to detect.
- **griffin_lim** (A11): phase is reconstructed iteratively, which gives phasey, "metallic" smearing. On a log-mel spectrogram
  this shows as blurred harmonics and noise filling the gaps between them, mainly in the mid and high bands.

## Change log
| date | change | reason | re-ran |
|------|--------|--------|--------|
| 2026-10-19 | none (v1.0 confirmed against Table 1) | (none) | (nothing needed) |
