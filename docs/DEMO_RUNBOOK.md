# Demo runbook

Everything runs from Yash's laptop. The checkpoints and clips are only there, so bring that laptop and its charger.
No GPU is needed, and each clip takes a few seconds.

## What is where
| What | Where |
|---|---|
| Project code (branch `b/train`: the work of A, B and C together) | `Study\ML\Project\cross-method-spoof-detection` |
| One-click launcher | `start_demo.bat` in that folder |
| Model used in the demo (VGGish-CBAM, official protocol, seed 1) | `checkpoints\p0_vggish_cbam_mel64_w400_s1\best.pt` |
| Demo clips | `C:\spoofdet-data\demo_clips\` |
| Backup screenshots of every clip | `C:\spoofdet-data\demo_clips\backup_screens\` |
| Result figures | `results\figures\` |
| Result tables | `results\tables\` |
| Experiment log (all 116 runs) | `docs\EXPERIMENT_LOG.md` |
| Older written report (October numbers, **do not quote it**) | `report\Results_Discussion_Inference.docx` |

## Starting the demo
1. In File Explorer, open the project folder and double-click `start_demo.bat`.
2. Wait until the black window shows `Running on local URL: http://127.0.0.1:7860` (about 20 seconds).
3. Open `http://127.0.0.1:7860` in a browser.
4. Leave the **Decision threshold slider** alone: it starts at the threshold chosen on the validation set.

Closing the black window stops the app. If the browser shows nothing, the black window is closed or still loading.

## Running a clip
1. Click the upload icon in the audio box, or drag a file in, from `C:\spoofdet-data\demo_clips\`.
2. Click **Analyze**.
3. Scroll down: the verdict is first, the three plots are at the bottom.
4. To load the next clip, click the **X** at the top right of the audio box.

### The five clips, in demo order
| File | What it is | Expected result | Point to make |
|---|---|---|---|
| `1_real_correct.flac` | Real speech | BONA FIDE, P(spoof) 0.020 | Correct on a real voice |
| `2_tts_griffinlim_A11_correct.flac` | Text-to-speech, Griffin-Lim | SPOOF, 0.980 | Generator never seen in training |
| `3_tts_neural_A10_correct.flac` | Text-to-speech, WaveRNN | SPOOF, 0.979 | Neural vocoder never seen in training |
| `4_vc_A17_MISSED.flac` | Voice conversion of real speech | BONA FIDE, 0.020 (**wrong, on purpose**) | Our main failure mode |
| `5_inthewild_real_FALSE_ALARM.wav` | Real clip from In-the-Wild | SPOOF, 0.980 (**wrong, on purpose**) | Real-world audio causes false alarms |

The threshold slider starts at 0.22, the value chosen on validation data.

Clips 1-3 show it working and clips 4-5 show the research finding. Say that 4 and 5 were chosen because they fail.
**These five were hand-picked from the test results, so they do not show the overall accuracy.** Say that too.

**Do not press play on clip 5.** In-the-Wild contains real public figures, and our project rules say not to play their
voices in public. Analyse it without playing it.

### What the screen shows
- **Class probabilities / Result:** the verdict and P(spoof). Higher means more likely fake.
- **Log-mel spectrogram (left):** time across, frequency up, brightness = energy. This image is the model's input.
  If a clip looks repeated, it was shorter than 4 s and was looped to fill 4 s.
- **Grad-CAM (middle):** the regions that pushed the score towards "spoof".
- **Mel-band occlusion (right):** how much P(spoof) drops when one frequency band is hidden. A longer bar means that band
  mattered more. A negative bar means hiding the band made the clip look more fake.

### Avoid the live microphone unless you present it as a risk
The app can record, but we have not tested that. A laptop-mic recording is noisy, and on noisy real-world audio this model
flags about 90% of genuine clips as fake. Your own voice may well come out as SPOOF. If someone asks to try it, say that first.

## If something goes wrong
| Problem | Fix |
|---|---|
| Black window closes at once | Open PowerShell in the project folder and run `.\start_demo.bat` to read the error |
| `Address already in use` / port 7860 busy | An old copy is running: close its black window, or restart the laptop |
| "Model unavailable" on the page | You moved or deleted `checkpoints\` |
| Analyze spins for over 30 seconds | Close other programs; refresh the page and retry |
| App won't work at all | Show the images in `backup_screens\`: they are real outputs from this model |

**The night before:** run all five clips once, switch off sleep mode and plug in the charger.

## Figures to show (all in `results\figures\`)
| File | Shows |
|---|---|
| `per_method_p0_vggish_cbam_mel64_w400_s1_test.png` | Error per attack, seed 1: most are low, A17 is about 30% (28% across three seeds) |
| `gap_p1_lofo.png` | Generalisation gap per held-out family |
| `lomo_heatmap.png` | Unseen error for each held-out attack |
| `roc_det_p0_itw.png` | ROC curves on In-the-Wild, one seed per model |
| `ablation.png` | Effect of silence removal and other settings |
| `xai_lofo_waveform_modification_vggish_cbam_mel64_w400_s1_A17_gradcam.png` | Grad-CAM on a held-out attack (only if asked) |

## Re-running the pipeline (only if asked; not needed for the demo)
In Git Bash, from the project folder:
```bash
source ~/.venvs/spoofdet/Scripts/activate
export SPOOFDET_DATA=/c/spoofdet-data PYTHONIOENCODING=utf-8
python -m pytest -q tests                 # 89 passed, about 2 min
bash scripts/smoke_test.sh                # SMOKE TEST PASSED, full pipeline on fake data, about 2 min
python scripts/check_splits.py            # ALL SPLIT CHECKS PASSED: no speaker or method leakage
python -m spoofdet.eval.evaluate --all    # metrics from the score files
python -m spoofdet.eval.aggregate         # summary tables
python scripts/make_figures.py            # figures
```
The pipeline order is: raw audio → `build_manifest.py` → `make_splits.py` → `extract_features.py` →
`spoofdet.train.run_experiment` → `evaluate` → `aggregate` → `make_figures.py` / `run_xai.py` → demo.

## What was and was not done (know this before questions)
**Done (116 runs, 104 of them CNNs, trained on a Kaggle T4 GPU in 87 minutes):** data pipeline on a seeded subset (4,180
ASVspoof + 600 In-the-Wild clips); SmallCNN, LCNN, ResNet-18 and VGGish-CBAM on the official protocol with 3 seeds each; MFCC +
logistic regression and MFCC + SVM baselines; leave-one-family-out for all 4 families (VGGish-CBAM 3 seeds, SmallCNN and SVM 1);
leave-one-method-out for all 17 attack groups with VGGish-CBAM (8 of them with 3 seeds, and with SmallCNN too); 33 ablation runs
(1 seed, 3 families); training on all attacks as a reference; silence-trim comparison; model profiling; Grad-CAM and band
occlusion on 21 models; working demo.

**Not done:** the full dataset; ASVspoof 2021; wav2vec2; LCNN and ResNet-18 beyond the official protocol and In-the-Wild. The
written report still has the October numbers.
