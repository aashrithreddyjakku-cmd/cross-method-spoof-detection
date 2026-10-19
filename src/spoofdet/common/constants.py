"""Project-wide constants. FROZEN at Day 0 - do not change without all three members agreeing."""

SR = 16000                      # sample rate (Hz) used everywhere
CLIP_SECONDS = 4.0              # every utterance is cropped / tiled to exactly 4 s
CLIP_SAMPLES = int(SR * CLIP_SECONDS)   # 64000

BONAFIDE = 0                    # label 0 = real human speech
SPOOF = 1                       # label 1 = synthetic / converted speech (the "positive" class)

# A score is always P(spoof): HIGHER score = MORE LIKELY SPOOF.

MANIFEST_COLUMNS = [
    "utt_id", "dataset", "partition", "path", "label", "attack_id",
    "method_id", "method_family", "speaker_id", "duration_s", "orig_sr",
]
# Passed through unchanged into every score file (plus "score").
SCORE_COLUMNS = ["utt_id", "score", "label", "dataset", "attack_id",
                 "method_id", "method_family", "speaker_id"]

DATASETS = ("la19", "la21", "df21", "itw")
