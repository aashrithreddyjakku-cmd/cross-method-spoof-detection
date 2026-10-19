"""Generate a TINY fake dataset in the EXACT raw formats of ASVspoof2019-LA and In-the-Wild,
so that every script can be developed and tested before the real downloads finish.

  python scripts/make_dummy_data.py              # writes into data/raw/  (SPOOFDET_DATA to change)

Bona fide  = harmonic "voice" + noise.  Each fake attack Axx adds a faint narrow-band artefact at an
attack-specific frequency, so a CNN CAN learn something (and held-out attacks are genuinely different).
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from spoofdet.common import paths

SR = 16000
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(paths.RAW))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.RandomState(a.seed)
raw = Path(a.out)


def voice(dur):
    t = np.arange(int(SR * dur)) / SR
    f0 = rng.uniform(100, 220)
    y = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) for h in range(1, 12))
    y = y * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))        # syllable-like envelope
    y = y + 0.05 * rng.randn(len(t))
    return y / (np.abs(y).max() + 1e-6) * 0.5, t


def make(attack, dur):
    y, t = voice(dur)
    if attack is not None:
        k = int(attack[1:])
        fc = 1000 + 350 * k
        y = y + 0.08 * np.sin(2 * np.pi * fc * t) * (1 + 0.5 * rng.randn(len(t)) * 0.1)
        y = y / (np.abs(y).max() + 1e-6) * 0.5
    return y.astype(np.float32)


la = raw / "asvspoof2019" / "LA"
proto_dir = la / "ASVspoof2019_LA_cm_protocols"
proto_dir.mkdir(parents=True, exist_ok=True)


def write_part(part, tag, proto_name, speakers, attacks, n_bona, n_per_attack, nontarget=()):
    adir = la / f"ASVspoof2019_LA_{part}" / "flac"
    adir.mkdir(parents=True, exist_ok=True)
    lines, uid = [], rng.randint(1000000, 2000000)
    for spk in list(speakers) + list(nontarget):
        for _ in range(n_bona):
            uid += 1; name = f"LA_{tag}_{uid}"
            sf.write(adir / f"{name}.flac", make(None, rng.uniform(1.5, 3.5)), SR)
            lines.append(f"LA_{spk:04d} {name} - - bonafide")
        if spk in nontarget:
            continue
        for at in attacks:
            for _ in range(n_per_attack):
                uid += 1; name = f"LA_{tag}_{uid}"
                sf.write(adir / f"{name}.flac", make(at, rng.uniform(1.5, 3.5)), SR)
                lines.append(f"LA_{spk:04d} {name} - {at} spoof")
    (proto_dir / proto_name).write_text("\n".join(lines) + "\n")


A_known = [f"A{i:02d}" for i in range(1, 7)]
A_eval = [f"A{i:02d}" for i in range(7, 20)]
write_part("train", "T", "ASVspoof2019.LA.cm.train.trn.txt", range(1, 7), A_known, 6, 4)
write_part("dev", "D", "ASVspoof2019.LA.cm.dev.trl.txt", range(7, 11), A_known, 6, 4, nontarget=(11, 12))
write_part("eval", "E", "ASVspoof2019.LA.cm.eval.trl.txt", range(13, 21), A_eval, 6, 2, nontarget=(21, 22, 23, 24))

# ---- In-the-Wild look-alike
itw = raw / "in_the_wild" / "release_in_the_wild"
itw.mkdir(parents=True, exist_ok=True)
rows = ["file,speaker,label"]
for i in range(80):
    spoof = i % 2 == 0
    sf.write(itw / f"{i}.wav", make(f"A{rng.randint(1, 20):02d}" if spoof else None, rng.uniform(2, 6)), SR)
    rows.append(f"{i}.wav,Speaker_{i % 8},{'spoof' if spoof else 'bona-fide'}")
(itw / "meta.csv").write_text("\n".join(rows) + "\n")
print("dummy data written to", raw)
