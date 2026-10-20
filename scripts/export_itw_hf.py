"""Task A.3.2 fallback: export the Hugging Face mirror of In-the-Wild (mueller91/In-The-Wild) to loose
16 kHz wav files + meta.csv (columns file,speaker,label) in the layout parse_itw expects.
  pip install datasets
  python scripts/export_itw_hf.py [--out data/raw/in_the_wild/release_in_the_wild] [--limit 100]
Check the first rows of meta.csv by hand afterwards and compare label counts with the paper.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from spoofdet.common import paths


def pick(cols, *cands):
    for c in cols:
        if c.lower() in cands:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="mueller91/In-The-Wild")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=str(paths.RAW / "in_the_wild" / "release_in_the_wild"))
    ap.add_argument("--limit", type=int, default=0, help="export only the first N rows (test run)")
    a = ap.parse_args()
    from datasets import load_dataset, Audio

    ds = load_dataset(a.repo, split=a.split)
    cols = ds.column_names
    acol = pick(cols, "audio")
    lcol = pick(cols, "label", "class", "key")
    scol = next((c for c in cols if "speaker" in c.lower()), None)
    fcol = pick(cols, "file", "filename", "path")
    if acol is None or lcol is None:
        raise SystemExit(f"cannot find audio/label columns in {cols}")
    ds = ds.cast_column(acol, Audio(sampling_rate=16000))
    names = ds.features[lcol].names if hasattr(ds.features[lcol], "names") else None
    print("columns:", cols, " label names:", names)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = ["file,speaker,label"]
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    for i in tqdm(range(n), desc="export"):
        r = ds[i]
        lab = names[r[lcol]] if names else str(r[lcol])
        lab = "bona-fide" if str(lab).lower().replace("_", "-") in ("bona-fide", "bonafide", "real", "0") else "spoof"
        fn = Path(str(r[fcol])).name if fcol and r[fcol] else f"{i}.wav"
        if not fn.endswith(".wav"):
            fn = f"{Path(fn).stem}.wav"
        sf.write(out / fn, np.asarray(r[acol]["array"], dtype=np.float32), 16000)
        spk = str(r[scol]).replace(",", " ") if scol else "unk"
        rows.append(f"{fn},{spk},{lab}")
    (out / "meta.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {n} files + meta.csv to {out}")


if __name__ == "__main__":
    main()
