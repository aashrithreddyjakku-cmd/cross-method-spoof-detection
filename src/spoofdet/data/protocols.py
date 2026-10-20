"""Parsers that turn each dataset's own protocol/metadata files into manifest rows."""
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from ..common.constants import MANIFEST_COLUMNS, BONAFIDE, SPOOF
from ..common.io import load_yaml
from ..common import paths


# ---------------------------------------------------------------- taxonomy
def load_taxonomy():
    tax = load_yaml(paths.CONFIGS / "attack_taxonomy.yaml")
    fam_of = {}
    for fam, members in tax["families"].items():
        for m in members:
            fam_of[m] = fam
    return tax.get("canonical", {}), fam_of


def add_method_columns(df: pd.DataFrame) -> pd.DataFrame:
    canon, fam_of = load_taxonomy()
    df["method_id"] = df["attack_id"].map(lambda a: "bonafide" if a == "bonafide" else canon.get(a, a))
    df["method_family"] = df["method_id"].map(
        lambda m: "bonafide" if m == "bonafide" else fam_of.get(m, "unknown"))
    return df


# ---------------------------------------------------------------- ASVspoof 2019 LA
LA19_PARTS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev":   ("ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval":  ("ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
}


def parse_la19(raw_root: Path, la_dir: str = "asvspoof2019/LA") -> pd.DataFrame:
    """Protocol line:  LA_0079 LA_T_1138215 - A01 spoof   (speaker, utt, -, attack|'-', key)"""
    rows = []
    base = Path(raw_root) / la_dir
    for part, (audio_dir, proto) in LA19_PARTS.items():
        pfile = base / "ASVspoof2019_LA_cm_protocols" / proto
        for line in open(pfile):
            f = line.split()
            if len(f) < 5:
                continue
            spk, utt, _, attack, key = f[:5]
            is_spoof = key == "spoof"
            rows.append(dict(
                utt_id=f"la19__{utt}", dataset="la19", partition=part,
                path=f"{la_dir}/{audio_dir}/flac/{utt}.flac",
                label=SPOOF if is_spoof else BONAFIDE,
                attack_id=attack if is_spoof else "bonafide",
                speaker_id=spk))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ASVspoof 2021 LA / DF
def parse_2021(trial_metadata: Path, dataset: str, audio_subdir: str) -> pd.DataFrame:
    """trial_metadata.txt columns (space separated):
       speaker, trial_id, codec, tx/source, attack, key, trim, subset, ...
       dataset: 'la21' or 'df21'; audio_subdir e.g. 'asvspoof2021/ASVspoof2021_LA_eval/flac'"""
    rows = []
    for line in open(trial_metadata):
        f = line.split()
        if len(f) < 8:
            continue
        spk, utt, codec, tx, attack, key = f[:6]
        is_spoof = key == "spoof"
        rows.append(dict(
            utt_id=f"{dataset}__{utt}", dataset=dataset, partition="eval",
            path=f"{audio_subdir}/{utt}.flac",
            label=SPOOF if is_spoof else BONAFIDE,
            attack_id=attack if is_spoof else "bonafide", speaker_id=spk))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- In-the-Wild
def parse_itw(raw_root: Path, itw_dir: str = "in_the_wild") -> pd.DataFrame:
    """meta.csv has (at least) columns file, speaker, label  (label: 'spoof' | 'bona-fide').
    We search for meta.csv below raw_root/itw_dir and are tolerant about column names."""
    root = Path(raw_root) / itw_dir
    metas = list(root.rglob("meta.csv"))
    if not metas:
        raise FileNotFoundError(f"No meta.csv under {root}. Check where you unzipped In-the-Wild.")
    meta_path = metas[0]
    meta = pd.read_csv(meta_path)
    cols = {c.lower().strip(): c for c in meta.columns}
    fcol = next(cols[c] for c in cols if c in ("file", "path", "filename"))
    scol = next((cols[c] for c in cols if "speaker" in c), None)
    lcol = next(cols[c] for c in cols if c in ("label", "class", "key"))
    base = meta_path.parent
    rows = []
    for _, r in meta.iterrows():
        lab = str(r[lcol]).lower()
        is_bona = lab in ("bona-fide", "bonafide", "bona_fide", "real", "genuine", "0")
        fname = str(r[fcol])
        rows.append(dict(
            utt_id=f"itw__{Path(fname).stem}", dataset="itw", partition="eval",
            path=(base / fname).relative_to(Path(raw_root)).as_posix(),   # '/' even on Windows (Kaggle is Linux)
            label=BONAFIDE if is_bona else SPOOF,
            attack_id="bonafide" if is_bona else "itw_spoof",
            speaker_id=str(r[scol]) if scol else "unk"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- shared
def add_audio_info(df: pd.DataFrame, raw_root: Path) -> pd.DataFrame:
    durs, srs = [], []
    for p in tqdm(df["path"], desc="reading audio headers"):
        try:
            info = sf.info(str(Path(raw_root) / p))
            durs.append(info.frames / info.samplerate); srs.append(info.samplerate)
        except Exception:
            durs.append(float("nan")); srs.append(-1)
    df["duration_s"], df["orig_sr"] = durs, srs
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_method_columns(df)
    return df[MANIFEST_COLUMNS]
