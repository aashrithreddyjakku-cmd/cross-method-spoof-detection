import json
from pathlib import Path

import pandas as pd
import yaml

from .constants import MANIFEST_COLUMNS, SCORE_COLUMNS


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_manifest(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"speaker_id": str, "attack_id": str,
                                  "method_id": str, "method_family": str})
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing manifest columns: {missing}")
    return df


def read_scores(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"speaker_id": str, "attack_id": str,
                                  "method_id": str, "method_family": str})
    missing = [c for c in SCORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing score columns: {missing}")
    return df
