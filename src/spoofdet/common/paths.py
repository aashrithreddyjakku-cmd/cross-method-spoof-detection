"""Central place for every directory. Override with environment variables on Kaggle/Colab."""
import os
from pathlib import Path


def _find_root() -> Path:
    if os.environ.get("SPOOFDET_ROOT"):
        return Path(os.environ["SPOOFDET_ROOT"]).resolve()
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


ROOT = _find_root()
DATA = Path(os.environ.get("SPOOFDET_DATA", ROOT / "data"))          # big, git-ignored
RAW = DATA / "raw"
MANIFESTS = DATA / "manifests"
FEATURES = DATA / "features"
SPLITS = Path(os.environ.get("SPOOFDET_SPLITS", ROOT / "splits"))     # small CSVs, tracked in git
CONFIGS = ROOT / "configs"
CKPT = Path(os.environ.get("SPOOFDET_CKPT", ROOT / "checkpoints"))   # git-ignored
RESULTS = Path(os.environ.get("SPOOFDET_RESULTS", ROOT / "results"))
SCORES = RESULTS / "scores"        # one folder per experiment (git-ignored)
TABLES = RESULTS / "tables"        # tracked
FIGURES = RESULTS / "figures"      # tracked


def ensure_dirs():
    for p in (RAW, MANIFESTS, FEATURES, SPLITS, CKPT, SCORES, TABLES, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
