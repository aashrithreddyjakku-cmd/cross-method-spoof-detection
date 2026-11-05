"""Member A: data pipeline tests (manifest, splits, feature store) on the dummy dataset."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from spoofdet.common.constants import MANIFEST_COLUMNS
from spoofdet.data.audio import fix_length
from spoofdet.data.features import FeatureStore, load_feat_cfg, n_frames
from spoofdet.data.protocols import add_method_columns, parse_itw

ROOT = Path(__file__).resolve().parents[1]


def run(script, env, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], env=env, cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    t = tmp_path_factory.mktemp("spoofdet")
    env = dict(os.environ, SPOOFDET_DATA=str(t / "data"), SPOOFDET_SPLITS=str(t / "splits"),
               SPOOFDET_RESULTS=str(t / "results"), PYTHONIOENCODING="utf-8")
    run("make_dummy_data.py", env)
    run("build_manifest.py", env)
    run("make_splits.py", env)
    return t, env


def test_taxonomy_mapping():
    df = add_method_columns(pd.DataFrame({"attack_id": ["bonafide", "A16", "A19", "A11", "A07"]}))
    assert list(df.method_id) == ["bonafide", "A04", "A06", "A11", "A07"]
    assert list(df.method_family) == ["bonafide", "waveform_modification", "waveform_modification",
                                      "griffin_lim", "source_filter_vocoder"]


def test_manifest_dummy(dummy):
    t, _ = dummy
    df = pd.read_csv(t / "data" / "manifests" / "master.csv")
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 672 and df.utt_id.is_unique
    assert (df[df.dataset == "la19"].label == 1).sum() > 0
    assert not ((df.label == 1) & (df.dataset == "la19") & (df.method_family == "unknown")).any()
    assert not df.path.str.contains("\\\\", regex=False).any()          # portable '/' paths
    assert df.duration_s.notna().all()


def test_splits_pass_checks(dummy):
    t, env = dummy
    out = run("check_splits.py", env)
    assert "ALL SPLIT CHECKS PASSED" in out
    assert len(list((t / "splits" / "P1_lomo").iterdir())) == 17
    assert {p.name for p in (t / "splits" / "P1_lofo").iterdir()} == {
        "neural_vocoder", "source_filter_vocoder", "waveform_modification", "griffin_lim"}


def test_check_splits_catches_leak(dummy, tmp_path):
    t, env = dummy
    import shutil
    bad = tmp_path / "splits"; shutil.copytree(t / "splits", bad)
    fold = bad / "P1_lomo" / "A10"
    tr = pd.read_csv(fold / "train.csv"); un = pd.read_csv(fold / "test_unseen.csv")
    pd.concat([tr, un[un.label == 1].head(1)]).to_csv(fold / "train.csv", index=False)   # inject a leak
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_splits.py")], cwd=ROOT,
                       env=dict(env, SPOOFDET_SPLITS=str(bad)), capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 1 and "held-out leaks" in r.stdout


def test_parse_itw_paths(tmp_path):
    d = tmp_path / "in_the_wild" / "release"; d.mkdir(parents=True)
    (d / "meta.csv").write_text("file,speaker,label\n0.wav,Alice,spoof\n1.wav,Bob,bona-fide\n")
    df = parse_itw(tmp_path)
    assert list(df.label) == [1, 0] and list(df.attack_id) == ["itw_spoof", "bonafide"]
    assert df.path.iloc[0] == "in_the_wild/release/0.wav"


def test_feature_store_roundtrip(tmp_path):
    cfg = load_feat_cfg("mel64_w400")
    rng = np.random.RandomState(0)
    for i, n in enumerate((8000, 64000, 90000)):
        sf.write(tmp_path / f"{i}.wav", (0.1 * rng.randn(n)).astype(np.float32), 16000)
    man = pd.DataFrame({"utt_id": ["a", "b", "c"], "path": ["0.wav", "1.wav", "2.wav"]})
    fs = FeatureStore.build(man, cfg, tmp_path / "store", raw_root=tmp_path, workers=1)
    assert fs.arr.shape == (3, 64, n_frames(cfg)) and fs.arr.dtype == np.float16
    x = fs.get("b")
    assert x.dtype == np.float32 and np.isfinite(x).all() and x.max() - x.min() <= 80.5
    assert not (tmp_path / "store" / "failed.csv").exists()


def test_tiling_keeps_signal():
    y = np.arange(1, 11, dtype=np.float32)
    assert np.array_equal(fix_length(y, 25)[10:20], y)
