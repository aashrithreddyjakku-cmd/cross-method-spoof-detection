"""Member B: training engine, scoring and baseline wiring on a tiny synthetic feature store."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spoofdet.common.constants import MANIFEST_COLUMNS, SCORE_COLUMNS
from spoofdet.models import REGISTRY, build_model, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = str(ROOT / "configs" / "experiments" / "_template.yaml")
FAST = ["model.name=smallcnn", "train.epochs=2", "train.batch_size=8", "train.num_workers=0", "protocol=P0_official"]


def manifest(prefix, n, speakers, method="A01"):
    rows = []
    for i in range(n):
        lab = i % 2
        rows.append(dict(utt_id=f"{prefix}{i}", dataset="la19", partition="train", path=f"{prefix}{i}.flac", label=lab,
                         attack_id=method if lab else "bonafide", method_id=method if lab else "bonafide",
                         method_family="neural_vocoder" if lab else "bonafide", speaker_id=speakers[i % len(speakers)],
                         duration_s=4.0, orig_sr=16000))
    return pd.DataFrame(rows)[MANIFEST_COLUMNS]


def write_store(root, name, frame, rng):
    d = root / "data" / "features" / name; d.mkdir(parents=True)
    arr = np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float16, shape=(len(frame), 64, 401))
    for i, lab in enumerate(frame.label):
        arr[i] = (rng.randn(64, 401) - 40 + 15 * lab * (np.arange(64) > 40)[:, None]).astype(np.float16)     # learnable cue
    arr.flush()
    pd.DataFrame({"utt_id": frame.utt_id, "row": np.arange(len(frame))}).to_csv(d / "index.csv", index=False)
    (d / "feat_config.yaml").write_text(yaml.safe_dump(dict(feat_id=name, sr=16000, clip_seconds=4.0, n_mels=64, n_fft=512, win_length=400,
                                                            hop_length=160, f_min=20, f_max=8000, power=2.0, top_db=80.0)))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    t = tmp_path_factory.mktemp("b")
    rng = np.random.RandomState(0)
    tr, va, te, itw = manifest("tr", 64, ["s1", "s2"]), manifest("va", 24, ["s3"]), manifest("te", 24, ["s4"]), manifest("itw", 16, ["x"])
    p = t / "splits" / "P0_official"; p.mkdir(parents=True)
    tr.to_csv(p / "train.csv", index=False); va.to_csv(p / "val.csv", index=False); te.to_csv(p / "test.csv", index=False)
    itw.to_csv(p / "test_itw.csv", index=False)
    la = pd.concat([tr, va, te], ignore_index=True)
    write_store(t, "full", pd.concat([la, itw], ignore_index=True), rng)
    write_store(t, "la_only", la, rng)                 # like an ablation store: no In-the-Wild
    env = dict(os.environ, SPOOFDET_DATA=str(t / "data"), SPOOFDET_SPLITS=str(t / "splits"), SPOOFDET_CKPT=str(t / "ckpt"),
               SPOOFDET_RESULTS=str(t / "results"), PYTHONIOENCODING="utf-8")
    return t, env


def run(env, *args):
    r = subprocess.run([sys.executable, *args], env=env, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_experiment_scores_contract_and_run_info(world):
    t, env = world
    out = run(env, "-m", "spoofdet.train.run_experiment", TEMPLATE, "exp_id=ok", "feat_id=full", *FAST)
    assert "scored itw: 16" in out
    d = pd.read_csv(t / "results" / "scores" / "ok" / "test.csv")
    assert list(d.columns) == SCORE_COLUMNS and d.score.between(0, 1).all()
    assert (t / "results" / "scores" / "ok" / "DONE").exists()
    assert {"best.pt", "last.pt", "config.yaml", "train_log.csv", "run_info.yaml"} <= {p.name for p in (t / "ckpt" / "ok").iterdir()}
    run(env, "scripts/check_scores.py", "--exp", "ok")


def test_ablation_store_skips_itw_instead_of_crashing(world):
    t, env = world
    out = run(env, "-m", "spoofdet.train.run_experiment", TEMPLATE, "exp_id=abl", "feat_id=la_only", *FAST)
    assert "skip itw" in out and (t / "results" / "scores" / "abl" / "DONE").exists()
    assert not (t / "results" / "scores" / "abl" / "itw.csv").exists()


def test_queue_skips_finished_and_reports_failure(world, tmp_path):
    t, env = world
    (t / "results" / "scores" / "doneexp").mkdir(parents=True); (t / "results" / "scores" / "doneexp" / "DONE").write_text("ok")
    (tmp_path / "doneexp.yaml").write_text("exp_id: doneexp\n")
    (tmp_path / "broken.yaml").write_text("exp_id: broken\n")
    r = subprocess.run([sys.executable, "scripts/run_queue.py", str(tmp_path / "*.yaml")], env=env, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert "1 already done, 1 to run" in r.stdout and r.returncode != 0 and "broken" in r.stderr


def test_classical_writes_contract_names_and_config(world):
    t, env = world
    run(env, "-m", "spoofdet.train.classical", "--feat", "full", "--clf", "lr", "--name", "mf", "--tests", "test", "test_itw")
    s = t / "results" / "scores" / "mf"
    assert (s / "itw.csv").exists() and not (s / "test_itw.csv").exists() and (s / "DONE").exists()
    assert yaml.safe_load((t / "ckpt" / "mf" / "config.yaml").read_text())["model"]["name"] == "mfcc_lr"


@pytest.mark.parametrize("name", list(REGISTRY))
def test_checkpoint_roundtrip_same_output(tmp_path, name):
    import torch
    m = build_model(name).eval(); x = torch.randn(2, 1, 64, 401)
    stats = {"mean": np.zeros(64, np.float32), "std": np.ones(64, np.float32)}
    save_checkpoint(tmp_path / "b.pt", m, name, {}, {"feat_id": "x"}, "global", stats, 1, 0.1, 1)
    m2, ck = load_checkpoint(tmp_path / "b.pt")
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x))
    assert ck["norm_stats"]["mean"].dtype == np.float32 and ck["model_name"] == name
