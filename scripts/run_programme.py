"""The experiment programme E1-E7 (playbook B.8) as one resumable command, in priority order.
  python scripts/run_programme.py --dry-run                 # list what would run
  python scripts/run_programme.py --parallel 3 --threads 4 --hours 8
  python scripts/run_programme.py --finish                  # score checks, tables, figures, explanations, log
Finished runs (results/scores/<exp>/DONE) are skipped, so stopping and restarting is safe. --hours stops STARTING new runs
after that time; stages come in the playbook's priority order, so whatever gets cut is the least important.
Logs: logs/programme/<exp>.log and logs/programme/status.csv
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from spoofdet.common import paths
from spoofdet.common.io import load_yaml

PY = sys.executable
FAMILIES = ["neural_vocoder", "source_filter_vocoder", "waveform_modification", "griffin_lim"]
LOMO_FIRST = ["A01", "A04", "A06", "A08", "A10", "A11", "A13", "A17"]
LOMO_REST = ["A02", "A03", "A05", "A07", "A09", "A12", "A14", "A15", "A18"]
MODELS = ["smallcnn", "lcnn", "resnet18", "vggish_cbam"]
LOGS = paths.ROOT / "logs" / "programme"


def done(exp):
    return (paths.SCORES / exp / "DONE").exists()


def matrix(*args):
    """Generate the configs with make_matrix.py and return their experiment ids."""
    r = subprocess.run([PY, str(paths.ROOT / "scripts" / "make_matrix.py"), *args], capture_output=True, text=True, check=True, cwd=paths.ROOT)
    return r.stdout.split()


def best_model():
    """Lowest mean validation EER on P0 over the finished seeds (playbook B.8); fixed before the held-out runs."""
    rows = []
    for m in MODELS:
        vals = [pd.read_csv(p).val_eer.min() for p in paths.CKPT.glob(f"p0_{m}_mel64_w400_s*/train_log.csv")]
        if vals:
            rows.append((sum(vals) / len(vals), m))
    return min(rows)[1] if rows else "lcnn"


def classical_jobs():
    jobs = [("p0_mfcc_lr", ["--protocol", "P0_official", "--clf", "lr", "--tests", "test", "test_itw"]),
            ("p0_mfcc_svm", ["--protocol", "P0_official", "--clf", "svm", "--tests", "test", "test_itw"])]
    for f in FAMILIES:
        for feat in ("mel64_w400", "mel64_w400_trim"):
            jobs.append((f"lofo_{f}_mfccsvm_{feat}", ["--protocol", f"P1_lofo/{f}", "--clf", "svm", "--feat", feat, "--tests", "test_seen", "test_unseen"]))
    for m in ("A10", "A17"):
        jobs.append((f"lomo_{m}_mfccsvm_mel64_w400", ["--protocol", f"P1_lomo/{m}", "--clf", "svm", "--tests", "test_seen", "test_unseen"]))
    return jobs


def run_one(exp, cmd, env, deadline):
    if done(exp):
        return exp, "skipped", 0.0
    if time.time() > deadline:
        return exp, "not started (time limit)", 0.0
    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOGS / f"{exp}.log", "w", encoding="utf-8", errors="replace") as log:
        code = subprocess.run(cmd, env=env, cwd=paths.ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    return exp, "ok" if code == 0 else f"FAILED ({code})", (time.time() - t0) / 60


def run_stage(name, jobs, a, env, deadline):
    todo = [j for j in jobs if not done(j[0])]
    print(f"\n== {name}: {len(jobs)} runs, {len(jobs) - len(todo)} already done", flush=True)
    if a.dry_run:
        for exp, _ in todo:
            print("  ", exp)
        return
    results = []
    with ThreadPoolExecutor(a.parallel) as ex:
        for exp, status, mins in ex.map(lambda j: run_one(j[0], j[1], env, deadline), jobs):
            if status != "skipped":
                print(f"   {exp}: {status} in {mins:.1f} min", flush=True)
                results.append(dict(time=time.strftime("%Y-%m-%d %H:%M"), stage=name, exp=exp, status=status, minutes=round(mins, 1)))
    if results:
        p = LOGS / "status.csv"
        pd.DataFrame(results).to_csv(p, mode="a", header=not p.exists(), index=False)


def finish():
    env = dict(os.environ)
    steps = [["scripts/check_splits.py"], ["scripts/check_scores.py", "--all"], ["-m", "spoofdet.eval.evaluate", "--all", "--boot", "300"],
             ["-m", "spoofdet.eval.aggregate"], ["scripts/make_figures.py"], ["-m", "spoofdet.train.profile_model", "--n-mels", "64"],
             ["scripts/update_experiment_log.py"]]
    failed = []
    for s in steps:
        print("\n>>", " ".join(s), flush=True)
        if subprocess.run([PY, *s], env=env, cwd=paths.ROOT).returncode:
            failed.append(" ".join(s))
    best = best_model()
    print("best model on P0 validation:", best, flush=True)
    for cfg in sorted(paths.CKPT.glob(f"lo[mf]o_*_{best}_mel64_w400_s1/config.yaml")):
        exp = cfg.parent.name
        protocol = load_yaml(cfg)["protocol"]
        if not (paths.SCORES / exp / "DONE").exists():
            continue
        for s in (["scripts/run_xai.py", "--exp", exp, "--split-csv", f"splits/{protocol}/test_unseen.csv", "--n", "40"],
                  ["scripts/xai_analysis.py", "--exp", exp, "--n", "40"]):
            if subprocess.run([PY, *s], env=env, cwd=paths.ROOT, stdout=subprocess.DEVNULL).returncode:
                failed.append(" ".join(s))
    print("\nFINISHED" if not failed else f"\nFINISHED WITH PROBLEMS: {failed}")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads per run")
    ap.add_argument("--hours", type=float, default=1e6, help="stop starting new runs after this many hours")
    ap.add_argument("--stages", nargs="*", default=["E1", "E1b", "E4", "E3", "E5", "E2", "E6", "E3b"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="seeds of the P0 comparison (E1)")
    ap.add_argument("--extra", nargs="*", default=[], help="overrides for every CNN run, e.g. train.epochs=20")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    if a.finish:
        sys.exit(finish())
    if os.name == "nt" and not a.dry_run:          # no idle sleep while this process runs (a closed lid still sleeps)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    deadline = time.time() + a.hours * 3600
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), PYTHONIOENCODING="utf-8")

    def cnn(exps):
        return [(e, [PY, "-m", "spoofdet.train.run_experiment", str(paths.CONFIGS / "experiments" / f"{e}.yaml"), "train.num_workers=0", *a.extra]) for e in exps]

    seeds = [str(s) for s in a.seeds]
    for st in a.stages:
        if st == "E1":
            run_stage("E1 P0 comparison", cnn(matrix("--kind", "p0", "--models", *MODELS, "--seeds", *seeds)), a, env, deadline)
        elif st == "E1b":
            run_stage("E1b classical baselines", [(n, [PY, "-m", "spoofdet.train.classical", "--name", n, *opt]) for n, opt in classical_jobs()], a, env, deadline)
        else:
            best = best_model()
            pair = sorted({best, "smallcnn"})
            if st == "E4":
                run_stage(f"E4 LOFO ({' '.join(pair)})", cnn(matrix("--kind", "lofo", "--models", *pair)), a, env, deadline)
            elif st == "E3":
                run_stage(f"E3 LOMO priority methods ({' '.join(pair)})", cnn(matrix("--kind", "lomo", "--models", *pair, "--methods", *LOMO_FIRST)), a, env, deadline)
            elif st == "E5":
                run_stage(f"E5 ablations ({best})", cnn(matrix("--kind", "ablate", "--model", best)), a, env, deadline)
            elif st == "E2":
                run_stage("E2 P1_all reference", cnn(matrix("--kind", "p1all", "--models", *pair)), a, env, deadline)
            elif st == "E6":
                exps = matrix("--kind", "lofo", "--models", best, "--seeds", "2", "3") + matrix("--kind", "lomo", "--models", best, "--methods", *LOMO_FIRST, "--seeds", "2", "3")
                run_stage(f"E6 extra seeds ({best})", cnn(exps), a, env, deadline)
            elif st == "E3b":
                run_stage(f"E3b LOMO remaining methods ({best})", cnn(matrix("--kind", "lomo", "--models", best, "--methods", *LOMO_REST)), a, env, deadline)
    print("\nProgramme stopped. Next: python scripts/run_programme.py --finish", flush=True)


if __name__ == "__main__":
    main()
