"""Generate experiment YAMLs from the template, e.g.
   python scripts/make_matrix.py --kind lomo --models lcnn smallcnn --methods A01 A04 A10 --seeds 1
   python scripts/make_matrix.py --kind p1all --models lcnn smallcnn
   python scripts/make_matrix.py --kind p0 --models smallcnn lcnn resnet18 vggish_cbam --seeds 1 2 3
   python scripts/make_matrix.py --kind ablate --model lcnn         (one-factor-at-a-time feature/norm/augment ablations on held-out folds)
"""
import argparse
import copy

from spoofdet.common import paths
from spoofdet.common.io import load_yaml, save_yaml

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["p0", "p1all", "lomo", "lofo", "ablate"], required=True)
ap.add_argument("--models", nargs="+", default=["lcnn"])
ap.add_argument("--model", default="lcnn")
ap.add_argument("--methods", nargs="*", default=[])
ap.add_argument("--families", nargs="*", default=["neural_vocoder", "source_filter_vocoder", "waveform_modification", "griffin_lim"])
ap.add_argument("--folds", nargs="*", default=["P1_lofo/neural_vocoder", "P1_lofo/source_filter_vocoder", "P1_lofo/waveform_modification"],
                help="protocols used for ablations (each ablation is run on each fold)")
ap.add_argument("--seeds", nargs="+", type=int, default=[1])
ap.add_argument("--feat", default="mel64_w400")
a = ap.parse_args()
base = load_yaml(paths.CONFIGS / "experiments" / "_template.yaml")
out = paths.CONFIGS / "experiments"; made = []


def emit(exp_id, **ch):
    c = copy.deepcopy(base); c["exp_id"] = exp_id
    for k, v in ch.items():
        if k in ("seed",): c["train"]["seed"] = v
        elif k == "model": c["model"] = {"name": v, "kwargs": {}}
        else: c[k] = v
    save_yaml(c, out / f"{exp_id}.yaml"); made.append(exp_id)


if a.kind == "p0":
    for m in a.models:
        for s in a.seeds:
            emit(f"p0_{m}_{a.feat}_s{s}", model=m, protocol="P0_official", feat_id=a.feat, seed=s)
elif a.kind == "p1all":
    for m in a.models:
        for s in a.seeds:
            emit(f"p1all_{m}_{a.feat}_s{s}", model=m, protocol="P1_all", feat_id=a.feat, seed=s)
elif a.kind in ("lomo", "lofo"):
    for m in a.models:
        for t in (a.methods if a.kind == "lomo" else a.families):
            for s in a.seeds:
                emit(f"{a.kind}_{t}_{m}_{a.feat}_s{s}", model=m, protocol=f"P1_{a.kind}/{t}", feat_id=a.feat, seed=s)
else:   # one-factor-at-a-time ablations around (mel64_w400, norm=utt, SpecAugment) on held-out folds
    m = a.model
    for fold in a.folds:
        tag = fold.split("/")[-1]
        emit(f"abl_base_{tag}_{m}", model=m, protocol=fold, feat_id="mel64_w400")
        for f in ("mel40_w400", "mel80_w400", "mel128_w400", "mel64_w200", "mel64_w800", "mel64_w400_trim"):
            emit(f"abl_feat_{f}_{tag}_{m}", model=m, protocol=fold, feat_id=f)
        for n in ("none", "utt_band", "global"):
            emit(f"abl_norm_{n}_{tag}_{m}", model=m, protocol=fold, feat_id="mel64_w400", norm=n)
        emit(f"abl_aug_none_{tag}_{m}", model=m, protocol=fold, feat_id="mel64_w400", augment={})
print("\n".join(made))
