"""Task A.11: sha256 checksums of every released artefact -> docs/checksums.txt (paste into DATA_CARD.md).
  python scripts/checksums.py
"""
import hashlib
from pathlib import Path

from spoofdet.common import paths


def sha256(p: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


files = [paths.MANIFESTS / "master.csv", paths.SPLITS / "split_summary.csv", paths.SPLITS / "_union_P1.csv"]
files += sorted(paths.FEATURES.glob("*/feats.npy")) + sorted(paths.FEATURES.glob("*/index.csv"))
lines = []
for p in files:
    if p.exists():
        base = next((b for b in (paths.DATA, paths.SPLITS.parent) if b in p.parents), p.parent)
        lines.append(f"{sha256(p)}  {p.relative_to(base).as_posix()}  ({p.stat().st_size / 1e6:.1f} MB)")
        print(lines[-1])
out = paths.ROOT / "docs" / "checksums.txt"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("->", out)
