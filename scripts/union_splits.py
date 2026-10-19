"""Write splits/_union_P1.csv = union of all rows used by P1_all (covers every P1 fold) -
use it as the --manifest for extracting ablation feature stores cheaply."""
import pandas as pd
from spoofdet.common import paths
d = pd.concat([pd.read_csv(paths.SPLITS / "P1_all" / f"{s}.csv") for s in ("train", "val", "test")])
d.to_csv(paths.SPLITS / "_union_P1.csv", index=False)
print(len(d), "rows ->", paths.SPLITS / "_union_P1.csv")
