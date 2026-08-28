"""Clustered CI for the PPR itself, so the headline and the detector estimates are
quoted on the same resampling basis. The corpus has many trajectories per task
instance, so resampling trajectories understates uncertainty; resample instances.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import econ

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
rows = json.load(open(os.path.join(DATA, "corpus_strat.json"), encoding="utf-8"))
failed = [e for e in rows if not e["resolved"]]


def ci(group, n_boot=4000, seed=0):
    by = {}
    for e in group:
        by.setdefault(e["iid"], []).append(e)
    iids = list(by)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        g = [e for k in rng.integers(0, len(iids), len(iids)) for e in by[iids[k]]]
        tot = sum(e["n"] for e in g) or 1
        v.append(sum(econ.waste_after(e["tstar"], e["n"]) for e in g) / tot)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def point(g):
    return sum(econ.waste_after(e["tstar"], e["n"]) for e in g) / (sum(e["n"] for e in g) or 1)


out = {}
for lab, g in (("all failed", failed),
               ("reached reference", [e for e in failed if e["maxprog"] > 0]),
               ("never reached", [e for e in failed if e["maxprog"] == 0])):
    lo, hi = ci(g)
    out[lab] = dict(point=point(g), lo=lo, hi=hi, n=len(g),
                    instances=len({e["iid"] for e in g}))
    print(f"{lab:20s} n={len(g):3d} over {out[lab]['instances']:3d} instances  "
          f"PPR {point(g):.3f}  clustered 95% CI [{lo:.3f}, {hi:.3f}]")
json.dump(out, open(os.path.join(DATA, "clustered_ppr.json"), "w"), indent=1)
