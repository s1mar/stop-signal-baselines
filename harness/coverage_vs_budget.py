"""Is the low coverage at the 5% budget a property of the detectors, or of the budget?

If coverage rises steeply as the false-abort budget is relaxed, the detectors do carry
signal and the binding constraint is separability against resolved runs (a calibration
problem). If coverage stays low at any budget, the detectors are blind (a feature problem).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import econ

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
rows = json.load(open(os.path.join(DATA, "corpus_strat.json"), encoding="utf-8"))
failed = [e for e in rows if not e["resolved"]]
resolved = [e for e in rows if e["resolved"]]
THR = json.load(open(os.path.join(DATA, "deep_analysis.json"), encoding="utf-8"))
GRID = {"syntactic": [1, 2, 3, 4, 5, 6, 7, 8],
        "convergence": list(np.round(np.linspace(0.50, 0.99, 25), 4)),
        "prm_proxy": list(range(1, 16)),
        "self_report": [1, 2, 3, 4, 5]}


def fire(sc, thr):
    if sc is None:
        return None
    for i, s in enumerate(sc):
        if s >= thr:
            return i
    return None


BUDGETS = [0.05, 0.10, 0.20, 0.50, 1.00]
print(f"{'detector':14s} " + " ".join(f"cov@{int(b*100)}%".rjust(8) for b in BUDGETS)
      + "   " + " ".join(f"sav@{int(b*100)}%".rjust(8) for b in BUDGETS))
out = {}
for d in GRID:
    cov, sav = [], []
    for b in BUDGETS:
        best = (0.0, None)
        for thr in GRID[d]:
            fa = sum(1 for e in resolved
                     if (lambda f: f is not None and f < e["n"] - 1)(fire(e["scores"][d], thr)))
            if fa / len(resolved) > b:
                continue
            s = sum(econ.waste_after(fire(e["scores"][d], thr), e["n"]) for e in failed)
            if s > best[0] * sum(e["n"] for e in failed) or best[1] is None:
                best = (s / sum(e["n"] for e in failed), thr)
        cov.append(np.mean([fire(e["scores"][d], best[1]) is not None for e in failed])
                   if best[1] is not None else float("nan"))
        sav.append(best[0])
    out[d] = dict(budgets=BUDGETS, coverage=[float(x) for x in cov], saved=sav)
    print(f"{d:14s} " + " ".join(f"{x:8.3f}" for x in cov)
          + "   " + " ".join(f"{x:8.3f}" for x in sav))
json.dump(out, open(os.path.join(DATA, "coverage_vs_budget.json"), "w"), indent=1)
