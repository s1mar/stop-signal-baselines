"""Test Kimi's soundness objection: is RWR=0.754 mechanically inflated by failed runs
that never approached the gold patch at all?

If a run never moves toward gold, t* is pinned near step 0 and its RWR is ~1.0 by
construction. If those runs dominate, the headline measures "agents that never got
close kept going", not "agents thrash after nearly solving it".

Stratify failed runs by max progress and report RWR within each stratum.
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import econ
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))

rows = []
for t in trajs:
    if t.resolved:
        continue
    orc = compute_oracle(t)
    if orc is None or orc.tstar is None:
        continue
    rows.append(dict(n=t.n_steps, tstar=orc.tstar, mx=orc.max_progress,
                     rec=econ.waste_after(orc.tstar, t.n_steps)))

print(f"failed runs with an oracle: {len(rows)}")
tot = sum(r["n"] for r in rows)
print(f"overall RWR = {sum(r['rec'] for r in rows)/tot:.3f}\n")

mx = np.array([r["mx"] for r in rows])
print(f"max-progress distribution: median {np.median(mx):.3f}  "
      f"frac==0 {np.mean(mx == 0):.3f}  frac<0.1 {np.mean(mx < 0.1):.3f}\n")

BANDS = [(0.0, 1e-9, "never moved (max=0)"), (1e-9, 0.1, "0 < max < 0.1"),
         (0.1, 0.3, "0.1 <= max < 0.3"), (0.3, 1.01, "max >= 0.3")]
print(f"{'stratum':22s} {'n':>4} {'steps':>7} {'RWR':>6} {'med t*/n':>9} {'share of steps':>15}")
for lo, hi, lab in BANDS:
    g = [r for r in rows if lo <= r["mx"] < hi]
    if not g:
        print(f"{lab:22s}    0"); continue
    st = sum(r["n"] for r in g)
    print(f"{lab:22s} {len(g):4d} {st:7d} {sum(r['rec'] for r in g)/st:6.3f} "
          f"{np.median([r['tstar']/max(1, r['n']) for r in g]):9.3f} {st/tot:15.3f}")

# The decisive number: RWR restricted to runs that demonstrably approached gold.
for cut in (0.05, 0.1, 0.2, 0.3):
    g = [r for r in rows if r["mx"] >= cut]
    st = sum(r["n"] for r in g) or 1
    print(f"\nRWR | max progress >= {cut}: {sum(r['rec'] for r in g)/st:.3f}  "
          f"(n={len(g)}, {st/tot:.1%} of failed steps)")
