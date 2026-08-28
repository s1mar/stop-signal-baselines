"""The panel's shared rejection risk: 205 failed runs have zero gold-token overlap, carry
0.571 of failed compute, and get PPR 0.906. The load-bearing premise is "zero overlap means
no progress". If that premise is wrong those runs may hold viable non-reference solutions.

Direct test: how often do RESOLVED runs reach zero overlap? A resolved run is known to have
solved the task. If resolved runs essentially never sit at zero overlap, then zero overlap is
an empirically near-certain marker of failure and the stratum is defensible. If a meaningful
share of resolved runs also show zero overlap, the premise fails and we must say so.
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))

R, F = [], []
for t in trajs:
    o = compute_oracle(t)
    if o is None or o.tstar is None:
        continue
    (R if t.resolved else F).append(o.max_progress)
R, F = np.array(R), np.array(F)

print(f"resolved n={len(R)}  failed n={len(F)}\n")
for lab, v in (("resolved", R), ("failed", F)):
    print(f"{lab:9s} frac max==0: {np.mean(v == 0):.4f}   "
          f"frac <0.05: {np.mean(v < 0.05):.4f}   median {np.median(v):.3f}")

n_r0 = int((R == 0).sum())
n_f0 = int((F == 0).sum())
print(f"\nzero-overlap runs: {n_f0} failed, {n_r0} resolved")
if n_f0 + n_r0:
    prec = n_f0 / (n_f0 + n_r0)
    print(f"P(failed | max overlap == 0) = {prec:.4f}")
    # Wilson 95% interval for that precision.
    n, p, z = n_f0 + n_r0, prec, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    print(f"Wilson 95% CI [{c-h:.4f}, {c+h:.4f}]")
print(f"\nBaseline P(failed) in corpus = {len(F)/(len(F)+len(R)):.4f}")
