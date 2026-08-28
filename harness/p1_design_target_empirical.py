"""A design target derived from the EMPIRICAL score distributions, not from a Gaussian.

An earlier draft's derivation was rejected in review: the previous "propagate the observed
1.4 to 1.7 coverage ratio" adjustment as an undefined operation: a ratio measured at coverage ~0.2
says nothing about the ratio at coverage 0.5. This script replaces it with a derivation.

METHOD (stated in the paper, one explicit assumption):
  The threshold is set by the RESOLVED population, whose empirical distribution we observe
  directly. Model a better detector as one whose doomed-run scores are the same shape shifted up
  by delta, which is the non-parametric counterpart of the equal-variance Gaussian assumption but
  keeps the real, skewed, discrete shape. Then for each shift delta:

     thr        = empirical (1-alpha) quantile of the resolved run-level maxima
     coverage   = P(R + delta >= thr),        R ~ empirical resolved distribution
     AUC        = P(R + delta > R') + 0.5 P(R + delta = R'),  R, R' iid empirical resolved

  Sweep delta, find the smallest one reaching the target coverage, and report its AUC. That AUC is
  the separation a detector of this shape must reach, measured against the tail the budget
  actually has to cut through.

Reported alongside the Gaussian number so a reader can see how much the idealization costs.
Writes data/p1_design_target_empirical.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ORDER = ["syntactic", "convergence", "prm_proxy", "self_report"]
LABEL = {"syntactic": "Syntactic repetition", "convergence": "Embedding diversity",
         "prm_proxy": "Process-reward proxy", "self_report": "Self-report"}
ALPHA = 0.05
TARGETS = [0.25, 0.50, 0.75]

try:
    from scipy.stats import norm
    Phi, Phinv = norm.cdf, norm.ppf
except Exception:
    from math import erf, sqrt as _sq
    Phi = lambda x: 0.5 * (1 + erf(x / _sq(2)))
    def Phinv(p):
        lo, hi = -10.0, 10.0
        for _ in range(200):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if Phi(m) < p else (lo, m)
        return (lo + hi) / 2


def auc_shift(R, delta):
    """P(R+delta > R') + 0.5 P(equal), R, R' iid from the empirical resolved sample."""
    a = R[:, None] + delta
    b = R[None, :]
    return float((a > b).mean() + 0.5 * (a == b).mean())


def main():
    rows = json.load(open(os.path.join(DATA, "corpus_strat.json"), encoding="utf-8"))
    resolved = [e for e in rows if e["resolved"]]
    failed = [e for e in rows if not e["resolved"]]

    out = {"alpha": ALPHA, "targets": TARGETS, "detectors": {}}
    print(f"empirical design target at a {ALPHA:.0%} false-abort budget")
    print(f"(resolved n={len(resolved)}, failed n={len(failed)})\n")
    header = "  ".join(f"{'C='+str(int(t*100))+'%':>18}" for t in TARGETS)
    print(f"{'detector':22s} {'observed AUC':>12}   {header}")

    per_target = {str(t): [] for t in TARGETS}
    for d in ORDER:
        R = np.array(sorted(max(x for x in e["scores"][d] if x is not None)
                            for e in resolved if e["scores"][d]), float)
        F = np.array([max(x for x in e["scores"][d] if x is not None)
                      for e in failed if e["scores"][d]], float)
        thr = float(np.percentile(R, 100 * (1 - ALPHA)))
        obs_auc = float((F[:, None] > R[None, :]).mean() + 0.5 * (F[:, None] == R[None, :]).mean())

        span = max(R.max() - R.min(), 1e-6)
        deltas = np.linspace(0, 4 * span, 4001)
        cov = np.array([float((R + dl >= thr).mean()) for dl in deltas])

        cells, res = [], {}
        for t in TARGETS:
            idx = np.argmax(cov >= t) if (cov >= t).any() else None
            if idx is None:
                cells.append(f"{'unreachable':>18}")
                res[str(t)] = None
                continue
            dl = float(deltas[idx])
            a = auc_shift(R, dl)
            res[str(t)] = dict(delta=dl, auc=a, coverage=float(cov[idx]))
            per_target[str(t)].append(a)
            cells.append(f"{a:18.3f}")
        out["detectors"][d] = dict(observed_auc=obs_auc, threshold=thr, required=res)
        print(f"{LABEL[d]:22s} {obs_auc:12.3f}   " + "  ".join(cells))

    print(f"\n{'target coverage':>16} {'empirical AUC range':>26} {'Gaussian AUC':>14}")
    summary = {}
    for t in TARGETS:
        vals = per_target[str(t)]
        g = Phi((Phinv(1 - ALPHA) + Phinv(t)) / np.sqrt(2))
        summary[str(t)] = dict(empirical_lo=min(vals) if vals else None,
                               empirical_hi=max(vals) if vals else None,
                               gaussian=float(g))
        rng = f"{min(vals):.3f} to {max(vals):.3f}" if vals else "n/a"
        print(f"{t:16.0%} {rng:>26} {g:14.3f}")
    out["summary"] = summary

    json.dump(out, open(os.path.join(DATA, "p1_design_target_empirical.json"), "w"), indent=1)
    print("\nwrote data/p1_design_target_empirical.json")


if __name__ == "__main__":
    main()
