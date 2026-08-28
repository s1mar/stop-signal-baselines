"""PRESCRIPTIVE: what separation would a usable stop-signal actually need?

The mechanism result says coverage collapses because the failed-run and resolved-run score
distributions overlap, not because alarms fire late. That turns the design question into a
quantitative one: how much separation is required to catch a given share of doomed runs while
wrongly aborting at most alpha of recoverable ones?

Under a two-Gaussian equal-variance model (resolved ~ N(0,1), failed ~ N(d',1)):
    threshold at the resolved (1-alpha) quantile:  thr = z_{1-alpha}
    coverage of failed runs:                       C   = Phi(d' - z_{1-alpha})
    so the required separation is                  d'  = z_{1-alpha} + z_C
and the equivalent run-level AUC is                AUC = Phi(d' / sqrt(2)).

This gives a falsifiable target a proposed detector can be measured against, rather than the
vague advice to "detect stagnation earlier". Compares the requirement to what the evaluated
families actually achieve.
"""
import json, os, sys
import numpy as np
from math import sqrt

try:
    from scipy.stats import norm
    Phi, Phinv = norm.cdf, norm.ppf
except Exception:                                   # no scipy: use erf-based fallback
    from math import erf
    def Phi(x): return 0.5 * (1 + erf(x / sqrt(2)))
    def Phinv(p):
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if Phi(mid) < p: lo = mid
            else: hi = mid
        return (lo + hi) / 2

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ALPHA = 0.05
TARGETS = [0.25, 0.50, 0.75, 0.90]


def main():
    mech = json.load(open(os.path.join(DATA, "mechanism.json"), encoding="utf-8"))
    obs = {d: v["auc_runlevel"] for d, v in mech["score_overlap"].items()}
    obs_cov = {d: v["failed_reaching"] for d, v in mech["score_overlap"].items()}

    z_a = Phinv(1 - ALPHA)
    print(f"required separation to hold false-abort at {ALPHA:.0%}\n")
    print(f"{'target coverage':>16} {'required d-prime':>17} {'required run-level AUC':>23}")
    req = {}
    for C in TARGETS:
        d = z_a + Phinv(C)
        a = Phi(d / sqrt(2))
        req[str(C)] = dict(d_prime=float(d), auc=float(a))
        print(f"{C:16.0%} {d:17.2f} {a:23.3f}")

    print(f"\nwhat the evaluated detector families actually achieve:")
    print(f"{'detector':14s} {'run-level AUC':>14} {'coverage@5%':>12}")
    for d in obs:
        print(f"{d:14s} {obs[d]:14.3f} {obs_cov[d]:12.3f}")
    best = max(obs, key=obs.get)
    print(f"\nbest observed run-level AUC: {obs[best]:.3f} ({best})")
    print(f"AUC needed for 50% coverage at {ALPHA:.0%} false-abort: {req['0.5']['auc']:.3f}")
    gap = req["0.5"]["auc"] - obs[best]
    print(f"gap: {gap:+.3f} AUC")

    out = dict(alpha=ALPHA, required=req, observed_auc=obs, observed_coverage=obs_cov,
               best_observed=dict(detector=best, auc=obs[best]),
               auc_gap_to_50pct_coverage=float(gap))
    json.dump(out, open(os.path.join(DATA, "design_target.json"), "w"), indent=1)
    print("\nwrote data/design_target.json")


if __name__ == "__main__":
    main()
