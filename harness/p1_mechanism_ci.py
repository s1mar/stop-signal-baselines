"""Three checks the review panel asked for on the mechanism / design-target section.

A. CLUSTERED CIs for the run-level separation AUC (bootstrap over task instances), so the
   "0.617 is the number to beat" claim carries an interval.

B. WITHIN-INSTANCE AUC. The corpus has ~16 trajectories per task instance, so a run-level AUC
   computed across instances could partly be measuring task difficulty (hard instances mostly
   fail) rather than run-level stagnation. We recompute the AUC comparing failed against resolved
   runs of the SAME instance only, pooling the within-instance comparisons. If it matches the
   pooled AUC, the confound is not driving the result.

C. MODEL-IMPLIED vs OBSERVED coverage. Under the equal-variance Gaussian model, an observed AUC
   implies d' = sqrt(2) Phi^-1(AUC) and coverage Phi(d' - z_{1-alpha}) at the detector's REALIZED
   false-abort rate (which is not exactly 5% because the scores are discrete). Compare with what
   the detector actually covers. This is the consistency check the design-target section needs.

Writes data/p1_mechanism_ci.json.
"""
from __future__ import annotations
import json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ORDER = ["syntactic", "convergence", "prm_proxy", "self_report"]
B = 2000
SEED = 20260724

try:
    from scipy.stats import norm
    Phi, Phinv = norm.cdf, norm.ppf
except Exception:                                        # erf fallback
    from math import erf, sqrt as _sq
    Phi = lambda x: 0.5 * (1 + erf(x / _sq(2)))
    def Phinv(p):
        lo, hi = -10.0, 10.0
        for _ in range(200):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if Phi(m) < p else (lo, m)
        return (lo + hi) / 2


def auc(pos, neg):
    """P(score(failed) > score(resolved)), ties at half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    s = np.concatenate([pos, neg])[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):                                    # average ranks within ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks[order] = r
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    rows = json.load(open(os.path.join(DATA, "corpus_strat.json"), encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    runmax, lab, iid = {}, [], []
    for e in rows:
        if not all(e["scores"][d] for d in ORDER):
            continue
        lab.append(not e["resolved"])                    # True = failed = positive class
        iid.append(e["iid"])
        for d in ORDER:
            runmax.setdefault(d, []).append(max(x for x in e["scores"][d] if x is not None))
    lab = np.array(lab); iid = np.array(iid)
    runmax = {d: np.array(v, float) for d, v in runmax.items()}
    n_if = len(set(iid[lab])); n_ir = len(set(iid[~lab]))
    n_both = len(set(iid[lab]) & set(iid[~lab]))
    print(f"n={len(lab)} runs ({lab.sum()} failed / {(~lab).sum()} resolved), "
          f"{len(set(iid))} task instances")
    print(f"instances: {n_if} with a failed run, {n_ir} with a resolved run, "
          f"{n_both} with both\n")

    by_inst = collections.defaultdict(list)
    for k, i in enumerate(iid):
        by_inst[i].append(k)
    insts = sorted(by_inst)

    out = {}
    print(f"{'detector':13s} {'AUC':>6} {'clustered 95% CI':>20} {'within-inst AUC':>16} "
          f"{'(n pairs)':>10}")
    for d in ORDER:
        s = runmax[d]
        point = auc(s[lab], s[~lab])

        # A. bootstrap over task instances
        boots = []
        for _ in range(B):
            pick = rng.choice(len(insts), size=len(insts), replace=True)
            idx = np.concatenate([by_inst[insts[p]] for p in pick])
            sl, ll = s[idx], lab[idx]
            if ll.all() or (~ll).all():
                continue
            boots.append(auc(sl[ll], sl[~ll]))
        lo, hi = np.percentile(boots, [2.5, 97.5])

        # B. within-instance AUC, pooled over instances that contain both classes
        num = den = 0.0
        n_inst_used = 0
        for i in insts:
            idx = np.array(by_inst[i])
            f, r = s[idx][lab[idx]], s[idx][~lab[idx]]
            if len(f) == 0 or len(r) == 0:
                continue
            n_inst_used += 1
            num += auc(f, r) * len(f) * len(r)
            den += len(f) * len(r)
        within = num / den if den else float("nan")

        # B2. the within-instance AUC is itself clustered and small: bootstrap it over the
        # instances that contain both classes, so we do not overclaim in the other direction.
        both = [i for i in insts
                if lab[np.array(by_inst[i])].any() and (~lab[np.array(by_inst[i])]).any()]
        wboots = []
        for _ in range(B):
            pick = rng.choice(len(both), size=len(both), replace=True)
            n2 = d2 = 0.0
            for p in pick:
                idx = np.array(by_inst[both[p]])
                f, r = s[idx][lab[idx]], s[idx][~lab[idx]]
                n2 += auc(f, r) * len(f) * len(r); d2 += len(f) * len(r)
            if d2:
                wboots.append(n2 / d2)
        wlo, whi = np.percentile(wboots, [2.5, 97.5])

        out[d] = dict(auc=float(point), ci=[float(lo), float(hi)],
                      within_instance_auc=float(within),
                      within_instance_ci=[float(wlo), float(whi)], n_pairs=int(den),
                      n_instances_both_classes=n_inst_used)
        print(f"{d:13s} {point:6.3f} [{lo:6.3f}, {hi:6.3f}]{'':>3} "
              f"{within:8.3f} [{wlo:.3f}, {whi:.3f}] {int(den):8d}")

    # C. model-implied vs observed coverage at the realized false-abort rate
    print(f"\n{'detector':13s} {'thr':>8} {'realized FA':>12} {'observed cov':>13} "
          f"{'model cov':>10} {'d-prime':>8}")
    for d in ORDER:
        s = runmax[d]
        thr = float(np.percentile(s[~lab], 95))
        fa = float(np.mean(s[~lab] >= thr))
        cov = float(np.mean(s[lab] >= thr))
        a = out[d]["auc"]
        dp = np.sqrt(2) * Phinv(a)
        model_cov = float(Phi(dp - Phinv(1 - fa))) if 0 < fa < 1 else float("nan")
        out[d].update(threshold=thr, realized_false_abort=fa, observed_coverage=cov,
                      model_implied_coverage=model_cov, d_prime=float(dp))
        print(f"{d:13s} {thr:8.3f} {fa:12.3f} {cov:13.3f} {model_cov:10.3f} {dp:8.2f}")

    best = max(ORDER, key=lambda d: out[d]["auc"])
    print(f"\nbest run-level AUC: {out[best]['auc']:.3f} ({best}), "
          f"d'={out[best]['d_prime']:.2f}, model-implied coverage at 5% false-abort "
          f"{Phi(out[best]['d_prime'] - Phinv(0.95)):.3f}")
    out["_meta"] = dict(n_runs=int(len(lab)), n_failed=int(lab.sum()),
                        n_resolved=int((~lab).sum()), n_instances=len(insts),
                        n_instances_failed=n_if, n_instances_resolved=n_ir,
                        n_instances_both=n_both,
                        bootstrap=B, seed=SEED,
                        best_detector=best,
                        model_cov_at_exactly_5pct=float(Phi(out[best]["d_prime"] - Phinv(0.95))))
    json.dump(out, open(os.path.join(DATA, "p1_mechanism_ci.json"), "w"), indent=1)
    print("\nwrote data/p1_mechanism_ci.json")


if __name__ == "__main__":
    main()
