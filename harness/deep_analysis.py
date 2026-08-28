"""Items 1-3 of the panel plan, on cached data, no new inference.

1. STRATIFIED FRONTIER. The paper compares an aggregate saved fraction against a
   stratum-specific PPR. Recompute every detector separately on the failed runs that
   reached the reference patch (max progress > 0) and those that never did, holding the
   resolved set fixed so false-abort is estimated on the same denominator throughout.
2. EVENT STUDY around t*. Coverage (does the detector ever fire on this run at the
   budget-feasible threshold?) and the alarm hazard in a window on t*, which separates a
   calibration problem from a blindness problem.
3. CLUSTERED BOOTSTRAP. Resample TASK INSTANCES, not trajectories, and reselect the
   threshold inside each replicate so the reported interval covers operating-point
   selection rather than conditioning on a full-data choice.

Writes data/deep_analysis.json.
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import detectors as det, econ, embed
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS = os.path.join(DATA, "corpus_strat.json")
THRESHOLDS = {
    "syntactic": [1, 2, 3, 4, 5, 6, 7, 8],
    "convergence": list(np.round(np.linspace(0.50, 0.99, 25), 4)),
    "prm_proxy": list(range(1, 16)),
    "self_report": [1, 2, 3, 4, 5],
}
ORDER = ["convergence", "syntactic", "prm_proxy", "self_report"]
BUDGET = 0.05


# ---------------------------------------------------------------- corpus
def build():
    if os.path.exists(CORPUS):
        return json.load(open(CORPUS, encoding="utf-8"))
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))
    rows = []
    for t in trajs:
        orc = compute_oracle(t)
        if orc is None or orc.tstar is None:
            continue
        sc = {}
        for name in THRESHOLDS:
            out = det.ALL_DETECTORS[name](t)
            sc[name] = [float(x) for x in out.scores] if out.available else None
        rows.append(dict(uid=t.uid, iid=t.instance_id, resolved=bool(t.resolved),
                         n=int(t.n_steps), tstar=int(orc.tstar),
                         maxprog=float(orc.max_progress), scores=sc))
    embed.save_cache()
    json.dump(rows, open(CORPUS, "w", encoding="utf-8"))
    return rows


def fire(scores, thr):
    if scores is None:
        return None
    for i, s in enumerate(scores):
        if s >= thr:
            return i
    return None


def op_point(failed, resolved, d, budget=BUDGET):
    """Best saved fraction subject to false-abort <= budget. Returns (saved, thr, fa)."""
    nR = len(resolved) or 1
    tot = sum(e["n"] for e in failed) or 1
    best = (0.0, None, 0.0)
    for thr in THRESHOLDS[d]:
        fa = sum(1 for e in resolved
                 if (lambda f: f is not None and f < e["n"] - 1)(fire(e["scores"][d], thr)))
        if fa / nR > budget:
            continue
        sv = sum(econ.waste_after(fire(e["scores"][d], thr), e["n"]) for e in failed)
        if sv / tot > best[0]:
            best = (sv / tot, thr, fa / nR)
    return best


def med_regret(failed, d, thr):
    r = [fire(e["scores"][d], thr) - e["tstar"] for e in failed
         if fire(e["scores"][d], thr) is not None]
    return float(np.median(r)) if r else float("nan")


# ---------------------------------------------------------------- 3. clustered bootstrap
def clustered_ci(failed, resolved, d, n_boot=1000, seed=0):
    """Resample task INSTANCES; reselect the threshold inside each replicate."""
    rng = np.random.default_rng(seed)
    by_iid = {}
    for e in failed + resolved:
        by_iid.setdefault(e["iid"], []).append(e)
    iids = list(by_iid)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(iids), len(iids))
        F, R = [], []
        for k in pick:
            for e in by_iid[iids[k]]:
                (R if e["resolved"] else F).append(e)
        if not F or not R:
            continue
        vals.append(op_point(F, R, d)[0])
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    rows = build()
    resolved = [e for e in rows if e["resolved"]]
    failed = [e for e in rows if not e["resolved"]]
    reached = [e for e in failed if e["maxprog"] > 0]
    never = [e for e in failed if e["maxprog"] == 0]
    n_iid = len({e["iid"] for e in rows})
    print(f"corpus: {len(rows)} trajectories over {n_iid} unique task instances")
    print(f"failed {len(failed)} = reached {len(reached)} + never {len(never)}; "
          f"resolved {len(resolved)}\n")

    out = {"n_instances": n_iid, "strata": {}, "event": {}, "ci": {}}

    # ---- 1. stratified frontier -------------------------------------------------
    print("=== 1. STRATIFIED FRONTIER (resolved set held fixed, false-abort <= 5%) ===")
    print(f"{'detector':14s} {'all-failed':>11} {'reached(212)':>13} {'never(205)':>11} "
          f"{'regret|all':>11}")
    for d in ORDER:
        sa, ta, _ = op_point(failed, resolved, d)
        sr, _, _ = op_point(reached, resolved, d)
        sn, _, _ = op_point(never, resolved, d)
        rg = med_regret(failed, d, ta) if ta is not None else float("nan")
        out["strata"][d] = dict(all=sa, reached=sr, never=sn, thr=ta, regret=rg)
        print(f"{d:14s} {sa:11.3f} {sr:13.3f} {sn:11.3f} {rg:11.1f}")
    ppr = lambda g: sum(econ.waste_after(e["tstar"], e["n"]) for e in g) / sum(e["n"] for e in g)
    print(f"{'PPR reference':14s} {ppr(failed):11.3f} {ppr(reached):13.3f} {ppr(never):11.3f}")
    out["ppr"] = dict(all=ppr(failed), reached=ppr(reached), never=ppr(never))

    # ---- 2. event study ---------------------------------------------------------
    print("\n=== 2. EVENT STUDY: coverage and alarm hazard around t* ===")
    print(f"{'detector':14s} {'coverage':>9} {'cov|reached':>12}   alarm-active rate at t*+k")
    offs = [-3, -1, 0, 1, 2, 3, 5, 10]
    print(f"{'':14s} {'':>9} {'':>12}   " + " ".join(f"{k:+d}".rjust(6) for k in offs))
    for d in ORDER:
        thr = out["strata"][d]["thr"]
        if thr is None:
            continue
        cov = np.mean([fire(e["scores"][d], thr) is not None for e in failed])
        covr = np.mean([fire(e["scores"][d], thr) is not None for e in reached])
        hz = []
        for k in offs:
            num = den = 0
            for e in reached:
                i = e["tstar"] + k
                if 0 <= i < e["n"] and e["scores"][d] is not None:
                    den += 1
                    num += int(e["scores"][d][i] >= thr)
            hz.append(num / den if den else float("nan"))
        out["event"][d] = dict(coverage=float(cov), coverage_reached=float(covr),
                               offsets=offs, hazard=[float(x) for x in hz])
        print(f"{d:14s} {cov:9.3f} {covr:12.3f}   " + " ".join(f"{x:6.3f}" for x in hz))

    # ---- 3. clustered CIs -------------------------------------------------------
    print("\n=== 3. CLUSTERED BOOTSTRAP (by task instance, threshold reselected) ===")
    print(f"{'detector':14s} {'saved@5%':>9}  {'95% CI (clustered)':>24}")
    for d in ORDER:
        lo, hi = clustered_ci(failed, resolved, d)
        out["ci"][d] = [lo, hi]
        print(f"{d:14s} {out['strata'][d]['all']:9.3f}  [{lo:.3f}, {hi:.3f}]")

    json.dump(out, open(os.path.join(DATA, "deep_analysis.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(DATA, 'deep_analysis.json')}")


if __name__ == "__main__":
    main()
