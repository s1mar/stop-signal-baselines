"""CONFIRMATORY transfer test of the coverage/separability diagnosis.

Pre-registered in notes/FROZEN_SPEC_transfer.md (written before any result here was computed).
Runs the FROZEN detector suite, unchanged, on all four held-out configurations and reports, per
configuration: the savings-vs-false-abort frontier, coverage at each budget, fraction active at t*,
unconstrained coverage, median Oracle Regret, oracle separation AUC, instance-clustered CIs, and n.

Also reports the two pre-specified extras:
  * budget sensitivity across {2,5,10,20,50,100}%
  * threshold TRANSFER: thresholds fitted on the original corpus applied unchanged here.

Usage:  python transfer_test.py            (all four configs)
        python transfer_test.py --only gpt-4o
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import detectors as det, econ, embed, ingest_traj
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(DATA, "transfer_cache")
BUDGETS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
THRESHOLDS = {
    "syntactic": [1, 2, 3, 4, 5, 6, 7, 8],
    "convergence": list(np.round(np.linspace(0.50, 0.99, 25), 4)),
    "prm_proxy": list(range(1, 16)),
    "self_report": [1, 2, 3, 4, 5],
}
ORDER = ["convergence", "syntactic", "prm_proxy", "self_report"]
SWEAGENT = ["claude-3.5-sonnet", "gpt-4o", "claude-4-sonnet"]


def fire(scores, thr):
    if not scores:
        return None
    for i, s in enumerate(scores):
        if s >= thr:
            return i
    return None


MIN_STEPS = 15          # same long-horizon floor as the development corpus


def build_rows(trajs):
    """-> list of per-trajectory dicts with n, tstar, resolved, iid, maxprog, scores.

    The MIN_STEPS floor matters and was missing. The development corpus is built by
    ingest.sample_trajectories(min_steps=15), and every claim in the paper is scoped to the
    long-horizon regime, but ingest_submission only drops runs shorter than 3 steps. Without this
    filter the held-out coverage figures were computed on a different population from the one they
    are compared against, so the replication compared unlike with unlike.
    """
    rows = []
    for t in trajs:
        if t.n_steps < MIN_STEPS:
            continue
        o = compute_oracle(t)
        if o is None or o.tstar is None:
            continue
        sc = {}
        for name in THRESHOLDS:
            out = det.ALL_DETECTORS[name](t)
            sc[name] = [float(x) for x in out.scores] if out.available else None
        rows.append(dict(uid=getattr(t, "uid", t.instance_id), iid=t.instance_id,
                         resolved=bool(t.resolved), n=int(t.n_steps), tstar=int(o.tstar),
                         maxprog=float(o.max_progress), scores=sc))
    embed.save_cache()
    return rows


def op_point(failed, resolved, d, budget):
    """Best saved fraction subject to false-abort <= budget. -> (saved, thr, fa, coverage)."""
    nR = len(resolved) or 1
    tot = sum(e["n"] for e in failed) or 1
    best = (0.0, None, 0.0, 0.0)
    for thr in THRESHOLDS[d]:
        fa = sum(1 for e in resolved
                 if (lambda f: f is not None and f < e["n"] - 1)(fire(e["scores"][d], thr)))
        if fa / nR > budget:
            continue
        sv = sum(econ.waste_after(fire(e["scores"][d], thr), e["n"]) for e in failed)
        cov = np.mean([fire(e["scores"][d], thr) is not None for e in failed]) if failed else 0.0
        if sv / tot > best[0]:
            best = (sv / tot, thr, fa / nR, float(cov))
    return best


def active_at_tstar(failed, d, thr):
    num = den = 0
    for e in failed:
        sc = e["scores"][d]
        if sc is None:
            continue
        i = e["tstar"]
        if 0 <= i < len(sc):
            den += 1
            num += int(sc[i] >= thr)
    return num / den if den else float("nan")


def med_regret(failed, d, thr):
    r = [fire(e["scores"][d], thr) - e["tstar"] for e in failed
         if fire(e["scores"][d], thr) is not None]
    return float(np.median(r)) if r else float("nan")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def clustered_ci(failed, resolved, d, budget, n_boot=400, seed=0):
    """Instance-clustered bootstrap CIs for BOTH the saved fraction and coverage.

    Coverage was previously left without an interval, which made the held-out replication read as a
    table of point estimates on 42 to 70 resolved runs. Coverage is the quantity the replication
    claim rests on, so it is the one that most needs an interval; op_point already returns it, so
    this costs nothing beyond collecting a second column from the same resamples. The operating
    threshold is reselected inside each replicate, as on the development corpus.
    """
    rng = np.random.default_rng(seed)
    by = {}
    for e in failed + resolved:
        by.setdefault(e["iid"], []).append(e)
    iids = list(by)
    sv, cv = [], []
    for _ in range(n_boot):
        F, R = [], []
        for k in rng.integers(0, len(iids), len(iids)):
            for e in by[iids[k]]:
                (R if e["resolved"] else F).append(e)
        if F and R:
            p = op_point(F, R, d, budget)
            sv.append(p[0]); cv.append(p[3])
    if not sv:
        nan = (float("nan"), float("nan"))
        return nan, nan
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(sv), q(cv)


def analyze(label, rows, orig_thr=None):
    failed = [e for e in rows if not e["resolved"]]
    resolved = [e for e in rows if e["resolved"]]
    n_inst = len({e["iid"] for e in rows})
    print(f"\n===== {label} =====")
    print(f"n: {len(failed)} failed / {len(resolved)} resolved over {n_inst} instances")
    if not failed or not resolved:
        print("  insufficient data; reported as such, not dropped.")
        return dict(label=label, n_failed=len(failed), n_resolved=len(resolved),
                    n_instances=n_inst, usable=False)
    pos = [e["maxprog"] for e in resolved]
    neg = [e["maxprog"] for e in failed]
    a = auc(pos, neg)
    print(f"oracle separation AUC: {a:.3f}")
    out = dict(label=label, n_failed=len(failed), n_resolved=len(resolved),
               n_instances=n_inst, usable=True, auc=float(a), detectors={})
    print(f"{'detector':14s} {'budget':>7} {'saved':>7} {'cover':>7} {'act@t*':>7} {'regret':>7}")
    for d in ORDER:
        rec = dict(budget={}, transfer=None)
        for b in BUDGETS:
            sv, thr, fa, cov = op_point(failed, resolved, d, b)
            at = active_at_tstar(failed, d, thr) if thr is not None else float("nan")
            rg = med_regret(failed, d, thr) if thr is not None else float("nan")
            rec["budget"][str(b)] = dict(saved=sv, thr=thr, false_abort=fa,
                                         coverage=cov, active_at_tstar=float(at),
                                         median_regret=rg)
            print(f"{d if b==BUDGETS[0] else '':14s} {b:7.2f} {sv:7.3f} {cov:7.3f} "
                  f"{at:7.3f} {rg:7.1f}")
        (lo, hi), (clo, chi) = clustered_ci(failed, resolved, d, 0.05)
        rec["ci95_saved_at_5pct"] = [lo, hi]
        rec["ci95_coverage_at_5pct"] = [clo, chi]
        print(f"{'':14s} {'CI@5%':>7} saved [{lo:.3f}, {hi:.3f}]  cov [{clo:.3f}, {chi:.3f}]")
        # threshold transfer from the original corpus
        if orig_thr and orig_thr.get(d) is not None:
            t0 = orig_thr[d]
            fa = np.mean([(lambda f: f is not None and f < e["n"] - 1)(fire(e["scores"][d], t0))
                          for e in resolved])
            cov = np.mean([fire(e["scores"][d], t0) is not None for e in failed])
            sv = sum(econ.waste_after(fire(e["scores"][d], t0), e["n"]) for e in failed) / \
                 (sum(e["n"] for e in failed) or 1)
            rec["transfer"] = dict(orig_threshold=t0, false_abort=float(fa),
                                   coverage=float(cov), saved=float(sv))
            print(f"{'':14s} {'xfer':>7} thr={t0} -> fa {fa:.3f}, cover {cov:.3f}, saved {sv:.3f}")
        out["detectors"][d] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-failed", type=int, default=120)
    ap.add_argument("--max-resolved", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    # thresholds chosen on the ORIGINAL corpus (for the transfer analysis)
    orig_thr = {}
    da = os.path.join(DATA, "deep_analysis.json")
    if os.path.exists(da):
        j = json.load(open(da, encoding="utf-8"))
        for d in ORDER:
            orig_thr[d] = j.get("strata", {}).get(d, {}).get("thr")
        print(f"original-corpus thresholds for transfer: {orig_thr}")

    configs = SWEAGENT + ["qwen3-openhands"]
    if a.only:
        configs = [a.only]
    results = []
    for label in configs:
        cpath = os.path.join(CACHE, f"{label}.pkl")
        if os.path.exists(cpath):
            rows = pickle.load(open(cpath, "rb"))
            print(f"[{label}] loaded {len(rows)} cached rows")
            # Enforce the floor HERE as well as in build_rows. Caches predating the filter would
            # otherwise bypass it silently and forever: adding the filter to build_rows alone
            # changed nothing on the first re-run because every config loaded pre-floor rows, and
            # the unchanged numbers looked like robustness rather than a check that never ran.
            n_before = len(rows)
            rows = [r for r in rows if r["n"] >= MIN_STEPS]
            if len(rows) != n_before:
                print(f"[{label}] applied the >= {MIN_STEPS}-step floor: "
                      f"{n_before} -> {len(rows)} rows")
        else:
            print(f"[{label}] ingesting ...", flush=True)
            try:
                if label == "qwen3-openhands":
                    from llab import ingest_openhands
                    trajs = ingest_openhands.ingest(max_failed=a.max_failed,
                                                    max_resolved=a.max_resolved)
                else:
                    trajs = ingest_traj.ingest_submission(label, max_failed=a.max_failed,
                                                          max_resolved=a.max_resolved)
                rows = build_rows(trajs)
                pickle.dump(rows, open(cpath, "wb"))
                print(f"[{label}] built {len(rows)} rows")
            except Exception as e:
                print(f"[{label}] INGEST FAILED: {type(e).__name__}: {e}")
                results.append(dict(label=label, usable=False, error=f"{type(e).__name__}: {e}"))
                continue
        results.append(analyze(label, rows, orig_thr))
    json.dump(results, open(os.path.join(DATA, "transfer_test.json"), "w"), indent=1)
    print("\nwrote data/transfer_test.json")


if __name__ == "__main__":
    main()
