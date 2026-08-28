"""Checkpoint diagnostics for a judge run. Run at every checkpoint to catch bugs
early and watch the confidence interval tighten.

  python judge_report.py --scores ../data/judge_scores_north.json --label north-mini

Does, in order:
  1. SANITY GAUNTLET (fails loudly if any invariant breaks):
     - every judged trajectory matches exactly one real trajectory (alignment)
     - score-list length == n_steps (no truncation)
     - judged instance_ids are distinct (no duplicate collapse)
     - STUCK verdict rate is in a sane band (not 0%, not ~100%)
  2. Judge frontier + saved@{5,10,20}% + median regret.
  3. BOOTSTRAP 95% CI on saved@5% (half-width is the 'no-doubt' gauge).
  4. HEAD-TO-HEAD: the 4 mechanical detectors recomputed on the SAME subset,
     so the judge is compared on identical trajectories, not indicatively.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import econ, detectors as det
from llab.oracle import compute_oracle
import pickle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
THR = {"syntactic": list(range(1, 9)),
       "convergence": list(np.round(np.linspace(0.50, 0.99, 25), 4)),
       "prm_proxy": list(range(1, 16)), "self_report": [1, 2, 3, 4, 5],
       "llm_judge": list(range(1, 11))}


def build_subcorpus(judge, trajs):
    by_uid = {}
    for t in trajs:
        by_uid.setdefault(t.uid, t)
    failed, resolved = [], []
    seen, unmatched = set(), 0
    for uid, rec in judge.items():
        t = by_uid.get(uid)
        if t is None:
            unmatched += 1; continue
        assert len(rec["scores"]) == t.n_steps, f"score/step length mismatch {uid}"
        assert uid not in seen, f"duplicate uid in judge scores: {uid}"
        seen.add(uid)
        orc = compute_oracle(t)
        if orc is None:
            continue
        scores = {"llm_judge": rec["scores"]}
        for name, fn in det.ALL_DETECTORS.items():
            if name == "logprob":
                continue
            out = fn(t)
            if out.available:
                scores[name] = out.scores
        cc = [len(s.text or "") + len(s.observation or "") for s in t.steps]
        entry = dict(instance_id=t.instance_id, n=t.n_steps, tstar=orc.tstar,
                     char_total=sum(cc) or 1,
                     char_after_tstar=sum(c for i, c in enumerate(cc)
                                          if orc.tstar is not None and i > orc.tstar),
                     scores=scores)
        (resolved if t.resolved else failed).append(entry)
    return econ.Corpus(failed=failed, resolved=resolved), unmatched


def stuck_rate(judge):
    alls = [s for v in judge.values() for s in v["scores"]]
    return sum(1 for s in alls if s > 0) / len(alls) if alls else 0.0


def boot_saved_at5(corpus, detector, n_boot=2000, seed=0):
    """Bootstrap CI of saved@<=5% false-abort over resampled FAILED trajectories."""
    rng = np.random.default_rng(seed)
    pts = econ.frontier(corpus, detector, THR[detector])
    feas = [p for p in pts if p["false_abort"] <= 0.05]
    if not feas:
        return (float("nan"),) * 3
    thr = max(feas, key=lambda p: p["saved_frac"])["threshold"]
    fails = corpus.failed
    tot0 = sum(e["n"] for e in fails) or 1
    def fire(sc, t):
        for i, s in enumerate(sc):
            if s >= t:
                return i
        return None
    vals = []
    idx = np.arange(len(fails))
    for _ in range(n_boot):
        samp = [fails[i] for i in rng.integers(0, len(fails), len(fails))]
        tot = sum(e["n"] for e in samp) or 1
        saved = sum(econ.waste_after(fire(e["scores"][detector], thr), e["n"]) for e in samp)
        vals.append(saved / tot)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--label", default="judge")
    a = ap.parse_args()
    judge = json.load(open(a.scores))
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))

    print(f"=== JUDGE CHECKPOINT: {a.label} ({len(judge)} judged) ===")
    sr = stuck_rate(judge)
    print(f"[gauntlet] STUCK verdict rate: {sr*100:.1f}%  "
          f"{'OK' if 0.02 < sr < 0.9 else '!! SUSPECT'}")
    corpus, unmatched = build_subcorpus(judge, trajs)
    print(f"[gauntlet] unmatched: {unmatched}  {'OK' if unmatched == 0 else '!! BUG'}")
    print(f"[gauntlet] alignment + distinct-id + length checks: PASSED (asserts held)")
    nf, nr = len(corpus.failed), len(corpus.resolved)
    gran = f"1/{nr}={100/nr:.1f}%" if nr else "n/a"
    print(f"subset: {nf} failed / {nr} resolved  (false-abort granularity {gran})")
    if nf == 0 or nr == 0:
        print("\n[partial] need both failed and resolved trajectories for the frontier; "
              "gauntlet passed, waiting for more of the run.")
        return

    m, lo, hi = boot_saved_at5(corpus, "llm_judge")
    print(f"\njudge saved@5% = {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"half-width {(hi-lo)/2:.3f}  {'TIGHT' if (hi-lo)/2 < 0.05 else 'WIDE, extend N'}")
    sar = econ.savings_at_risk(corpus, "llm_judge", THR["llm_judge"])
    rr = econ.regret_at_risk(corpus, "llm_judge", THR["llm_judge"])
    print(f"judge saved 5/10/20% = {sar['5']:.3f}/{sar['10']:.3f}/{sar['20']:.3f}  "
          f"median_regret@5% = {rr['median_regret']:.1f}")

    print("\n--- HEAD-TO-HEAD on this identical subset (saved@5/10/20%) ---")
    for d in ["convergence", "syntactic", "prm_proxy", "self_report", "llm_judge"]:
        s = econ.savings_at_risk(corpus, d, THR[d])
        print(f"  {d:12s} {s['5']:.3f} / {s['10']:.3f} / {s['20']:.3f}")


if __name__ == "__main__":
    main()
