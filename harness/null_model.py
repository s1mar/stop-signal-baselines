"""Null-model control for the Post-Peak Ratio (reviewer ask, 2026-07-23).

Concern: t* is "the last edit step that set a new maximum of a progress signal", and
PPR = (n-1-t*)/n. Even for a signal with NO genuine early-peaking structure, a last-maximum
statistic tends to land before the end, so PPR > 0 mechanically. How much of the observed PPR
(0.754 overall, 0.552 reached-reference) exceeds that mechanical expectation?

Null: for each trajectory keep n, the edit-step positions, and the MULTISET of per-step
progress values, but randomly PERMUTE which edit carries which value. This destroys the
temporal "peak early then decline" structure while preserving everything mechanical. Recompute
t* and PPR on the permuted signal. If observed >> null, early-peaking is a real property of
agents; if observed ~ null, PPR is a property of the estimator.

Reports the length-weighted corpus PPR observed vs the permutation-null distribution, per
stratum (all failed / reached-reference / zero-overlap), with a permutation p-style CI.
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab.oracle import compute_oracle, _last_new_max

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
R = 1000  # permutations


def main():
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))
    rng = np.random.default_rng(0)
    rows = []
    for t in trajs:
        if t.resolved:
            continue
        o = compute_oracle(t)
        if o is None or o.tstar is None or not o.prog_primary:
            continue
        prog = np.array(o.prog_primary, dtype=float)
        idx = np.array(o.edit_idx)
        n = t.n_steps
        obs_waste = n - 1 - o.tstar
        # R permutation-null wastes for this trajectory
        nw = np.empty(R)
        for r in range(R):
            tp = _last_new_max(list(rng.permutation(prog)))
            tstar_null = int(idx[tp]) if tp is not None else 0
            nw[r] = max(0, n - 1 - tstar_null)
        rows.append(dict(n=n, maxprog=o.max_progress, obs_waste=obs_waste, null_wastes=nw))

    def report(g, label):
        if not g:
            print(f"{label:20s} (empty)"); return None
        totn = sum(r["n"] for r in g)
        obs = sum(r["obs_waste"] for r in g) / totn
        # length-weighted corpus null PPR for each permutation draw
        null_draws = np.array([sum(r["null_wastes"][k] for r in g) for k in range(R)]) / totn
        lo, hi = np.percentile(null_draws, [2.5, 97.5])
        nm = null_draws.mean()
        print(f"{label:20s} n={len(g):3d}  observed {obs:.3f}   null {nm:.3f} "
              f"[{lo:.3f}, {hi:.3f}]   excess {obs-nm:+.3f}")
        return dict(n=len(g), observed=float(obs), null_mean=float(nm),
                    null_ci=[float(lo), float(hi)], excess=float(obs - nm))

    print(f"permutations per trajectory: {R}\n")
    out = {}
    out["all"] = report(rows, "all failed")
    out["reached"] = report([r for r in rows if r["maxprog"] > 0], "reached-reference")
    out["zero"] = report([r for r in rows if r["maxprog"] == 0], "zero-overlap")
    json.dump(out, open(os.path.join(DATA, "null_model.json"), "w"), indent=1)
    print("\nwrote null_model.json")


if __name__ == "__main__":
    main()
