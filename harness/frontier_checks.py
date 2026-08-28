"""Independent blind-spot checks on the frontier RWR: (1) is post-t* thrashing or
exploration for frontier models? (2) bootstrap CIs + Kruskal-Wallis across models
(is the difference even significant?); (3) intersection absolute step counts
(is Claude-4's low RWR premature halting, not better stopping?)."""
import json, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from llab import ingest_traj, embed
from llab.oracle import compute_oracle
from run_frontier import MODELS, is_truncated

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def meanpair(V):
    if len(V) < 2:
        return np.nan
    s = [float(np.dot(V[a], V[b])) for a in range(len(V)) for b in range(a + 1, len(V))]
    return float(np.mean(s))


def analyze():
    per_model = {}
    for m in MODELS:
        trajs = ingest_traj.ingest_submission(m, max_failed=150, max_resolved=0)
        rows = []
        for t in trajs:
            if t.resolved:
                continue
            orc = compute_oracle(t)
            if orc is None or orc.tstar is None:
                continue
            # thrashing: pre vs post t* action similarity
            texts = [s.text or "" for s in t.steps]
            pre_sim = post_sim = np.nan
            if len(texts) >= 4:
                V = embed.encode(texts)
                pre = [V[i] for i in range(len(texts)) if i <= orc.tstar]
                post = [V[i] for i in range(len(texts)) if i > orc.tstar]
                if len(pre) >= 2 and len(post) >= 2:
                    pre_sim, post_sim = meanpair(pre), meanpair(post)
            rows.append(dict(iid=t.instance_id, n=t.n_steps, tstar=orc.tstar,
                             truncated=is_truncated(t.exit_status),
                             wf=(t.n_steps - 1 - orc.tstar) / (t.n_steps - 1) if t.n_steps > 1 else 0,
                             pre_sim=pre_sim, post_sim=post_sim))
        per_model[m] = rows
        embed.save_cache()

    print("\n=== (1) THRASHING vs EXPLORATION at the frontier (non-truncated failed) ===")
    for m in MODELS:
        r = [x for x in per_model[m] if not x["truncated"] and not np.isnan(x["post_sim"])]
        if not r:
            print(f"  {ingest_traj.SUBMISSIONS[m][1]}: no data"); continue
        pre = np.median([x["pre_sim"] for x in r]); post = np.median([x["post_sim"] for x in r])
        frac = np.mean([x["post_sim"] > x["pre_sim"] for x in r])
        print(f"  {ingest_traj.SUBMISSIONS[m][1]:20s} pre={pre:.3f} post={post:.3f} "
              f"post>pre in {frac*100:.0f}%  (n={len(r)})  "
              f"{'THRASHING' if post>pre else 'exploration?'}")

    print("\n=== (2) RWR CIs + across-model test (non-truncated failed) ===")
    boot = {}
    for m in MODELS:
        r = [x for x in per_model[m] if not x["truncated"]]
        ns = np.array([x["n"] for x in r], float)
        rec = np.array([x["n"] - 1 - x["tstar"] for x in r], float)
        rng = np.random.default_rng(0)
        vals = [rec[idx].sum() / ns[idx].sum() for idx in
                (rng.integers(0, len(ns), len(ns)) for _ in range(2000))] if len(ns) else []
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
        rwr = rec.sum() / ns.sum() if ns.sum() else np.nan
        boot[m] = [x["wf"] for x in r]
        print(f"  {ingest_traj.SUBMISSIONS[m][1]:20s} RWR={rwr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(r)}")
    groups = [boot[m] for m in MODELS if len(boot[m]) > 1]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis across models: H={H:.2f}, p={p:.3f}  "
              f"{'DIFFER' if p < 0.05 else 'NOT significantly different'}")

    print("\n=== (3) Intersection: absolute steps (is low RWR premature halting?) ===")
    failed_sets = {m: set(x["iid"] for x in per_model[m]) for m in MODELS}
    inter = set.intersection(*failed_sets.values()) if failed_sets else set()
    print(f"  intersection n={len(inter)}")
    for m in MODELS:
        r = [x for x in per_model[m] if x["iid"] in inter]
        if not r:
            continue
        print(f"  {ingest_traj.SUBMISSIONS[m][1]:20s} median total steps={np.median([x['n'] for x in r]):.0f} "
              f"median t*={np.median([x['tstar'] for x in r]):.0f} "
              f"median waste-frac={np.median([x['wf'] for x in r]):.3f}")


if __name__ == "__main__":
    analyze()
