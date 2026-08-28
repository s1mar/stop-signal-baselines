"""Full analysis of the modern-model (Qwen3-Coder-480B, OpenHands, SWE-rebench)
replay: RWR + bootstrap CI + oracle AUC + the thrashing-vs-exploration check.
A cross-scaffold, cross-taskset MODERN data point (heavily confounded for direct
magnitude comparison, but tells us the phenomenon's shape at the current frontier).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import ingest_openhands as oh, embed
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def meanpair(V):
    if len(V) < 2:
        return np.nan
    return float(np.mean([float(np.dot(V[a], V[b]))
                          for a in range(len(V)) for b in range(a + 1, len(V))]))


def run():
    trajs = oh.ingest(max_failed=150, max_resolved=90, scan_cap=6000)
    rows, res_mp, fail_mp = [], [], []
    for t in trajs:
        orc = compute_oracle(t)
        if orc is None:
            continue
        (res_mp if t.resolved else fail_mp).append(orc.max_progress)
        if t.resolved or orc.tstar is None:
            continue
        texts = [s.text or "" for s in t.steps]
        pre_sim = post_sim = np.nan
        if len(texts) >= 4:
            V = embed.encode(texts)
            pre = [V[i] for i in range(len(texts)) if i <= orc.tstar]
            post = [V[i] for i in range(len(texts)) if i > orc.tstar]
            if len(pre) >= 2 and len(post) >= 2:
                pre_sim, post_sim = meanpair(pre), meanpair(post)
        rows.append(dict(n=t.n_steps, tstar=orc.tstar, exit=t.exit_status,
                         pre_sim=pre_sim, post_sim=post_sim))
    embed.save_cache()

    ns = np.array([r["n"] for r in rows], float)
    rec = np.array([r["n"] - 1 - r["tstar"] for r in rows], float)
    rwr = rec.sum() / ns.sum() if ns.sum() else float("nan")
    rng = np.random.default_rng(0)
    boot = [rec[i].sum() / ns[i].sum() for i in (rng.integers(0, len(ns), len(ns)) for _ in range(2000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    def auc(pos, neg):
        return (sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
                if pos and neg else float("nan"))

    thr = [r for r in rows if not np.isnan(r["post_sim"])]
    pre = np.median([r["pre_sim"] for r in thr]); post = np.median([r["post_sim"] for r in thr])
    frac = np.mean([r["post_sim"] > r["pre_sim"] for r in thr])
    from collections import Counter
    print(f"\n=== Qwen3-Coder-480B (OpenHands / SWE-rebench) ===")
    print(f"failed n={len(rows)} | exit_status: {dict(Counter(r['exit'] for r in rows))}")
    print(f"RWR = {rwr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"oracle AUC (resolved vs failed max-progress) = {auc(res_mp, fail_mp):.3f} "
          f"(res n={len(res_mp)}, fail n={len(fail_mp)})")
    print(f"thrashing: pre-t* sim={pre:.3f} post-t* sim={post:.3f} post>pre in {frac*100:.0f}% "
          f"-> {'THRASHING' if post > pre else 'EXPLORATION'}  (n={len(thr)})")
    json.dump(dict(model="Qwen3-Coder-480B", rwr=rwr, ci=[lo, hi], n_failed=len(rows),
                   oracle_auc=auc(res_mp, fail_mp), pre_sim=pre, post_sim=post,
                   post_gt_pre=float(frac)), open(os.path.join(DATA, "openhands_qwen.json"), "w"),
              indent=1, default=float)
    print("wrote data/openhands_qwen.json")


if __name__ == "__main__":
    run()
