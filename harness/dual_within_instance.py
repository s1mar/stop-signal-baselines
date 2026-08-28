"""Within-instance separability, measured properly on a difficulty-matched corpus.

The paper reports within-instance AUC on 7 instances and 576 pairs, with intervals so wide that it
can only claim a direction. `build_dual_corpus.py` assembles a corpus where every instance carries
both outcomes, which lets the same quantity be estimated with real power.

The comparison this settles: how much of the pooled run-level separation (0.55 to 0.62 AUC) is the
detector recognizing a stalled RUN, and how much is it recognizing a hard TASK? Pooled AUC mixes
both. Within-instance AUC holds the task fixed, so it isolates the part a deployed detector could
actually use when it cannot choose its tasks.

Writes data/dual_within_instance.json.
"""
from __future__ import annotations
import collections, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import detectors as det, embed

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ORDER = ["syntactic", "convergence", "prm_proxy", "self_report"]
B = 2000
SEED = 20260724


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def run_max(scores):
    s = [x for x in scores if x is not None]
    return max(s) if s else 0.0


def main():
    trajs = pickle.load(open(os.path.join(DATA, "dual_corpus.pkl"), "rb"))
    print(f"loaded {len(trajs)} trajectories")

    rows = []
    for k, t in enumerate(trajs):
        outs = {"syntactic": det.syntactic_repetition(t), "convergence": det.convergence_monitor(t),
                "prm_proxy": det.prm_proxy(t), "self_report": det.self_report(t)}
        rows.append(dict(iid=t.instance_id, resolved=bool(t.resolved), n=t.n_steps,
                         maxes={d: run_max(o.scores) for d, o in outs.items()}))
        if (k + 1) % 100 == 0:
            print(f"  scored {k+1}/{len(trajs)}", flush=True)

    by_inst = collections.defaultdict(list)
    for r in rows:
        by_inst[r["iid"]].append(r)
    both = [i for i, v in by_inst.items()
            if any(not r["resolved"] for r in v) and any(r["resolved"] for r in v)]
    pairs = sum(sum(not r["resolved"] for r in by_inst[i]) * sum(r["resolved"] for r in by_inst[i])
                for i in both)
    print(f"\n{len(by_inst)} instances, {len(both)} with both outcomes, {pairs} within-instance pairs")

    rng = np.random.default_rng(SEED)
    out = {"_meta": dict(n_trajectories=len(rows), n_instances=len(by_inst),
                         n_instances_both=len(both), n_pairs=int(pairs),
                         n_failed=sum(not r["resolved"] for r in rows),
                         n_resolved=sum(r["resolved"] for r in rows), bootstrap=B, seed=SEED)}

    print(f"\n{'detector':14s} {'pooled AUC':>11} {'within-instance AUC':>21} {'delta':>7}")
    for d in ORDER:
        f = [r["maxes"][d] for r in rows if not r["resolved"]]
        s = [r["maxes"][d] for r in rows if r["resolved"]]
        pooled = auc(f, s)

        num = den = 0.0
        for i in both:
            v = by_inst[i]
            fi = [r["maxes"][d] for r in v if not r["resolved"]]
            ri = [r["maxes"][d] for r in v if r["resolved"]]
            num += auc(fi, ri) * len(fi) * len(ri)
            den += len(fi) * len(ri)
        within = num / den

        boots = []
        for _ in range(B):
            pick = rng.choice(len(both), size=len(both), replace=True)
            n2 = d2 = 0.0
            for p in pick:
                v = by_inst[both[p]]
                fi = [r["maxes"][d] for r in v if not r["resolved"]]
                ri = [r["maxes"][d] for r in v if r["resolved"]]
                n2 += auc(fi, ri) * len(fi) * len(ri); d2 += len(fi) * len(ri)
            if d2:
                boots.append(n2 / d2)
        lo, hi = np.percentile(boots, [2.5, 97.5])

        out[d] = dict(pooled_auc=pooled, within_instance_auc=within,
                      within_ci=[float(lo), float(hi)], delta=float(within - pooled))
        print(f"{d:14s} {pooled:11.3f} {within:11.3f} [{lo:.3f}, {hi:.3f}] {within-pooled:+7.3f}")

    json.dump(out, open(os.path.join(DATA, "dual_within_instance.json"), "w"), indent=1)
    print("\nwrote data/dual_within_instance.json")


if __name__ == "__main__":
    main()
