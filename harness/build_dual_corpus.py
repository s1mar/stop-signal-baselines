"""Build a DIFFICULTY-MATCHED corpus: many runs per task instance, both outcomes present.

Why this exists. The paper's mechanism (run-level separation of doomed from recoverable runs) is
confounded with task difficulty, because the original sample has only 7 instances carrying both
outcomes. That was assumed to be a property of the corpus. It is not: `probe_dual_outcome.py`
found 348 dual-outcome instances and 35,083 within-instance pairs in the first 40k rows alone. The
original sampler simply never looked for pairs, filling a failed quota and a resolved quota
independently and stopping.

With a matched corpus, within-instance AUC can be measured properly instead of estimated on 7
clusters, which is the difference between "we cannot separate stagnation from task difficulty" and
a clean answer either way.

Selection is declared here before any result is computed:
  * instances are ranked by min(failed, resolved), so we take those most able to support a
    within-instance comparison, not those with the most extreme outcome ratio;
  * per instance we keep at most CAP_PER_CLASS runs of each outcome, so no single easy or hard
    instance dominates the pooled statistic;
  * runs need >= MIN_STEPS steps, the same long-horizon scope as the main corpus.

Run: python build_dual_corpus.py --instances 60 --cap 8
"""
from __future__ import annotations
import argparse, collections, itertools, json, os, pickle, sys

sys.path.insert(0, os.path.dirname(__file__))
from datasets import load_dataset
from llab import ingest

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
NEBIUS = "nebius/SWE-agent-trajectories"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=60)
    ap.add_argument("--cap", type=int, default=8, help="max runs per outcome per instance")
    ap.add_argument("--min-steps", type=int, default=15)
    ap.add_argument("--scan", type=int, default=40000)
    ap.add_argument("--out", default=os.path.join(DATA, "dual_corpus.pkl"))
    a = ap.parse_args()

    probe = json.load(open(os.path.join(DATA, "dual_outcome_probe.json"), encoding="utf-8"))
    cand = probe["dual_outcome_instances"]
    # rank by the number of within-instance pairs the cap can actually realise
    ranked = sorted(cand.items(),
                    key=lambda kv: -min(kv[1]["failed"], a.cap) * min(kv[1]["resolved"], a.cap))
    want = {k for k, _ in ranked[:a.instances]}
    print(f"targeting {len(want)} instances, cap {a.cap} per outcome, >= {a.min_steps} steps")

    ds = load_dataset(NEBIUS, split="train", streaming=True)
    kept = collections.defaultdict(lambda: {"failed": [], "resolved": []})
    n_seen = 0
    for row in itertools.islice(ds, a.scan):
        n_seen += 1
        iid = row.get("instance_id", "")
        if iid not in want:
            continue
        traj = row.get("trajectory")
        if not isinstance(traj, list):
            continue
        if sum(1 for s in traj if s.get("role") == "ai") < a.min_steps:
            continue
        slot = "resolved" if row.get("target") else "failed"
        if len(kept[iid][slot]) >= a.cap:
            continue
        kept[iid][slot].append(ingest.normalize(row))
        if n_seen % 10000 == 0:
            tot = sum(len(v["failed"]) + len(v["resolved"]) for v in kept.values())
            print(f"  scanned {n_seen:6d}  collected {tot:5d}", flush=True)

    # keep only instances that still carry both outcomes after the cap
    trajs = []
    n_inst = 0
    for iid, v in kept.items():
        if v["failed"] and v["resolved"]:
            n_inst += 1
            trajs.extend(v["failed"] + v["resolved"])
    print(f"\n{n_inst} instances retained, {len(trajs)} trajectories "
          f"({sum(not t.resolved for t in trajs)} failed / {sum(t.resolved for t in trajs)} resolved)")

    gold = ingest.build_gold_index({t.instance_id for t in trajs})
    out = []
    for t in trajs:
        if t.instance_id in gold:
            t.gold_patch = gold[t.instance_id]
            out.append(t)
    inst_final = {t.instance_id for t in out}
    both = sum(1 for i in inst_final
               if any(t.instance_id == i and not t.resolved for t in out)
               and any(t.instance_id == i and t.resolved for t in out))
    pairs = 0
    for i in inst_final:
        f = sum(1 for t in out if t.instance_id == i and not t.resolved)
        r = sum(1 for t in out if t.instance_id == i and t.resolved)
        pairs += f * r
    print(f"after gold join: {len(out)} trajectories, {len(inst_final)} instances, "
          f"{both} with both outcomes, {pairs} within-instance pairs")

    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print(f"pickled -> {os.path.normpath(a.out)}")


if __name__ == "__main__":
    main()
