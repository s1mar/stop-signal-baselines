"""How many task instances in the public corpus have BOTH a failed and a resolved long run?

The paper's mechanism result is confounded with task difficulty because only 7 of its 80 instances
contribute runs of both outcomes. That was attributed to the corpus. It may instead be an artifact
of OUR sampler, which fills a failed quota and a resolved quota independently and stops, never
trying to pair outcomes within an instance.

This probe answers the question before anyone spends effort generating new agent runs: stream the
corpus, bucket long-enough runs by instance_id, and count how many instances carry both outcomes.
Metadata only, no trajectory normalization, no gold join, no inference.

Run: python probe_dual_outcome.py --scan 40000
"""
from __future__ import annotations
import argparse, collections, itertools, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from datasets import load_dataset

NEBIUS = "nebius/SWE-agent-trajectories"
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=40000)
    ap.add_argument("--min-steps", type=int, default=15)
    a = ap.parse_args()

    ds = load_dataset(NEBIUS, split="train", streaming=True)
    by_inst = collections.defaultdict(lambda: {"failed": 0, "resolved": 0, "models": set()})
    seen = kept = 0
    for row in itertools.islice(ds, a.scan):
        seen += 1
        traj = row.get("trajectory")
        if not isinstance(traj, list):
            continue
        if sum(1 for s in traj if s.get("role") == "ai") < a.min_steps:
            continue
        kept += 1
        e = by_inst[row.get("instance_id", "")]
        e["resolved" if row.get("target") else "failed"] += 1
        e["models"].add(row.get("model_name", ""))
        if seen % 5000 == 0:
            both = sum(1 for v in by_inst.values() if v["failed"] and v["resolved"])
            print(f"  scanned {seen:6d}  kept {kept:6d}  instances {len(by_inst):5d}  "
                  f"dual-outcome {both:5d}", flush=True)

    both = {k: v for k, v in by_inst.items() if v["failed"] and v["resolved"]}
    pairs = sum(v["failed"] * v["resolved"] for v in both.values())
    print(f"\nscanned {seen} rows, kept {kept} runs of >= {a.min_steps} steps")
    print(f"instances seen              : {len(by_inst)}")
    print(f"instances with BOTH outcomes: {len(both)}")
    print(f"failed x resolved PAIRS available within instances: {pairs}")
    if both:
        top = sorted(both.items(), key=lambda kv: -(kv[1]["failed"] * kv[1]["resolved"]))[:10]
        print(f"\n{'instance':45s} {'failed':>7} {'resolved':>9} {'pairs':>6}")
        for k, v in top:
            print(f"{k[:45]:45s} {v['failed']:7d} {v['resolved']:9d} "
                  f"{v['failed']*v['resolved']:6d}")

    out = dict(scanned=seen, kept=kept, min_steps=a.min_steps,
               n_instances=len(by_inst), n_dual_outcome=len(both), n_within_pairs=pairs,
               dual_outcome_instances={k: dict(failed=v["failed"], resolved=v["resolved"])
                                       for k, v in both.items()})
    json.dump(out, open(os.path.join(DATA, "dual_outcome_probe.json"), "w"), indent=1)
    print("\nwrote data/dual_outcome_probe.json")


if __name__ == "__main__":
    main()
