"""Prepare the gold-blind adjudication sample.

Select failed, reached-reference trajectories with an INTERIOR Oracle Stop (so there is a
real judgment to make), fetch each instance's problem_statement from the gold datasets, and
cache a compact, judge-ready record. No gold patch goes into the judge payload.
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
from datasets import load_dataset
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(DATA, "adj_sample.json")
GOLD_SOURCES = ["nebius/SWE-bench-extra", "princeton-nlp/SWE-bench"]
N_TARGET = 70
PER_INSTANCE = 2
MIN_STEPS, MAX_STEPS = 6, 60


def compact_steps(t):
    rows = []
    for s in t.steps:
        txt = (s.text or "").strip().replace("\n", " ")
        act = getattr(s.action, "raw", "") or ""
        obs = (s.observation or "").strip().replace("\n", " ")
        rows.append(dict(i=s.index,
                         say=txt[:320],
                         cmd=act[:160],
                         obs=obs[:180]))
    return rows


def main():
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))
    cand = []
    for t in trajs:
        if t.resolved:
            continue
        o = compute_oracle(t)
        if o is None or o.tstar is None:
            continue
        if o.max_progress <= 0:                      # reached-reference stratum only
            continue
        if not (MIN_STEPS <= t.n_steps <= MAX_STEPS):
            continue
        if not (1 <= o.tstar <= t.n_steps - 2):      # interior t*
            continue
        cand.append((t, o.tstar))
    # Maximize instance diversity: round-robin across instances, <=PER_INSTANCE each.
    import collections
    by_iid = collections.defaultdict(list)
    for t, ts in sorted(cand, key=lambda x: x[0].uid):
        by_iid[t.instance_id].append((t, ts))
    picked, r = [], 0
    while len(picked) < N_TARGET and any(len(v) > r for v in by_iid.values()):
        for iid in sorted(by_iid):
            if len(by_iid[iid]) > r and len(picked) < N_TARGET:
                picked.append(by_iid[iid][r])
        r += 1
        if r >= PER_INSTANCE:
            break
    cand = picked
    want = {t.instance_id for t, _ in cand}
    print(f"selected {len(cand)} trajectories over {len(want)} instances; fetching issues")

    issues = {}
    for src in GOLD_SOURCES:
        if not want - set(issues):
            break
        split = "train" if "extra" in src else "test"
        try:
            ds = load_dataset(src, split=split, streaming=True)
            for g in ds:
                iid = g.get("instance_id")
                if iid in want and iid not in issues:
                    ps = g.get("problem_statement") or ""
                    if ps.strip():
                        issues[iid] = ps
                if not want - set(issues):
                    break
        except Exception as e:
            print(f"[issues] {src} failed: {e!r}")
    print(f"got problem_statement for {len(issues)}/{len(want)} instances")

    recs = []
    for t, tstar in cand:
        if t.instance_id not in issues:
            continue
        recs.append(dict(uid=t.uid, iid=t.instance_id, n=t.n_steps, tstar=tstar,
                         issue=issues[t.instance_id][:4000],
                         steps=compact_steps(t)))
    json.dump(recs, open(OUT, "w"), indent=1)
    print(f"wrote {len(recs)} judge-ready records -> {OUT}")


if __name__ == "__main__":
    main()
