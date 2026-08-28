"""Merge the LLM-judge subset scores into the frontier; print Table-2 judge row."""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
from llab import econ
from llab.oracle import compute_oracle

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")

judge = json.load(open(os.path.join(DATA, "judge_scores.json")))
trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))
# Multiple trajectories share an instance_id; match the EXACT judged trajectory by
# (instance_id, resolved, n_steps) so scores align with the right oracle.
by_uid = {}
for t in trajs:
    by_uid.setdefault(t.uid, t)

failed, resolved = [], []
mismatch = 0
for uid, rec in judge.items():
    t = by_uid.get(uid)
    if t is None:
        mismatch += 1
        continue
    iid = t.instance_id
    assert t.n_steps == rec["n"] == len(rec["scores"]), "score/step length mismatch"
    orc = compute_oracle(t)
    if orc is None:
        continue
    entry = dict(instance_id=iid, n=t.n_steps, tstar=orc.tstar,
                 char_total=1, char_after_tstar=0,
                 scores={"llm_judge": rec["scores"]})
    (resolved if rec["resolved"] else failed).append(entry)

corpus = econ.Corpus(failed=failed, resolved=resolved)
thr = list(range(1, 11))
sar = econ.savings_at_risk(corpus, "llm_judge", thr)
rr = econ.regret_at_risk(corpus, "llm_judge", thr, target_risk=0.05)
print(f"LLM-judge subset: {len(failed)} failed / {len(resolved)} resolved "
      f"(unmatched: {mismatch})")
print(f"  saved@5%={sar['5']:.3f}  @10%={sar['10']:.3f}  @20%={sar['20']:.3f}  "
      f"median_regret@5%={rr['median_regret']:.1f}")
print("\nPaste into Table 2 (replace [[MEASURE:j5/j10/j20/jr]]):")
print(f"  {sar['5']:.3f} & {sar['10']:.3f} & {sar['20']:.3f} & {rr['median_regret']:.1f}")

# persist for the record
out = os.path.join(DATA, "judge_frontier.json")
json.dump(dict(n_failed=len(failed), n_resolved=len(resolved),
               savings_at_risk=sar, regret_at_risk=rr), open(out, "w"), indent=1, default=float)
print(f"\nwrote {out}")
