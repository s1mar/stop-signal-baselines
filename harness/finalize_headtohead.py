"""Authoritative head-to-head: all detectors + both judge models on the identical
subset. Prints Table 3 numbers and saves data/headtohead.json."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import econ, detectors as det
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
THR = {"syntactic": list(range(1, 9)),
       "convergence": list(np.round(np.linspace(0.50, 0.99, 25), 4)),
       "prm_proxy": list(range(1, 16)), "self_report": [1, 2, 3, 4, 5],
       "judge_north": list(range(1, 11)), "judge_laguna": list(range(1, 11))}

north = json.load(open(os.path.join(DATA, "judge_scores_north.json")))
laguna = json.load(open(os.path.join(DATA, "judge_scores_laguna.json")))
trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))
by_uid = {t.uid: t for t in trajs}

# shared subset = uids judged by BOTH models (identical trajectories)
shared = [u for u in north if u in laguna]
# Drop trajectories whose verdicts are not fully cache-backed: those contain steps
# scored by the old error path (silent 0) rather than by the judge.
_sus = os.path.join(DATA, "suspect_north-mini.json")
if os.path.exists(_sus) and os.environ.get("KEEP_SUSPECT") != "1":
    bad = set(json.load(open(_sus)))
    before = len(shared)
    shared = [u for u in shared if u not in bad]
    print(f"excluded {before - len(shared)} integrity-suspect trajectories")
failed, resolved = [], []
for u in shared:
    t = by_uid.get(u)
    if t is None:
        continue
    orc = compute_oracle(t)
    if orc is None:
        continue
    assert len(north[u]["scores"]) == len(laguna[u]["scores"]) == t.n_steps
    scores = {"judge_north": north[u]["scores"], "judge_laguna": laguna[u]["scores"]}
    for name, fn in det.ALL_DETECTORS.items():
        if name == "logprob":
            continue
        out = fn(t)
        if out.available:
            scores[name] = out.scores
    entry = dict(instance_id=t.instance_id, n=t.n_steps, tstar=orc.tstar,
                 char_total=1, char_after_tstar=0, scores=scores)
    (resolved if t.resolved else failed).append(entry)

corpus = econ.Corpus(failed=failed, resolved=resolved)
print(f"shared identical subset: {len(failed)} failed / {len(resolved)} resolved\n")
order = ["convergence", "syntactic", "prm_proxy", "self_report", "judge_north", "judge_laguna"]
rows = {}
print(f"{'detector':16s} {'5%':>7} {'10%':>7} {'20%':>7} {'regret@5%':>10}")
for d in order:
    sar = econ.savings_at_risk(corpus, d, THR[d])
    rr = econ.regret_at_risk(corpus, d, THR[d])
    rows[d] = dict(saved=sar, regret=rr["median_regret"], operating=rr["operating"])
    print(f"{d:16s} {sar['5']:7.3f} {sar['10']:7.3f} {sar['20']:7.3f} {rr['median_regret']:10.1f}")

json.dump(dict(n_failed=len(failed), n_resolved=len(resolved), rows=rows),
          open(os.path.join(DATA, "headtohead.json"), "w"), indent=1, default=float)
print("\nwrote data/headtohead.json")
