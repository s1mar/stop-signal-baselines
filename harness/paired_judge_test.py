"""Paired bootstrap for the judge-vs-judge contrast at the strict 5% budget.

Comparing one judge's CI to another
detector's POINT estimate is not a test, so we use a paired statistic. Both judges scored the SAME trajectories,
so the honest statistic is a paired bootstrap of the difference: resample
trajectories, recompute each judge's best saved-fraction subject to false-abort
<= 5% within the resample, and take the difference.
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import econ, detectors as det
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
THR = list(range(1, 11))

north = json.load(open(os.path.join(DATA, "judge_scores_north.json")))
laguna = json.load(open(os.path.join(DATA, "judge_scores_laguna.json")))
trajs = {t.uid: t for t in pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))}
bad = set(json.load(open(os.path.join(DATA, "suspect_north-mini.json"))))

failed, resolved = [], []
for u in north:
    if u not in laguna or u in bad or u not in trajs:
        continue
    t = trajs[u]
    orc = compute_oracle(t)
    if orc is None:
        continue
    e = dict(n=t.n_steps, tstar=orc.tstar,
             a=north[u]["scores"], b=laguna[u]["scores"])
    (resolved if t.resolved else failed).append(e)
print(f"paired subset: {len(failed)} failed / {len(resolved)} resolved")


def fire(sc, thr):
    for i, s in enumerate(sc):
        if s >= thr:
            return i
    return None


def saved_at5(F, R, key):
    """Best saved-fraction subject to false-abort <= 5% on this (resampled) set."""
    best = 0.0
    nR = len(R) or 1
    tot = sum(e["n"] for e in F) or 1
    for thr in THR:
        fa = sum(1 for e in R if (lambda f: f is not None and f < e["n"] - 1)(fire(e[key], thr)))
        if fa / nR > 0.05:
            continue
        sv = sum(econ.waste_after(fire(e[key], thr), e["n"]) for e in F)
        best = max(best, sv / tot)
    return best


obs_a, obs_b = saved_at5(failed, resolved, "a"), saved_at5(failed, resolved, "b")
print(f"observed saved@5%: conservative={obs_a:.3f}  eager={obs_b:.3f}  diff={obs_a-obs_b:+.3f}")

rng = np.random.default_rng(0)
diffs = []
for _ in range(2000):
    fi = rng.integers(0, len(failed), len(failed))
    ri = rng.integers(0, len(resolved), len(resolved))
    F = [failed[i] for i in fi]
    R = [resolved[i] for i in ri]
    diffs.append(saved_at5(F, R, "a") - saved_at5(F, R, "b"))
diffs = np.array(diffs)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"paired bootstrap of (conservative - eager) saved@5%: "
      f"mean {diffs.mean():+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"P(conservative > eager) = {(diffs > 0).mean():.3f}")
print("VERDICT:", "difference EXCLUDES zero (resolved)" if lo > 0 or hi < 0
      else "difference INCLUDES zero (not resolved)")
json.dump(dict(obs_conservative=obs_a, obs_eager=obs_b, mean_diff=float(diffs.mean()),
               ci=[float(lo), float(hi)], p_gt=float((diffs > 0).mean())),
          open(os.path.join(DATA, "paired_judge_test.json"), "w"), indent=1)
