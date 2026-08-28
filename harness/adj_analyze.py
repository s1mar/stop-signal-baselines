"""Does the gold-blind panel agree with the token-based Oracle Stop t*?

Reports, per judge and for the panel median: how close the blind judgment lands to t*, and
crucially how that compares to POSITIONAL NULLS (guess the midpoint; guess uniformly at
random). If judges only matched a midpoint prior, agreement would be trivial. Beating the
nulls is the evidence that the oracle tracks progress a capable reader independently sees.
Clusters are acknowledged by also reporting an instance-level aggregate.
"""
import json, os, sys, collections
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
JUDGES = ["gpt-5.6-sol", "kimi-k3", "claude-fable-5"]


def load(tag):
    p = os.path.join(DATA, f"adj_{tag}.json")
    return {r["uid"]: r for r in json.load(open(p, encoding="utf-8")) if r.get("judge") is not None}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d else float("nan")


data = {t: load(t) for t in JUDGES}
all_uids = set().union(*[set(d) for d in data.values()])
# reference facts (n, tstar) from any judge record
ref = {}
for d in data.values():
    for u, r in d.items():
        ref[u] = (r["n"], r["tstar"], r["iid"])

print("=== per-judge agreement with token-based t* (gold-blind judges) ===")
print(f"{'judge':16s} {'n':>4} {'med|J-t*|':>9} {'<=2':>6} {'<=3':>6} {'spearman':>9} "
      f"{'null_mid':>9} {'null_rand':>9}")
rows = {}
for t in JUDGES:
    js, ts, ns, iids = [], [], [], []
    for u, r in data[t].items():
        js.append(r["judge"]); ts.append(r["tstar"]); ns.append(r["n"]); iids.append(r["iid"])
    js, ts, ns = np.array(js), np.array(ts), np.array(ns)
    absd = np.abs(js - ts)
    mid = np.abs((ns // 2) - ts)                 # midpoint-guess error
    # expected uniform-random error E|U-t*| over integer steps 0..n-1
    rand = np.array([np.mean(np.abs(np.arange(n) - tt)) for n, tt in zip(ns, ts)])
    rows[t] = dict(n=len(js), med=float(np.median(absd)),
                   w2=float(np.mean(absd <= 2)), w3=float(np.mean(absd <= 3)),
                   sp=spearman(js/ns, ts/ns),
                   nmid=float(np.median(mid)), nrand=float(np.median(rand)),
                   js=js, ts=ts, ns=ns, iids=iids)
    r = rows[t]
    print(f"{t:16s} {r['n']:>4} {r['med']:>9.1f} {r['w2']:>6.2f} {r['w3']:>6.2f} "
          f"{r['sp']:>9.2f} {r['nmid']:>9.1f} {r['nrand']:>9.1f}")

# --- panel median (only uids all judges scored) ---
common = set.intersection(*[set(d) for d in data.values()])
pj, pt, pn = [], [], []
for u in common:
    pj.append(np.median([data[t][u]["judge"] for t in JUDGES]))
    pt.append(ref[u][1]); pn.append(ref[u][0])
pj, pt, pn = np.array(pj), np.array(pt), np.array(pn)
absd = np.abs(pj - pt)
print(f"\n=== panel median (n={len(common)} trajectories all three judged) ===")
print(f"median |panel - t*| = {np.median(absd):.1f} steps; within 2 = {np.mean(absd<=2):.2f}; "
      f"within 3 = {np.mean(absd<=3):.2f}; spearman(pos) = {spearman(pj/pn, pt/pn):.2f}")

# --- inter-judge: do judges agree with each other as much as with t*? ---
print("\n=== inter-judge agreement (median pairwise |Ji-Jj|) ===")
pairs = [("gpt-5.6-sol", "kimi-k3"), ("gpt-5.6-sol", "claude-fable-5"), ("kimi-k3", "claude-fable-5")]
for a, b in pairs:
    sh = set(data[a]) & set(data[b])
    dd = [abs(data[a][u]["judge"] - data[b][u]["judge"]) for u in sh]
    print(f"{a:16s} vs {b:16s}  median {np.median(dd):.1f}  (n={len(sh)})")

# --- instance-level (cluster-aware): average |J-t*| per instance, then across instances ---
print("\n=== instance-level (cluster by task, panel median) ===")
byi = collections.defaultdict(list)
for u in common:
    byi[ref[u][2]].append(abs(np.median([data[t][u]["judge"] for t in JUDGES]) - ref[u][1]))
per_inst = [np.mean(v) for v in byi.values()]
print(f"{len(byi)} instances; mean over instances of mean|panel-t*| = {np.mean(per_inst):.1f} steps")

out = dict(per_judge={t: {k: rows[t][k] for k in ("n","med","w2","w3","sp","nmid","nrand")} for t in JUDGES},
           panel=dict(n=len(common), med=float(np.median(absd)),
                      w2=float(np.mean(absd<=2)), w3=float(np.mean(absd<=3)),
                      spearman=spearman(pj/pn, pt/pn)),
           instances=len(byi), per_instance_mean=float(np.mean(per_inst)))
json.dump(out, open(os.path.join(DATA, "adj_result.json"), "w"), indent=1)
print("\nwrote adj_result.json")
