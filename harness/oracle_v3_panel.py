"""Does the current-state oracle (v3) beat the cumulative oracle (v2) as an instrument?

Two checks the paper's oracle-validation section needs, neither of which oracle_v3.py saved:

A. DEGENERACY. If v3's t* almost always coincides with the LAST applied gold edit, then v3 is not
   locating a peak at all: it is reporting "the last time the agent touched a gold file", which no
   detector could be scored against meaningfully.

B. EXTERNAL AGREEMENT. On the 70-trajectory gold-blind adjudication set, compare |panel - t*| for
   v2 and v3 on the SAME trajectories. The oracle that agrees better with readers who never saw the
   gold patch is the better available instrument.

Writes data/oracle_v3_panel.json.
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab.oracle import compute_oracle
from oracle_v3 import oracle_v3, current_patch_series
from llab.patchtools import gold_target_files

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
JUDGES = ["gpt-5.6-sol", "kimi-k3", "claude-fable-5"]


def main():
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))

    # ---- A. is v3's t* just the last applied gold edit? ---------------------------
    n_deg = n_tot = 0
    v3_by_uid, v2_by_uid = {}, {}
    for t in trajs:
        if t.resolved:
            continue
        v3 = oracle_v3(t)
        if v3 is None or v3["tstar"] is None or len(v3["prog_line"]) < 2:
            continue
        v2 = compute_oracle(t)
        if v2 is None or v2.tstar is None:
            continue
        n_tot += 1
        last_edit = v3["edit_idx"][-1]
        n_deg += int(v3["tstar"] == last_edit)
        v3_by_uid[t.uid] = v3["tstar"]
        v2_by_uid[t.uid] = v2.tstar

    deg = n_deg / n_tot if n_tot else float("nan")
    print(f"A. degeneracy: v3 t* == last applied gold edit in {n_deg}/{n_tot} = {deg:.1%}")

    # ---- B. panel agreement, v2 vs v3, same trajectories --------------------------
    data = {}
    for tag in JUDGES:
        p = os.path.join(DATA, f"adj_{tag}.json")
        data[tag] = {r["uid"]: r for r in json.load(open(p, encoding="utf-8"))
                     if r.get("judge") is not None}
    common = set.intersection(*[set(d) for d in data.values()])
    both = sorted(u for u in common if u in v3_by_uid and u in v2_by_uid)

    panel = {u: float(np.median([data[t][u]["judge"] for t in JUDGES])) for u in both}
    d2 = np.array([abs(panel[u] - v2_by_uid[u]) for u in both])
    d3 = np.array([abs(panel[u] - v3_by_uid[u]) for u in both])

    print(f"\nB. gold-blind panel agreement on the {len(both)} trajectories both oracles score")
    print(f"   v2 (cumulative)    median |panel - t*| = {np.median(d2):.1f} steps; "
          f"within 2 = {np.mean(d2 <= 2):.2f}")
    print(f"   v3 (current-state) median |panel - t*| = {np.median(d3):.1f} steps; "
          f"within 2 = {np.mean(d3 <= 2):.2f}")
    better = float(np.mean(d2 < d3))
    print(f"   v2 strictly closer on {better:.1%} of them; v3 closer on {np.mean(d3 < d2):.1%}")

    out = dict(n_failed_scored=n_tot, n_tstar_at_last_edit=n_deg, degeneracy_frac=deg,
               panel_n=len(both),
               v2_median_abs=float(np.median(d2)), v3_median_abs=float(np.median(d3)),
               v2_within2=float(np.mean(d2 <= 2)), v3_within2=float(np.mean(d3 <= 2)),
               v2_closer_frac=better, v3_closer_frac=float(np.mean(d3 < d2)))
    json.dump(out, open(os.path.join(DATA, "oracle_v3_panel.json"), "w"), indent=1)
    print("\nwrote data/oracle_v3_panel.json")


if __name__ == "__main__":
    main()
