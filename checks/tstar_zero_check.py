"""Check the zero-progress t* convention on the dev corpus (r24 supervisor finding T6).

Paper line (The Instrument): "t*=0 when no step attains positive progress".
Code (llab/oracle.py _last_new_max with best=-1.0): the first edit step always sets a
new maximum, so t* = index of the FIRST EDIT STEP, not 0, on zero-progress runs.
This script measures which convention the shipped numbers actually reflect.
"""
import os, pickle, sys, statistics

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
sys.path.insert(0, LAB)
from llab.patchtools import gold_target_files, gold_added_tokens, agent_goldfile_states
from llab.oracle import compute_oracle

CACHE = os.path.abspath(os.path.join(LAB, "..", "data", "sample_large.pkl"))
print("reading:", CACHE)
trajs = pickle.load(open(CACHE, "rb"))

zero_rows, mismatch = [], 0
fail_fracs = []
for t in trajs:
    orc = compute_oracle(t)
    if orc is None:
        continue
    if not t.resolved:
        fail_fracs.append(orc.tstar / t.n_steps if orc.tstar is not None else 0.0)
        if orc.max_progress == 0.0:
            first_edit = next((s.index for s in t.steps if s.action and s.action.is_edit), None)
            zero_rows.append((orc.tstar, first_edit, t.n_steps))
            if orc.tstar != first_edit:
                mismatch += 1

print(f"failed runs used: {len(fail_fracs)}, median t*/n = {statistics.median(fail_fracs):.4f} "
      f"(paper says 0.21)")
print(f"zero-progress failed runs: {len(zero_rows)} (paper says 205)")
print(f"  t* == first edit step index on {len(zero_rows) - mismatch} of {len(zero_rows)} "
      f"(mismatches: {mismatch})")
print(f"  t* literally 0 on {sum(1 for ts, _, _ in zero_rows if ts == 0)} of {len(zero_rows)}")
print(f"  median t* among zero-progress: {statistics.median([ts for ts, _, _ in zero_rows])}")
print(f"  median t*/n among zero-progress: "
      f"{statistics.median([ts / n for ts, _, n in zero_rows]):.4f}")
