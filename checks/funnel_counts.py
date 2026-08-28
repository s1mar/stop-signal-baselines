"""Recover the dev-corpus funnel counts from the recorded cache (no network, no re-ingest).

Reads data/sample_large.pkl (the joined sample the N708 analysis loaded, per
results_N708.json config.cache_file) and classifies every trajectory by the same
predicates run_study.build_corpus applies via compute_oracle:
  drop_gold_empty : gold patch yields no target files or no added-token set
  drop_no_states  : non-empty gold token set but no reconstructable gold-file states
  used            : everything else (should be 708: 417 failed / 291 resolved)
Prints the absolute path of the file it read.
"""
import os, pickle, sys

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
sys.path.insert(0, LAB)
from llab.patchtools import gold_target_files, gold_added_tokens, agent_goldfile_states

CACHE = os.path.abspath(os.path.join(LAB, "..", "data", "sample_large.pkl"))
print("reading:", CACHE)
trajs = pickle.load(open(CACHE, "rb"))
print("cached joined trajectories:", len(trajs))

n_failed_in = sum(1 for t in trajs if not t.resolved)
n_res_in = len(trajs) - n_failed_in
print(f"  of which failed={n_failed_in} resolved={n_res_in}")

drop_gold_empty = []
drop_no_states = []
used_f = used_r = 0
for t in trajs:
    gfiles = gold_target_files(t.gold_patch)
    gtokens = gold_added_tokens(t.gold_patch)
    if not gfiles or not gtokens:
        drop_gold_empty.append(t)
        continue
    states = agent_goldfile_states(t, gfiles)
    if not states:
        drop_no_states.append(t)
        continue
    if t.resolved:
        used_r += 1
    else:
        used_f += 1

def split(lst):
    f = sum(1 for t in lst if not t.resolved)
    return f"{len(lst)} (failed={f} resolved={len(lst)-f})"

print("dropped, empty gold target-file/token set:", split(drop_gold_empty))
print("dropped, no reconstructable gold-file states:", split(drop_no_states))
print(f"used: {used_f + used_r} (failed={used_f} resolved={used_r})")
print("EXPECT used=708 failed=417 resolved=291 to match results_N708.json")

# distinct step-count floor check: everything cached should already be >=15 ai steps
short = sum(1 for t in trajs if t.n_steps < 15)
print("cached with <15 steps (should be 0):", short)
print("unique instance ids in cache:", len({t.instance_id for t in trajs}))
