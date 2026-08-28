"""How do zero-overlap RESOLVED runs solve the task? (r24 feedback item 4)

The paper says 13.1% (38 of 291) of resolved runs finish with zero token overlap,
"having solved the task by a route the reference does not describe". Before adding any
concrete intuition to that sentence, measure which route: did those runs edit any gold
target file at all (same-file different-identifiers), or only other files entirely?
Prints the absolute path it reads.
"""
import os, pickle, sys

LAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
sys.path.insert(0, LAB)
from llab.patchtools import gold_target_files, gold_added_tokens, agent_goldfile_states

CACHE = os.path.abspath(os.path.join(LAB, "..", "data", "sample_large.pkl"))
print("reading:", CACHE)
trajs = pickle.load(open(CACHE, "rb"))

n_res = n_zero = touched_gold = only_other = no_edits_at_all = 0
for t in trajs:
    gfiles = gold_target_files(t.gold_patch)
    gtokens = gold_added_tokens(t.gold_patch)
    if not gfiles or not gtokens:
        continue
    states = agent_goldfile_states(t, gfiles)
    if not states:
        continue
    if not t.resolved:
        continue
    n_res += 1
    max_prog = max((len(tok & gtokens) / len(gtokens) for _, tok, _ in states), default=0.0)
    if max_prog > 0.0:
        continue
    n_zero += 1
    edited_gold = any(s.action and s.action.is_edit and s.action.target_file in gfiles
                      for s in t.steps)
    if edited_gold:
        touched_gold += 1
    else:
        only_other += 1

print(f"resolved runs used: {n_res} (expect 291)")
print(f"zero-overlap resolved: {n_zero} (paper says 38)")
print(f"  edited a gold target file, just with none of the reference's identifiers: {touched_gold}")
print(f"  never edited any gold target file (fixed elsewhere entirely): {only_other}")
