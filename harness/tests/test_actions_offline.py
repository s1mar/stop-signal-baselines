"""Offline validation of the action parser on the real saved trajectory.
Run: python tests/test_actions_offline.py  (no network, no ML deps)."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llab.actions import parse_action, attach_current_file

ex = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "data",
                                 "probe_example.json"), encoding="utf-8"))
traj = ex["trajectory"]
ai = [s for s in traj if s.get("role") == "ai"]
actions = [parse_action(s.get("text") or "") for s in ai]
attach_current_file([a for a in actions if a])

parsed = [a for a in actions if a]
edits = [a for a in parsed if a and a.is_edit]
navs = [a for a in parsed if a and a.is_nav]
cmds = {}
for a in parsed:
    cmds[a.cmd] = cmds.get(a.cmd, 0) + 1

print(f"ai steps: {len(ai)} | parsed actions: {len(parsed)} | "
      f"edits: {len(edits)} | navs: {len(navs)}")
print("command histogram:", dict(sorted(cmds.items(), key=lambda x: -x[1])))
print("\nEdit actions (file <- content preview):")
for a in edits[:8]:
    preview = a.edit_content.replace("\n", " / ")[:80]
    print(f"  [{a.cmd} -> {a.target_file}] {preview!r}")

# sanity assertions
assert len(parsed) >= 0.7 * len(ai), "parser should recover most ai actions"
assert len(edits) >= 1, "expected at least one edit action"
assert any(a.edit_content for a in edits), "edits should carry content"
assert any(a.target_file for a in edits), "edits should map to a file"
print("\nOK: action parser validated on real trajectory.")
