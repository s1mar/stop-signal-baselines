"""Spike 2: (a) inspect AI action format in the saved example,
(b) confirm gold-patch availability for the instance via SWE-bench-extra."""
import json, itertools, re
from datasets import load_dataset

ex = json.load(open("../data/probe_example.json", encoding="utf-8"))
traj = ex["trajectory"]
iid = ex["instance_id"]
print("INSTANCE:", iid, "| n_steps:", len(traj), "| resolved:", ex["target"])

# (a) Show the TAIL of each ai step (where the SWE-agent command lives) for first 12 ai steps
ai_steps = [(i, s) for i, s in enumerate(traj) if s.get("role") == "ai"]
print(f"\n# ai steps: {len(ai_steps)}")
edit_like = 0
for i, s in ai_steps:
    txt = s.get("text") or ""
    tail = txt.strip().splitlines()[-6:]
    joined = "\n".join(tail)
    if re.search(r"\b(edit|create|str_replace|insert|append)\b", txt):
        edit_like += 1
    if i < 30:  # only print early ones
        print(f"\n--- ai step {i} TAIL ---")
        print("   " + joined.replace("\n", "\n   ")[:500])
print(f"\nai steps containing edit-like keywords: {edit_like}/{len(ai_steps)}")

# (b) Gold patch: search nebius/SWE-bench-extra for this instance_id (streaming)
print("\n=== Looking up gold patch in nebius/SWE-bench-extra ===")
found = None
try:
    gold_ds = load_dataset("nebius/SWE-bench-extra", split="train", streaming=True)
    for g in itertools.islice(gold_ds, 5000):
        if g.get("instance_id") == iid:
            found = g
            break
    if found:
        print("FOUND in SWE-bench-extra. keys:", list(found.keys()))
        for k in ("instance_id", "base_commit", "repo"):
            if k in found:
                print(f"  {k}: {found[k]}")
        gp = found.get("patch") or found.get("gold_patch") or found.get("solution_patch")
        print("  GOLD PATCH (first 500):\n", str(gp)[:500])
    else:
        print("NOT found in first 5000 of SWE-bench-extra (may be in SWE-bench dev split).")
        # peek at schema anyway
        g0 = next(iter(load_dataset("nebius/SWE-bench-extra", split="train", streaming=True)))
        print("  SWE-bench-extra schema keys:", list(g0.keys()))
except Exception as e:
    print("ERR:", repr(e))
