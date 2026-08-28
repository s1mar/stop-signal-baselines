"""Feasibility spike: stream ONE Nebius trajectory and inspect structure.
Does NOT download the full corpus (streaming=True + take)."""
import json, itertools
from datasets import load_dataset

ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

# Grab a small handful, prefer a FAILED, long one (that's our population of interest).
picked = None
scanned = 0
for ex in itertools.islice(ds, 200):
    scanned += 1
    traj = ex.get("trajectory")
    n = len(traj) if isinstance(traj, list) else -1
    if ex.get("target") is False and isinstance(traj, list) and n >= 20:
        picked = ex
        break
if picked is None:
    # fallback: first record
    picked = next(iter(itertools.islice(load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True), 1)))

print("SCANNED:", scanned)
print("TOP-LEVEL KEYS:", list(picked.keys()))
for k, v in picked.items():
    if k == "trajectory":
        continue
    s = str(v)
    print(f"\n== {k} ({type(v).__name__}) ==")
    print(s[:600])

traj = picked["trajectory"]
print("\n\n===== TRAJECTORY: n_steps =", len(traj), "=====")
print("ITEM TYPE:", type(traj[0]).__name__)
if isinstance(traj[0], dict):
    print("STEP KEYS:", list(traj[0].keys()))

# Dump first 4 steps and last 2 steps, truncated
def show(i, item):
    print(f"\n----- step {i} -----")
    if isinstance(item, dict):
        for kk, vv in item.items():
            print(f"  [{kk}]: {str(vv)[:900]}")
    else:
        print("  ", str(item)[:900])

for i in range(min(4, len(traj))):
    show(i, traj[i])
print("\n... (tail) ...")
for i in range(max(0, len(traj)-2), len(traj)):
    show(i, traj[i])

# Save the full picked example for offline inspection
with open("../data/probe_example.json", "w", encoding="utf-8") as f:
    json.dump(picked, f, ensure_ascii=False, indent=1)
print("\nSAVED ../data/probe_example.json")
