"""Smoke-test oracle + detectors on the one saved trajectory (needs gold).
Fetches ONLY this instance's gold patch (tiny)."""
import json, os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from datasets import load_dataset
from llab.ingest import normalize
from llab.oracle import compute_oracle
from llab import detectors as det

ex = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "data",
                                 "probe_example.json"), encoding="utf-8"))
traj = normalize(ex)
print("instance:", traj.instance_id, "| resolved:", traj.resolved,
      "| ai steps:", traj.n_steps, "| edit steps:", len(traj.edit_steps))

# fetch this instance's gold patch
iid = traj.instance_id
gds = load_dataset("nebius/SWE-bench-extra", split="train", streaming=True)
for g in itertools.islice(gds, 8000):
    if g.get("instance_id") == iid:
        traj.gold_patch = g["patch"]
        break
assert traj.gold_patch, "gold not found"
print("gold patch chars:", len(traj.gold_patch))

orc = compute_oracle(traj)
assert orc is not None
print(f"\nORACLE: t*={orc.tstar} (struct t*={orc.tstar_struct}) contested={orc.contested}")
print(f"  n edit-states={len(orc.d_embed)} | monotone_frac={orc.monotone_frac:.3f}")
print(f"  d_embed first/min/last = {orc.d_embed[0]:.3f} / "
      f"{min(orc.d_embed):.3f} / {orc.d_embed[-1]:.3f}")
waste = max(0, traj.n_steps - 1 - (orc.tstar or 0))
print(f"  recoverable waste after t*: {waste} of {traj.n_steps} steps")

print("\nDETECTORS (first fire step at a sample threshold):")
for name, fn in det.ALL_DETECTORS.items():
    out = fn(traj)
    if not out.available:
        print(f"  {name:12s} UNAVAILABLE ({out.note})")
        continue
    mx = max(out.scores) if out.scores else 0
    print(f"  {name:12s} max_score={mx:.2f} scores[-5:]="
          f"{[round(s,2) for s in out.scores[-5:]]}")
print("\nOK: oracle + detectors ran end-to-end on a real trajectory.")
