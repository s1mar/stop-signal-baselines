"""Stream + join a sample once, pickle it for fast offline oracle iteration."""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
from llab import ingest

ap = argparse.ArgumentParser()
ap.add_argument("--failed", type=int, default=300)
ap.add_argument("--resolved", type=int, default=200)
ap.add_argument("--scan", type=int, default=12000)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "sample.pkl"))
a = ap.parse_args()

trajs = ingest.load_joined(n_failed=a.failed, n_resolved=a.resolved, scan_cap=a.scan)
with open(a.out, "wb") as f:
    pickle.dump(trajs, f)
print(f"pickled {len(trajs)} trajectories -> {a.out}")
