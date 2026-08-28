"""Compute LLM-judge scores for a seeded subset; save incrementally to judge_scores.json."""
import argparse, json, os, pickle, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from llab import llm_judge
try:
    import ollama_lock          # optional advisory GPU lock from the authors' setup;
                                # not part of this artifact, safely skipped when absent
except Exception:
    ollama_lock = None

ap = argparse.ArgumentParser()
ap.add_argument("--cache-file", default=os.path.join(os.path.dirname(__file__), "..", "data", "sample_large.pkl"))
ap.add_argument("--n-failed", type=int, default=25)
ap.add_argument("--n-resolved", type=int, default=25)
ap.add_argument("--max-steps", type=int, default=45, help="cap steps judged per trajectory")
ap.add_argument("--max-len", type=int, default=120, help="skip trajectories longer than this")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="output scores file (default judge_scores.json)")
a = ap.parse_args()

trajs = pickle.load(open(a.cache_file, "rb"))
rng = random.Random(a.seed)
failed = [t for t in trajs if not t.resolved and t.n_steps <= a.max_len]
resolved = [t for t in trajs if t.resolved and t.n_steps <= a.max_len]
rng.shuffle(failed); rng.shuffle(resolved)

# Per-trajectory (not per-instance): each trajectory is its own data point, exactly
# as the mechanical detectors count them. Scores are keyed by the unique t.uid, so
# multiple trajectories of the same issue no longer collapse.
fsel, rsel = failed[:a.n_failed], resolved[:a.n_resolved]
# Interleave so every incremental checkpoint has both classes.
subset = []
for i in range(max(len(fsel), len(rsel))):
    if i < len(fsel):
        subset.append(fsel[i])
    if i < len(rsel):
        subset.append(rsel[i])
print(f"judging {len(subset)} trajectories (<= {a.max_len} steps, cap {a.max_steps}) "
      f"with {llm_judge.JUDGE_MODEL}", flush=True)

path = a.out or os.path.join(os.path.dirname(__file__), "..", "data", "judge_scores.json")
out = json.load(open(path)) if os.path.exists(path) else {}

# Claim the shared GPU so other sessions do not swap the model out from under us.
if ollama_lock is not None:
    if not ollama_lock.claim(llm_judge.JUDGE_MODEL,
                             f"judge run ({os.path.basename(a.out or 'judge')})",
                             eta=f"~{len(subset)} trajectories"):
        print("Not starting: another process holds the local Ollama model lock.")
        sys.exit(1)
t0 = time.time()
for i, t in enumerate(subset):
    if t.uid in out:
        continue
    try:
        scores = llm_judge.judge_scores(t, max_steps=a.max_steps)
    except llm_judge.JudgeUnavailable as e:
        # Do NOT record a partial/degraded trajectory: a failed call must never
        # be saved as a verdict. Stop cleanly; the run is resumable.
        print(f"\nABORT at {i+1}/{len(subset)}: judge unavailable ({e}).\n"
              f"Nothing corrupt was saved. Likely another process swapped the "
              f"loaded Ollama model. Re-run to resume from {len(out)}.", flush=True)
        break
    out[t.uid] = dict(instance_id=t.instance_id, resolved=t.resolved, n=t.n_steps,
                      scores=scores)
    llm_judge.save_cache()
    json.dump(out, open(path, "w"), indent=1)          # incremental: partial is usable
    el = time.time() - t0
    print(f"  {i+1}/{len(subset)} done ({el:.0f}s)", flush=True)
print(f"saved {len(out)} -> {path}")
# Release the GPU for other sessions. If this process is hard-killed instead, the
# lock self-heals: the recorded pid is gone, so the next session sees it as STALE.
if ollama_lock is not None:
    ollama_lock.drop()
