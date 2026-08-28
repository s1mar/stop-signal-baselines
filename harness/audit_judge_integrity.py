"""Integrity audit for judge score files.

The old judge silently returned 0 ("not stuck") when an Ollama call failed, and did
NOT cache that. So: any judged step whose prompt is absent from the verdict cache got
its score from the error path and is fabricated-by-accident. This finds them.
"""
import hashlib, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))
from llab import llm_judge

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FILES = {"north-mini": ("judge_scores_north.json", "north-mini-code-1.0:latest"),
         "laguna": ("judge_scores_laguna.json", "laguna-xs-2.1:latest"),
         "gemma4": ("judge_scores_gemma4.json", "gemma4:latest")}
MAX_STEPS = 120


def key(model, prompt):
    return hashlib.sha1((llm_judge._CALL_VERSION + "||" + model + "||" + prompt)
                        .encode("utf-8", "ignore")).hexdigest()


cache = json.load(open(os.path.join(DATA, "judge_cache.json")))
trajs = {t.uid: t for t in pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))}
print(f"cache entries: {len(cache)}\n")
print(f"{'file':12s} {'trajs':>6} {'steps':>7} {'cache-backed':>13} {'from-error':>11}  verdict")
for label, (fn, model) in FILES.items():
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        print(f"{label:12s} (missing)"); continue
    d = json.load(open(p))
    judged = missing = 0
    bad_trajs = []
    for uid, rec in d.items():
        t = trajs.get(uid)
        if t is None:
            continue
        prompts = llm_judge.build_prompts(t)
        limit = min(MAX_STEPS, len(prompts))
        miss_here = 0
        for i in range(limit):
            judged += 1
            if key(model, prompts[i]) not in cache:
                missing += 1
                miss_here += 1
        if miss_here:
            bad_trajs.append((uid, miss_here))
    pct = 100 * (judged - missing) / judged if judged else 0
    verdict = "CLEAN" if missing == 0 else f"SUSPECT ({len(bad_trajs)} trajs affected)"
    print(f"{label:12s} {len(d):6d} {judged:7d} {pct:12.1f}% {missing:11d}  {verdict}")
    if bad_trajs:
        json.dump([u for u, _ in bad_trajs],
                  open(os.path.join(DATA, f"suspect_{label}.json"), "w"), indent=1)
