"""Frontier-judge calibration: does a frontier reference judge agree with the
cheap local judges? Samples steps the local judges already scored, has
gpt-5.6-terra (OpenRouter) judge the identical prompts, reports Cohen's kappa +
agreement. Answers the reviewer critique that open-weight judges are untrustworthy.
"""
import argparse, hashlib, json, os, pickle, random, sys, time, urllib.request, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import llm_judge

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
def _api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    p = os.path.join(os.path.dirname(__file__), "..", "open_router_config.txt")
    try:
        return re.search(r"sk-or-v1-\w+", open(p, encoding="utf-8").read()).group(0)
    except Exception:
        sys.exit("No API key: set OPENROUTER_API_KEY, or put a line containing your "
                 "sk-or-v1-... key in open_router_config.txt at the repository root. "
                 "(The recorded calibration already ships in data/calibration.json, so this "
                 "script only needs to run if you want to re-collect it.)")

KEY = _api_key()
CACHE_PATH = os.path.join(DATA, "ref_judge_cache.json")
LOCAL = {"north-mini": "judge_scores_north.json", "laguna": "judge_scores_laguna.json",
         "gemma4": "judge_scores_gemma4.json"}

_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def ref_judge(prompt: str, model: str) -> int:
    k = hashlib.sha1((model + "||" + prompt).encode()).hexdigest()
    if k in _cache:
        return _cache[k]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 12, "temperature": 0}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        txt = resp["choices"][0]["message"]["content"].upper()
        v = 1 if "STUCK" in txt else 0
    except Exception as e:
        print("  ref call error:", str(e)[:80]); return -1
    _cache[k] = v
    json.dump(_cache, open(CACHE_PATH, "w"))
    return v


def kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = np.mean(a == b)
    pe = sum((np.mean(a == c) * np.mean(b == c)) for c in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), float(po)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5.6-terra")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    trajs = {t.uid: t for t in pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))}
    local = {}
    for name, fn in LOCAL.items():
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            d = json.load(open(p))
            if len(d) >= 100:                 # skip incomplete judge runs
                local[name] = d
            else:
                print(f"skipping {name}: only {len(d)} trajectories (incomplete)")
    if not local:
        print("no local judge scores found"); return
    shared = set.intersection(*[set(d) for d in local.values()])
    shared = [u for u in shared if u in trajs]
    print(f"local judges: {list(local)} | shared trajectories: {len(shared)}")

    rng = random.Random(0)
    pairs = []
    for u in shared:
        n = trajs[u].n_steps
        for i in range(n):
            pairs.append((u, i))
    rng.shuffle(pairs)
    pairs = pairs[:a.n]

    prompts_cache = {}
    ref, locs = [], {name: [] for name in local}
    t0 = time.time()
    for j, (u, i) in enumerate(pairs):
        if u not in prompts_cache:
            prompts_cache[u] = llm_judge.build_prompts(trajs[u])
        v = ref_judge(prompts_cache[u][i], a.model)
        if v < 0:
            continue
        ref.append(v)
        for name in local:
            locs[name].append(1 if local[name][u]["scores"][i] > 0 else 0)
        if (j + 1) % 50 == 0:
            print(f"  {j+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nreference judge = {a.model}; n={len(ref)}; ref STUCK-rate={np.mean(ref):.3f}")
    print(f"{'local judge':14s} {'agree%':>7} {'kappa':>7} {'localSTUCK%':>11}")
    out = {"model": a.model, "n": len(ref), "ref_stuck_rate": float(np.mean(ref)), "agreement": {}}
    for name in local:
        k, po = kappa(ref, locs[name])
        print(f"{name:14s} {po*100:7.1f} {k:7.3f} {np.mean(locs[name])*100:11.1f}")
        out["agreement"][name] = dict(agreement=po, kappa=k, local_stuck_rate=float(np.mean(locs[name])))
    json.dump(out, open(os.path.join(DATA, "calibration.json"), "w"), indent=1, default=float)
    print("\nwrote data/calibration.json")


if __name__ == "__main__":
    main()