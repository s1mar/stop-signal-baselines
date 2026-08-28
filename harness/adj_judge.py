"""Gold-blind adjudication panel: does a frontier reader, WITHOUT the reference solution,
place the last-progress step where the token oracle places t*?

Each judge sees the issue and the numbered agent steps only. It never sees the gold patch.
Agreement between an independent gold-blind judge and the gold-based t* is evidence the
oracle tracks genuine progress rather than lexical overlap. Operating rules: content-only prompt, capped reasoning, save-before-print.
"""
import argparse, json, os, re, sys, threading, urllib.request

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
                 "(The recorded panel outputs already ship in data/, so this script only "
                 "needs to run if you want to re-collect them.)")

KEY = _api_key()
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE = os.path.join(DATA, "adj_sample.json")

PROMPT = r"""You are analyzing the trajectory of an autonomous software agent that attempted to
fix a bug. The agent FAILED to resolve the issue. Your job is to identify the last step at
which the agent was still making genuine progress toward a correct fix, after which its
remaining actions were unproductive (repeating, thrashing, or going in circles).

You do NOT have the reference solution. Judge progress only from the issue and the agent's
own reasoning, commands, and the outputs it saw. "Progress" means moving toward a correct
fix: locating the real fault, making a code change that addresses it, or gaining information
that a correct fix requires. Re-reading files already seen, re-running the same failing test,
re-applying edits already tried, or editing unrelated code is NOT progress.

ISSUE:
{issue}

AGENT STEPS (index: reasoning | command | result):
{steps}

Return ONLY a JSON object, no other text:
{{"last_progress_step": <integer index>, "confidence": <0.0-1.0>, "reason": "<one sentence>"}}
The index must be one of the step indices shown above."""


def render_steps(steps):
    out = []
    for s in steps:
        line = f"{s['i']}: {s['say']}"
        if s.get("cmd"):
            line += f" | $ {s['cmd']}"
        if s.get("obs"):
            line += f" | -> {s['obs']}"
        out.append(line)
    return "\n".join(out)


def ask(model, prompt, max_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens}
    if effort:
        body["reasoning"] = {"effort": effort}
    r = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Authorization": f"Bearer {KEY}",
                                        "Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    ch = d["choices"][0]
    m = ch.get("message", {}) or {}
    return (m.get("content") or ""), ch.get("finish_reason"), d.get("usage", {})


def parse(txt):
    m = re.search(r'\{[^{}]*"last_progress_step"[^{}]*\}', txt, re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        return int(o["last_progress_step"])
    except Exception:
        m2 = re.search(r'"last_progress_step"\s*:\s*(\d+)', txt)
        return int(m2.group(1)) if m2 else None


def judge_one(model, rec, results, lock):
    prompt = PROMPT.format(issue=rec["issue"][:4000], steps=render_steps(rec["steps"]))
    idx = None
    cost = 0.0
    for mt, eff in ((3000, "low"), (8000, "medium")):
        try:
            txt, fin, usage = ask(model, prompt, mt, eff)
        except Exception as e:
            with lock:
                print(f"  [{rec['uid'][:16]}] ERROR {type(e).__name__}", flush=True)
            return
        cost += usage.get("cost", 0) or 0
        idx = parse(txt)
        if idx is not None:
            break
    with lock:
        results.append(dict(uid=rec["uid"], iid=rec["iid"], n=rec["n"],
                            tstar=rec["tstar"], judge=idx, cost=cost))
        print(f"  [{rec['uid'][:16]}] t*={rec['tstar']:3d} judge={idx} n={rec['n']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    recs = json.load(open(SAMPLE, encoding="utf-8"))
    if a.limit:
        recs = recs[:a.limit]
    tag = a.model.split("/")[-1]
    out = os.path.join(DATA, f"adj_{tag}.json")
    results, lock = [], threading.Lock()
    if a.resume and os.path.exists(out):
        results = [r for r in json.load(open(out, encoding="utf-8")) if r.get("judge") is not None]
        have = {r["uid"] for r in results}
        recs = [r for r in recs if r["uid"] not in have]
        print(f"[{tag}] resume: {len(have)} kept, {len(recs)} remaining")
    print(f"[{tag}] judging {len(recs)} trajectories, {a.workers} workers")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(lambda r: judge_one(a.model, r, results, lock), recs))
    json.dump(results, open(out, "w"), indent=1)
    got = [r for r in results if r["judge"] is not None]
    print(f"[{tag}] parsed {len(got)}/{len(results)}  cost ${sum(r['cost'] for r in results):.3f}  -> {out}")


if __name__ == "__main__":
    main()
