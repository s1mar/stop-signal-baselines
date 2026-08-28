"""LLM-as-judge stop-signal (SHP-style), served by a local Ollama model.

A cheap deployable judge reads the issue and the last few (action, observation)
pairs and says whether the agent is still PROGRESSING or STUCK. This is the only
detector that costs inference at run time; that cost asymmetry versus the frozen
embedding / heuristic detectors is itself part of the economic story.
"""
from __future__ import annotations
import hashlib, json, os, time, urllib.request
from .schema import Trajectory

OLLAMA = "http://localhost:11434/api/generate"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3:latest")
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "judge_cache.json")
_CACHE: dict | None = None

_PROMPT = """You monitor a software-engineering agent fixing a GitHub issue. \
At the current step, is the agent still making PROGRESS toward a fix, or is it \
STUCK (repeating itself, thrashing, or not advancing)? Reply with exactly one \
word: PROGRESSING or STUCK.

ISSUE:
{issue}

RECENT STEPS:
{recent}

Answer:"""


def _load():
    global _CACHE
    if _CACHE is None:
        _CACHE = json.load(open(_CACHE_PATH)) if os.path.exists(_CACHE_PATH) else {}
    return _CACHE


def save_cache():
    if _CACHE is not None:
        json.dump(_CACHE, open(_CACHE_PATH, "w"))


def _key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


OLLAMA_CHAT = "http://localhost:11434/api/chat"
_CALL_VERSION = "chat-think-off-v2"   # bump to invalidate stale cached verdicts


def _call(prompt: str) -> str:
    # Use the chat endpoint with thinking disabled: several local models
    # (north-mini-code, laguna-xs) are reasoning models that otherwise spend the
    # whole token budget on hidden reasoning and emit an empty answer.
    body = json.dumps({"model": JUDGE_MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": 8}}).encode()
    req = urllib.request.Request(OLLAMA_CHAT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]


class JudgeUnavailable(RuntimeError):
    """The judge could not be reached / returned nothing usable.

    Never degrade this to a verdict: an Ollama failure (e.g. another session
    swapping the loaded model) silently scoring as PROGRESSING would fabricate
    data that no downstream sanity check can distinguish from a real signal.
    """


def judge_step(issue: str, recent: str, retries: int = 3) -> int:
    """Return 1 if STUCK else 0 (cached). Raises JudgeUnavailable on failure."""
    cache = _load()
    prompt = _PROMPT.format(issue=issue[:500], recent=recent[:900])
    k = _key(_CALL_VERSION + "||" + JUDGE_MODEL + "||" + prompt)
    if k in cache:
        return cache[k]
    last = None
    for attempt in range(retries):
        try:
            resp = (_call(prompt) or "").strip().upper()
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))   # back off; model may be reloading
            continue
        if "STUCK" in resp or "PROGRESS" in resp:
            verdict = 1 if "STUCK" in resp else 0
            cache[k] = verdict
            return verdict
        last = ValueError(f"unparseable judge reply: {resp[:60]!r}")
        time.sleep(1)
    raise JudgeUnavailable(f"{JUDGE_MODEL}: {last}")


def build_prompts(traj: Trajectory) -> list[str]:
    """The exact per-step judge prompt for each step (issue + last 3 act/obs).
    Used both by judge_scores and by the frontier-judge calibration so the
    reference model sees identical inputs."""
    issue = ""
    for s in traj.steps:
        if s.observation and "ISSUE" in (s.observation or "")[:40].upper():
            issue = s.observation
            break
    if not issue and traj.steps:
        issue = traj.steps[0].observation or ""
    prompts, hist = [], []
    for s in traj.steps:
        act = (s.action.raw if s.action else s.text)[:250]
        obs = (s.observation or "")[:250]
        hist.append(f"[action] {act}\n[result] {obs}")
        prompts.append(_PROMPT.format(issue=issue[:500], recent="\n".join(hist[-3:])[:900]))
    return prompts


def judge_scores(traj: Trajectory, max_steps: int | None = None) -> list[float]:
    """Per-step score = run length of consecutive STUCK verdicts.
    max_steps caps how many steps are judged (early steps decide the fire step);
    remaining steps carry the last score forward."""
    issue = ""
    # the issue text is the first user turn's content, captured as an early obs
    for s in traj.steps:
        if s.observation and "ISSUE" in (s.observation or "")[:40].upper():
            issue = s.observation
            break
    if not issue and traj.steps:
        issue = traj.steps[0].observation or ""
    scores, run = [], 0
    hist: list[str] = []
    limit = max_steps if max_steps is not None else len(traj.steps)
    for j, s in enumerate(traj.steps):
        if j >= limit:
            scores.append(scores[-1] if scores else 0.0)
            continue
        act = (s.action.raw if s.action else s.text)[:250]
        obs = (s.observation or "")[:250]
        hist.append(f"[action] {act}\n[result] {obs}")
        recent = "\n".join(hist[-3:])
        stuck = judge_step(issue, recent)
        run = run + 1 if stuck else 0
        scores.append(float(run))
    return scores
