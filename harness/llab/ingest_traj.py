"""Ingest frontier SWE-agent .traj trajectories from the public SWE-bench
experiments submissions (SWE-bench Verified split). Same scaffold as the Nebius
corpus, different (frontier) action-generator models.

Data sources (all public, free):
  - trajectories: s3://swe-bench-submissions/verified/<folder>/trajs/<iid>.traj
  - resolved/failed labels: experiments repo <folder>/results/results.json
  - gold patches: princeton-nlp/SWE-bench_Verified (HF)
"""
from __future__ import annotations
import json, os, re, urllib.request
from .schema import Trajectory, Step
from .actions import parse_command, attach_current_file

S3 = "https://swe-bench-submissions.s3.amazonaws.com"
REPO = "https://raw.githubusercontent.com/SWE-bench/experiments/main/evaluation/verified"
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "frontier_trajs")

# label -> (submission folder, display model name)
SUBMISSIONS = {
    "devstral-small":     ("20250725_sweagent_devstral_small_2507",   "Devstral-Small (open)"),
    "gpt-4o":             ("20240728_sweagent_gpt4o",                 "GPT-4o"),
    "claude-3.5-sonnet":  ("20240620_sweagent_claude3.5sonnet",       "Claude 3.5 Sonnet"),
    "claude-3.7-sonnet":  ("20250225_sweagent_claude-3-7-sonnet",     "Claude 3.7 Sonnet"),
    "claude-4-sonnet":    ("20250522_sweagent_claude-4-sonnet-20250514", "Claude 4 Sonnet"),
}

_GOLD = None


def _get(url: str, timeout=60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def load_gold_verified() -> dict[str, str]:
    global _GOLD
    if _GOLD is None:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        _GOLD = {r["instance_id"]: r["patch"] for r in ds}
    return _GOLD


def results(folder: str) -> dict:
    return json.loads(_get(f"{REPO}/{folder}/results/results.json"))


def list_traj_keys(folder: str) -> list[tuple[str, str]]:
    """Return [(instance_id, s3_key)] for a submission's trajs (single page; ~500)."""
    xml = _get(f"{S3}/?list-type=2&prefix=verified/{folder}/trajs/&max-keys=1000").decode()
    keys = re.findall(r"<Key>([^<]+\.traj)</Key>", xml)
    out = []
    for k in keys:
        iid = os.path.basename(k)[:-5]  # strip .traj
        out.append((iid, k))
    return out


def download_traj(key: str, folder: str) -> str:
    d = os.path.join(CACHE, folder)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, os.path.basename(key))
    if not os.path.exists(path):
        data = _get(f"{S3}/{urllib.parse.quote(key)}")
        with open(path, "wb") as f:
            f.write(data)
    return path


def parse_traj(path: str, instance_id: str, model: str, resolved: bool) -> Trajectory | None:
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    traj = d.get("trajectory") or []
    steps = []
    for i, st in enumerate(traj):
        act = (st.get("action") or "").strip()
        action = parse_command(act) if act else None
        text = st.get("thought") or st.get("response") or act
        steps.append(Step(index=i, raw_index=i, role="ai", text=text,
                          action=action, observation=st.get("observation") or ""))
    attach_current_file([s.action for s in steps if s.action])
    info = d.get("info") or {}
    return Trajectory(instance_id=instance_id, model_name=model, resolved=resolved,
                      steps=steps, generated_patch=info.get("submission") or "",
                      exit_status=info.get("exit_status") or "", source="swebench-verified")


def ingest_submission(label: str, max_failed=120, max_resolved=80, seed=0) -> list[Trajectory]:
    import random
    folder, model = SUBMISSIONS[label]
    res = results(folder)
    resolved_ids = set(res.get("resolved", []))
    skip = set(res.get("no_generation", [])) | set(res.get("no_logs", []))
    gold = load_gold_verified()
    keys = list_traj_keys(folder)
    rng = random.Random(seed)
    rng.shuffle(keys)
    failed_k = [(i, k) for i, k in keys if i not in resolved_ids and i not in skip and i in gold]
    res_k = [(i, k) for i, k in keys if i in resolved_ids and i in gold]
    picked = failed_k[:max_failed] + res_k[:max_resolved]
    out = []
    for iid, key in picked:
        path = download_traj(key, folder)
        t = parse_traj(path, iid, model, iid in resolved_ids)
        if t is None or t.n_steps < 3:
            continue
        t.gold_patch = gold[iid]
        out.append(t)
    n_f = sum(1 for t in out if not t.resolved)
    print(f"[{label}] {model}: {len(out)} trajectories ({n_f} failed / {len(out)-n_f} resolved)")
    return out


import urllib.parse  # noqa: E402 (used in download_traj)
