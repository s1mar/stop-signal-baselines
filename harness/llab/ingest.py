"""Ingest Nebius SWE-agent trajectories + join gold patches.

Streaming only: we never download the full 80k corpus. A bounded scan collects
a class-balanced sample (failed long trajectories = the population of interest,
plus resolved trajectories for calibration and the false-abort risk axis).
"""
from __future__ import annotations
import itertools
from datasets import load_dataset
from .schema import Trajectory, Step
from .actions import parse_action, attach_current_file

NEBIUS = "nebius/SWE-agent-trajectories"
GOLD_SOURCES = ["nebius/SWE-bench-extra", "princeton-nlp/SWE-bench"]


def normalize(row: dict) -> Trajectory:
    traj_raw = row.get("trajectory") or []
    # observation for an ai step = text of the next user step
    ai_steps: list[Step] = []
    ai_counter = 0
    for i, s in enumerate(traj_raw):
        if s.get("role") != "ai":
            continue
        obs = ""
        for j in range(i + 1, len(traj_raw)):
            if traj_raw[j].get("role") == "user":
                obs = traj_raw[j].get("text") or ""
                break
            if traj_raw[j].get("role") == "ai":
                break
        text = s.get("text") or ""
        step = Step(index=ai_counter, raw_index=i, role="ai", text=text,
                    action=parse_action(text), observation=obs)
        ai_steps.append(step)
        ai_counter += 1
    attach_current_file([s.action for s in ai_steps if s.action])
    return Trajectory(
        instance_id=row.get("instance_id", ""),
        model_name=row.get("model_name", ""),
        resolved=bool(row.get("target")),
        steps=ai_steps,
        generated_patch=row.get("generated_patch", "") or "",
        exit_status=row.get("exit_status", "") or "",
        source="nebius",
    )


def sample_trajectories(n_failed=400, n_resolved=150, min_steps=15,
                        scan_cap=12000, seed=0) -> list[Trajectory]:
    """Bounded streaming scan → normalized trajectories (unjoined to gold)."""
    ds = load_dataset(NEBIUS, split="train", streaming=True)
    failed, resolved = [], []
    for row in itertools.islice(ds, scan_cap):
        traj = row.get("trajectory")
        if not isinstance(traj, list):
            continue
        n_ai = sum(1 for s in traj if s.get("role") == "ai")
        if n_ai < min_steps:
            continue
        if row.get("target") is False and len(failed) < n_failed:
            failed.append(normalize(row))
        elif row.get("target") is True and len(resolved) < n_resolved:
            resolved.append(normalize(row))
        if len(failed) >= n_failed and len(resolved) >= n_resolved:
            break
    return failed + resolved


def build_gold_index(instance_ids: set[str], scan_cap=60000) -> dict[str, str]:
    """Map instance_id -> gold patch, scanning the gold sources once each."""
    want = set(instance_ids)
    gold: dict[str, str] = {}
    for src in GOLD_SOURCES:
        if not want - set(gold):
            break
        try:
            splits = ["train"] if "extra" in src else ["dev"]
            for split in splits:
                gds = load_dataset(src, split=split, streaming=True)
                for g in itertools.islice(gds, scan_cap):
                    iid = g.get("instance_id")
                    if iid in want and iid not in gold:
                        patch = g.get("patch") or g.get("gold_patch") or ""
                        if patch:
                            gold[iid] = patch
                    if not want - set(gold):
                        break
        except Exception as e:
            print(f"[gold] source {src} failed: {e!r}")
    return gold


def load_joined(**kw) -> list[Trajectory]:
    """Sample trajectories and attach gold patches; drop those without gold."""
    trajs = sample_trajectories(**kw)
    gold = build_gold_index({t.instance_id for t in trajs})
    out = []
    for t in trajs:
        if t.instance_id in gold:
            t.gold_patch = gold[t.instance_id]
            out.append(t)
    print(f"[ingest] sampled {len(trajs)}, joined to gold {len(out)} "
          f"({sum(t.resolved for t in out)} resolved / "
          f"{sum(not t.resolved for t in out)} failed)")
    return out
