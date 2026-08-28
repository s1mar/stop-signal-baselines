"""The detector zoo: every published stop-signal family behind one interface.

A Detector maps a trajectory to a per-step alarm score (higher = 'more likely a
doomed/stuck run, stop now'). fire_step(threshold) is the first step whose score
reaches the threshold. Sweeping the threshold traces the savings-vs-false-abort
frontier. Each detector is a faithful reimplementation of its source family; see
notes/novelty_memo.md for citations.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
from .schema import Trajectory
from . import embed


@dataclass
class DetectorOutput:
    name: str
    scores: list[float]           # aligned to ai-step indices 0..n-1
    available: bool = True
    note: str = ""

    def fire_step(self, threshold: float) -> int | None:
        for i, s in enumerate(self.scores):
            if s >= threshold:
                return i
        return None


# ---------------------------------------------------------------- detectors ---

def syntactic_repetition(traj: Trajectory, window: int = 8) -> DetectorOutput:
    """Reflexion-style: how many times the current normalized action has
    recurred in the last `window` steps. Exact/normalized match, so paraphrased
    stagnation is undercounted by design."""
    norm = [s.action.normalized() if s.action else f"__noop_{i}"
            for i, s in enumerate(traj.steps)]
    scores = []
    for i in range(len(norm)):
        lo = max(0, i - window + 1)
        scores.append(float(norm[lo:i + 1].count(norm[i])))
    return DetectorOutput("syntactic", scores)


def convergence_monitor(traj: Trajectory, w: int = 5) -> DetectorOutput:
    """Published Convergence Monitor: embedding diversity over a sliding window.
    Score = mean pairwise cosine SIMILARITY in the last w action embeddings
    (high similarity = low diversity = stuck). Encoder all-MiniLM-L6-v2."""
    texts = [s.text or "" for s in traj.steps]
    if not texts:
        return DetectorOutput("convergence", [])
    V = embed.encode(texts)
    scores = []
    for i in range(len(texts)):
        lo = max(0, i - w + 1)
        win = V[lo:i + 1]
        if len(win) < 2:
            scores.append(0.0)
            continue
        sims = [float(np.dot(win[a], win[b]))
                for a in range(len(win)) for b in range(a + 1, len(win))]
        scores.append(float(np.mean(sims)))
    return DetectorOutput("convergence", scores)


_FAIL_TOK = re.compile(r"traceback|error|exception|no such file|not found|"
                       r"failed|fatal|cannot|command not found|syntaxerror",
                       re.IGNORECASE)
_PASS_TOK = re.compile(r"\bpassed\b|\bok\b|success|no errors|tests? pass",
                       re.IGNORECASE)


def prm_proxy(traj: Trajectory) -> DetectorOutput:
    """Process-reward proxy (transparent, not a trained PRM): the length of the
    current run of consecutive steps whose observation shows a failure signal and
    no pass signal. A long failing run = low process reward = stop."""
    run = 0
    scores = []
    for s in traj.steps:
        obs = s.observation or ""
        fail = bool(_FAIL_TOK.search(obs)) and not bool(_PASS_TOK.search(obs))
        run = run + 1 if fail else 0
        scores.append(float(run))
    return DetectorOutput("prm_proxy", scores,
                          note="heuristic proxy for a process reward model")


_GIVEUP = re.compile(
    r"i'?m not sure|not sure why|unable to|can'?t seem|still (failing|not)|"
    r"doesn'?t (seem to )?work|isn'?t working|different approach|let me try another|"
    r"unfortunately|i apologize|seems like i|stuck|no luck|revert", re.IGNORECASE)


def self_report(traj: Trajectory, window: int = 5) -> DetectorOutput:
    """BAGEN-style self-reported non-progress: accumulation of give-up/uncertainty
    markers in the agent's own text over the last `window` steps."""
    flags = [1.0 if _GIVEUP.search(s.text or "") else 0.0 for s in traj.steps]
    scores = []
    for i in range(len(flags)):
        lo = max(0, i - window + 1)
        scores.append(float(sum(flags[lo:i + 1])))
    return DetectorOutput("self_report", scores)


def logprob_unavailable(traj: Trajectory) -> DetectorOutput:
    """AgentStop-style token log-prob confidence. Nebius trajectories do not carry
    token log-probs, so this family cannot be evaluated on the replay corpus. It is
    reported as UNAVAILABLE and measured on a local logprob-emitting subset."""
    return DetectorOutput("logprob", [0.0] * traj.n_steps, available=False,
                          note="no token log-probs in Nebius; needs local runs")


ALL_DETECTORS = {
    "syntactic": syntactic_repetition,
    "convergence": convergence_monitor,
    "prm_proxy": prm_proxy,
    "self_report": self_report,
    "logprob": logprob_unavailable,
}
# llm_judge is added by the driver only when enabled (needs model calls).
