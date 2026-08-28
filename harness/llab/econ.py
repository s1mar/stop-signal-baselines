"""Economic evaluation: Recoverable-Waste Ratio, Oracle Regret, and the frontier.

Cost is measured in agent STEPS (unambiguous; a char-proxy for tokens is also
reported). For a trajectory of n steps with Oracle Stop t*, the recoverable waste
is the steps strictly after t*. A detector that fires at step f recovers the steps
after f (on failed runs) but risks aborting a run that would have succeeded (on
resolved runs). Sweeping each detector's threshold traces (false-abort, saved).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def waste_after(t: int | None, n: int) -> int:
    if t is None:
        return 0
    return max(0, n - 1 - t)


@dataclass
class Corpus:
    """Precomputed per-trajectory facts, split by outcome."""
    # each entry: dict(n, tstar, char_after_tstar, char_total, scores_by_detector)
    failed: list[dict]
    resolved: list[dict]

    def rwr_steps(self) -> float:
        tot = sum(e["n"] for e in self.failed) or 1
        rec = sum(waste_after(e["tstar"], e["n"]) for e in self.failed)
        return rec / tot

    def rwr_chars(self) -> float:
        tot = sum(e["char_total"] for e in self.failed) or 1
        rec = sum(e["char_after_tstar"] for e in self.failed)
        return rec / tot


def bootstrap_rwr(corpus: "Corpus", n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap 95% CI for the step-RWR over failed trajectories."""
    rng = np.random.default_rng(seed)
    ns = np.array([e["n"] for e in corpus.failed], dtype=float)
    rec = np.array([waste_after(e["tstar"], e["n"]) for e in corpus.failed], dtype=float)
    m = len(ns)
    if m == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        tot = ns[idx].sum()
        vals.append(rec[idx].sum() / tot if tot else 0.0)
    return dict(mean=float(np.mean(vals)),
                lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)))


def savings_at_risk(corpus: "Corpus", detector: str, thresholds: list[float],
                    risk_levels=(0.05, 0.10, 0.20)) -> dict:
    pts = frontier(corpus, detector, thresholds)
    out = {}
    for r in risk_levels:
        feasible = [p["saved_frac"] for p in pts if p["false_abort"] <= r]
        out[f"{int(r*100)}"] = max(feasible) if feasible else 0.0
    return out


def _fire(scores: list[float], thr: float) -> int | None:
    for i, s in enumerate(scores):
        if s >= thr:
            return i
    return None


def frontier(corpus: Corpus, detector: str, thresholds: list[float]) -> list[dict]:
    """Trace (false_abort, saved_frac, median_regret) over thresholds."""
    tot_failed_steps = sum(e["n"] for e in corpus.failed) or 1
    n_resolved = len(corpus.resolved) or 1
    pts = []
    for thr in thresholds:
        saved = 0
        regrets = []
        for e in corpus.failed:
            f = _fire(e["scores"][detector], thr)
            if f is not None:
                saved += waste_after(f, e["n"])
                if e["tstar"] is not None:
                    regrets.append(f - e["tstar"])
        false_abort = 0
        for e in corpus.resolved:
            f = _fire(e["scores"][detector], thr)
            if f is not None and f < e["n"] - 1:
                false_abort += 1
        pts.append(dict(
            threshold=thr,
            false_abort=false_abort / n_resolved,
            saved_frac=saved / tot_failed_steps,
            median_regret=float(np.median(regrets)) if regrets else float("nan"),
            fire_rate=sum(1 for e in corpus.failed
                          if _fire(e["scores"][detector], thr) is not None)
                     / (len(corpus.failed) or 1),
        ))
    return pts


def pareto_upper_left(pts: list[dict]) -> list[dict]:
    """Keep points that are not dominated (lower false_abort AND higher saved)."""
    order = sorted(pts, key=lambda p: (p["false_abort"], -p["saved_frac"]))
    best = -1.0
    keep = []
    for p in order:
        if p["saved_frac"] > best:
            keep.append(p)
            best = p["saved_frac"]
    return keep


def regret_at_risk(corpus: Corpus, detector: str, thresholds: list[float],
                   target_risk: float = 0.05) -> dict:
    """Operating point: the threshold with false_abort closest to (but <=) target.
    Report the Oracle Regret distribution there."""
    pts = frontier(corpus, detector, thresholds)
    feasible = [p for p in pts if p["false_abort"] <= target_risk]
    if feasible:
        op = max(feasible, key=lambda p: p["saved_frac"])
    else:
        op = min(pts, key=lambda p: p["false_abort"])
    regrets = []
    for e in corpus.failed:
        f = _fire(e["scores"][detector], op["threshold"])
        if f is not None and e["tstar"] is not None:
            regrets.append(f - e["tstar"])
    return dict(detector=detector, operating=op,
                median_regret=float(np.median(regrets)) if regrets else float("nan"),
                mean_regret=float(np.mean(regrets)) if regrets else float("nan"),
                n_fired=len(regrets))
