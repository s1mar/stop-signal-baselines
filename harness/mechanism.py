"""MECHANISM: why do detectors have usable AUC but almost no coverage at a strict budget?

Reviewer ask (Fable): "Show it's an ROC-shape/tail-behavior phenomenon, e.g. failed and resolved
runs are indistinguishable early and separable only late, when the savings are gone." That converts
"we measured a failure" into "we understand a failure".

Two pre-specified analyses, both on existing data:

A. SEPARABILITY OVER TIME. At each relative position p in the run (p = step/n), compute the AUC of
   the detector's running score for failed vs resolved runs. If early AUC is ~0.5 and only rises
   late, the signal is a lagging indicator: by the time it separates, most of the run is spent.

B. THE PRICE OF SEPARATION. For each detector, sweep the threshold and plot the achievable
   (false-abort, coverage) pair, then report how much compute remains at the point where the
   detector first fires on failed runs. If firing early enough to save compute forces the threshold
   into a region where resolved runs also trip it, the bottleneck is the score DISTRIBUTIONS
   overlapping, not the timing rule.

Writes data/mechanism.json.
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import detectors as det, embed
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS = os.path.join(DATA, "corpus_strat.json")
ORDER = ["convergence", "syntactic", "prm_proxy", "self_report"]
POSITIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def running_max(scores, upto):
    s = [x for x in scores[:upto] if x is not None]
    return max(s) if s else 0.0


def main():
    rows = json.load(open(CORPUS, encoding="utf-8"))
    failed = [e for e in rows if not e["resolved"]]
    resolved = [e for e in rows if e["resolved"]]
    print(f"corpus: {len(failed)} failed / {len(resolved)} resolved\n")

    out = {"separability_over_time": {}, "score_overlap": {}}

    # ---- A. separability over time -------------------------------------------------
    print("=== A. separability over time: AUC(failed vs resolved) of the running score ===")
    print(f"{'detector':14s} " + " ".join(f"{int(p*100):>5}%" for p in POSITIONS))
    for d in ORDER:
        aucs = []
        for p in POSITIONS:
            fp, rp = [], []
            for e in failed:
                sc = e["scores"][d]
                if sc:
                    fp.append(running_max(sc, max(1, int(round(p * len(sc))))))
            for e in resolved:
                sc = e["scores"][d]
                if sc:
                    rp.append(running_max(sc, max(1, int(round(p * len(sc))))))
            aucs.append(auc(fp, rp))
        out["separability_over_time"][d] = [float(x) for x in aucs]
        print(f"{d:14s} " + " ".join(f"{x:6.3f}" for x in aucs))

    # ---- B. score-distribution overlap at the operating region ---------------------
    print("\n=== B. why the threshold cannot be lowered: score overlap ===")
    print(f"{'detector':14s} {'thr@5%fa':>9} {'failed>=thr':>12} {'resolved>=thr':>14} {'overlap':>8}")
    for d in ORDER:
        fmax = [running_max(e["scores"][d], len(e["scores"][d])) for e in failed if e["scores"][d]]
        rmax = [running_max(e["scores"][d], len(e["scores"][d])) for e in resolved if e["scores"][d]]
        if not fmax or not rmax:
            continue
        # threshold that keeps false-abort <= 5% on resolved run-level maxima
        thr = float(np.percentile(rmax, 95))
        f_hit = float(np.mean([x >= thr for x in fmax]))
        r_hit = float(np.mean([x >= thr for x in rmax]))
        # overlap coefficient of the two run-level max distributions
        lo, hi = min(min(fmax), min(rmax)), max(max(fmax), max(rmax))
        bins = np.linspace(lo, hi, 41) if hi > lo else np.array([lo, lo + 1])
        hf, _ = np.histogram(fmax, bins=bins, density=True)
        hr, _ = np.histogram(rmax, bins=bins, density=True)
        w = np.diff(bins)
        ov = float(np.sum(np.minimum(hf, hr) * w))
        out["score_overlap"][d] = dict(thr_at_5pct_fa=thr, failed_reaching=f_hit,
                                       resolved_reaching=r_hit, overlap=ov,
                                       auc_runlevel=float(auc(fmax, rmax)))
        print(f"{d:14s} {thr:9.3f} {f_hit:12.3f} {r_hit:14.3f} {ov:8.3f}")

    json.dump(out, open(os.path.join(DATA, "mechanism.json"), "w"), indent=1)
    print("\nwrote data/mechanism.json")


if __name__ == "__main__":
    main()
