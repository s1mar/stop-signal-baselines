"""FALSE-ABORT SENSITIVITY: does the coverage collapse survive a more forgiving accounting?

The frontier counts any alarm before completion on a RESOLVED run as a lost resolution. That is
the safe deployment assumption, but it is pessimistic: a resolved run that has already written the
patch that makes the tests pass, and is then doing test runs or cleanup, loses little if aborted.
Because the operating threshold is set by the extreme tail of the resolved population, even a few
such cases could move it, so the level of the whole frontier depends on this convention.

We recompute every operating point under two additional accountings, both computed by replay from
cached scores with no new inference and no use of the gold patch:

  strict   (as published) an alarm anywhere before the last step is a false abort;
  lenient  an alarm at or after the resolved run's LAST APPLIED EDIT is not a false abort, on the
           grounds that the code that resolved the task was already written;
  editwin  as lenient, but the exemption starts 3 steps BEFORE the last applied edit, to allow for
           the patch being complete slightly before the final edit action.

Only the false-abort bookkeeping changes. Coverage and saved fraction on failed runs are computed
exactly as before, so any movement in them comes only from the threshold the budget now permits.

Writes data/p1_falseabort_sensitivity.json.
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import econ

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ORDER = ["syntactic", "convergence", "prm_proxy", "self_report"]
LABEL = {"syntactic": "Syntactic repetition", "convergence": "Embedding diversity",
         "prm_proxy": "Process-reward proxy", "self_report": "Self-report"}
BUDGETS = [0.05, 0.10, 0.20]
WINDOW = 3


def fire(scores, thr):
    for i, s in enumerate(scores):
        if s is not None and s >= thr:
            return i
    return None


def grid(rows, det):
    """The CANONICAL threshold grid, imported so it cannot drift from the main frontier.

    An earlier version of this script built its own grid from the observed score values, taking a
    60-point quantile grid whenever a detector had more than 60 distinct values. For prm_proxy
    (262 distinct values) that grid stepped 12.0 -> 13.27 and SKIPPED the optimum at 12, so the
    strict column understated the detector (0.115/0.163 instead of 0.125/0.185) and the paper
    explained the gap away as a harmless difference of granularity. It was not harmless: a
    self-built grid that misses the feasible optimum biases every column that uses it. Using the
    same thresholds as deep_analysis.py makes the strict column reproduce Table 1 exactly, so the
    comparison across accountings needs no caveat at all.
    """
    from deep_analysis import THRESHOLDS
    return list(THRESHOLDS[det])


def main():
    rows = json.load(open(os.path.join(DATA, "corpus_strat.json"), encoding="utf-8"))
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))

    # last applied edit step per uid, from the trajectory objects (no gold patch involved)
    last_edit = {}
    for t in trajs:
        idx = [s.index for s in t.steps if s.action and s.action.is_edit]
        last_edit[t.uid] = max(idx) if idx else None

    failed = [e for e in rows if not e["resolved"]]
    resolved = [e for e in rows if e["resolved"]]
    n_missing = sum(1 for e in resolved if e["uid"] not in last_edit)
    n_noedit = sum(1 for e in resolved if last_edit.get(e["uid"]) is None)
    print(f"{len(failed)} failed / {len(resolved)} resolved; "
          f"{n_missing} resolved uids absent from the pickle, {n_noedit} with no edit action\n")

    tot_failed_steps = sum(e["n"] for e in failed) or 1

    def false_abort(det, thr, mode):
        bad = 0
        for e in resolved:
            f = fire(e["scores"][det], thr)
            if f is None or f >= e["n"] - 1:
                continue
            if mode == "strict":
                bad += 1
                continue
            le = last_edit.get(e["uid"])
            if le is None:                              # cannot exempt: count it
                bad += 1
                continue
            cut = le if mode == "lenient" else max(0, le - WINDOW)
            if f < cut:
                bad += 1
        return bad / (len(resolved) or 1)

    def failed_side(det, thr):
        saved = cov = 0
        for e in failed:
            f = fire(e["scores"][det], thr)
            if f is not None:
                cov += 1
                saved += econ.waste_after(f, e["n"])
        return saved / tot_failed_steps, cov / (len(failed) or 1)

    out = {"window": WINDOW, "modes": {}}
    for mode in ("strict", "lenient", "editwin"):
        print(f"=== accounting: {mode} ===")
        print(f"{'family':22s} " + "  ".join(f"{'sv@'+str(int(b*100))+'%':>7} {'cov':>5}"
                                             for b in BUDGETS))
        out["modes"][mode] = {}
        for det in ORDER:
            ths = grid(rows, det)
            pts = [(t, false_abort(det, t, mode)) + failed_side(det, t) for t in ths]
            row = {}
            cells = []
            for b in BUDGETS:
                ok = [p for p in pts if p[1] <= b]
                best = max(ok, key=lambda p: p[2]) if ok else None
                row[str(b)] = (dict(threshold=float(best[0]), false_abort=best[1],
                                    saved=best[2], coverage=best[3])
                               if best else dict(threshold=None, false_abort=None,
                                                 saved=0.0, coverage=0.0))
                cells.append(f"{row[str(b)]['saved']:7.3f} {row[str(b)]['coverage']:5.2f}")
            out["modes"][mode][det] = row
            print(f"{LABEL[det]:22s} " + "  ".join(cells))
        print()

    # headline comparison at the 5% budget
    print("=== headline: 5% budget, strict -> lenient -> editwin ===")
    for det in ORDER:
        s = [out["modes"][m][det]["0.05"] for m in ("strict", "lenient", "editwin")]
        print(f"{LABEL[det]:22s} saved {s[0]['saved']:.3f} -> {s[1]['saved']:.3f} -> "
              f"{s[2]['saved']:.3f}   coverage {s[0]['coverage']:.3f} -> "
              f"{s[1]['coverage']:.3f} -> {s[2]['coverage']:.3f}")
    best = {m: max(out["modes"][m][d]["0.05"]["coverage"] for d in ORDER)
            for m in ("strict", "lenient", "editwin")}
    print(f"\nbest coverage at the 5% budget: strict {best['strict']:.3f}, "
          f"lenient {best['lenient']:.3f}, editwin {best['editwin']:.3f}")
    out["best_coverage_at_5pct"] = best
    out["n_resolved_no_edit"] = n_noedit

    json.dump(out, open(os.path.join(DATA, "p1_falseabort_sensitivity.json"), "w"), indent=1)
    print("\nwrote data/p1_falseabort_sensitivity.json")


if __name__ == "__main__":
    main()
