"""Do failed agents ever UNDO their alignment with the gold patch?

The paper's primary progress signal is cumulative gold-token coverage, which is monotone by
construction and therefore cannot witness regression. This script measures regression directly,
by edit-replay rather than from the (frequently windowed) file dumps:

  per gold target file, maintain the agent's ACTIVE edit blocks keyed by their line range;
  a new successful edit to [lo,hi] DROPS any prior block overlapping that range (it has been
  overwritten) and installs the new one. Failed edits are not applied (the scaffold reports
  "introduced new syntax error" / "would have looked" and leaves the file unchanged).

Current-state alignment = |identifiers(active blocks) & gold_added_tokens| / |gold_added_tokens|.
Because overwrites REMOVE prior blocks, this signal can decrease. We report how often it does.

Known approximation (stated in the paper): line ranges across successive edits live in shifting
coordinates (an insertion renumbers later lines), so overlap detection is approximate. It is used
only to characterize how common regression is, not to define t*.
"""
import json, os, pickle, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab.patchtools import gold_target_files, gold_added_tokens, identifiers

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EDIT_RANGE = re.compile(r"^edit\s+(\d+):(\d+)", re.M)
FAIL_MARKERS = ("introduced new syntax error", "would have looked")


def alignment_sequence(traj, gfiles, gtokens):
    """Per-step current-state alignment after each APPLIED edit to a gold file."""
    active = {}           # file -> list of (lo, hi, content)
    seq, unranged = [], 0
    for s in traj.steps:
        a = s.action
        if not (a and a.is_edit and a.target_file in gfiles and a.edit_content):
            continue
        obs = s.observation or ""
        if any(m in obs for m in FAIL_MARKERS):
            continue                      # edit rejected; file unchanged
        m = EDIT_RANGE.search(a.raw or "")
        blocks = active.setdefault(a.target_file, [])
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            blocks[:] = [b for b in blocks if not (b[0] <= hi and lo <= b[1])]
            blocks.append((lo, hi, a.edit_content))
        else:
            unranged += 1                 # 2025 format: cannot model overwrite
            blocks.append((10**9, 10**9, a.edit_content))
        cur = set()
        for _, _, c in blocks:
            cur |= identifiers(c)
        seq.append(len(cur & gtokens) / len(gtokens))
    return seq, unranged


def main():
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))
    rows, skipped_unranged = [], 0
    for t in trajs:
        if t.resolved:
            continue
        gf, gt = gold_target_files(t.gold_patch), gold_added_tokens(t.gold_patch)
        if not gf or not gt:
            continue
        seq, unranged = alignment_sequence(t, gf, gt)
        if len(seq) < 2:
            continue
        if unranged:                       # exclude runs we cannot model overwrites for
            skipped_unranged += 1
            continue
        drops = [seq[i - 1] - seq[i] for i in range(1, len(seq)) if seq[i] < seq[i - 1] - 1e-9]
        rows.append(dict(uid=t.uid, n_edits=len(seq), n_drops=len(drops),
                         max_drop=max(drops) if drops else 0.0,
                         final=seq[-1], peak=max(seq)))
    n = len(rows)
    with_drop = [r for r in rows if r["n_drops"] > 0]
    out = dict(
        n_analyzable=n,
        n_excluded_unrangeable=skipped_unranged,
        n_with_regression=len(with_drop),
        regression_rate=len(with_drop) / n if n else float("nan"),
        median_max_drop_among_regressors=float(np.median([r["max_drop"] for r in with_drop])) if with_drop else 0.0,
        median_drops_among_regressors=float(np.median([r["n_drops"] for r in with_drop])) if with_drop else 0.0,
        median_peak_minus_final=float(np.median([r["peak"] - r["final"] for r in rows])),
        frac_ending_below_peak=float(np.mean([r["final"] < r["peak"] - 1e-9 for r in rows])),
    )
    print(f"analyzable failed runs (>=2 applied gold edits, rangeable): {n}")
    print(f"excluded (2025 format, overwrite not modelable):            {skipped_unranged}")
    print(f"runs whose gold-alignment EVER decreases:                   {len(with_drop)} "
          f"({out['regression_rate']:.1%})")
    if with_drop:
        print(f"  among regressors: median max drop {out['median_max_drop_among_regressors']:.3f} "
              f"of gold tokens, median {out['median_drops_among_regressors']:.0f} drop event(s)")
    print(f"runs ending BELOW their own peak alignment:                 {out['frac_ending_below_peak']:.1%}")
    json.dump(out, open(os.path.join(DATA, "regression_rate.json"), "w"), indent=1)
    print("\nwrote data/regression_rate.json")


if __name__ == "__main__":
    main()
