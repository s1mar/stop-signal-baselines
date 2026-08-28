"""Oracle v3: a CURRENT-STATE progress signal, so t* is a genuine peak that can decline.

v2 (llab/oracle.py) scores the UNION of every gold identifier the agent has ever written, which is
monotone by construction: t* is then only "the last step that introduced a new gold identifier",
and the signal cannot witness an agent undoing correct work (the reviewer's objection).

v3 scores the agent's CURRENT patch instead of its cumulative one. Per gold target file we keep the
agent's ACTIVE edit blocks; a new edit to line range [lo,hi] DROPS any block it overlaps (that text
has been replaced) and installs the new one. Edits the scaffold rejected are not applied. The
agent's current patch is the concatenation of active blocks, so the signal falls when correct work
is overwritten.

Progress is measured two ways against the gold patch's added lines:
  * primary  : LINE-level recall (normalized added lines of gold present in the current patch),
               which Terra and Kimi both preferred over identifier sets because it catches
               right-token-wrong-place and is robust to identifier reuse;
  * secondary: identifier Jaccard on the current patch (the v2 metric, current-state), kept as a
               robustness check.

t* = the LAST step at which the signal attains its GLOBAL maximum (Terra's definition; cleaner than
"last record-setting step" under ties and plateaus).

Known approximation, stated in the paper: edit line ranges live in shifting coordinates once
insertions renumber later lines, so overlap detection is approximate.

Run: python oracle_v3.py
"""
from __future__ import annotations
import json, os, pickle, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import econ
from llab.oracle import compute_oracle
from llab.patchtools import gold_target_files, gold_added_tokens, gold_added_content, identifiers

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EDIT_RANGE = re.compile(r"^edit\s+(\d+):(\d+)", re.M)
FAIL_MARKERS = ("introduced new syntax error", "would have looked")


def norm_lines(text: str) -> set[str]:
    return {ln.strip() for ln in (text or "").splitlines() if ln.strip()}


def current_patch_series(traj, gfiles):
    """[(step_index, current_patch_text)] after each APPLIED edit to a gold file."""
    active: dict[str, list] = {}
    out = []
    for s in traj.steps:
        a = s.action
        if not (a and a.is_edit and a.target_file in gfiles and a.edit_content):
            continue
        if any(m in (s.observation or "") for m in FAIL_MARKERS):
            continue                                   # rejected: file unchanged
        blocks = active.setdefault(a.target_file, [])
        m = EDIT_RANGE.search(a.raw or "")
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            blocks[:] = [b for b in blocks if not (b[0] <= hi and lo <= b[1])]
            blocks.append((lo, hi, a.edit_content))
        else:
            blocks.append((10**9, 10**9, a.edit_content))   # unrangeable: append only
        text = "\n".join(c for f in active for (_, _, c) in active[f])
        out.append((s.index, text))
    return out


def last_global_max(vals):
    if not vals:
        return None
    mx = max(vals)
    for i in range(len(vals) - 1, -1, -1):
        if vals[i] >= mx - 1e-12:
            return i
    return None


def oracle_v3(traj):
    gf = gold_target_files(traj.gold_patch)
    gtok = gold_added_tokens(traj.gold_patch)
    glines = norm_lines(gold_added_content(traj.gold_patch))
    if not gf or not gtok or not glines:
        return None
    series = current_patch_series(traj, gf)
    if not series:
        return None
    idx = [i for i, _ in series]
    prog_line = [len(norm_lines(t) & glines) / len(glines) for _, t in series]
    prog_tok = [len(identifiers(t) & gtok) / len(gtok) for _, t in series]
    k = last_global_max(prog_line)
    return dict(edit_idx=idx, prog_line=prog_line, prog_tok=prog_tok,
                tstar=idx[k] if k is not None else None,
                max_line=max(prog_line), n=traj.n_steps)


def main():
    trajs = pickle.load(open(os.path.join(DATA, "sample_large.pkl"), "rb"))
    rows = []
    for t in trajs:
        if t.resolved:
            continue
        v3 = oracle_v3(t)
        if v3 is None or v3["tstar"] is None or len(v3["prog_line"]) < 2:
            continue
        v2 = compute_oracle(t)
        if v2 is None or v2.tstar is None:
            continue
        p = v3["prog_line"]
        drops = [p[i - 1] - p[i] for i in range(1, len(p)) if p[i] < p[i - 1] - 1e-12]
        rows.append(dict(uid=t.uid, n=t.n_steps,
                         tstar_v2=v2.tstar, tstar_v3=v3["tstar"],
                         ppr_v2=econ.waste_after(v2.tstar, t.n_steps) / t.n_steps,
                         ppr_v3=econ.waste_after(v3["tstar"], t.n_steps) / t.n_steps,
                         n_drops=len(drops), max_drop=max(drops) if drops else 0.0,
                         max_line=v3["max_line"],
                         ends_below_peak=p[-1] < max(p) - 1e-12))
    n = len(rows)
    dec = [r for r in rows if r["n_drops"] > 0]
    tot_n = sum(r["n"] for r in rows) or 1
    ppr2 = sum(econ.waste_after(r["tstar_v2"], r["n"]) for r in rows) / tot_n
    ppr3 = sum(econ.waste_after(r["tstar_v3"], r["n"]) for r in rows) / tot_n
    d = np.array([r["tstar_v3"] - r["tstar_v2"] for r in rows], float)
    out = dict(
        n=n,
        non_monotone_frac=len(dec) / n if n else float("nan"),
        median_max_drop=float(np.median([r["max_drop"] for r in dec])) if dec else 0.0,
        frac_ending_below_peak=float(np.mean([r["ends_below_peak"] for r in rows])),
        ppr_v2=float(ppr2), ppr_v3=float(ppr3),
        tstar_delta_median=float(np.median(d)),
        tstar_agree_within_2=float(np.mean(np.abs(d) <= 2)),
        tstar_v3_earlier_frac=float(np.mean(d < 0)),
    )
    print(f"failed runs with a v3 oracle: {n}")
    print(f"\n--- is the v3 signal actually non-monotone? ---")
    print(f"runs with >=1 decrease        : {len(dec)} ({out['non_monotone_frac']:.1%})")
    print(f"median largest drop (decliners): {out['median_max_drop']:.3f} of gold lines")
    print(f"runs ending below their peak  : {out['frac_ending_below_peak']:.1%}")
    print(f"\n--- v2 (cumulative) vs v3 (current-state) ---")
    print(f"PPR  v2 {ppr2:.3f}   ->   v3 {ppr3:.3f}   (delta {ppr3-ppr2:+.3f})")
    print(f"t* delta (v3-v2): median {out['tstar_delta_median']:+.1f} steps; "
          f"within 2 steps {out['tstar_agree_within_2']:.1%}; v3 earlier {out['tstar_v3_earlier_frac']:.1%}")
    json.dump(out, open(os.path.join(DATA, "oracle_v3.json"), "w"), indent=1)
    print("\nwrote data/oracle_v3.json")


if __name__ == "__main__":
    main()
