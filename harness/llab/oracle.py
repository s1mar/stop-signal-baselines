"""The gold-patch progress oracle (v2: gold-file-restricted, variation-robust).

Progress at an edit step is measured two ways that must AGREE to be trusted:
  - primary  : token Jaccard between the identifiers the agent has written into
               the GOLD target file(s) and the identifiers added by the gold patch.
               Robust to reformatting/renaming-light textual variation; restricted
               to gold files so throwaway scaffolding does not count as progress.
  - secondary: cosine(embed(agent gold-file content), embed(gold added text)).

The Oracle Stop t* is the last edit step that reached a new MAX progress (the last
time the agent got strictly closer to the gold fix than ever before). Steps after
t* on a FAILED run produced no further movement toward the known solution and no
resolution: recoverable waste. Discriminativeness and the oracle's error are
reported in the study (see notes/feasibility.md and the paper's calibration).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .schema import Trajectory
from .patchtools import (gold_target_files, gold_added_tokens,
                         gold_added_content, agent_goldfile_states)
from . import embed


@dataclass
class OracleResult:
    instance_id: str
    resolved: bool
    n_steps: int
    edit_idx: list[int] = field(default_factory=list)
    prog_primary: list[float] = field(default_factory=list)   # token Jaccard
    prog_secondary: list[float] = field(default_factory=list)  # embedding cos
    tstar: int | None = None            # canonical (primary)
    tstar_secondary: int | None = None
    max_progress: float = 0.0
    contested: bool = False
    edited_gold_file: bool = False


def _last_new_max(vals: list[float], eps: float = 1e-6) -> int | None:
    best, last = -1.0, None
    for i, v in enumerate(vals):
        if v > best + eps:
            best, last = v, i
    return last


def compute_oracle(traj: Trajectory) -> OracleResult | None:
    gfiles = gold_target_files(traj.gold_patch)
    gtokens = gold_added_tokens(traj.gold_patch)
    if not gfiles or not gtokens:
        return None
    states = agent_goldfile_states(traj, gfiles)
    if not states:
        return None
    idx = [i for i, _, _ in states]
    prog_primary = [len(tok & gtokens) / len(gtokens) for _, tok, _ in states]

    # secondary: embedding cosine of cumulative gold-file content vs gold text
    contents = [c for _, _, c in states]
    if any(contents):
        gold_text = gold_added_content(traj.gold_patch)
        vecs = embed.encode([gold_text] + contents)
        g = vecs[0]
        prog_secondary = [max(0.0, float(np.dot(g, vecs[k + 1])))
                          for k in range(len(contents))]
    else:
        prog_secondary = [0.0] * len(states)

    tp = _last_new_max(prog_primary)
    ts = _last_new_max(prog_secondary)
    tstar = idx[tp] if tp is not None else None
    tstar_secondary = idx[ts] if ts is not None else None

    contested = False
    if tstar is not None and tstar_secondary is not None and traj.n_steps > 0:
        contested = abs(tstar - tstar_secondary) / traj.n_steps > 0.2

    return OracleResult(
        instance_id=traj.instance_id, resolved=traj.resolved, n_steps=traj.n_steps,
        edit_idx=idx, prog_primary=prog_primary, prog_secondary=prog_secondary,
        tstar=tstar, tstar_secondary=tstar_secondary,
        max_progress=max(prog_primary) if prog_primary else 0.0,
        contested=contested, edited_gold_file=any(v > 0 for v in prog_primary),
    )
