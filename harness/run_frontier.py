"""Frontier RWR: compute RWR + oracle validation for SWE-agent runs across the
capability ladder (open Devstral -> frontier Claude 4) on SWE-bench Verified,
same scaffold. Applies additional controls: truncation-stratified (exit_status), and
a failure-intersection pass (identical hard tasks) to kill hard-task survivorship.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from llab import ingest_traj, embed
from llab.oracle import compute_oracle

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
# Devstral and Claude-3.7 submissions did not upload trajectories to S3; the three
# below have public .traj files (2024 GPT-4o/Claude-3.5 + 2025 Claude-4).
MODELS = ["gpt-4o", "claude-3.5-sonnet", "claude-4-sonnet"]


def is_truncated(exit_status: str) -> bool:
    # exit_status == "submitted" means the agent stopped on its own; anything else
    # (exit_cost, exit_context, exit_api, max_steps, ...) is a cap/error, i.e. truncated.
    return exit_status.strip().lower() != "submitted"


def rwr(rows, iids=None, nontrunc=False):
    f = [r for r in rows if not r["resolved"]
         and (iids is None or r["iid"] in iids)
         and (not nontrunc or not r["truncated"])]
    tot = sum(r["n"] for r in f)
    rec = sum(max(0, r["n"] - 1 - r["tstar"]) for r in f if r["tstar"] is not None)
    return (rec / tot if tot else float("nan")), len(f)


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def run(max_failed=150, max_resolved=80):
    facts = {}
    for m in MODELS:
        trajs = ingest_traj.ingest_submission(m, max_failed=max_failed, max_resolved=max_resolved)
        rows = []
        for t in trajs:
            orc = compute_oracle(t)
            if orc is None:
                continue
            rows.append(dict(iid=t.instance_id, resolved=t.resolved, n=t.n_steps,
                             tstar=orc.tstar, exit=t.exit_status,
                             truncated=is_truncated(t.exit_status),
                             maxprog=orc.max_progress))
        facts[m] = rows
        embed.save_cache()

    # failure intersection: instances that EVERY model failed (from full results.json)
    full_failed = {}
    for m in MODELS:
        folder = ingest_traj.SUBMISSIONS[m][0]
        res = ingest_traj.results(folder)
        skip = set(res.get("no_generation", [])) | set(res.get("no_logs", []))
        # a task is "failed" if not resolved and the model did generate something
        full_failed[m] = None  # placeholder; intersection uses sampled rows below
    inter = None
    for m in MODELS:
        s = set(r["iid"] for r in facts[m] if not r["resolved"])
        inter = s if inter is None else (inter & s)

    print("\n=== FRONTIER RWR (SWE-bench Verified, same SWE-agent scaffold) ===")
    print(f"{'model':22s} {'RWRall':>7} {'nF':>4} {'RWR(no-trunc)':>13} {'nNT':>4} "
          f"{'%trunc':>6} {'oracleAUC':>9}")
    out = {"models": {}}
    for m in MODELS:
        rows = facts[m]
        rall, nf = rwr(rows)
        rnt, nnt = rwr(rows, nontrunc=True)
        ntr = sum(1 for r in rows if not r["resolved"] and r["truncated"])
        res_mp = [r["maxprog"] for r in rows if r["resolved"]]
        fail_mp = [r["maxprog"] for r in rows if not r["resolved"]]
        a = auc(res_mp, fail_mp)
        pct_tr = ntr / nf if nf else float("nan")
        print(f"{ingest_traj.SUBMISSIONS[m][1]:22s} {rall:7.3f} {nf:4d} {rnt:13.3f} {nnt:4d} "
              f"{pct_tr:6.2f} {a:9.3f}")
        out["models"][m] = dict(display=ingest_traj.SUBMISSIONS[m][1], rwr_all=rall, n_failed=nf,
                                rwr_nontrunc=rnt, n_nontrunc=nnt, pct_truncated=pct_tr, oracle_auc=a)

    print(f"\nfailure-intersection (tasks ALL {len(MODELS)} models failed, sampled): "
          f"n={len(inter) if inter else 0}")
    out["intersection_n"] = len(inter) if inter else 0
    if inter:
        print(f"{'model':22s} {'RWR@intersection':>16} {'n':>4}")
        out["intersection"] = {}
        for m in MODELS:
            ri, ni = rwr(facts[m], iids=inter)
            print(f"{ingest_traj.SUBMISSIONS[m][1]:22s} {ri:16.3f} {ni:4d}")
            out["intersection"][m] = dict(rwr=ri, n=ni)

    json.dump(out, open(os.path.join(DATA, "frontier_rwr.json"), "w"), indent=1, default=float)
    print("\nwrote data/frontier_rwr.json")


if __name__ == "__main__":
    run()
