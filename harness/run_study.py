"""End-to-end P2 study driver.

  python run_study.py --failed 250 --resolved 120 --scan 9000

Outputs (deterministic given the same scan window):
  data/corpus_facts.json   per-trajectory facts (ids, n, t*, contested, ...)
  data/results.json        all headline numbers + frontier points
  paper/figures/frontier.{png,pdf}
Prints a summary the paper's Results section is written around.
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import ingest, detectors as det, econ, embed
from llab.oracle import compute_oracle

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "paper", "figures")

THRESHOLDS = {
    "syntactic": [1, 2, 3, 4, 5, 6, 7, 8],
    "convergence": list(np.round(np.linspace(0.50, 0.99, 25), 4)),
    "prm_proxy": list(range(1, 16)),
    "self_report": [1, 2, 3, 4, 5],
}
DEFENSIBLE_ORDER = ["convergence", "syntactic", "prm_proxy", "self_report"]


def char_cost(traj):
    return [len(s.text or "") + len(s.observation or "") for s in traj.steps]


def build_corpus(trajs):
    failed, resolved, oracle_stats = [], [], []
    contested = 0
    used = 0
    for t in trajs:
        orc = compute_oracle(t)
        if orc is None:
            continue
        used += 1
        contested += int(orc.contested)
        oracle_stats.append(dict(instance_id=t.instance_id, resolved=t.resolved,
                                 n=t.n_steps, tstar=orc.tstar,
                                 tstar_secondary=orc.tstar_secondary,
                                 contested=orc.contested,
                                 max_progress=orc.max_progress,
                                 edited_gold_file=orc.edited_gold_file))
        # detector scores
        scores = {}
        for name, fn in det.ALL_DETECTORS.items():
            out = fn(t)
            scores[name] = out.scores if out.available else None
        cc = char_cost(t)
        tot_c = sum(cc) or 1
        after = sum(c for i, c in enumerate(cc)
                    if orc.tstar is not None and i > orc.tstar)
        entry = dict(instance_id=t.instance_id, n=t.n_steps, tstar=orc.tstar,
                     char_total=tot_c, char_after_tstar=after, scores=scores)
        (resolved if t.resolved else failed).append(entry)
    return econ.Corpus(failed=failed, resolved=resolved), oracle_stats, contested, used


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def run(args):
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    cache = args.cache_file or os.path.join(DATA, "sample.pkl")
    if args.from_cache and os.path.exists(cache):
        trajs = pickle.load(open(cache, "rb"))
        print(f"[cache] loaded {len(trajs)} trajectories from {cache}")
    else:
        trajs = ingest.load_joined(n_failed=args.failed, n_resolved=args.resolved,
                                   min_steps=args.min_steps, scan_cap=args.scan)
    corpus, oracle_stats, contested, used = build_corpus(trajs)
    embed.save_cache()

    n_failed, n_resolved = len(corpus.failed), len(corpus.resolved)
    print(f"\n=== CORPUS: {used} with gold+oracle "
          f"({n_failed} failed / {n_resolved} resolved), contested={contested} ===")

    rwr_steps = corpus.rwr_steps()
    rwr_chars = corpus.rwr_chars()
    rwr_ci = econ.bootstrap_rwr(corpus)
    print(f"RWR (steps) = {rwr_steps:.3f} [95% CI {rwr_ci['lo']:.3f}-{rwr_ci['hi']:.3f}] "
          f"| RWR (char-proxy) = {rwr_chars:.3f}")

    # oracle calibration: discriminativeness (resolved vs failed)
    res = [o for o in oracle_stats if o["resolved"]]
    fail = [o for o in oracle_stats if not o["resolved"]]
    auc_prog = _auc([o["max_progress"] for o in res], [o["max_progress"] for o in fail])
    def tfrac(rows):
        return [o["tstar"] / (o["n"] - 1) for o in rows
                if o["tstar"] is not None and o["n"] > 1]
    r_tf, f_tf = tfrac(res), tfrac(fail)
    contested_frac = contested / used if used else float("nan")
    print(f"Oracle calibration: max-progress AUC(res>fail)={auc_prog:.3f} | "
          f"t*/n resolved={np.median(r_tf):.3f} failed={np.median(f_tf):.3f} | "
          f"contested_frac={contested_frac:.3f}")
    print(f"  resolved max-progress median="
          f"{np.median([o['max_progress'] for o in res]):.3f}, "
          f"failed median={np.median([o['max_progress'] for o in fail]):.3f}")

    results = dict(
        config=vars(args),
        sample=dict(n_failed=n_failed, n_resolved=n_resolved, used=used,
                    contested=contested, contested_frac=contested_frac),
        rwr_steps=rwr_steps, rwr_chars=rwr_chars, rwr_ci=rwr_ci,
        oracle_calibration=dict(
            max_progress_auc=auc_prog,
            tstar_frac_resolved_median=float(np.median(r_tf)) if r_tf else None,
            tstar_frac_failed_median=float(np.median(f_tf)) if f_tf else None,
            resolved_maxprog_median=float(np.median([o['max_progress'] for o in res])),
            failed_maxprog_median=float(np.median([o['max_progress'] for o in fail])),
        ),
        frontiers={}, regret_at_risk={}, unavailable=["logprob"],
    )

    print("\n--- Oracle Regret @ 5% false-abort ---")
    for name in DEFENSIBLE_ORDER:
        pts = econ.frontier(corpus, name, THRESHOLDS[name])
        results["frontiers"][name] = pts
        rr = econ.regret_at_risk(corpus, name, THRESHOLDS[name], target_risk=0.05)
        rr["savings_at_risk"] = econ.savings_at_risk(corpus, name, THRESHOLDS[name])
        results["regret_at_risk"][name] = rr
        op = rr["operating"]
        print(f"  {name:12s} saved={op['saved_frac']:.3f} "
              f"risk={op['false_abort']:.3f} median_regret={rr['median_regret']:.1f} "
              f"(fired {rr['n_fired']})")

    with open(os.path.join(DATA, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    with open(os.path.join(DATA, "corpus_facts.json"), "w") as f:
        json.dump(oracle_stats, f, indent=1, default=float)

    make_figure(corpus, results, rwr_steps)
    print("\nWrote data/results.json, data/corpus_facts.json, paper/figures/frontier.*")


def make_figure(corpus, results, rwr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    markers = dict(convergence="o", syntactic="s", prm_proxy="^", self_report="D")
    for name in DEFENSIBLE_ORDER:
        pts = econ.pareto_upper_left(results["frontiers"][name])
        xs = [p["false_abort"] for p in pts]
        ys = [p["saved_frac"] for p in pts]
        ax.plot(xs, ys, marker=markers.get(name, "o"), ms=4, lw=1.4, label=name)
    ax.axhline(rwr, ls="--", c="k", lw=1.0)
    ax.text(0.28, rwr + 0.015, f"PPR reference ({rwr:.2f})", fontsize=8)
    ax.axvspan(0.0, 0.10, color="green", alpha=0.06)
    ax.text(0.005, 0.02, "usable\n(risk<0.1)", fontsize=7, color="green")
    ax.set_xlabel("False-abort rate on resolved trajectories")
    ax.set_ylabel("Post-peak compute saved (fraction of failed)")
    ax.set_title("Stop-signal detectors vs. the Oracle Stop")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(-0.01, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "frontier.png"), dpi=160)
    fig.savefig(os.path.join(FIGS, "frontier.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed", type=int, default=250)
    ap.add_argument("--resolved", type=int, default=120)
    ap.add_argument("--min-steps", type=int, default=15)
    ap.add_argument("--scan", type=int, default=9000)
    ap.add_argument("--from-cache", action="store_true",
                    help="load a pickled sample instead of streaming")
    ap.add_argument("--cache-file", default=None)
    run(ap.parse_args())
