"""Figures for Paper 1 (the diagnosis) -> work/p1-agenticdev/paper/figures.

Pure redraw of cached results; no agent is run and no inference is called.

  fig 1  frontier.pdf       savings-versus-false-abort, development corpus
  fig 2  mechanism.pdf      (a) run-level score distributions overlap
                            (b) separability AUC does not improve with time
  fig 3  design_target.pdf  required run-level AUC vs target coverage at 5% false-abort,
                            against what the evaluated families achieve
  fig 4  transfer.pdf       coverage vs false-abort budget on the four held-out configs
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from llab import econ

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "..", "p1-agenticdev", "paper", "figures")
ORDER = ["syntactic", "convergence", "prm_proxy", "self_report"]
LABELS = {"convergence": "embedding diversity", "syntactic": "syntactic repetition",
          "prm_proxy": "process-reward proxy", "self_report": "self-report"}
MARK = {"convergence": "o", "syntactic": "s", "prm_proxy": "^", "self_report": "D"}
COL = {"convergence": "#1f77b4", "syntactic": "#d62728",
       "prm_proxy": "#2ca02c", "self_report": "#9467bd"}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})

W1, W2 = 3.4, 7.1                                   # IEEE single / double column width


def load(name):
    return json.load(open(os.path.join(DATA, name), encoding="utf-8"))


def running_max(scores, upto):
    s = [x for x in scores[:upto] if x is not None]
    return max(s) if s else 0.0


# ---------------------------------------------------------------- fig 1: frontier
def fig_frontier():
    r = load("results.json")
    ppr = r["rwr_steps"]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    for name in ORDER:
        pts = econ.pareto_upper_left(r["frontiers"][name])
        ax.plot([p["false_abort"] for p in pts], [p["saved_frac"] for p in pts],
                marker=MARK[name], ms=3, lw=1.2, color=COL[name], label=LABELS[name])
    ax.axhline(ppr, ls="--", c="k", lw=0.9)
    ax.text(0.30, ppr + 0.02, f"PPR reference ({ppr:.2f})", fontsize=7)
    ax.axvspan(0.0, 0.05, color="green", alpha=0.10)
    ax.text(0.06, 0.86, "deployable\nbudget", fontsize=7, color="green")
    ax.set_xlabel("False-abort rate on resolved runs")
    ax.set_ylabel("Post-peak compute saved")
    ax.set_xlim(-0.01, 1.0); ax.set_ylim(-0.01, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, "frontier.pdf"))
    plt.close(fig)
    print(f"frontier.pdf (PPR={ppr:.3f})")


# --------------------------------------------------------------- fig 2: mechanism
def fig_mechanism():
    rows = load("corpus_strat.json")
    mech = load("mechanism.json")
    failed = [e for e in rows if not e["resolved"]]
    resolved = [e for e in rows if e["resolved"]]

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))

    # (a) survival curves of the run-level score: the tail a threshold has to cut
    d = "syntactic"
    fmax = np.sort([running_max(e["scores"][d], len(e["scores"][d])) for e in failed if e["scores"][d]])
    rmax = np.sort([running_max(e["scores"][d], len(e["scores"][d])) for e in resolved if e["scores"][d]])
    info = mech["score_overlap"][d]
    thr = info["thr_at_5pct_fa"]
    grid = np.linspace(min(fmax.min(), rmax.min()), np.percentile(np.r_[fmax, rmax], 99), 300)
    surv = lambda a: np.array([np.mean(a >= g) for g in grid])
    ax = axes[0]
    ax.plot(grid, surv(rmax), lw=1.6, color="#2ca02c", label="resolved (recoverable)")
    ax.plot(grid, surv(fmax), lw=1.6, color="#d62728", label="failed (doomed)")
    ax.axvline(thr, ls="--", c="k", lw=1.0)
    ax.annotate(f"5% of resolved runs", xy=(thr, 0.05), xytext=(thr + 1.2, 0.30), fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate(f"only {info['failed_reaching']:.0%} of failed runs", xy=(thr, info["failed_reaching"]),
                xytext=(thr + 1.2, 0.62), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(f"run-level alarm score, {LABELS[d]}")
    ax.set_ylabel("fraction of runs reaching it")
    ax.set_title(f"(a) the two distributions overlap ({info['overlap']:.2f}), "
                 f"AUC {info['auc_runlevel']:.2f}")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)

    # (b) separability does not improve with time
    ax = axes[1]
    pos = [0.1 * i for i in range(1, 11)]
    for name in ORDER:
        ax.plot(pos, mech["separability_over_time"][name], marker=MARK[name], ms=3, lw=1.2,
                color=COL[name], label=LABELS[name])
    ax.axhline(0.5, ls=":", c="k", lw=0.9)
    ax.text(0.12, 0.505, "chance", fontsize=7)
    ax.set_xlabel("fraction of the run elapsed")
    ax.set_ylabel("AUC (failed vs resolved)")
    ax.set_ylim(0.45, 0.80)
    ax.set_title("(b) waiting does not sharpen the signal")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, "mechanism.pdf"))
    plt.close(fig)
    print(f"mechanism.pdf (panel a: {d}, overlap {info['overlap']:.3f})")


# ----------------------------------------------------------- fig 3: design target
def fig_design_target():
    dt = load("design_target.json")
    try:
        from scipy.stats import norm
        Phi, Phinv = norm.cdf, norm.ppf
    except Exception:
        from math import erf, sqrt as _s
        Phi = lambda x: 0.5 * (1 + erf(x / _s(2)))
        def Phinv(p):
            lo, hi = -10.0, 10.0
            for _ in range(200):
                m = (lo + hi) / 2
                lo, hi = (m, hi) if Phi(m) < p else (lo, m)
            return (lo + hi) / 2

    alpha = dt["alpha"]
    z_a = Phinv(1 - alpha)
    cov = np.linspace(0.02, 0.95, 200)
    auc_req = np.array([Phi((z_a + Phinv(c)) / np.sqrt(2)) for c in cov])

    fig, ax = plt.subplots(figsize=(W1, 2.6))
    ax.plot(cov, auc_req, lw=1.6, color="k", label=f"required at {alpha:.0%} false-abort")
    for name in ORDER:
        a = dt["observed_auc"][name]
        ax.axhline(a, lw=0.9, ls="--", color=COL[name], alpha=0.85)
        ax.plot([dt["observed_coverage"][name]], [a], marker=MARK[name], ms=5, color=COL[name],
                label=f"{LABELS[name]} (observed)")
    a50 = dt["required"]["0.5"]["auc"]
    ax.plot([0.5], [a50], marker="*", ms=10, color="k")
    ax.annotate(f"50% coverage needs AUC {a50:.2f}", xy=(0.5, a50), xytext=(0.16, 0.93),
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("coverage of doomed runs")
    ax.set_ylabel("run-level separation (AUC)")
    ax.set_ylim(0.5, 1.0); ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, "design_target.pdf"))
    plt.close(fig)
    print("design_target.pdf")


# --------------------------------------------------------------- fig 4: transfer
def fig_transfer():
    T = load("transfer_test.json")
    budgets = [0.05, 0.1, 0.2, 0.5, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.0), sharey=True)
    for ax, cfg in zip(axes, T):
        for name in ORDER:
            b = cfg["detectors"][name]["budget"]
            ax.plot([100 * x for x in budgets], [b[str(x)]["coverage"] for x in budgets],
                    marker=MARK[name], ms=3, lw=1.2, color=COL[name], label=LABELS[name])
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.set_xticklabels(["5", "10", "20", "50", "100"])
        ax.axvspan(4.0, 5.0, color="green", alpha=0.10)
        ax.set_title(f"{cfg['label']}\n(AUC {cfg['auc']:.2f})", fontsize=7)
        ax.set_xlabel("false-abort budget (%)")
        ax.set_ylim(-0.02, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("coverage of failed runs")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, "transfer.pdf"))
    plt.close(fig)
    print("transfer.pdf")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_frontier()
    fig_mechanism()
    fig_design_target()
    fig_transfer()
    print("\nwrote figures to", os.path.normpath(FIGS))
