"""Self-contained verifier for this packet. Run it: `python checks/verify.py`

Everything is resolved RELATIVE TO THIS FILE, so it reads the manuscript and the data shipped in
this packet and nothing else. It prints both paths before doing anything, because the other scripts
in this directory carry absolute paths from the author's machine and would silently check different
files (see NOTES-ON-THE-CHECKS in BRIEF.md).

It re-derives 27 numeric claims from the source JSON and checks each against the wording in
main.tex, then reports how much of the paper's numeric surface is pinned by the author's gate.
No network, no API keys, standard library only.
"""
import io
import json
import os
import re
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
PAPER = os.path.join(PACK, "main.tex")
DATA = os.path.join(PACK, "data")
print(f"manuscript: {PAPER}\ndata      : {DATA}\n")

RAW = io.open(PAPER, encoding="utf-8").read()
TEX = " ".join(RAW.split())
FAM = ("syntactic", "convergence", "prm_proxy", "self_report")


def J(f):
    return json.load(open(os.path.join(DATA, f), encoding="utf-8"))


rows = []


def check(claim, quote, expect, got):
    """`quote` is a literal substring, or a regex if it is wrapped in re.compile().

    Use the regex form for claims whose wording is likely to be rephrased. An r18 trim
    changed "violated it on 9 of 16 held-out pairs" to "violated the budget on 9 of our 16
    held-out pairs" and this check failed on a claim the paper still makes correctly. A check
    that fires on rewording trains you to ignore it.
    """
    found = bool(quote.search(TEX)) if hasattr(quote, "search") else quote in TEX
    shown = quote.pattern if hasattr(quote, "pattern") else quote
    rows.append((claim, found, str(expect) == str(got), expect, got, shown))


# 1. matched-corpus within-instance deltas
d = J("dual_within_instance.json")
check("max within-vs-pooled AUC delta is +0.021",
      "by at most $+0.021$", "0.021", f'{max(d[k]["delta"] for k in FAM):.3f}')
check("self-report's delta is +0.000, so no range may start above it",
      "by $+0.000$ for self-report", "0.000", f'{d["self_report"]["delta"]:.3f}')
check("all four pooled AUCs fall inside their within-instance CI",
      "pooled value falls inside its", 4,
      sum(d[k]["within_ci"][0] <= d[k]["pooled_auc"] <= d[k]["within_ci"][1] for k in FAM))

# 2. best-separation referents across the two corpora
mech = J("p1_mechanism_ci.json")
dev_best = max((mech[k]["auc"], k) for k in FAM)
mat_best = max((d[k]["pooled_auc"], k) for k in FAM)
check("development-corpus best AUC is the process-reward proxy at 0.617",
      "0.617 for the process-reward proxy on the development corpus",
      ("prm_proxy", "0.617"), (dev_best[1], f"{dev_best[0]:.3f}"))
check("matched-corpus best AUC is syntactic repetition at 0.613",
      "0.613 for syntactic repetition", ("syntactic", "0.613"),
      (mat_best[1], f"{mat_best[0]:.3f}"))
check("those are different families, which the paper must say",
      "the best family is not the same one", True, dev_best[1] != mat_best[1])
check("process-reward proxy reads 0.572 on the matched corpus",
      "0.572 here", "0.572", f'{d["prm_proxy"]["pooled_auc"]:.3f}')

# 3. threshold transfer
T = J("transfer_test.json")
viol = [(c["detectors"][x]["transfer"]["false_abort"], x, c["label"]) for c in T for x in FAM]
worst_mech = max(v for v in viol if v[1] != "self_report")
worst_all = max(viol)
check("worst mechanical transfer violation is 0.262",
      "by more than five times", "0.262", f"{worst_mech[0]:.3f}")
check("0.262 exceeds the 0.05 budget by more than five times",
      "by more than five times", True, worst_mech[0] / 0.05 > 5)
check("worst overall violation is nearly thirteen times the budget",
      "once by nearly thirteen times", True, 12 < worst_all[0] / 0.05 < 13)
check("9 of the 16 held-out pairs violate the transferred budget",
      re.compile(r"violated (?:it|the budget) on 9 of (?:our )?16 held-out pairs"), 9, sum(1 for v, _, _ in viol if v > 0.05))
check("held-out configurations have 42 to 70 resolved runs",
      "(42 to 70 runs)", (42, 70), (min(c["n_resolved"] for c in T), max(c["n_resolved"] for c in T)))
check("held-out median coverage at the 5% budget is 0.043",
      "median coverage at the 5\\% budget is 0.043", "0.043",
      f'{statistics.median([c["detectors"][x]["budget"]["0.05"]["coverage"] for c in T for x in FAM]):.3f}')

# 4. coverage on the development corpus, constrained and unconstrained
cvb = J("coverage_vs_budget.json")
cov5 = {k: cvb[k]["coverage"][0] for k in FAM}
unc = {k: max(cvb[k]["coverage"]) for k in FAM}
sav = {k: max(cvb[k]["saved"]) for k in FAM}
band = [k for k in FAM if k != "self_report"]
check("best saved fraction at the 5% budget is 0.129",
      "recovers 0.129 of failed-run", "0.129", f'{max(cvb[k]["saved"][0] for k in FAM):.3f}')
check("best coverage at the 5% budget is 0.18",
      "0.18 of failed runs", "0.18", f"{max(cov5.values()):.2f}")
check("mechanical coverage band on the development corpus is 0.16 to 0.18",
      "0.16 to 0.18 for the mechanical families",
      ("0.16", "0.18"), (f"{min(cov5[k] for k in band):.2f}", f"{max(cov5[k] for k in band):.2f}"))
check("self-report sits below that band, so the band must be labelled mechanical",
      "mechanical band of 0.16 to 0.18", True, cov5["self_report"] < 0.16)
check("ONLY TWO families reach 1.00 unconstrained, not three",
      "mechanical families cover 0.99 to 1.00", 2, sum(1 for k in FAM if unc[k] >= 0.999))
check("unconstrained mechanical coverage spans 0.99 to 1.00",
      "cover 0.99 to 1.00 of failed runs", ("0.99", "1.00"),
      (f"{min(unc[k] for k in band):.2f}", f"{max(unc[k] for k in band):.2f}"))
check("unconstrained mechanical saved fraction spans 0.88 to 0.97",
      "save 0.88 to 0.97 of their compute", ("0.88", "0.97"),
      (f"{min(sav[k] for k in band):.2f}", f"{max(sav[k] for k in band):.2f}"))
check("self-report never exceeds 0.34 coverage at any budget",
      "never exceeding 0.34 coverage", True, unc["self_report"] <= 0.34)

# 5. the zero-reaching intervals, which restate Table 1
tab = RAW[RAW.index(r"\label{tab:frontier}") - 1400: RAW.index(r"\label{tab:frontier}")]
zero = [n for n, lo in re.findall(r"^([A-Z][\w\- ]+?)\s*&.*?& (\d\.\d{3})--", tab, re.M) if lo == "0.000"]
check("only syntactic repetition and embedding diversity have CIs reaching zero",
      "for syntactic repetition and embedding diversity the interval reaches zero",
      ["Syntactic repetition", "Embedding diversity"], zero)

# 6. the more forgiving false-abort accountings
S = J("p1_falseabort_sensitivity.json")["modes"]
check("lenient accounting best saved 0.144",
      "from 0.129 to 0.144", "0.144", f'{max(S["lenient"][k]["0.05"]["saved"] for k in FAM):.3f}')
check("edit-window accounting best coverage 0.24",
      "from 0.18 to 0.24", "0.24", f'{max(S["editwin"][k]["0.05"]["coverage"] for k in FAM):.2f}')
check("edit-window accounting best saved 0.168 at 0.235 coverage",
      "0.168 saved at 0.235 coverage", "0.168", f'{max(S["editwin"][k]["0.05"]["saved"] for k in FAM):.3f}')
check("the strict column of the sensitivity table reproduces Table 1 exactly",
      "Process-reward proxy & 0.125", "0.125", f'{S["strict"]["prm_proxy"]["0.05"]["saved"]:.3f}')

# 7. the design target
emp = J("p1_design_target_empirical.json")["summary"]["0.5"]
gau = J("design_target.json")["required"]["0.5"]["auc"]
check("empirical half-coverage target 0.914 to 0.954, stated rounded",
      "0.91 to 0.95 run-level AUC", ("0.91", "0.95"),
      (f'{emp["empirical_lo"]:.2f}', f'{emp["empirical_hi"]:.2f}'))
check("Gaussian half-coverage target 0.878, stated rounded as 0.88",
      "0.88 under the more forgiving equal-variance Gaussian idealization", "0.88", f"{gau:.2f}")
check("the empirical band sits 0.04 to 0.08 above the Gaussian value",
      "sits 0.04 to 0.08 above the Gaussian value", ("0.04", "0.08"),
      (f'{emp["empirical_lo"] - gau:.2f}', f'{emp["empirical_hi"] - gau:.2f}'))

# 8. the instrument
check("the current-state oracle degenerates on 116 of 121 runs, 95.9%",
      "116 of them, 95.9\\%", "95.9", f'{100 * J("oracle_v3_panel.json")["degeneracy_frac"]:.1f}')

w = max(len(r[0]) for r in rows)
bad = 0
for claim, in_paper, agrees, expect, got, quote in rows:
    ok = in_paper and agrees
    bad += not ok
    print(f"{'ok   ' if ok else ('NO-QUOTE' if not in_paper else 'MISMATCH')} "
          f"{claim.ljust(w)}  paper={expect}  data={got}")
    if not in_paper:
        print(f"        wording not found in main.tex: {quote!r}")
print(f"\n{len(rows)} claims re-derived from the shipped data, {bad} failing")

# ---- how much of the paper's numeric surface is pinned at all
body = RAW[RAW.index(r"\begin{abstract}"): RAW.index(r"\bibliographystyle")]
body = re.sub(r"\\begin\{CCSXML\}.*?\\end\{CCSXML\}", " ", body, flags=re.S)
prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.S)
nums = {m.group(0).rstrip("\\%") for m in
        re.finditer(r"(?<![\w.])\d+(?:[.,]\d+)*\\?%?", re.sub(r"\s+", " ", prose))}
print(f"\ndistinct numbers in the body prose: {len(nums)}")
print("the author's gate pins 22 of them. This script re-derives the ones above.")
print("EVERY OTHER NUMBER IN THE PAPER IS UNVERIFIED BY ANY AUTOMATED CHECK.")
