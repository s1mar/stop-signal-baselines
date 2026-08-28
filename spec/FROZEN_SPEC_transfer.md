# FROZEN ANALYSIS SPEC: transfer test of the separability/coverage diagnosis

**Written 2026-07-23, BEFORE any transfer result was computed.** Purpose: defend against the
"adaptively mined story" critique (Sol). The original-corpus diagnosis is labelled EXPLORATORY;
this transfer analysis is the confirmatory test, and it is specified here in advance.

## Status of prior findings

- **EXPLORATORY** (discovered during adversarial-review rounds on the Nebius corpus, same data used
  to shape metric names, controls and claims): the coverage/separability diagnosis, i.e. that at a
  5% false-abort budget detectors fire on only 0.16-0.19 of failed runs and are active at t* on
  <0.03, while unconstrained coverage reaches 1.00 for three of four detectors.
- **CONFIRMATORY** (this spec): whether that pattern replicates on held-out model/scaffold
  configurations that were not used to develop the diagnosis.

## Hypothesis (pre-registered)

H1. On held-out configurations, detector coverage of failed runs is LOW at a 5% false-abort budget
    and RISES substantially as the constraint is relaxed, reaching near-full coverage unconstrained
    for the mechanical detectors.
H2. The dominant loss is coverage (detectors not firing at all), not latency (firing late).

## Held-out configurations: ALL FOUR are evaluated. No subset selection.

1. Claude 3.5 Sonnet, SWE-agent, SWE-bench Verified
2. GPT-4o, SWE-agent, SWE-bench Verified
3. Claude 4 Sonnet, SWE-agent 2025 str_replace_editor interface, SWE-bench Verified
4. Qwen3-Coder-480B, OpenHands, SWE-rebench

Chosen by diversity of (model family x scaffold x task set), declared here before results. If any
configuration fails to yield usable data, that fact is reported with the reason; it is NOT silently
dropped and NOT replaced by a more favorable configuration.

## Detectors: FROZEN, unchanged from the main corpus

syntactic repetition; embedding diversity (Convergence Monitor); process-reward proxy; self-report.
Same implementations, same score definitions, same threshold grids as `run_study.py`/`deep_analysis.py`.
No detector is re-tuned or re-specified for the transfer configurations.

## Quantities reported per configuration (all of them, for every config)

1. savings-versus-false-abort frontier (saved fraction of failed compute at 5%, 10%, 20% budgets)
2. coverage: fraction of failed runs on which the detector fires at all, at each budget
3. fraction of runs whose alarm is active at t*
4. unconstrained coverage (no false-abort constraint)
5. median Oracle Regret at the 5% budget
6. oracle separation AUC on that configuration
7. bootstrap CIs clustered by task instance, and n (failed / resolved / unique instances)
8. reached-reference and never-reached strata where sample size permits

## Additional pre-specified analyses

- **Budget sensitivity.** Report coverage and saved fraction across budgets {2, 5, 10, 20, 50, 100}%.
  If the pattern exists only at exactly 5%, H1/H2 are NOT supported.
- **Threshold transfer.** Fit each detector's operating threshold on the ORIGINAL corpus and apply
  it unchanged to each held-out configuration; report coverage and false-abort there. Per-config
  retuning is reported separately. Local retuning alone does not count as transfer.

## Decision rule (fixed in advance)

- **Headline SURVIVES** only if the qualitative pattern of H1 and H2 replicates on **at least two
  meaningfully distinct held-out configurations** (distinct in model family and/or scaffold).
- **Headline DIES / withdraw the main-track framing** if separability turns out to be easy on the
  held-out configurations (i.e. detectors achieve high coverage at a strict budget there).
- **Narrow the claim** if the pattern is heterogeneous: report the heterogeneity honestly and scope
  the claim to the configurations where it holds. Non-transfer is NOT to be presented as robustness.
- **Downscope (workshop) or withdraw** if any of: fails to replicate on >=2 configs; depends on the
  5% operating point specifically; clustered CIs cannot distinguish "bottleneck" from "insufficient
  data"; the conclusion requires gold information unavailable to a deployed policy; the rewrite is
  still a benchmark table with no general framework or actionable implication.

## Reporting commitments

- Every eligible configuration is reported, including unfavorable ones.
- The exploratory/confirmatory split above is stated in the paper.
- No universal language ("always", "in general") without >=2-config replication plus budget
  sensitivity.
- Deployable quantities are separated from gold-based diagnostic quantities throughout.

## Detector attempt (subordinate, only after the above)

If a new detector is attempted, it is ONE detector, specified in writing before evaluation,
motivated by the separability diagnosis, developed on the original corpus and evaluated ONCE on the
held-out configurations. Comparison uses a PAIRED, instance-clustered CI for the difference against
the incumbent, not CI overlap. If it fails, the failure is reported.
