# Run order: which script produced which result file

Every result file in `../data/` was produced by a script in this directory. All paths below are
run from `harness/`. Step 0 is required once for any trajectory-level analysis; everything else
reads either the cache or an earlier result file.

```bash
# 0. Rebuild the trajectory cache (deterministic prefix of the public corpus; network)
python cache_sample.py --failed 450 --resolved 300 --scan 24000 --out ../data/sample_large.pkl

# 1. Development-corpus frontier and oracle statistics
python run_study.py --from-cache --cache-file ../data/sample_large.pkl
#    writes data/results.json and data/corpus_facts.json; the shipped
#    data/results_N708.json is the RECORDED copy of exactly this output from the
#    paper's run, kept under a distinct name so a rebuild never overwrites the
#    record. Compare your fresh results.json against it.
python coverage_vs_budget.py                                             # -> coverage_vs_budget.json
python p1_falseabort_sensitivity.py                                      # -> p1_falseabort_sensitivity.json
python clustered_ppr.py                                                  # -> clustered_ppr.json
python deep_analysis.py                                                  # -> deep_analysis.json
python null_model.py                                                     # -> null_model.json

# 2. Preregistered transfer replication (held-out configurations; see ../spec/)
python transfer_test.py                                                  # -> transfer_test.json

# 3. Mechanism and the difficulty-matched corpus
python p1_mechanism_ci.py                                                # -> p1_mechanism_ci.json
python mechanism.py                                                      # -> mechanism.json
python build_dual_corpus.py && python dual_within_instance.py            # -> dual_within_instance.json
python probe_dual_outcome.py                                             # -> dual_outcome_probe.json

# 4. Design target
python design_target.py                                                  # -> design_target.json
python p1_design_target_empirical.py                                     # -> p1_design_target_empirical.json

# 5. Oracle validation
python oracle_v3.py && python oracle_v3_panel.py                         # -> oracle_v3*.json
python regression_rate.py                                                # -> regression_rate.json
python zero_overlap_check.py

# 6. LLM-as-judge family (local Ollama; pull the models first, see ../MODELS.md
#    for exact tags and digests; the server must be up at localhost:11434)
JUDGE_MODEL=north-mini-code-1.0:latest python run_judge.py --n-failed 200 --n-resolved 140 \
    --max-steps 120 --max-len 150 --out ../data/judge_scores_north.json
JUDGE_MODEL=laguna-xs-2.1:latest      python run_judge.py --n-failed 200 --n-resolved 140 \
    --max-steps 120 --max-len 150 --out ../data/judge_scores_laguna.json
python judge_report.py --scores ../data/judge_scores_north.json --label north-mini
python audit_judge_integrity.py                                          # sanity gauntlet
python finalize_headtohead.py                                            # -> headtohead.json
python paired_judge_test.py                                              # -> paired_judge_test.json
python integrate_judge.py            # merges a judge frontier into results.json when run
python run_frontier.py                                                   # -> frontier_rwr.json
python run_calibration.py            # -> calibration.json (remote inference: needs your
                                     #    OPENROUTER_API_KEY or open_router_config.txt at
                                     #    the repo root; recorded copy already shipped)

# 7. Gold-blind panel (remote inference; raw outputs already shipped in ../data/adj_*.json)
python adj_prepare.py        # rebuilds the panel input sample from the public corpus
python adj_judge.py          # runs the panel (needs your own API key)
python adj_analyze.py        # -> adj_result.json

# 8. Figures
python p1_figures.py
```

Determinism: fixed seeds throughout; the corpus scan is a deterministic prefix of the dataset
stream for a fixed dataset revision, and the embedding cache makes oracle secondary views
reproducible bit-for-bit once computed.
