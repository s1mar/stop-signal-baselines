# The replay harness

Zero-new-inference, laptop-reproducible study behind *They Do Not Fire Late, They Barely Fire*
(AgenticDev 2026). Everything is retrospective replay of public agent trajectories.

## Install
```bash
python -m venv .venv && ./.venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements.txt
```

## Reproduce the paper's numbers
```bash
# 1. Cache a class-balanced sample (streams Nebius; no full download)
python cache_sample.py --failed 450 --resolved 300 --scan 24000 --out ../data/sample_large.pkl
# 2. Run the study (oracle + detectors + econ + figure)
python run_study.py --from-cache --cache-file ../data/sample_large.pkl
#    -> ../data/results.json, ../data/corpus_facts.json, ../paper/figures/frontier.{png,pdf}
# 3. LLM-judge detector (local Ollama; fills the paper's judge rows; see ../MODELS.md)
python run_judge.py --n-failed 30 --n-resolved 30      # needs `ollama serve` + a small model
python integrate_judge.py                              # merges judge frontier into results
```

## Layout
- `llab/schema.py`      common Trajectory/Step/Action types
- `llab/actions.py`     SWE-agent command parser (edit/create/open/...)
- `llab/ingest.py`      stream Nebius + join gold from SWE-bench-extra
- `llab/patchtools.py`  gold parsing + agent cumulative gold-file edits
- `llab/oracle.py`      progress oracle (token recall + embedding), Oracle Stop t*
- `llab/detectors.py`   6 stop-signal families behind one interface
- `llab/llm_judge.py`   LLM-as-judge via local Ollama
- `llab/econ.py`        RWR, Oracle Regret, frontier, bootstrap CI
- `run_study.py`        end-to-end driver
- `oracle_lab.py`       oracle-definition comparison (discriminativeness)
- `tests/`              offline validation on a saved real trajectory

## Determinism
Fixed seeds; on-disk embedding cache (`../data/emb_cache.pkl`). Same scan window +
same sample → same numbers. Streaming order from HF is stable for a fixed revision.

## Data provenance
- Trajectories: `nebius/SWE-agent-trajectories` (CC-BY-4.0).
- Gold patches: `nebius/SWE-bench-extra` / `princeton-nlp/SWE-bench`.
No agent is executed and no paid API is called.

## Oracle validation & model disclosure (reproducibility)

The paper validates the token-based Oracle Stop t* two ways and names every model used as a
measured instrument. Exact provenance:

**Local LLM-judge detector (the paper's LLM-as-judge family).** Two open ~30B Mixture-of-Experts
coding models run locally via Ollama:
- `north-mini-code-1.0` (Cohere North Mini Code 1.0; 30B/3B-active; Apache-2.0) = CONSERVATIVE judge (18% stuck)
- `laguna-xs-2.1` (Laguna XS 2.1; 33B/3B-active; OpenMDW-1.1) = EAGER judge (32% stuck)
Judge prompt: `llab/llm_judge.py` (`build_prompts`). Raw per-step scores:
`data/judge_scores_north.json`, `data/judge_scores_laguna.json`. Runner: `run_judge.py`
(claims the Ollama lock automatically).

**Gold-blind adjudication panel (paper section "Does the Instrument Hold Up?").**
Three frontier API models from distinct vendors, accessed July 2026: GPT-5.6 (`openai/gpt-5.6-sol`),
Kimi-K3 (`moonshotai/kimi-k3`), Fable-5 (`anthropic/claude-fable-5`). The panel sees the issue and
the numbered agent steps but NOT the gold patch.
- Exact prompt: `adj_judge.py` (`PROMPT`); sample builder `adj_prepare.py`; scorer `adj_analyze.py`.
- Raw panel outputs: `data/adj_gpt-5.6-sol.json`, `data/adj_kimi-k3.json`, `data/adj_claude-fable-5.json`.
- Headline: median |panel - t*| = 1 step (clustered 95% CI 1-2); midpoint-null 5, random-null 6.

**Reproducing the paid layer.** The adjudication uses OpenRouter; supply your own API key via
the `OPENROUTER_API_KEY` environment variable, or put a line containing `sk-or-v1-...` in a file
`open_router_config.txt` at the repository root (no key is hardcoded or included in this
release). The raw panel outputs are already in `../data/`, so nothing needs re-running or
re-paying, and the main results (frontier, transfer, mechanism) need no inference and no key
at all.

**Offline tests.** `tests/` validates the action parser and the oracle on one saved example
trajectory. The example is not redistributed (it is third-party corpus text); generate it first
with `python probe_schema.py` (streams a single trajectory from the public dataset into
`../data/probe_example.json`), then run the tests.
