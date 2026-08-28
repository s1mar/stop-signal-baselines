# Artifact: They Do Not Fire Late, They Barely Fire

Research artifact for the paper **"They Do Not Fire Late, They Barely Fire: Four
Surface-Behavior Stopping Baselines for Long-Horizon Coding Agents, and How to Report One"**
(AgenticDev 2026, the International Workshop on Agentic AI for Next-Generation Software
Development, co-located with ASE 2026, Munich).

The paper evaluates four inexpensive fixed-threshold stop-signal detector families for
long-horizon coding agents by retrospective replay of public agent trajectories, grounded on a
reference-patch progress oracle (the Oracle Stop). This repository contains the replay harness,
the frozen pre-registration for the transfer replication, the recorded result files every number
in the paper traces to, and self-contained verification scripts.

## Layout

```
paper/main.tex   the manuscript source the checks verify against
harness/         the replay harness (llab/ package) and every analysis script
                 that produced a file in data/; RUNSHEET.md is the run order
spec/            FROZEN_SPEC_transfer.md, the pre-registered analysis specification
                 frozen before any held-out result was computed
data/            recorded result files (JSON): frontier, transfer, mechanism,
                 design target, oracle validation, gold-blind panel outputs,
                 local-judge scores and calibration
checks/          self-contained verifiers (standard library only, no network)
MODELS.md        exact tags, digests and licenses of the local judge models
```

Requirements: Python 3.11 or newer. The verification step below needs only the standard
library; the rebuild and pipeline steps need `pip install -r harness/requirements.txt`.

## Verify the paper's numbers (no dependencies, no network)

```
python checks/verify.py
```

This re-derives 30 numeric claims from `data/` and checks each against the wording in
`paper/main.tex`. Expected output ends with `30 claims re-derived from the shipped data,
0 failing`.

## Rebuild the trajectory corpus (network; needed only for trajectory-level checks)

The development corpus is a bounded, deterministic sample of the public
[nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
dataset (CC-BY-4.0, 80,036 rows). The trajectory texts themselves are not redistributed here;
rebuild the cache with:

```
pip install datasets
python harness/cache_sample.py --failed 450 --resolved 300 --scan 24000 --out data/sample_large.pkl
```

The scan is a deterministic prefix of the dataset stream, so this reproduces the corpus the
paper used: 746 trajectories join to a gold patch (446 failed, 300 resolved). Then:

```
python checks/funnel_counts.py            # the 80,036 -> 708 corpus funnel, count by count
python checks/tstar_zero_check.py         # the zero-progress t* convention, run by run
python checks/zero_overlap_route_check.py # how zero-overlap resolved runs actually solved tasks
```

## Reproduce the full pipeline

`harness/RUNSHEET.md` lists the run order. The main pipeline runs no agents and calls no
inference; only the optional LLM-judge rows (local models via Ollama, see `MODELS.md`) and the
gold-blind validation panel call models, and their raw outputs are already in `data/` so nothing
needs re-running or re-paying.

## Attribution and licenses

- Code and recorded measurement data in this repository: MIT License (see `LICENSE`).
- Source trajectory corpus: [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories),
  CC-BY-4.0, not redistributed here. Gold patches join from SWE-bench-extra / SWE-bench.
- Local judge models: see `MODELS.md` for exact tags, digests and their own licenses.
