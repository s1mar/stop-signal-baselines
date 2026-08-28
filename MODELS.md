# Local judge models (the LLM-as-judge detector family)

Both judges are open-weight models served locally through [Ollama](https://ollama.com) under an
identical prompt. The prompt is `_PROMPT` in `harness/llab/llm_judge.py`; per-step scoring and
caching are in the same file. No remote inference is used by the judge family.

| role | Ollama tag | weight digest (ollama) | architecture | quantization | license |
|---|---|---|---|---|---|
| conservative judge (18% of steps called stuck) | `north-mini-code-1.0:latest` | `d8b269ad5c7c` | cohere2moe, 30.5B params (3B active) | Q4_K_M | Apache-2.0 |
| eager judge (32% of steps called stuck) | `laguna-xs-2.1:latest` | `a8562dfd0cad` | laguna, 33.4B params (3B active) | Q4_K_M | OpenMDW-1.1 |

Reproduce a judge run: pull each model first (`ollama pull north-mini-code-1.0:latest`,
`ollama pull laguna-xs-2.1:latest`; roughly 18 GB and 20 GB of weights, so a machine that can
host a ~30B Q4 model is needed) and have the server running at `localhost:11434`. Then:

```
JUDGE_MODEL=north-mini-code-1.0:latest python harness/run_judge.py \
    --n-failed 200 --n-resolved 140 --max-steps 120 --max-len 150 \
    --out data/judge_scores_north.json
```

The recorded score files (`data/judge_scores_north.json`, `data/judge_scores_laguna.json`) and
their integrity audit (`harness/audit_judge_integrity.py`) are included, so the paper's judge
numbers verify without re-running any model.

# Gold-blind validation panel

The oracle-validation panel (paper, "Does the Instrument Hold Up?") used three frontier models
from distinct vendors, gold patch withheld. Their raw per-trajectory outputs are
`data/adj_claude-fable-5.json`, `data/adj_gpt-5.6-sol.json`, `data/adj_kimi-k3.json`, with the
adjudicated result in `data/adj_result.json`. The panel input sample is rebuilt from the public
corpus by `harness/adj_prepare.py` (it embeds third-party trajectory text and is therefore not
redistributed).
