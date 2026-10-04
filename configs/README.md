# gnosis configs

Every gnosis feature is a flag (safe defaults: off). A **config file** turns a
named set of them on.

**gnosis auto-loads [`default.yaml`](default.yaml) when `GNOSIS_CONFIG_FILE` is
unset** — so out of the box it runs the preferred (best-scoring) config. To
change that:

```bash
GNOSIS_CONFIG_FILE=configs/runs/run11.yaml gnosis   # load a different config
GNOSIS_CONFIG_FILE="" gnosis                         # opt out: safe minimal defaults
```

Precedence, highest first: explicit `GNOSIS_*` env vars → `.env` → the YAML
config file → code defaults. So the file sets the baseline and a single env var
still overrides one key.

LLM-planned graph QA is gated by `GNOSIS_GRAPHQA_ENABLED` (default off, and not
set by any file here; see [docs/security.md](../docs/security.md#graph-qa-safety)).
With it off, graph-QA fusion - including the multi-hop route of adaptive routing
in `default.yaml` - is a no-op. To reproduce a run measured with graph-QA fusion
(runs 10, 11, 14-21), set `GNOSIS_GRAPHQA_ENABLED=true` alongside the config.

## default.yaml — the preferred config (Run 18)

[`default.yaml`](default.yaml) is the LOCOMO benchmark-best: fact extraction +
entity graph at write; adaptive routing + route-aware Chain-of-Note at read.
LOCOMO subset-3 J **74.8 / 76.7**; full-LOCOMO (Run 23) excl-adv J **66.9–68.9**
— competitive with mem0, leading single-hop / temporal / adversarial. Needs a
capable `GNOSIS_LLM` (extraction/routing make LLM calls; the `gemma4` default is
not adequate for extraction).

## All measured runs

Each `runs/runN.yaml` reproduces that run's **flags** on the current code. Runs
14/15/17/18 share the same four flags — they differ by Chain-of-Note *code*
(naive → route-aware → hardened → likelihood carve-out), all now in `main`, so
those files all load today's (Run 18) CoN. Runs 1–3 were code changes with no
flags. Full per-run detail: [docs/BENCHMARKS.md](../docs/BENCHMARKS.md).

| Run | Config | Change | ctx J | Verdict |
|---|---|---|---|---|
| 1 | [run1](runs/run1.yaml) | baseline — verbatim dated RAG | 37.4 | starting line |
| 2 | [run2](runs/run2.yaml) | cross-session reads + dates + budget | 41.0 | kept (code) |
| 3 | [run3](runs/run3.yaml) | relevance ranking + compact rendering | 59.5 | kept (code) |
| 4 | [run4](runs/run4.yaml) | LLM recall filter | 58.7 | rejected |
| 5 | [run5](runs/run5.yaml) | **fact extraction at ingest** | 71.2 | **kept (+11.7)** |
| 6 | [run6](runs/run6.yaml) | + hybrid BM25 retrieval | 71.4 | wash |
| 7 | [run7](runs/run7.yaml) | abstention prompt | 69.6 | tunable |
| 8 | [run8](runs/run8.yaml) | facts→verbatim expansion | 71.2 | flat |
| 9 | [run9](runs/run9.yaml) | hybrid + verbatim + supersession stacked | 57.9 | **crashed −13.3** |
| 10 | [run10](runs/run10.yaml) | entity graph + graph-QA fusion | 70.9 | multi-hop flat |
| 11 | [run11](runs/run11.yaml) | adaptive per-query routing | 74.3 | new best +2.9 |
| 12 | [run12](runs/run12.yaml) | entity-anchored graph traversal | 70.7 | rejected |
| 13 | [run13](runs/run13.yaml) | Chain-of-Note reading instruction | 72.0 | kept |
| 14 | [run14](runs/run14.yaml) | routing + CoN (naive) | 71.4 | doesn't compose |
| 15 | [run15](runs/run15.yaml) | routing + route-aware CoN | 73.2 | composes |
| 16 | [run16](runs/run16.yaml) | + directed bridge traversal | 72.5 | rejected |
| 17 | [run17](runs/run17.yaml) | hardened CoN | 72.2 | adversarial 83.0 |
| **18** | **[run18](runs/run18.yaml)** | **+ likelihood carve-out** | **74.8** | **= default.yaml** |
| 19 | [run19](runs/run19.yaml) | 2× coverage item budget | 72.5 | rejected |
| 20 | [run20](runs/run20.yaml) | speculative-inference CoN | 74.3 | tunable |
| 21 | [run21](runs/run21.yaml) | enumeration CoN | 71.7 | rejected |
| 22 | [run22](runs/run22.yaml) | entity-grouped rendering (GRAVITY) | 71.9 | rejected |
| 23 | [run23](runs/run23.yaml) | full-LOCOMO re-measure of Run 18 | 66.9–68.9¹ | apples-to-apples |
| 24 | [run24](runs/run24.yaml) | + LLM reranker (LME_S target) | pending | reranker not yet benchmarked |
| 25 | [run25](runs/run25.yaml) | + community graph + multi-query rewrite (LME_S) | pending | open-domain + multi-hop target |

¹ Run 23 is the full-10-conversation measurement (excl-adv J, two judges); all
other rows are the subset-3 dev gate. Runs 24–25 are queued LongMemEval_S
experiments; ctx J column will be filled after measurement.

**Note:** the LOCOMO run numbers above (Runs 1–25) are separate from the LongMemEval_S
experiment labels (L-0 through L-31+) in gnosis-membench. LME_S experiment history:

| LME_S Label | gnosis-membench | Status | Score | Notes |
|---|---|---|---|---|
| L-21 | Run 21 equivalent | Complete | — | Full-500 ingest; gemini-embedding-001 |
| L-23 | — | Complete (2026-07-31) | 69.8% | Claude-Sonnet-4-6 backbone + judge; established competitive baseline |
| L-25 | Run 25 partial | Complete (2026-08-05) | 72.4% | edu-v2.0 + relation_slots KU fix |
| **L-25b** | — | **Complete (2026-08-06)** | **73.6%** | + singleton-only supersession; **current best**; gpt-4o backbone + judge |
| L-27 | — | Rejected (2026-08-06) | 73.4% | Community graph: -0.2pp overall, SSA -5.3pp |
| L-28 | — | Rejected (2026-08-06) | 71.8% | Stronger CoN recency clause: over-fires outside KU |
| L-29 | — | Tied (2026-08-06) | 73.6% | knowledge_update route + recency injection: KU +4.2pp, SSA -5.3pp cancel |
| L-30 | — | Rejected (2026-08-06) | 73.0% | Tighter KU router guide: temporal -2.4pp from over-restriction |
| **L-31** | — | **In progress (2026-08-06)** | pending | Write-time SUPERSEDES + valid_to filter; structural KU fix |

See [gnosis-membench RESULTS.md](https://github.com/nolgiainc/gnosis-membench/blob/main/RESULTS.md).
