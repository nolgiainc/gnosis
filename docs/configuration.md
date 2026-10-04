# Configuration reference

Every setting is an environment variable (`GNOSIS_FOO` ↔ field `gnosis_foo`) or a
key in a YAML config file. Precedence, highest first: **env vars → `.env` → the
YAML config file → code defaults**.

- gnosis **auto-loads `configs/default.yaml`** (the preferred config) when
  `GNOSIS_CONFIG_FILE` is unset. Set `GNOSIS_CONFIG_FILE` to another path, or to
  `""` to opt out to code defaults. Named presets for every measured run live in
  [`configs/runs/`](../configs/README.md).
- Every feature flag defaults **off** — the code default is a minimal, cheap
  baseline. `configs/default.yaml` turns on the [preferred config](#the-preferred-config).

## Required

| Setting | Purpose |
|---|---|
| `GNOSIS_TOKEN` | service bearer token (normal callers) |
| `GNOSIS_READ_OPERATOR_TOKEN` / `WRITE` / `EXPORT` / `ADMIN` `_OPERATOR_TOKEN` | operator token classes (see [security.md](security.md)) |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | graph store |
| `LITELLM_BASE_URL` / `LITELLM_API_KEY` | OpenAI-compatible LLM + embedding endpoint |
| `GNOSIS_TENANT_ID` | default tenant (`nolgia`) |

## Models & backends

| Setting | Default | Purpose |
|---|---|---|
| `GNOSIS_LLM` | `openai/gemma4` | extraction / routing / graph-QA model. **Set a capable model** — `gemma4` is not adequate for extraction. |
| `GNOSIS_EMBEDDING` | `local-qwen3-embedding-0.6b` | embedding model |
| `GNOSIS_EMBEDDING_DIMENSIONS` | `1024` | embedding dim (e.g. 3072 for `gemini-embedding-001`) |
| `GNOSIS_CONFIG_FILE` | *(unset → `configs/default.yaml`)* | YAML config path; `""` opts out |

## Write behavior

| Setting | Default | Purpose |
|---|---|---|
| `GNOSIS_WRITE_MODE` | `sync` | `sync` or `buffered` writes |
| `GNOSIS_MAX_PENDING` | `200` | buffered-write queue bound |
| `GNOSIS_FACT_DEDUPLICATION_ENABLED` | `true` | non-destructive dedup on add |
| `GNOSIS_TRACE_EMBEDDING_ENABLED` | `true` | embed reasoning traces |
| `GNOSIS_CONVERSATION_TTL_DAYS` | *(none)* | optional message TTL |
| `GNOSIS_AUDIT_READ` | `false` | write a `MemoryReadAudit` per read |
| `GNOSIS_MEMORY_EDIT_ENABLED` | `false` | allow `PATCH`/`DELETE /v1/memories/{id}` |

## Fact extraction (write path)

| Setting | Default | Purpose |
|---|---|---|
| `GNOSIS_FACT_EXTRACTION_ENABLED` | `false` | **extract atomic dated fact units at ingest** (the largest lever) |
| `GNOSIS_FACT_EXTRACTION_MODE` | `sync` | `sync` or `background` (bounded async queue, drop-not-block) |
| `GNOSIS_FACT_EXTRACTION_MODEL` | *(→ `GNOSIS_LLM`)* | override extraction model |
| `GNOSIS_FACT_EXTRACTION_CONTEXT_TURNS` | `10` | prior-turn window for extraction |
| `GNOSIS_FACT_EXTRACTION_MAX_CONCURRENCY` | `2` | background worker concurrency |
| `GNOSIS_FACT_EXTRACTION_MAX_PENDING` | `200` | background queue bound |
| `GNOSIS_ENTITY_GRAPH_ENABLED` | `false` | **materialize the `Entity`/`MENTIONS`/`RELATES` graph** from triples |
| `GNOSIS_EXTRACT_ENTITIES_ENABLED` / `_RELATIONS_ENABLED` / `_PREVIEW_ENABLED` | `false` | document-extraction toggles |
| `GNOSIS_EXTRACTION_BATCH_SIZE` / `_MAX_CONCURRENCY` / `_CHUNK_SIZE` / `_CHUNK_OVERLAP` | 25 / 1 / 4000 / 200 | document-extraction tuning |

## Read-path features

All default `false` / `1`. See [CAPABILITIES.md](CAPABILITIES.md) for each
technique. With `GNOSIS_ADAPTIVE_ROUTING_ENABLED` on, several of these are
applied per-route rather than globally.

| Setting | Default | Technique |
|---|---|---|
| `GNOSIS_ADAPTIVE_ROUTING_ENABLED` | `false` | **adaptive per-query routing** (composition core) |
| `GNOSIS_ROUTING_MODEL` | *(→ `GNOSIS_LLM`)* | router model |
| `GNOSIS_CHAIN_OF_NOTE_ENABLED` | `false` | **route-aware hardened Chain-of-Note** |
| `GNOSIS_CON_SPECULATIVE_INFERENCE_ENABLED` | `false` | CoN speculative-inference widening (tunable) |
| `GNOSIS_CON_ENUMERATION_ENABLED` | `false` | CoN exhaustive-enumeration clause |
| `GNOSIS_HYBRID_RETRIEVAL_ENABLED` | `false` | BM25 + dense RRF fusion |
| `GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED` | `false` | scope-narrowed dense (multi-user stores) |
| `GNOSIS_DENSE_SCOPE_POOL` | `4000` | scoped-dense over-fetch pool |
| `GNOSIS_READ_SUPERSESSION_ENABLED` | `false` | deterministic read-time newest-wins |
| `GNOSIS_RERANK_ENABLED` | `false` | listwise LLM reranker |
| `GNOSIS_RERANK_MODEL` / `_CANDIDATE_CAP` | *(→ `GNOSIS_LLM`)* / `50` | reranker model / how many to reorder |
| `GNOSIS_RECALL_FILTER_ENABLED` / `_CANDIDATES` | `false` / `30` | LLM recall filter (rejected on LOCOMO) |
| `GNOSIS_COMMUNITY_GRAPH_ENABLED` | `false` | community graph / subgraph summaries for open-domain aggregation (tried L-27, rejected: -0.2pp overall, SSA -5.3pp) |
| `GNOSIS_QUERY_REWRITE_ENABLED` | `false` | multi-query reformulation at retrieval time (tried L-7, rejected globally: MS -11.1pp; may be re-enabled per-route) |
| `GNOSIS_SUFFICIENCY_CHECK_ENABLED` / `_MODEL` | `false` | sufficiency autorater signal |
| `GNOSIS_ABSTENTION_PROMPT_ENABLED` | `false` | abstention grounding instruction |
| `GNOSIS_FACT_VERBATIM_EXPANSION_ENABLED` / `_MAX` | `false` / `5` | render source turns under top facts |
| `GNOSIS_GRAPHQA_ENABLED` | `false` | LLM-planned Cypher for `/v1/graph/context`, `include_graph`, and fusion; off = fixed parameterised queries only (see [Security](security.md#graph-qa-safety)) |
| `GNOSIS_GRAPHQA_FUSION_ENABLED` / `_TIMEOUT_SECONDS` | `false` / `20` | dual-route graph-QA fusion (no-op unless `GNOSIS_GRAPHQA_ENABLED`) |
| `GNOSIS_GRAPH_TRAVERSAL_ENABLED` / `GNOSIS_BRIDGE_TRAVERSAL_ENABLED` | `false` | entity / bridge traversal |
| `GNOSIS_COVERAGE_BUDGET_MULTIPLIER` | `1` | item-budget multiplier on enumeration routes (1–5) |

## Prompt enrichment, maintenance, media

| Setting | Default | Purpose |
|---|---|---|
| `GNOSIS_PROMPT_ENTITIES_ENABLED` / `_PREFERENCES_ENABLED` / `_REASONING_ENABLED` | `false` | add entity / preference / reasoning sections to context |
| `GNOSIS_CONSOLIDATION_SCHEDULE_ENABLED` | `false` | scheduled consolidation |
| `GNOSIS_OCR_ENABLED` / `_MODEL` / `_MAX_IMAGE_BYTES` | `false` | image OCR at ingest |
| `GNOSIS_RUSTFS_ENABLED` / `_BUCKET` / `_PREFIX` / `_ENDPOINT` / `_RETENTION_DAYS` | `false` | S3-compatible blob storage for media |

## MCP & federation

| Setting | Default | Purpose |
|---|---|---|
| `GNOSIS_MCP_ENABLED` / `GNOSIS_MCP_AGENT_ID` | `false` / `mcp-client` | mount the MCP server at `/mcp` |
| `GNOSIS_FEDERATION_TOKEN` | *(empty = inbound federation off)* | inbound federated-caller token class |
| `GNOSIS_PEERS` | `[]` | JSON list of remote peers (`name`, `base_url`, `direction`, `remote_tenant_id`) |
| `GNOSIS_PEER_<NAME>_TOKEN` | — | outbound token per peer (= that peer's `GNOSIS_FEDERATION_TOKEN`) |

See [security.md](security.md) for token classes and [CAPABILITIES.md](CAPABILITIES.md#federation) for the federation model.

## The preferred config

`configs/default.yaml` (Run 18) enables exactly four flags — everything else
stays at its safe default:

```yaml
gnosis_fact_extraction_enabled: true    # write: extract dated fact units
gnosis_entity_graph_enabled: true       # write: materialize the entity graph
gnosis_adaptive_routing_enabled: true   # read: per-query routing
gnosis_chain_of_note_enabled: true      # read: route-aware Chain-of-Note
```

It requires a capable `GNOSIS_LLM`. Full per-run flag sets and scores:
[BENCHMARKS.md](BENCHMARKS.md) and [`configs/`](../configs/README.md).
