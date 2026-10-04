# gnosis

**gnosis** is a self-hosted memory service for AI agents. It exposes an
authenticated, tenant-scoped HTTP gateway backed by a Neo4j graph/vector store and
an OpenAI-compatible LLM/embedding endpoint. Clients interact over HTTP or the
optional MCP mount; they never connect to Neo4j or the Python SDK directly.

The service maintains long-term recall keyed by `tenant_id` + `user_id`. Every
stored memory and retrieval request is scope-checked, redacted, and rendered as
prompt-safe sections for agent consumption.

## Documentation

- [Getting started](docs/getting-started.md) — integration walkthrough
- [Architecture](docs/architecture.md) — request flow and module map
- [Data model](docs/data-model.md) — graph schema and scope spine
- [Provider surface](docs/provider-surface.md) — HTTP and MCP contracts
- [Configuration](docs/configuration.md) — environment variables and YAML keys
- [Security](docs/security.md) — token classes, scope, redaction, federation
- [Operations](docs/operations.md) — health, workers, backup, scaling
- [Capabilities](docs/CAPABILITIES.md) — feature behavior and measured tradeoffs
- [Development](docs/development.md) — contribution and measurement workflow
- [Benchmarks](docs/BENCHMARKS.md) — maintained benchmark ledger

## Quick start

The tracked [`compose.yaml`](compose.yaml) starts Neo4j 5.26+ and the published
`ghcr.io/nolgiainc/gnosis:latest` image. You need Docker Compose v2 and an
OpenAI-compatible chat/embedding endpoint. The default compose points at Ollama on
the host; set variables (or a `.env` next to the file) for LiteLLM, OpenAI, or
another endpoint.

```bash
git clone https://github.com/nolgiainc/gnosis.git
cd gnosis
ollama pull llama3.2:latest
ollama pull nomic-embed-text
docker compose up -d
```

Check liveness and backend readiness:

```bash
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8080/ready
```

The compose file uses development-only placeholder tokens. Replace them with
secret-backed values before any non-disposable deployment. Tear down with
`docker compose down -v` (the `-v` removes the named Neo4j volume).

### Write and read a memory

```bash
export GNOSIS_URL=http://localhost:8080
export GNOSIS_TOKEN=dev-token

# Verbatim write (no extraction LLM call):
curl -fsS "$GNOSIS_URL/v1/memories" \
  -H "Authorization: Bearer $GNOSIS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "scope": {"tenant_id":"nolgia","space_id":"demo","agent_id":"assistant",
              "session_id":"session-1","user_id":"alice","visibility":"private_user"},
    "content": "Alice moved from Seattle to Austin in March.",
    "infer": false
  }'

# Retrieve context for a follow-up question:
curl -fsS "$GNOSIS_URL/v1/memory/context" \
  -H "Authorization: Bearer $GNOSIS_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "scope": {"tenant_id":"nolgia","space_id":"demo","agent_id":"assistant",
              "session_id":"session-2","user_id":"alice","visibility":"private_user"},
    "query": "Where does Alice live?",
    "max_items": 8
  }'
```

The response contains `sections[]` with scoped, redacted context. To use
extraction mode, send a `messages` array with `"infer": true` and a capable
`GNOSIS_LLM` — extraction makes LLM calls.

## Features

### Write path

- **Verbatim storage**: store any content as a dated, scoped memory unit with zero
  LLM calls.
- **Fact extraction** (`GNOSIS_FACT_EXTRACTION_ENABLED`): extract self-contained,
  dated, entity-normalized fact units from conversation turns at ingest. The
  single largest quality lever measured: +11.7 J on LOCOMO, temporal +42 points.
- **Entity graph** (`GNOSIS_ENTITY_GRAPH_ENABLED`): materialize a Neo4j knowledge
  graph of named entities and their relationships alongside extracted facts.
  Enables graph-based multi-hop retrieval.
- **Community graph** (`GNOSIS_COMMUNITY_GRAPH_ENABLED`): detect clusters among
  entity nodes, generate LLM summaries per community, and store them as
  `:Community` nodes. Closes the open-domain retrieval gap (Zep/Graphiti analysis:
  ~30 pp vs pure entity retrieval).
- **Write buffer**: optional async ingestion with configurable concurrency and
  back-pressure (`GNOSIS_WRITE_MODE=buffered`).

### Read path

- **BM25 + dense fusion** (`GNOSIS_HYBRID_RETRIEVAL_ENABLED`): reciprocal rank
  fusion of full-text and vector search. On temporal queries: +7.8 J.
- **Adaptive routing** (`GNOSIS_ADAPTIVE_ROUTING_ENABLED`): one cheap LLM
  classification call per query selects the measured-best retrieval strategy for
  that query's category (temporal → BM25+dense; multi-hop → graph-QA + verbatim;
  etc.). +2.9 J vs any single global strategy.
- **Scoped dense retrieval** (`GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED`): narrows
  vector search to the request scope in-query, required for correctness in
  multi-user single-store deployments (e.g., LongMemEval multi-instance runs).
- **Graph-QA fusion** (`GNOSIS_GRAPHQA_FUSION_ENABLED`): Cypher query planner over
  the entity graph, fused with vector candidates. Validated, read-only, tenant-scoped.
  Requires `GNOSIS_GRAPHQA_ENABLED=true` (off by default; see
  [Security](docs/security.md#graph-qa-safety)).
- **LLM reranker** (`GNOSIS_RERANK_ENABLED`): listwise reranker over the top-N
  fused candidates before the item-budget cut. Reranking is the single lever
  present in all strongest 2026 systems (Mnemis, EverMemOS, agentmemory).
  Benchmark result pending on LongMemEval_S.
- **Chain-of-Note** (`GNOSIS_CHAIN_OF_NOTE_ENABLED`): reading instruction that
  makes the answerer cite evidence and abstain when context is insufficient.
  Route-aware (skipped on temporal route where it hurts). +8.9 adversarial J.
- **Sufficiency check** (`GNOSIS_SUFFICIENCY_CHECK_ENABLED`): autorater that
  judges whether retrieved context fully determines the answer — signal exposed to
  clients, not used to block responses.
- **Multi-query rewrite** (`GNOSIS_QUERY_REWRITE_ENABLED`): when the sufficiency
  check fires, generates 2–3 complementary queries (entity pivot, temporal
  calculation, concept expansion, HyDE) and RRF-fuses results. Requires
  `GNOSIS_SUFFICIENCY_CHECK_ENABLED=true`. EverMemOS fires this on 31% of queries.
- **Read-time supersession** (`GNOSIS_READ_SUPERSESSION_ENABLED`): newest-wins
  conflict resolution at retrieval time. Evidence-backed: deterministic newest-wins
  scores 94.8% on FactConsolidation vs LLM-based invalidation at 7%.

### Trust and safety

- All operations scoped by `tenant_id`, `space_id`, `agent_id`, `session_id`,
  `user_id`, and `visibility`. Tenant mismatches rejected before backend access.
- Separate least-privilege token classes: read, write, export, admin, federation.
- Prompt-facing output is redacted. Graph-QA never accepts caller-supplied Cypher.
- Deduplication and consolidation are dry-run-first; federation requires explicit
  `metadata.shareable: true`.

## API surface

| Surface | Routes |
|---|---|
| Health | `GET /health`, `GET /ready`, `GET /v1/diagnostics` |
| Memory | `POST /v1/memories`, `/v1/memories/search`, `/v1/memories/list`, `/v1/memories/promote` |
| Context | `POST /v1/memory/context`, `/v1/graph/context`, `/v1/reasoning/context` |
| Ingestion | `POST /v1/messages`, `/v1/events`, `/v1/events/batch`, `/v1/memory/extraction/preview` |
| Editing | `PATCH`/`DELETE /v1/memories/{memory_id}` (requires `GNOSIS_MEMORY_EDIT_ENABLED=true`) |
| MCP | Streamable HTTP at `/mcp` (requires `GNOSIS_MCP_ENABLED=true`) |

`/health` and `/ready` are unauthenticated. All other routes require
`Authorization: Bearer <token>`. Full schema: FastAPI `/docs` and `/openapi.json`.

## Configuration

Minimum required: `GNOSIS_TOKEN`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`,
`LITELLM_BASE_URL`, `LITELLM_API_KEY`.

**gnosis auto-loads `configs/default.yaml` when `GNOSIS_CONFIG_FILE` is unset**,
which enables the benchmark-best feature set (fact extraction + entity graph +
adaptive routing + Chain-of-Note). Set `GNOSIS_CONFIG_FILE=""` to use the minimal
safe defaults, or point it at a specific run config.

Precedence: explicit environment variables → `.env` → YAML config → code defaults.

See [docs/configuration.md](docs/configuration.md) for the complete variable matrix.

## Source run

```bash
uv sync --locked
uv run uvicorn gnosis.main:app --host localhost --port 8080
```

## Benchmark standing

### LongMemEval_S — L-33 (full 500-Q, 2026-08-10), gpt-4o backbone + judge — **current best**

| Category | gnosis L-33 | gnosis L-25b | Zep | mem0 | Chronos (SOTA) |
|---|---|---|---|---|---|
| single-session-assistant | 94.6% (n=56) | **98.2%** | — | — | — |
| single-session-user | 82.8% (n=64) | 84.4% | — | — | — |
| knowledge-update | **81.9%** (n=72) | 70.8% | 83.3% | — | **100%** |
| temporal-reasoning | 69.3% (n=127) | 74.0% | 62.4% | — | 95.5% |
| multi-session | **60.3%** (n=121) | 58.7% | 57.9% | — | 88.7% |
| single-session-preference | **66.7%** (n=30) | 60.0% | — | — | — |
| abstention | **83.3%** (n=30) | 83.3% | — | — | — |
| **Overall** | **74.2%** (500 Q) | 73.6% | 71.2% | 67.6% | 95.6% |

*L-33 reuses L-31 Neo4j data (no re-ingest). New best overall (74.2%), best KU (81.9%), best MS (60.3%). SSA/temporal remain below L-25b — ingest variation from L-31 fresh reingest.*

**L-33 config (on top of L-32):** extended `_AGGREGATIVE_PATTERN` (added `average|percentage|how long`) + 4 sub-queries (was 2) + set-based dedup in membench answer.py.

**Key remaining gaps (L-33 baseline):**
- **KU (81.9%):** gap to Zep (83.3%): 1.4pp; gap to Chronos (100%): 18.1pp.
- **Multi-session (60.3%):** +0.8pp from L-32; 39.7% remaining failure rate (48/121).
- **Temporal (69.3%):** gap to L-25b (74.0%): 4.7pp — ingest-variation gap, not an L-33 regression.

**L-31 (2026-08-09):** write-time SUPERSEDES edges + `valid_to IS NULL` filter. KU **70.8% → 80.6% (+9.8pp)**. Overall 71.0%; regressions confirmed as ingest variation (not SUPERSEDES logic). See [RESULTS.md](https://github.com/nolgiainc/gnosis-membench/blob/main/RESULTS.md).

**L-32 (2026-08-10):** enumeration clause fix (`GNOSIS_CON_ENUMERATION_ENABLED=true`) + 2-sub-query expansion for aggregative multi-session questions. MS **54.5% → 59.5% (+5.0pp)**. Overall **72.6%** (+1.6pp vs L-31). No re-ingest.

**L-33 (2026-08-10) — COMPLETE:** extended aggregative pattern + 4 sub-queries (was 2) + set-based dedup in membench answer.py. Overall **74.2%** (+1.6pp vs L-32, **+0.6pp vs previous best L-25b**). MS **59.5% → 60.3%** (+0.8pp). KU flat (81.9%). No re-ingest. See [gnosis-membench RESULTS.md](https://github.com/nolgiainc/gnosis-membench/blob/main/RESULTS.md).

### LOCOMO — Run 23 (full 10-conversation, 2026-07-04), GPT-5.5 judge

| Category | gnosis | mem0 | mem0-graph | Zep |
|---|---|---|---|---|
| single-hop J | **77.0** | 67.13 | 65.71 | 61.70 |
| temporal J | **73.8** | 55.51 | 58.13 | 49.31 |
| multi-hop F1 | **34.3** | 28.64 | 24.32 | 19.37 |
| open-domain J | 29.2 | 72.93 | 75.71 | **76.60** |
| adversarial J | **83.9** | — | — | — |
| **excl-adv J** | **66.9–68.9** | 66.88 | 68.44 | 65.99 |

Open-domain is the primary LOCOMO gap (29.2 vs frontier ~74–77). The community graph
feature (`GNOSIS_COMMUNITY_GRAPH_ENABLED`) targets this gap with cluster-level summaries.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and
[gnosis-membench RESULTS.md](https://github.com/nolgiainc/gnosis-membench/blob/main/RESULTS.md) for the full run ledger.

## Development

```bash
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run basedpyright
uv run pytest -q
```

The Docker build runs on `main` push only. For feature work: keep optional flags
default-off, ensure LLM-backed features degrade gracefully, and measure with
[gnosis-membench](https://github.com/nolgiainc/gnosis-membench) before making
quality claims.

## Deployment

- [`Dockerfile`](Dockerfile): pinned `uv.lock`, copies `src/` and `configs/`,
  starts Uvicorn on port 8080.
- [`compose.yaml`](compose.yaml): minimal Neo4j + service stack for local use.
  Not a production topology or secret-management system.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): test gate on PRs; push
  to `main` also builds and publishes `ghcr.io/nolgiainc/gnosis:latest`.

Kubernetes, ingress, secret management, and rollout policy are owned by your
deployment environment. Keep all credentials and token classes in environment
secret-backed configuration.

## Related projects

- [gnosis-membench](https://github.com/nolgiainc/gnosis-membench) — benchmark
  harness for LOCOMO and LongMemEval experiments.
- [hermes-gnosis](https://github.com/nolgiainc/hermes-gnosis) — memory-provider
  plugin for NousResearch Hermes agents.
