# Architecture

gnosis is a **policy gateway** in front of a Neo4j knowledge graph. Clients speak
HTTP; only gnosis touches Neo4j, Bolt, or the `neo4j-agent-memory` SDK. Every
request passes scope enforcement, policy, and redaction, and every response is
prompt-safe `sections[]` rather than raw storage payloads.

## Layers

```
        HTTP clients (Discord bots, hermes agents via hermes-gnosis, MCP)
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │  Gateway  (FastAPI: main.py + routes/*)            │  HTTP contracts
        ├────────────────────────────────────────────────────┤
        │  Policy   (auth, scope_policy, redaction, ingestion)│  auth · scope · redaction
        ├────────────────────────────────────────────────────┤
        │  Orchestration (backend.py, context_assembly)       │  which legs run, in what order
        │    write: fact_extraction/worker, entity_graph      │
        │    read:  query_router, recall_filter, supersession,│
        │           reranker, sufficiency, *_traversal, graph │
        ├────────────────────────────────────────────────────┤
        │  SDK / store (sdk_client, graph_memory_store)       │  neo4j-agent-memory 0.5.0
        └───────────────┬───────────────────────┬────────────┘
                        │                       │
                   Neo4j graph            LiteLLM (LLM + embeddings)
```

## Request flow — read (`POST /v1/memory/context`)

1. **Auth + scope.** The service token is verified; the request scope
   (`tenant`/`space`/`agent`/`session`/`user`/`visibility`) is validated.
2. **Route** (if `GNOSIS_ADAPTIVE_ROUTING_ENABLED`). `query_router` classifies the
   query and produces a `RouteDecision` — the effective feature set for this query.
3. **Retrieval legs run** (some in parallel via `asyncio.gather`): dense
   similarity (optionally scope-narrowed and BM25-fused via RRF), graph-QA
   traversal (`graph_query_qa`, planned + validated + executed read-only), and
   entity/bridge traversal — per the route.
4. **Fuse** the candidate legs (`context_assembly.fuse_graph_facts`), dedup by id
   and rendered line.
5. **Filter & resolve.** Optional LLM recall filter; deterministic read-time
   supersession (newest-wins per slot); optional listwise **rerank** — all
   **before** the item-budget cut, so they decide which facts reach the prompt.
6. **Budget cut** (`cut_with_graph_reserve`), then optional facts→verbatim
   expansion.
7. **Render** compact dated `sections[]`; prepend the Chain-of-Note / abstention
   instruction section if enabled.
8. **Sufficiency** (optional) attaches a `{assessed, sufficient, reason}` block.
9. **Redact** and return.

Every optional step degrades to the simpler path on failure or timeout, so a read
never fails because an enhancement did.

## Request flow — write (`POST /v1/memories`)

1. Auth + scope; ingestion policy applied.
2. The turn-pair is stored as verbatim `Message` nodes (episodic layer).
3. If `GNOSIS_FACT_EXTRACTION_ENABLED`, extraction runs (inline or on the
   background worker): one LLM call per turn-pair emits dated `Fact` units, and —
   if `GNOSIS_ENTITY_GRAPH_ENABLED` — `(head, relation, tail)` triples that
   `entity_graph` materializes into `Entity`/`MENTIONS`/`RELATES`.
4. Non-destructive dedup runs on add; storage stays append-only.

## Graph-QA is planned, validated, executed under policy

Off unless `GNOSIS_GRAPHQA_ENABLED=true` (see [Security](security.md#graph-qa-safety)).
Callers never write Cypher. For a graph question, gnosis: plans Cypher with an LLM
(`graph_query_qa`), **validates** it against a scope-checked schema guide
(`graph_query_validation` / `graph_query_rules` — every node pattern pinned by
`tenant_id: $tenant_id` in its property map, `Entity`/`Fact` also by `user_id`,
no `OR`/`XOR`/`NOT`), logs it, then executes it in a **READ** transaction and
drops any row whose returned tenant columns differ from the caller's
(`graph_query_execution`). Invalid or unsafe plans are rejected and the read
degrades to dense-only.

## Module map

| Area | Modules |
|---|---|
| Gateway / app | `main.py`, `routes/` (`system`, `memory_provider`, `operator`, `reasoning`, `events_skills`), `mcp_server.py` |
| Policy | `auth.py`, `scope_policy.py`, `ingestion_policy.py`, `redaction.py`, `json_redaction.py` |
| Orchestration | `backend.py`, `backend_protocols.py`, `context_assembly.py`, `models.py`, `memory_provider.py`, `memory_filters.py` |
| Write path | `fact_extraction.py`, `extraction_worker.py`, `entity_graph.py`, `event_facts.py`, `dedup_consolidation.py` |
| Read path | `query_router.py`, `recall_filter.py`, `supersession.py`, `reranker.py`, `sufficiency.py`, `entity_traversal.py`, `bridge_traversal.py` |
| Graph QA | `graph_query_qa.py`, `graph_query_validation.py`, `graph_query_rules.py`, `graph_query_execution.py`, `graph_schema.py`, `graph_cypher.py`, `graph_context.py`, `graph_upsert*.py`, `graph_*` |
| Store / SDK | `sdk_client.py`, `graph_memory_store.py`, `graph_store.py` |
| Federation | `federation.py` |
| Reasoning / skills | `reasoning_support.py`, `skill_registry.py` |
| Settings | `settings.py` (+ YAML config loading) |

## Gateway boundary

`gnosis` is the only service in the stack that talks to Neo4j and the Python SDK
directly. It exists to add the layer the SDK lacks: HTTP contracts, an auth model,
scope enforcement, rollout controls, and redaction. See [security.md](security.md)
for the trust boundary and [data-model.md](data-model.md) for the schema.
