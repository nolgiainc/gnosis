# gnosis capabilities & techniques

A reference to what gnosis does and *how* — the algorithms and the peer-reviewed
work each one is grounded in. gnosis is a policy gateway in front of a Neo4j
knowledge graph that gives agents scoped, auditable, benchmarked long-term
memory over one HTTP API. Every technique below is a discrete, measured
capability rather than an undifferentiated "RAG pipeline."

- **Scores and per-run history:** [BENCHMARKS.md](BENCHMARKS.md)
- **HTTP contract:** [provider-surface.md](provider-surface.md)
- **Configuration:** [../configs/README.md](../configs/README.md) (the preferred config is `configs/default.yaml`)

---

## Design principles

Four principles shape every capability here.

1. **Gateway boundary.** Clients never touch Neo4j, Bolt, or the SDK. Every read
   and write passes scope enforcement, redaction, and rollout policy first, and
   the response is prompt-safe `sections[]`, not raw storage payloads.

2. **Everything is a flag, off by default, byte-identical when off.** Each
   technique is independently toggleable; a disabled feature adds zero bytes to
   the read/write path. This makes every change measurable in isolation and
   keeps a minimal, cheap default.

3. **Measured, not vibes.** Quality is tracked on LOCOMO
   ([arXiv 2402.17753](https://arxiv.org/abs/2402.17753)) and LongMemEval
   ([arXiv 2410.10813](https://arxiv.org/abs/2410.10813)) with the *official*
   judging protocols. Kept **and** rejected changes are recorded in
   [BENCHMARKS.md](BENCHMARKS.md); the harness re-scores every release against a
   frozen judge so numbers cannot silently regress.

4. **The route table is the composition mechanism.** Stacking the individually
   best features globally *destroys* the score (measured: −13.3 J when
   extraction + hybrid + verbatim + supersession were all enabled at once —
   undated raw turns displaced dated extracted facts). gnosis therefore does
   **not** run every feature on every query. A per-query router classifies the
   query and applies only *that category's* measured-best feature set. This is
   the central architectural decision and the reason gnosis composes cleanly
   where naive stacking regresses.

5. **Graceful degradation.** Every LLM-augmented step (routing, reranking,
   filtering, graph traversal, sufficiency) degrades to the simpler path on any
   failure or timeout and logs a structured warning — a read is never blocked by
   an optional enhancement.

---

## Capability map

| Capability | Flag | Grounded in | Path | Status |
|---|---|---|---|---|
| Fact extraction — atomic dated units (edu-v2.0) | `GNOSIS_FACT_EXTRACTION_ENABLED` | EMem (2511.17208), EverMemOS, Memanto (2604.22085) | write | **kept** — +11.7 J; edu-v2.0 adds Rule 15 (assistant-turn extraction) |
| **Write-time structural supersession (SUPERSEDES edges + valid\_to)** | *(part of extraction path)* | arXiv 2607.26520 (Graph-Native Bitemporal), arXiv 2607.21962 (Ground Truth First) | write | **new (L-31)** — structural KU fix; marks stale same-slot facts with `valid_to` at ingest; filters at retrieval |
| Relation-class-aware supersession (relation_slots) | `GNOSIS_READ_SUPERSESSION_ENABLED` | "Don't Ask the LLM to Track Freshness" (2606.01435) | write+read | **kept** — KU fix; groups facts by entity+relation slot for newest-wins |
| Entity-graph materialization | `GNOSIS_ENTITY_GRAPH_ENABLED` | HippoRAG 2 (2502.14802), Graphiti (2501.13956) | write | kept (drives graph recall) |
| Bi-temporal fact model with write-time SUPERSEDES | *(always on)* | Zep/Graphiti (2501.13956), arXiv 2607.26520 | write | kept — now includes deterministic `valid_to` invalidation at write time (L-31) |
| Selective addition + dedup | `GNOSIS_FACT_DEDUPLICATION_ENABLED` | Selective memory addition (2505.16067) | write | kept |
| Relevance-ranked dated assembly | *(baseline)* | — | read | kept — +18.5 J over verbatim RAG |
| Hybrid retrieval (BM25 + dense, RRF) | `GNOSIS_HYBRID_RETRIEVAL_ENABLED` | RRF; EverMemOS (k=60) | read | routed (wash globally on short LOCOMO) |
| Scope-narrowed dense retrieval | `GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED` | correctness for multi-user stores | read | kept for shared stores |
| **Adaptive per-query routing (6 routes incl. knowledge_update)** | `GNOSIS_ADAPTIVE_ROUTING_ENABLED` | Adaptive-RAG (2403.14403) | read | **kept** — +2.9 J; the composition mechanism; L-29 adds knowledge_update route |
| **Chain-of-Note (read-then-reason)** | `GNOSIS_CHAIN_OF_NOTE_ENABLED` | Chain-of-Note (2311.09210) | read | **kept** — adversarial 83.0 (peak) |
| Read-time supersession (newest-wins) | `GNOSIS_READ_SUPERSESSION_ENABLED` | "Don't Ask the LLM to Track Freshness" (2606.01435) | read | kept (routed) |
| Sufficiency signalling | `GNOSIS_SUFFICIENCY_CHECK_ENABLED` | Sufficient Context (2411.06037) | read | client hint |
| Abstention prompting | `GNOSIS_ABSTENTION_PROMPT_ENABLED` | AbstentionBench (2506.09038) | read | superseded by CoN |
| Facts→verbatim expansion | `GNOSIS_FACT_VERBATIM_EXPANSION_ENABLED` | EverMemOS facts→episodes | read | routed |
| Graph-QA fusion (dual-route) | `GNOSIS_GRAPHQA_FUSION_ENABLED` | Mnemis dual-route (2602.15313) | read | routed |
| Entity / bridge traversal | `GNOSIS_GRAPH_TRAVERSAL_ENABLED`, `GNOSIS_BRIDGE_TRAVERSAL_ENABLED` | Self-Ask (2210.03350), IRCoT (2212.10509), HippoRAG (2405.14831) | read | routed (fires rarely on LOCOMO) |
| Listwise LLM reranker | `GNOSIS_RERANK_ENABLED` | RankGPT; Mnemis reranker ablation | read | **new**, default-off, unmeasured |
| LLM recall filter | `GNOSIS_RECALL_FILTER_ENABLED` | EMem (2511.17208) | read | rejected on LOCOMO (flat, +6s/read) |
| Community graph / subgraph summaries | `GNOSIS_COMMUNITY_GRAPH_ENABLED` | GraphRAG (2404.16130), Graphiti/Zep | read | **tried (L-27), rejected** — -0.2pp overall; SSA -5.3pp, MS -5.0pp; temporal +2.8pp, SSP +3.3pp; gains did not outweigh regressions |
| Multi-query rewrite (query reformulation) | `GNOSIS_QUERY_REWRITE_ENABLED` | Adaptive-RAG | read | **tried (L-7), rejected** — helped T +5.3pp but hurt MS -11.1pp globally; routing may resurrect selectively |

"Routed" = enabled per-query by the router for the categories it helps, not globally.

---

## The write path (ingest)

### Fact extraction — atomic, dated, self-contained units

**Approach.** On each conversational turn-pair, an LLM extracts atomic
*event-centric fact units* — self-contained statements carrying their own
absolute date — and stores them alongside the verbatim turns. This turns a
transcript into durable, individually-retrievable knowledge instead of a bag of
messages.

**Algorithm.** One extraction call per turn-pair, with a bounded context window
of prior turns. The current prompt (edu-v2.0) emits `{subject, predicate, object,
event_date, entities}` and, when the entity graph is enabled, `(head, relation,
tail)` triples. `session_date` is consumed as the conversation date so relative
references ("last week") resolve to absolute ranges. Extraction runs on a
bounded async queue (`GNOSIS_FACT_EXTRACTION_MODE=background`, drop-not-block) so
it never blocks the write response; malformed model JSON is re-sampled rather
than dropped.

**Prompt versions.** `edu-v1` extracted only user-stated facts (human-only exemplar).
`edu-v2.0` adds Rule 15 — explicitly extract from assistant turns (recommendations,
commitments, stated facts, how-to guidance) with attribution "The assistant
recommended/stated/committed …". The exemplar was replaced with a user+assistant
exchange demonstrating 3 assistant-attributed extractions. Version is stored in each
fact's `extraction_version` metadata, so old and new facts coexist without migration.

**Research.** Event-centric fact units with an LLM recall filter beat
Mem0/Zep/full-context at ~740 tokens in EMem
([arXiv 2511.17208](https://arxiv.org/abs/2511.17208)); EverMemOS's "MemCells"
and the RL-trained extractor of Memory-R2 point the same way — *extraction
quality dominates backbone size*. Memanto ([arXiv 2604.22085](https://arxiv.org/abs/2604.22085))
identifies the assistant-perspective extraction gap: SSA is suppressed when only
user turns are extracted.

**Impact.** +11.7 J on LOCOMO (temporal 42→84) — the single largest measured
lever. With extraction off, gnosis is a dated-RAG store.

### Entity-graph materialization

**Approach.** Materialize a per-user knowledge graph next to the extracted
facts, so multi-hop questions can be answered by *walking relationships* rather
than hoping the chain of intermediate facts survives a dense top-k.

**Algorithm.** From each extracted `(head, relation, tail)` triple, gnosis
writes `(:Entity)` nodes (deduplicated within `tenant_id`+`user_id` scope),
`(:Fact)-[:MENTIONS]->(:Entity)` provenance edges, and directed
`(:Entity)-[:RELATES {relation, fact_id, event_date}]->(:Entity)` edges. The
graph-QA validator scopes every `Entity`/`Fact` alias by both tenant and user.
Because it is a write-path change, the payoff only appears on a fresh ingest.

**Research.** HippoRAG 2 ([arXiv 2502.14802](https://arxiv.org/abs/2502.14802),
ICML 2025) and Graphiti/Zep
([arXiv 2501.13956](https://arxiv.org/abs/2501.13956)) — passages/entities as
graph nodes for multi-hop recall.

### Bi-temporal fact model + write-time SUPERSEDES edges (L-31)

**Approach.** Every fact carries two timestamps: `event_date` (when the thing
happened) and `created_at` (when it was written). Reads date-anchor and, for
supersession, reason over recency. As of L-31, gnosis also writes structural
`SUPERSEDES` edges at ingest time and marks stale facts with `valid_to`.

**Write-time structural supersession (L-31).** When a new extracted fact occupies
the same `relation_slot` as an existing fact in scope (e.g. "alice:works_at"),
gnosis writes a `(new_fact)-[:SUPERSEDES]->(old_fact)` edge in Neo4j and sets
`old_fact.valid_to = datetime()`. For the `knowledge_update` retrieval route, the
vector and BM25 Cypher queries filter `f.valid_to IS NULL` — so structurally
superseded facts are *excluded from the embedding search* rather than just dropped
after retrieval. This is the fix for the core KU problem: old facts outrank new
facts in embedding space (the update is phrased in a different conversational
context), so post-retrieval supersession could not help when the new fact was
already below rank 20.

**Research.** arXiv 2607.26520 (Graph-Native Bitemporal Memory) implements this
pattern natively — immutable identity nodes + versioned content nodes with
`valid_time`/`transaction_time` + write-time semantic edges. arXiv 2607.21962
(Ground Truth First) provides the volatility-class taxonomy (singleton vs.
additive relations) that determines which facts participate in supersession. The
freshness study "Don't Ask the LLM to Track Freshness" (arXiv 2606.01435) validates
append-only + deterministic invalidation over LLM-based write-time invalidation
(94.8% vs 7%); gnosis's approach IS the deterministic variant, now applied at the
DB query layer rather than only at the application layer.

**Relation to the prior design.** Zep/Graphiti ([2501.13956](https://arxiv.org/abs/2501.13956))
use an LLM to set `invalid_at` at write time. gnosis rejects LLM-based
invalidation (the freshness study measures it at ~7%) and uses deterministic
slot-matching instead: the `relation_slot` key `"alice:works_at"` is computed
from the extracted `(head, relation_class)` triple, not from a model judgment call.
Storage remains append-only — no facts are deleted or mutated beyond the `valid_to`
timestamp on the superseded fact.

### Selective addition & non-destructive dedup

**Approach.** Adds are selective, and consolidation is used only to deduplicate,
never to destroy. Append-only storage with read-time resolution keeps every
write auditable.

**Research.** Adding everything indiscriminately degrades quality (67.5→55.5 in
[arXiv 2505.16067](https://arxiv.org/abs/2505.16067)); the right selectivity is
at the store boundary, not destructive edits after the fact.

---

## The read path

A memory-context request assembles prompt-facing facts through several stages.
The default (baseline) path already ranks and dates facts; the flags below add
targeted capability, gated by the router.

### Retrieval

**Relevance-ranked, date-anchored assembly (baseline).** When the request
carries a query, candidate facts come from the same embedding-similarity search
`/v1/memories/search` uses, and each is rendered as one compact dated line
(`- [7 May 2023] ...`), preferring a stored `session_date`/`date` over
`created_at`. Without a query or embedder, it falls back to recency. This alone
took gnosis from 37.4 → 59.5 J over verbatim, dateless RAG.

**Hybrid retrieval (BM25 + dense, RRF).** `GNOSIS_HYBRID_RETRIEVAL_ENABLED` runs
a Neo4j BM25 full-text search beside the embedding search and fuses the two
rankings with Reciprocal Rank Fusion (k=60, the value EverMemOS uses) before
scope re-checks — in both context assembly and search. Keyword-heavy or
long-haystack queries benefit; it was a wash on short LOCOMO conversations
globally, so the router applies it where it helps.

**Scope-narrowed dense retrieval.** `GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED` is a
correctness fix for multi-user single-store deployments (e.g. LongMemEval, where
many users' near-identical facts share one store). The SDK ranks the fact-vector
index *globally* and only post-filters by scope, so the requesting user's
candidates get crowded out; when enabled, dense candidates come from a
scope-narrowed vector query (over-fetch `GNOSIS_DENSE_SCOPE_POOL`, filter to
scope in-query, keep the top candidates). Single-user reads are byte-identical.

### Adaptive per-query routing — the composition core

**Approach.** Rather than run one fixed pipeline, gnosis classifies each query
and applies the feature set measured-best for *that* category — the mechanism
that lets independently-good features coexist without the interference that
sinks global stacking.

**Algorithm.** `GNOSIS_ADAPTIVE_ROUTING_ENABLED` sends the query to an LLM
classifier that returns one of six routes. The resulting `RouteDecision` carries
the effective read-path feature set — which retrieval legs run, the Chain-of-Note
variant, the item-budget multiplier, the supersession filter — so temporal queries
route to dated/hybrid retrieval, multi-hop/aggregative get graph traversal and
expanded budgets, and knowledge-update gets structural `valid_to` filtering. Routing
failures degrade to the globally-configured flags.

**Routes (as of L-31):**

| Route | When | Key features |
|---|---|---|
| `temporal` | asks WHEN, elapsed time, date ordering | hybrid BM25+dense; CoN off (verbatim date leakage); supersession off |
| `multi_hop` | two or more chained facts needed | graph-QA fusion; verbatim expansion; dense-only |
| `aggregative` | count/frequency/synthesis across memories | hybrid BM25+dense; expanded budget; supersession off |
| `knowledge_update` | current value of a fact that changes over time | hybrid BM25+dense; `valid_to IS NULL` filter; **recency injection** (top-5 most-recently-ingested facts merged into dense top-20 before reranking); supersession on |
| `unanswerable_risk` | topic definitively absent from personal memory | abstention prompt; sufficiency check; supersession off |
| `single_hop` | everything else — one fact answers it | hybrid BM25+dense (L-20) |

**Research.** Adaptive-RAG
([arXiv 2403.14403](https://arxiv.org/abs/2403.14403)) — match retrieval strategy
to query complexity.

**Impact.** +2.9 J (new best at the time); it is what makes the per-category
peaks (single-hop, temporal, adversarial) hold *simultaneously* instead of
trading off.

### Reranking & filtering

**Listwise LLM reranker.** `GNOSIS_RERANK_ENABLED` reorders the fused candidate
pool by query relevance **before** the item-budget cut — so the reranker, not
raw vector proximity, decides which facts reach the prompt. One structured-output
call returns a ranked permutation of the top candidates (RankGPT-style listwise
reranking); a cross-encoder would be cheaper but none is exposed on the
deployment's LLM router. It never drops a candidate (omitted/out-of-range indices
fall back to retrieval order) and never blocks a read. Retrieval is the
long-haystack bottleneck (LongMemEval: full-context 0.606 vs oracle retrieval
0.870), and a reranker is the lever common to the strongest 2026 systems. New,
default-off, measurement pending.

**LLM recall filter.** `GNOSIS_RECALL_FILTER_ENABLED` screens the top candidates
with one LLM call, keeping only those that could help answer the query (can
remove, never add). EMem ([arXiv 2511.17208](https://arxiv.org/abs/2511.17208))
reports the filter as its single biggest component; on short LOCOMO it was flat
and cost ~6s/read, so it is off by default and documented as rejected — an honest
negative result kept in the record.

### Read-time supersession — deterministic newest-wins

**Approach.** When several facts fill the same "slot" (the same thing stated at
different times), keep only the newest at read time — without ever mutating
storage. As of L-31, this is supplemented by a write-time structural layer (see
[Bi-temporal fact model + write-time SUPERSEDES](#bi-temporal-fact-model--write-time-supersedes-edges-l-31)).

**Algorithm.** `GNOSIS_READ_SUPERSESSION_ENABLED` runs a deterministic pass:
same slot = same normalized subject plus normalized predicate (typed facts); for
extracted facts, the slot is the `relation_slots` metadata key written at ingest
time — `"{normalized_head}:{normalized_relation_class}"` (e.g. `"alice:works_at"`).
This makes supersession relation-class-aware: two facts about Alice's employer are
correctly grouped and the newer one wins, even if they use different phrasings.
Newest wins by `event_date`, else `created_at`; ties, cross-user, cross-scope,
`said_*`, and no-entity facts are conservatively kept. Append-only storage is
untouched — this is a *read-time* resolution.

**Singleton-only restriction (L-25b).** Only *singleton* relation classes
participate in slot supersession — relations where at most one value is true at
a time: `works_at`, `lives_in`, `relationship_status`, `studies_at`, and similar
biographical anchors. *Additive* relations — `likes`, `prefers`, `has_visited`,
`owns` — can have multiple concurrent valid values and are excluded from
supersession. The `is_singleton_relation_class()` predicate in `supersession.py`
encodes this taxonomy (informed by arXiv:2607.21962 "Ground Truth First" volatility
classes). This restriction fixed an over-supersession regression in L-25 (before
the fix, preference and event-accumulation facts were incorrectly superseded).

**L-31 structural layer.** Read-time supersession can only act on candidates that
reach the application layer. If the new fact is below rank 20 in embedding space
(because the old, stale phrasing matches the query better), it never enters the
candidate pool and read-time supersession cannot help. The write-time SUPERSEDES
edges + `valid_to` filter (L-31) solve this by excluding structurally superseded
facts from the vector and BM25 Cypher queries for the `knowledge_update` route,
so the current (non-superseded) fact naturally reaches the top-20.

**Research.** "Don't Ask the LLM to Track Freshness"
([arXiv 2606.01435](https://arxiv.org/abs/2606.01435)): deterministic read-time
newest-wins scores 78–94.8% on FactConsolidation where LLM write-time
invalidation scores ~7%. This is a deliberate, measured reversal of the
bi-temporal-invalidation orthodoxy. L-31 extends this into the query layer —
deterministic `valid_to IS NULL` filtering rather than application-layer dropping.

### Reading aids (instruction-level)

**Chain-of-Note (read-then-reason).** `GNOSIS_CHAIN_OF_NOTE_ENABLED` prepends a
standing instruction as a leading `instructions` section telling the reader to
silently note, per memory, whether it is relevant, what it says, who it is
about, and whether it contradicts another memory — then answer only from the
relevant ones and never guess. It is **route-aware** (skipped on the temporal
route, where "state what the memory says" made readers parrot relative dates),
**hardened** (explicit attribution + never-guess clauses), and carries a
**likelihood carve-out** (infer the most plausible answer only when the question
itself asks what is *likely*). Grounded in Chain-of-Note
([arXiv 2311.09210](https://arxiv.org/abs/2311.09210)); it produced the adversarial
83.0 peak and is the highest-leverage read-side seam in the system — three
consecutive prompt-only refinements moved the totals more than any retrieval
change since extraction.

**Abstention prompting.** `GNOSIS_ABSTENTION_PROMPT_ENABLED` prepends a grounding
instruction so clients abstain when the memories do not contain the answer
(AbstentionBench, [arXiv 2506.09038](https://arxiv.org/abs/2506.09038)). Kept for
reference; Chain-of-Note dominates it (it buys abstention without over-abstaining
on answerable questions).

**Sufficiency signalling.** `GNOSIS_SUFFICIENCY_CHECK_ENABLED` adds one
structured-output call that judges whether the assembled context *fully
determines* the answer, and returns an additive `{assessed, sufficient, reason}`
block. gnosis is the memory service, not the answering model, so it *exposes*
the signal rather than deciding to abstain itself. Google's Sufficient Context
study ([arXiv 2411.06037](https://arxiv.org/abs/2411.06037)) shows richer context
makes answerers over-confident even when it is insufficient — a cheap sufficiency
autorater is the fix; retrieval-score thresholds are unreliable.

### Graph-augmented recall

**Graph-QA fusion (dual-route).** `GNOSIS_GRAPHQA_FUSION_ENABLED` (together
with `GNOSIS_GRAPHQA_ENABLED`, which is off by default) runs the
LLM-planned, validation-gated, scope-safe, read-only graph-QA route
(entity → relationship → answer) in parallel (`asyncio.gather`) with dense
retrieval and unions its derived nodes into the candidate set before supersession
and the item budget. A node already surfaced by dense retrieval is not
double-added; the route is bounded by a timeout and degrades to dense-only on any
planner/validation/timeout failure. This is the Mnemis dual-route technique
([arXiv 2602.15313](https://arxiv.org/abs/2602.15313)): a System-2 traversal route
unioned with the System-1 vector route to catch multi-hop chains dense matching
displaces.

**Entity & bridge traversal.** `GNOSIS_GRAPH_TRAVERSAL_ENABLED` walks
`RELATES`/`MENTIONS` from a query-anchored entity; `GNOSIS_BRIDGE_TRAVERSAL_ENABLED`
does a directed second hop from hop-1's bridge node ("which city have both X and
Y visited?" → the connecting entity). The self-ask
([arXiv 2210.03350](https://arxiv.org/abs/2210.03350)) / IRCoT
([arXiv 2212.10509](https://arxiv.org/abs/2212.10509)) / HippoRAG
([arXiv 2405.14831](https://arxiv.org/abs/2405.14831)) line motivates the
decomposition. Measured honestly: the mechanism works (textbook bridge repairs)
but fires on too few LOCOMO questions to move the aggregate, so it is
router-scoped rather than global — LOCOMO's multi-hop misses are cross-session
*enumerations*, not bridge chains.

**Facts→verbatim expansion.** `GNOSIS_FACT_VERBATIM_EXPANSION_ENABLED` renders,
beneath a top-ranked extracted fact, its linked source verbatim turn as an
indented `quote:` line — matching on the precise atomic fact while assembling the
raw text's nuance alongside — via one scope-narrowed, scope-re-checked batch
lookup that never surfaces a cross-scope turn or double-renders. EverMemOS
facts→episodes; degrades to the compact fact alone on any lookup failure.

---

## Federation

Two sovereign gnosis deployments can selectively share memory without merging
stores. Federation is off in both directions by default and one peer concept
backs both paths.

- **Consent is a tag, not a policy toggle.** A memory crosses a boundary only if
  its metadata carries `"shareable": true`; the gateway conjoins that filter onto
  every federated read and promote scan server-side, regardless of caller filters.
- **Sharing is never transitive.** Promotion strips `shareable` from the pushed
  copy, so the receiving tenant decides its own consent tags.
- **Promote (push) is review-first.** `POST /v1/memories/promote` defaults to
  `dry_run=true` (returns the candidate manifest, no side effects); a real push
  stamps `{promoted_from, source_memory_id, promoted_at}` provenance.
- **Federated search (pull)** fans a query to named pull-capable peers, tags each
  result `origin: local | <peer>`, merges by score, and degrades a failed peer to
  a `peer_errors` entry rather than a 5xx. A dedicated inbound
  `GNOSIS_FEDERATION_TOKEN` class can reach only three routes and cannot itself
  name peers, so federation cannot loop.

The bi-temporal + consent-tagged sharing design draws on Zep/Graphiti's
federation ideas ([2501.13956](https://arxiv.org/abs/2501.13956)); the
review-first, non-transitive consent model is gnosis's own.

---

## Policy & safety

- **Scope spine.** Every record is keyed by `tenant_id` / `space_id` / `agent_id`
  / `session_id` / `user_id` / `visibility`. Long-term recall is keyed by
  `tenant_id`+`user_id`; `agent_id` and caller metadata are write-side audit tags
  that do not partition recall and are redacted from prompt-facing content.
  Different business entities run their own deployment with their own tenant and
  storage.
- **Graph-QA is planned, validated, and executed under policy.** Callers ask
  natural-language questions; only gnosis plans the Cypher, validates it against a
  scope-checked schema guide (every alias bound by tenant+user), logs it, and
  executes it read-only. Clients never issue Cypher.
- **Redaction & review-first ops.** Export, dedup, and consolidation responses
  are redacted before leaving the gateway; dedup and consolidation are dry-run by
  default. Reasoning traces are stored for audit but hidden chain-of-thought is
  kept out of prompt recall and public memory.

---

## Configuration & measurement

- **Select a config with `GNOSIS_CONFIG_FILE`.** `configs/default.yaml` (the
  preferred, best-scoring config) auto-loads out of the box; every measured run's
  flag set is in `configs/runs/runN.yaml`. Precedence: env vars → `.env` → YAML
  config → code defaults. See [../configs/README.md](../configs/README.md).
- **The preferred config (Run 18)** is extraction + entity graph at write;
  adaptive routing + route-aware hardened Chain-of-Note at read. It needs a
  capable `GNOSIS_LLM` (extraction and routing make LLM calls).
- **All scores, per-category history, and honest deviations** live in
  [BENCHMARKS.md](BENCHMARKS.md).

---

## Research bibliography

| arXiv | Work | Applied to |
|---|---|---|
| [2311.09210](https://arxiv.org/abs/2311.09210) | Chain-of-Note | read-then-reason reading instruction |
| [2403.14403](https://arxiv.org/abs/2403.14403) | Adaptive-RAG | per-query routing (composition core) |
| [2511.17208](https://arxiv.org/abs/2511.17208) | EMem | atomic fact units; recall filter |
| [2604.22085](https://arxiv.org/abs/2604.22085) | Memanto | assistant-turn extraction gap (edu-v2.0 Rule 15) |
| [2502.14802](https://arxiv.org/abs/2502.14802) | HippoRAG 2 (ICML 2025) | entity-graph recall |
| [2405.14831](https://arxiv.org/abs/2405.14831) | HippoRAG / Graphiti approach | entity/bridge traversal |
| [2501.13956](https://arxiv.org/abs/2501.13956) | Zep / Graphiti | bi-temporal model; federation |
| [2607.26520](https://arxiv.org/abs/2607.26520) | Graph-Native Bitemporal Memory Store | write-time SUPERSEDES edges + valid_time/transaction_time model (L-31) |
| [2607.21962](https://arxiv.org/abs/2607.21962) | Ground Truth First | validity intervals + volatility-class taxonomy (singleton vs. additive) |
| [2608.04746](https://arxiv.org/abs/2608.04746) | Scrub Jay Episodic Memory | per-memory perishability score + temporal decay for retrieval |
| [2608.00009](https://arxiv.org/abs/2608.00009) | AgentMemBench | GEM + CBS for multi-session memory; benchmark coverage |
| [2607.16848](https://arxiv.org/abs/2607.16848) | Beyond Memory Leaderboards | retrieval budget comparability warning; leaderboard context |
| [2606.01435](https://arxiv.org/abs/2606.01435) | "Don't Ask the LLM to Track Freshness" | read-time supersession (over invalidation) |
| [2411.06037](https://arxiv.org/abs/2411.06037) | Sufficient Context | sufficiency signalling |
| [2506.09038](https://arxiv.org/abs/2506.09038) | AbstentionBench | abstention prompting |
| [2602.15313](https://arxiv.org/abs/2602.15313) | Mnemis | dual-route graph-QA fusion |
| [2210.03350](https://arxiv.org/abs/2210.03350) | Self-Ask | multi-hop decomposition |
| [2212.10509](https://arxiv.org/abs/2212.10509) | IRCoT | interleaved retrieve-and-reason |
| [2505.16067](https://arxiv.org/abs/2505.16067) | Selective memory addition | selective store-time add |
| [2504.19413](https://arxiv.org/abs/2504.19413) | mem0 | comparison baseline; extraction guardrails |
| [2402.17753](https://arxiv.org/abs/2402.17753) | LOCOMO | primary benchmark |
| [2410.10813](https://arxiv.org/abs/2410.10813) | LongMemEval | second benchmark |
| [2304.03442](https://arxiv.org/abs/2304.03442) | Generative Agents | reflection / importance (studied) |
| [2305.10250](https://arxiv.org/abs/2305.10250) | MemoryBank | decay / reinforcement (studied) |
| [2310.08560](https://arxiv.org/abs/2310.08560) | MemGPT | layered memory-as-tools (studied) |

"Studied" = evaluated and informing design direction, not yet a shipped flag.
