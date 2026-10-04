import asyncio
import hmac
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal, cast
from uuid import UUID, uuid4

from neo4j.exceptions import Neo4jError
from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.llm.adapters.litellm import LiteLLMEmbeddingProvider
from neo4j_agent_memory.memory.reasoning import ToolCallStatus
from neo4j_agent_memory.schema.models import EntityRef
from openai import OpenAIError
from pydantic import TypeAdapter, ValidationError

from gnosis.backend_protocols import (
    BackendCapabilityUnavailable,
    BackendRequestError,
    CommunityCapableBackend,
    ExtractionPreviewBackend,
    MemoryBackend,
    MemoryNotFoundError,
    RecallFilteringBackend,
)
from gnosis.bridge_traversal import (
    BRIDGE_MENTION_CYPHER,
    BridgeNamer,
    LiteLLMBridgeNamer,
    bridge_parameters,
    parse_bridge_names,
)
from gnosis.community_graph import (
    COMMUNITY_FACT_CAP as _COMMUNITY_FACT_CAP,
)
from gnosis.community_graph import (
    CREATE_COMMUNITY_INDEX_CYPHER as _CREATE_COMMUNITY_INDEX_CYPHER,
)
from gnosis.community_graph import (
    FETCH_COMMUNITY_SUMMARIES_CYPHER as _FETCH_COMMUNITY_SUMMARIES_CYPHER,
)
from gnosis.community_graph import (
    FETCH_ENTITIES_CYPHER as _FETCH_ENTITIES_CYPHER,
)
from gnosis.community_graph import (
    FETCH_ENTITY_FACTS_CYPHER as _FETCH_ENTITY_FACTS_CYPHER,
)
from gnosis.community_graph import (
    FETCH_RELATIONS_CYPHER as _FETCH_RELATIONS_CYPHER,
)
from gnosis.community_graph import (
    CommunityRecord,
    community_id_for,
    community_write_statements,
    summarize_community,
    weakly_connected_components,
)
from gnosis.context_assembly import (
    MEMORY_SEARCH_CANDIDATE_LIMIT as _MEMORY_SEARCH_CANDIDATE_LIMIT,
)
from gnosis.context_assembly import (
    append_context_section as _append_context_section,
)
from gnosis.context_assembly import (
    cut_with_graph_reserve as _cut_with_graph_reserve,
)
from gnosis.context_assembly import (
    dedupe_graph_context as _dedupe_graph_context,
)
from gnosis.context_assembly import (
    fact_context_line as _fact_context_line,
)
from gnosis.context_assembly import (
    fact_freshness as _fact_freshness,
)
from gnosis.context_assembly import (
    fact_from_memory as _fact_from_memory,
)
from gnosis.context_assembly import (
    fact_markers as _fact_markers,
)
from gnosis.context_assembly import (
    fact_matches_scope as _fact_matches_scope,
)
from gnosis.context_assembly import (
    fuse_graph_facts as _fuse_graph_facts,
)
from gnosis.context_assembly import (
    graph_facts_to_candidates as _graph_facts_to_candidates,
)
from gnosis.context_assembly import (
    legacy_context_request as _legacy_context_request,
)
from gnosis.context_assembly import (
    legacy_context_response as _legacy_context_response,
)
from gnosis.context_assembly import (
    log_supersession as _log_supersession,
)
from gnosis.context_assembly import (
    long_term_enrichment_enabled as _long_term_enrichment_enabled,
)
from gnosis.context_assembly import (
    memory_freshness as _memory_freshness,
)
from gnosis.context_assembly import (
    memory_record_line as _memory_record_line,
)
from gnosis.context_assembly import (
    metadata_fragments as _metadata_fragments,
)
from gnosis.context_assembly import (
    query_recent_facts as _query_recent_facts,
)
from gnosis.context_assembly import (
    stored_memory_line as _stored_memory_line,
)
from gnosis.context_assembly import (
    verbatim_expansion_targets as _verbatim_expansion_targets,
)
from gnosis.dedup_consolidation import (
    CONSOLIDATION_APPLY_REQUIRED_DETAIL as _CONSOLIDATION_APPLY_REQUIRED_DETAIL,
)
from gnosis.dedup_consolidation import (
    CONSOLIDATION_IDEMPOTENCY_DETAIL as _CONSOLIDATION_IDEMPOTENCY_DETAIL,
)
from gnosis.dedup_consolidation import (
    CONSOLIDATION_TOKEN_TTL as _CONSOLIDATION_TOKEN_TTL,
)
from gnosis.dedup_consolidation import (
    CONSOLIDATION_UNAVAILABLE_DETAIL as _CONSOLIDATION_UNAVAILABLE_DETAIL,
)
from gnosis.dedup_consolidation import (
    DEDUP_APPLY_REQUIRED_DETAIL as _DEDUP_APPLY_REQUIRED_DETAIL,
)
from gnosis.dedup_consolidation import (
    DEDUP_IDEMPOTENCY_DETAIL as _DEDUP_IDEMPOTENCY_DETAIL,
)
from gnosis.dedup_consolidation import (
    DEDUP_TOKEN_TTL as _DEDUP_TOKEN_TTL,
)
from gnosis.dedup_consolidation import (
    DEDUP_UNAVAILABLE_DETAIL as _DEDUP_UNAVAILABLE_DETAIL,
)
from gnosis.dedup_consolidation import (
    ConsolidationCapableMemoryClient,
    ConsolidationDryRunState,
    ConsolidationIdempotencyRecord,
    ConsolidationMemory,
    ConsolidationTokenClaims,
    DedupCandidateState,
    DedupCapableLongTermMemory,
    DedupIdempotencyRecord,
    DedupTokenClaims,
)
from gnosis.dedup_consolidation import (
    apply_dedup_operation as _apply_dedup_operation,
)
from gnosis.dedup_consolidation import (
    consolidation_apply_fingerprint as _consolidation_apply_fingerprint,
)
from gnosis.dedup_consolidation import (
    consolidation_request_fingerprint as _consolidation_request_fingerprint,
)
from gnosis.dedup_consolidation import (
    consolidation_token as _consolidation_token,
)
from gnosis.dedup_consolidation import (
    dedup_candidate as _dedup_candidate,
)
from gnosis.dedup_consolidation import (
    dedup_snapshot_hash as _dedup_snapshot_hash,
)
from gnosis.dedup_consolidation import (
    dedup_stats_payload as _dedup_stats_payload,
)
from gnosis.dedup_consolidation import (
    dedup_token as _dedup_token,
)
from gnosis.dedup_consolidation import (
    require_consolidation_token as _require_consolidation_token,
)
from gnosis.dedup_consolidation import (
    require_current_consolidation_dry_run as _require_current_consolidation_dry_run,
)
from gnosis.dedup_consolidation import (
    require_current_dedup_candidate as _require_current_dedup_candidate,
)
from gnosis.dedup_consolidation import (
    require_dedup_token as _require_dedup_token,
)
from gnosis.dedup_consolidation import (
    run_consolidation_operation as _run_consolidation_operation,
)
from gnosis.dedup_consolidation import (
    safe_consolidation_report as _safe_consolidation_report,
)
from gnosis.entity_graph import (
    CREATE_ENTITY_SCOPE_INDEX_CYPHER,
    RelationTriple,
    entity_graph_statements,
    normalize_entity_name,
)
from gnosis.entity_traversal import (
    ENTITY_TRAVERSAL_CYPHER,
    query_seed_candidates,
    traversal_parameters,
)
from gnosis.event_facts import EventFactPromoter
from gnosis.extraction_worker import BackgroundExtractionQueue
from gnosis.fact_extraction import (
    EXTRACTION_VERSION,
    ConversationTurn,
    LiteLLMMemoryUnitExtractor,
    MemoryUnit,
    MemoryUnitExtractor,
    extract_memory_units,
    unit_relations,
)
from gnosis.graph_probe import StructuredGraphStore, direct_neo4j_driver_factory
from gnosis.graph_query_qa import LiteLLMGraphQueryPlanner
from gnosis.graph_store import DirectNeo4jGraphStore, Neo4jGraphExecutor
from gnosis.ingestion_policy import (
    ExtractionPolicy,
)
from gnosis.ingestion_policy import (
    conversation_date as _conversation_date,
)
from gnosis.ingestion_policy import (
    extraction_policy as _extraction_policy,
)
from gnosis.ingestion_policy import (
    fact_extraction_model as _fact_extraction_model,
)
from gnosis.ingestion_policy import (
    message_extraction_policy as _message_extraction_policy,
)
from gnosis.ingestion_policy import (
    preview_candidates as _preview_candidates,
)
from gnosis.ingestion_policy import (
    preview_document_count as _preview_document_count,
)
from gnosis.ingestion_policy import (
    preview_extraction_policy as _preview_extraction_policy,
)
from gnosis.ingestion_policy import (
    preview_source_ids as _preview_source_ids,
)
from gnosis.ingestion_policy import (
    require_ingestion_sources_allowed as _require_ingestion_sources_allowed,
)
from gnosis.ingestion_policy import (
    require_memory_add_mode as _require_memory_add_mode,
)
from gnosis.ingestion_policy import (
    require_preview_enabled as _require_preview_enabled,
)
from gnosis.ingestion_policy import (
    require_preview_sources_allowed as _require_preview_sources_allowed,
)
from gnosis.ingestion_policy import (
    routing_model as _routing_model,
)
from gnosis.ingestion_policy import (
    sufficiency_model as _sufficiency_model,
)
from gnosis.json_redaction import (
    hash_json as _hash_json,
)
from gnosis.json_redaction import (
    json_object as _json_object,
)
from gnosis.json_redaction import (
    redacted_object as _redacted_object,
)
from gnosis.json_redaction import (
    redacted_optional_text as _redacted_optional_text,
)
from gnosis.json_redaction import (
    redacted_text as _redacted_text,
)
from gnosis.memory_filters import (
    FilterValidationError,
    MemoryFilter,
    build_cypher_filter,
    matches_filters,
    parse_filters,
)
from gnosis.memory_provider import (
    CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER,
    CREATE_MEMORY_CYPHER,
    DELETE_MEMORY_CYPHER,
    EXTRACTED_FACT_PREDICATE,
    LEXICAL_MEMORY_SEARCH_CYPHER,
    LOOKUP_LATEST_MEMORY_CYPHER,
    LOOKUP_MEMORIES_BY_IDS_CYPHER,
    LOOKUP_MEMORY_CYPHER,
    RECENT_TURN_MEMORIES_CYPHER,
    SCOPED_DENSE_MEMORY_SEARCH_CYPHER,
    TURN_MEMORY_PREDICATE_PREFIX,
    UPDATE_MEMORY_CYPHER,
    VERBATIM_MEMORY_PREDICATE,
    WRITE_TIME_SUPERSEDE_CYPHER,
    StoredMemory,
    fuse_memory_rankings,
    lexical_stored_memory,
    list_memories_cypher,
    memory_add_event,
    memory_filter_fields,
    memory_matches_scope,
    memory_record,
    memory_score,
    merged_memory_metadata,
    public_memory_metadata,
    sanitize_lucene_query,
    scope_read_fragments,
    scored_stored_memory_from_row,
    session_read_fragments,
    stored_memories_from_sdk,
    stored_memory_from_row,
    stored_memory_from_sdk,
)
from gnosis.models import (
    BackendReadiness,
    BufferFlushResponse,
    BufferStatus,
    ClientEvent,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    CommunityRebuildRequest,
    CommunityRebuildResponse,
    ConsolidationApplyRequest,
    ConsolidationApplyResponse,
    ConsolidationDryRunRequest,
    ConsolidationDryRunResponse,
    ContextRequest,
    ContextResponse,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidateRequest,
    DedupCandidateResponse,
    DedupStatsRequest,
    DedupStatsResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    EventIngestResult,
    ExtractionPreviewMetrics,
    ExtractionPreviewProvenance,
    ExtractionPreviewRequest,
    ExtractionPreviewResponse,
    FactRecord,
    FactSearchRequest,
    FactSearchResponse,
    FactWriteRequest,
    GraphContextRequest,
    GraphContextResponse,
    GraphExportRequest,
    GraphExportResponse,
    JsonObject,
    JsonValue,
    MemoryAddEvent,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryAddResult,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MessageWriteRequest,
    MessageWriteResponse,
    PreferenceRecord,
    PreferenceSearchRequest,
    PreferenceSearchResponse,
    PreferenceWriteRequest,
    ReasoningContextRequest,
    ReasoningContextResponse,
    ReasoningSimilarTracesRequest,
    ReasoningSimilarTracesResponse,
    ReasoningStepRequest,
    ReasoningStepResponse,
    ReasoningStepSearchRequest,
    ReasoningStepSearchResponse,
    ReasoningToolCallRequest,
    ReasoningToolCallResponse,
    ReasoningToolStatsRequest,
    ReasoningToolStatsResponse,
    ReasoningTraceCompleteRequest,
    ReasoningTraceCompleteResponse,
    ReasoningTraceDetailRequest,
    ReasoningTraceDetailResponse,
    ReasoningTraceListRequest,
    ReasoningTraceListResponse,
    ReasoningTraceStartRequest,
    ReasoningTraceStartResponse,
    SdkStatsRequest,
    SdkStatsResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
    SufficiencyAssessment,
)
from gnosis.query_rewrite import LiteLLMQueryRewriter
from gnosis.query_router import (
    LiteLLMQueryRouter,
    QueryRoute,
    QueryRouter,
    RouteDecision,
)
from gnosis.reasoning_support import (
    REASONING_READ_UNAVAILABLE_DETAIL as _REASONING_READ_UNAVAILABLE_DETAIL,
)
from gnosis.reasoning_support import (
    get_reasoning_trace as _get_reasoning_trace,
)
from gnosis.reasoning_support import (
    reasoning_step_matches_scope as _reasoning_step_matches_scope,
)
from gnosis.reasoning_support import (
    reasoning_step_record as _reasoning_step_record,
)
from gnosis.reasoning_support import (
    reasoning_trace_matches_scope as _reasoning_trace_matches_scope,
)
from gnosis.reasoning_support import (
    reasoning_trace_summary as _reasoning_trace_summary,
)
from gnosis.reasoning_support import (
    safe_reasoning_context as _safe_reasoning_context,
)
from gnosis.reasoning_support import (
    scoped_reasoning_traces as _scoped_reasoning_traces,
)
from gnosis.recall_filter import (
    LiteLLMRecallFilter,
    RecallFilter,
    keep_relevant_candidates,
)
from gnosis.redaction import redact_secrets
from gnosis.reranker import (
    MIN_RERANK_CANDIDATES,
    LiteLLMReranker,
    Reranker,
    apply_rerank,
    rerank_candidate_cap,
    rerank_model,
)
from gnosis.scope_policy import (
    memory_edit_audit as _memory_edit_audit,
)
from gnosis.scope_policy import (
    reasoning_write_metadata as _reasoning_write_metadata,
)
from gnosis.scope_policy import (
    record_matches_filters as _record_matches_filters,
)
from gnosis.scope_policy import (
    redacted_entity as _redacted_entity,
)
from gnosis.scope_policy import (
    redacted_fact as _redacted_fact,
)
from gnosis.scope_policy import (
    redacted_preference as _redacted_preference,
)
from gnosis.scope_policy import (
    scope_json_metadata as _scope_json_metadata,
)
from gnosis.scope_policy import (
    scope_metadata as _scope_metadata,
)
from gnosis.scope_policy import (
    scoped_filters as _scoped_filters,
)
from gnosis.scope_policy import (
    session_id as _session_id,
)
from gnosis.scope_policy import (
    user_identifier as _user_identifier,
)
from gnosis.scope_policy import (
    write_metadata as _write_metadata,
)
from gnosis.sdk_client import (
    BufferErrorCapableMemoryClient,
    BufferFlushCapableMemoryClient,
    BufferPendingCapableMemoryClient,
    CypherQuery,
    GraphCapableMemoryClient,
    GraphNodeLike,
    GraphRelationshipLike,
    GraphWriteQuery,
    LongTermFactMemory,
    LongTermMemory,
    MemoryClientContext,
    MemoryClientFactory,
    MemoryConfigKwargs,
    MemoryGraphLike,
    ReasoningMemory,
    ShortTermMemory,
    StatsCapableMemoryClient,
    TextEmbedder,
    litellm_embedding_model,
)
from gnosis.sdk_client import (
    build_memory_settings as _build_memory_settings,
)
from gnosis.sdk_client import (
    graph_export_response as _graph_export_response,
)
from gnosis.sdk_client import (
    graph_write_query as _graph_write_query,
)
from gnosis.sdk_client import (
    memory_client_context as _memory_client_context,
)
from gnosis.sdk_client import (
    memory_embedding as _memory_embedding,
)
from gnosis.settings import Settings
from gnosis.skill_registry import InMemorySkillRegistry, SkillRegistry
from gnosis.sufficiency import (
    LiteLLMSufficiencyAssessor,
    SufficiencyAssessor,
    bounded_reason,
)
from gnosis.supersession import drop_superseded, is_singleton_relation_class

if TYPE_CHECKING:
    from gnosis.graph_types import CypherParameters

__all__ = [
    "BackendCapabilityUnavailable",
    "BackendRequestError",
    "BufferErrorCapableMemoryClient",
    "BufferFlushCapableMemoryClient",
    "BufferPendingCapableMemoryClient",
    "CommunityCapableBackend",
    "ConsolidationCapableMemoryClient",
    "ConsolidationDryRunState",
    "ConsolidationIdempotencyRecord",
    "ConsolidationMemory",
    "ConsolidationTokenClaims",
    "CypherQuery",
    "DedupCandidateState",
    "DedupCapableLongTermMemory",
    "DedupIdempotencyRecord",
    "DedupTokenClaims",
    "ExtractionPolicy",
    "ExtractionPreviewBackend",
    "GraphCapableMemoryClient",
    "GraphNodeLike",
    "GraphRelationshipLike",
    "GraphWriteQuery",
    "LongTermFactMemory",
    "LongTermFactsContext",
    "LongTermMemory",
    "MemoryBackend",
    "MemoryClientContext",
    "MemoryClientFactory",
    "MemoryConfigKwargs",
    "MemoryGraphLike",
    "MemoryNotFoundError",
    "Neo4jAgentMemoryBackend",
    "ReasoningMemory",
    "RecallFilteringBackend",
    "ShortTermMemory",
    "StatsCapableMemoryClient",
    "TextEmbedder",
    "build_direct_graph_store",
    "litellm_embedding_model",
]

_JSON_OBJECT_ADAPTER: Final[TypeAdapter[JsonObject]] = TypeAdapter(JsonObject)
_ENTITY_RECORD_ADAPTER: Final[TypeAdapter[EntityRecord]] = TypeAdapter(EntityRecord)
_ENTITY_RECORDS_ADAPTER: Final[TypeAdapter[list[EntityRecord]]] = TypeAdapter(
    list[EntityRecord],
)
_FACT_RECORD_ADAPTER: Final[TypeAdapter[FactRecord]] = TypeAdapter(FactRecord)
_FACT_RECORDS_ADAPTER: Final[TypeAdapter[list[FactRecord]]] = TypeAdapter(
    list[FactRecord],
)
_PREFERENCE_RECORD_ADAPTER: Final[TypeAdapter[PreferenceRecord]] = TypeAdapter(
    PreferenceRecord,
)
_PREFERENCE_RECORDS_ADAPTER: Final[TypeAdapter[list[PreferenceRecord]]] = TypeAdapter(
    list[PreferenceRecord]
)
_SDK_STATS_UNAVAILABLE_DETAIL: Final[str] = "SDK stats are unavailable."
_SDK_BUFFER_FLUSH_UNAVAILABLE_DETAIL: Final[str] = "SDK buffer flush is unavailable."
_SDK_BUFFER_WAIT_UNAVAILABLE_DETAIL: Final[str] = "SDK buffer wait is unavailable."
# Bounded shutdown drain for background extraction: enough for in-flight
# extraction LLM calls to finish, short enough not to stall pod rotation.
_EXTRACTION_DRAIN_TIMEOUT_SECONDS: Final[float] = 10.0
_NO_EXCLUDED_MEMORY_IDS: Final[frozenset[str]] = frozenset()
_SDK_GRAPH_UNAVAILABLE_DETAIL: Final[str] = "SDK graph export is unavailable."
_QUERY_EMBEDDER_UNAVAILABLE_DETAIL: Final[str] = "SDK query embedder is unavailable."
_MEMORY_UPDATE_FIELDS_DETAIL: Final[str] = "Memory updates require content or metadata."
_MEMORY_ID_UNAVAILABLE_DETAIL: Final[str] = "SDK did not expose a stable memory id."
_MEMORY_LIST_SCAN_LIMIT: Final[int] = 2000
_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_ABSTENTION_INSTRUCTION: Final[str] = (
    "Answer only from the memories below; if they do not contain the answer, "
    "say you don't know."
)
# Chain-of-Note read-then-reason (arXiv 2311.09210; LongMemEval arXiv
# 2410.10813 measures up to +10 QA points from structured reading on
# conversational memory). The note step lets the reader reject retrieved
# similar-but-wrong memories instead of answering from them, and raises
# rejection of unanswerable questions - reading is the lever, not retrieval.
# The attribution and yes/no clauses target the two residual adversarial
# failure patterns measured in Run 15 (2026-07-04, 24 misses): answering
# with a *different person's* similar fact, and answering yes/no about a
# thing the memories never mention. The likelihood clause carves the one
# measured over-abstention cost back out (Run 17: open-domain "would X
# likely..." questions expect an inference from known traits, and the bare
# never-guess rule refused them). The instruction is assembled from the
# base plus one inference clause so the optional widenings below stay
# byte-identical to the historic instruction while their flags are off.
_CHAIN_OF_NOTE_BASE: Final[str] = (
    "Before answering, silently take notes on each memory below: state "
    "whether it is relevant to the question, what it says, who it is about, "
    "and whether it contradicts another memory. Ignore memories that are "
    "merely similar to the question but do not answer it, especially "
    "memories about a different person than the question asks about. Then "
    "answer using only the relevant memories; if no memory states the "
    "answer, say you don't know - never guess, and never answer yes or no "
    "about something the memories never mention."
)
_CON_LIKELIHOOD_CLAUSE: Final[str] = (
    " Only when the question itself asks what is likely or probable, infer "
    "the most plausible answer from the relevant memories instead of saying "
    "you don't know."
)
# Widened inference carve-out (GNOSIS_CON_SPECULATIVE_INFERENCE_ENABLED).
# LOCOMO Run 18 open-domain miss analysis (2026-07-04): 8 of 12 misses were
# abstentions on speculative judgment questions phrased WITHOUT the word
# "likely" ("Would Caroline pursue writing?", "Does John live close to a
# beach or the mountains?") whose gold answers are themselves hedged
# inferences. The likelihood clause never fires on them. This clause widens
# the carve-out to speculative judgments about a person while explicitly
# keeping the refusal of specific unstated facts, which is what the
# adversarial category probes.
_CON_SPECULATIVE_CLAUSE: Final[str] = (
    " When the question asks what is likely or probable, or asks for a "
    "speculative judgment about a person - what they would enjoy, prefer, "
    "or do, what they might be or believe, whether they would be considered "
    "something - infer the most plausible answer from what the relevant "
    "memories show about them instead of saying you don't know. Still never "
    "invent specific facts (names, places, events, dates, numbers) the "
    "memories never state."
)
# Exhaustive-enumeration reading clause (GNOSIS_CON_ENUMERATION_ENABLED),
# applied only on multi-hop/aggregative routed reads. LOCOMO Run 19 proved
# the enumeration misses are a READER problem, not retrieval: doubling the
# item budget raised gold-item coverage 50%->60% and repaired 0 of 27 -
# even fully-covered questions answered with one salient item ("peach
# cobbler" when both gold desserts were in context). Run 19 changed the
# budget; this changes the reading behavior itself.
_CON_ENUMERATION_CLAUSE: Final[str] = (
    " When the question asks which items, what things, or otherwise asks "
    "for a list, enumerate every distinct real-world item or event the "
    "relevant memories support, not only the most prominent one. When "
    "the question asks how many, count unique items or events — if the "
    "same item, event, or person appears in multiple memories, count it "
    "only once. Do not count the number of memory records; count the "
    "number of distinct things those records describe."
)
_ENUMERATION_CLAUSE_ROUTES: Final[frozenset[str]] = frozenset(
    {"multi_hop", "aggregative"},
)
# Temporal conflict resolution clause (GNOSIS_CON_RECENCY_PREFERENCE_ENABLED).
# LME_S L-11 root cause: when the same fact appears across multiple sessions
# with different values (e.g. 3 → 4 → 5 Korean restaurants across sessions),
# the CoN model sometimes picks the wrong one. The base instruction notes
# contradictions but gives no resolution rule; adding "prefer most recent date"
# directly fixes the observed failure mode where the model chose "higher and
# more definitive" (an earlier session's value) over the most recent one.
# L-12 lesson: the clause must (a) override the never-guess rule explicitly
# ("state that value directly") and (b) be skipped on temporal-routed reads
# where "most recent fact" ≠ "the specific past event the question asks about".
# Also adds an anti-extrapolation rule to prevent forward projection from rates
# (LME_S 1cea1afa: model calculated 600 + 10/week x 17d ~= 624 from two
# complementary facts instead of reporting the last known count of 600).
# L-13 lesson: "the same fact" was too loose — the model conflated related-but-
# different facts (baseball ↔ football, guitar ↔ violin) and fired the clause
# when it shouldn't. Anchoring to "the same specific fact the question is asking
# about" prevents this. Also added "unless the question asks about a past or
# initial state" to stop the clause from overriding correctly-retrieved initial
# values when the question uses words like "initially", "originally", "at first".
# L-14 lesson: "same specific fact the question is asking about" was STILL not
# tight enough — GPT-4o treated "guitar practice time" as the same specific fact
# as "violin practice time" because both are "daily practice time." The fix is to
# restructure the clause so it fires ONLY on memories already identified as
# relevant. In L-9/L-12 (without strong clause), GPT-4o correctly filtered guitar
# memories as non-relevant to a violin question; the "state that value directly"
# override then bypassed that correct judgment. By leading with "among the
# memories you've identified as relevant," the model applies its own relevance
# filter first, and the recency rule is confined to that already-filtered set.
_CON_RECENCY_CLAUSE: Final[str] = (
    " Among the memories you have identified as relevant above, if two or more "
    "give different values for the same fact, the most recently-dated value is "
    "the correct current state — state it directly as your answer, unless the "
    "question asks about a past or initial state. Never calculate or project a "
    "current value from rates, trends, or growth patterns stated in the memories."
)
# The recency clause must not fire on temporal-routed reads: temporal questions
# ask about a SPECIFIC past event ("what happened X days ago?"), not the current
# state, so preferring the most recent memory is wrong there.
_RECENCY_CLAUSE_EXCLUDED_ROUTES: Final[frozenset[str]] = frozenset({"temporal"})
# Number of most-recently-created facts injected for knowledge_update route
# (L-29). Kept small so only the latest ingest is guaranteed a slot before
# the budget cut, without flooding the context with unrelated recent facts.
_RECENCY_INJECTION_LIMIT: Final[int] = 5
# Absence-implies-unknown clause (GNOSIS_CON_ABSTENTION_ENABLED).
# LME_S L-15 stable-wrong analysis identified two recurring absence-of-evidence
# errors on abstention-category questions:
# (1) "Zero instead of not-enough-info": model correctly notes no memory
#     mentions an activity but then concludes the count is zero (88432d0a_abs:
#     "no egg tart baking mentioned → zero times"; 0ddfec37_abs: "no football
#     records → zero footballs"). The CoN base already says "say you don't know"
#     but models bypass it by INFERRING zero from absence rather than FINDING zero
#     in a memory. This clause makes the prohibition concrete.
# (2) "Partial comparison": model finds data about one party in a who-did-X-first
#     or who-did-X-more question, but not the other, and answers "Party A first"
#     instead of abstaining (gpt4_fe651585_abs: Alex's parenting date known,
#     Tom's unknown → model says "Alex first"). Gold expects "not enough info."
# No route exclusion: temporal questions ask about confirmed events (activity IS
# mentioned); the clause fires only when no memory mentions the activity at all,
# which is not the temporal-question pattern.
_CON_ABSTENTION_CLAUSE: Final[str] = (
    " Do not infer that the count of an activity is zero simply because no memory "
    "mentions it — the user may not have shared those details; say instead that "
    "you do not have enough information. If the question asks who among two or more "
    "people did something first, most, or at a specific time, and you only have "
    "relevant information about some of them and not others, say you do not have "
    "enough information to make that comparison."
)

# Recommendation-response clause (GNOSIS_CON_RECOMMENDATION_ENABLED).
# LME_S L-17 stable-wrong analysis: SSP questions ("Can you recommend cultural
# events?", "Any suggestions for Denver?") fail because the model describes the
# user's preferences in the third person ("The user would prefer responses that
# suggest...") instead of making concrete first/second-person recommendations.
# Root cause: the model treats these as meta-questions about response preferences
# rather than as requests to actually recommend based on recalled preferences.
# The hypothesis evidence confirms the model finds the right memories (assistant
# previously suggested language festivals / Denver attractions) but then pivots
# to a preference-profile format. This clause redirects the final answer shape.
_CON_RECOMMENDATION_CLAUSE: Final[str] = (
    " When the question asks you to recommend, suggest, or advise (e.g. 'Can you "
    "recommend...', 'Any suggestions for...', 'What should I...'), respond with "
    "concrete suggestions derived from the memories. Do not describe what the user "
    "would prefer in the third person; instead, make actual recommendations in "
    "first or second person (e.g. 'You might enjoy X' or 'I would suggest Y') "
    "based on what the memories reveal about their interests and past experiences."
)


@dataclass(frozen=True, slots=True)
class LongTermFactsContext:
    context: str = ""
    markers: set[str] = field(default_factory=set)


def build_direct_graph_store(settings: Settings) -> DirectNeo4jGraphStore:
    embedding_provider = LiteLLMEmbeddingProvider(
        litellm_embedding_model(settings.gnosis_embedding),
        dimensions=settings.gnosis_embedding_dimensions,
        api_base=settings.litellm_base_url,
        api_key=settings.litellm_api_key,
    )
    # GNOSIS_GRAPHQA_ENABLED gates the LLM Cypher planner (see settings): with
    # it off no planner is wired, so graph context only ever runs the fixed,
    # parameterised queries.
    graph_query_planner = (
        LiteLLMGraphQueryPlanner(
            model=settings.gnosis_llm,
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        if settings.gnosis_graphqa_enabled
        else None
    )
    return DirectNeo4jGraphStore(
        executor=Neo4jGraphExecutor(
            driver_factory=direct_neo4j_driver_factory(settings),
            embedding_dimensions=settings.gnosis_embedding_dimensions,
            embedding_provider=embedding_provider,
            graph_query_planner=graph_query_planner,
        ),
    )


class Neo4jAgentMemoryBackend:
    def __init__(  # noqa: PLR0913 - One injection seam per LLM collaborator.
        self,
        settings: Settings,
        memory_client_factory: MemoryClientFactory | None = None,
        graph_store: StructuredGraphStore | None = None,
        skill_registry: SkillRegistry | None = None,
        recall_filter: RecallFilter | None = None,
        fact_extractor: MemoryUnitExtractor | None = None,
        sufficiency_assessor: SufficiencyAssessor | None = None,
        reranker: Reranker | None = None,
        query_router: QueryRouter | None = None,
        bridge_namer: BridgeNamer | None = None,
    ) -> None:
        self._app_settings: Settings = settings
        self._settings: MemorySettings = _build_memory_settings(settings)
        self._memory_client_factory: MemoryClientFactory | None = memory_client_factory
        self._graph_store: StructuredGraphStore = (
            graph_store or build_direct_graph_store(settings)
        )
        self._recall_filter: RecallFilter = recall_filter or LiteLLMRecallFilter(
            model=settings.gnosis_llm,
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._fact_extractor: MemoryUnitExtractor = (
            fact_extractor
            or LiteLLMMemoryUnitExtractor(
                model=_fact_extraction_model(settings),
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
                # The entity graph needs (head, relation, tail) triples, so the
                # extractor emits them only when materialization is enabled;
                # off, the edu-v1 prompt and schema stay byte-identical.
                emit_relations=settings.gnosis_entity_graph_enabled,
            )
        )
        self._sufficiency_assessor: SufficiencyAssessor = (
            sufficiency_assessor
            or LiteLLMSufficiencyAssessor(
                model=_sufficiency_model(settings),
                base_url=settings.litellm_base_url,
                api_key=settings.litellm_api_key,
            )
        )
        self._reranker: Reranker = reranker or LiteLLMReranker(
            model=rerank_model(settings),
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._query_router: QueryRouter = query_router or LiteLLMQueryRouter(
            model=_routing_model(settings),
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._bridge_namer: BridgeNamer = bridge_namer or LiteLLMBridgeNamer(
            model=_routing_model(settings),
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        _rewrite_model = settings.gnosis_query_rewrite_model or settings.gnosis_llm
        self._query_rewriter: LiteLLMQueryRewriter = LiteLLMQueryRewriter(
            model=_rewrite_model,
            base_url=settings.litellm_base_url,
            api_key=settings.litellm_api_key,
        )
        self._skill_registry: SkillRegistry = skill_registry or InMemorySkillRegistry()
        self._extraction_queue: BackgroundExtractionQueue = BackgroundExtractionQueue(
            max_concurrency=settings.gnosis_fact_extraction_max_concurrency,
            max_pending=settings.gnosis_fact_extraction_max_pending,
        )
        self._event_fact_promoter: EventFactPromoter = EventFactPromoter()
        self._fulltext_index_ready: bool = False
        self._entity_graph_schema_ready: bool = False
        self._dedup_candidates: dict[str, DedupCandidateState] = {}
        self._dedup_idempotency: dict[str, DedupIdempotencyRecord] = {}
        self._consolidation_dry_runs: dict[str, ConsolidationDryRunState] = {}
        self._consolidation_idempotency: dict[
            str,
            ConsolidationIdempotencyRecord,
        ] = {}

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        metadata = _scope_metadata(request.scope)
        policy = _message_extraction_policy(request, self._app_settings)
        _require_ingestion_sources_allowed(request, self._app_settings)
        extraction_enabled = self._app_settings.gnosis_fact_extraction_enabled
        extract_inline = extraction_enabled and not self._background_extraction()
        source_memory_ids: list[str] = []
        async with self._memory_client() as client:
            context_turns: list[ConversationTurn] = []
            if extract_inline:
                context_turns = await self._recent_session_turns(
                    client,
                    request.scope,
                )
            _ = await client.short_term.add_message(
                session_id=_session_id(request.scope),
                role=request.role.value,
                content=request.content,
                user_identifier=_user_identifier(request.scope),
                metadata=metadata,
                extract_entities=policy.extract_entities,
                extract_relations=policy.extract_relations,
            )
            raw_record = await client.long_term.add_fact(
                subject=_user_identifier(request.scope),
                predicate=f"said_{request.role.value}",
                obj=request.content,
                metadata=_scope_json_metadata(request.scope),
                generate_embedding=True,
            )
            if extraction_enabled:
                memory = stored_memory_from_sdk(raw_record)
                source_memory_ids = [memory.memory_id] if memory is not None else []
            if extract_inline:
                _ = await self._extracted_unit_results(
                    client,
                    request.scope,
                    caller_metadata={},
                    context_turns=context_turns,
                    new_turns=[
                        ConversationTurn(
                            speaker=request.role.value,
                            content=request.content,
                        ),
                    ],
                    source_memory_ids=source_memory_ids,
                )
        if extraction_enabled and not extract_inline:
            self._enqueue_background_extraction(
                request.scope,
                caller_metadata={},
                new_turns=[
                    ConversationTurn(
                        speaker=request.role.value,
                        content=request.content,
                    ),
                ],
                source_memory_ids=source_memory_ids,
            )
        return MessageWriteResponse(accepted=True)

    async def preview_extraction(
        self,
        request: ExtractionPreviewRequest,
    ) -> ExtractionPreviewResponse:
        _require_preview_enabled(self._app_settings)
        policy = _preview_extraction_policy(request, self._app_settings)
        _require_preview_sources_allowed(request, self._app_settings)
        documents = _preview_document_count(request)
        candidates = _preview_candidates(request, self._app_settings)
        return ExtractionPreviewResponse(
            candidates=candidates,
            metrics=ExtractionPreviewMetrics(
                documents=documents,
                chunks=max(documents, 1),
                ocr_images=len(request.ocr_image_references),
                rustfs_objects=len(request.rustfs_source_references),
                batch_size=self._app_settings.gnosis_extraction_batch_size,
                max_concurrency=self._app_settings.gnosis_extraction_max_concurrency,
            ),
            provenance=ExtractionPreviewProvenance(
                source_ids=_preview_source_ids(request),
                rustfs_objects=request.rustfs_source_references,
            ),
            extract_entities=policy.extract_entities,
            extract_relations=policy.extract_relations,
        )

    async def readiness(self) -> BackendReadiness:
        graph_readiness = await self._graph_store.readiness()
        buffer_status = await self.buffer_status()
        return graph_readiness.model_copy(
            update={"buffer_status": buffer_status.status},
        )

    async def buffer_status(self) -> BufferStatus:
        write_errors = await self._buffer_write_error_count()
        return BufferStatus(
            write_mode=self._app_settings.gnosis_write_mode,
            max_pending=self._app_settings.gnosis_max_pending,
            pending_writes=None,
            write_errors=write_errors,
            status=_buffer_readiness_status(write_errors),
        )

    async def flush_buffer(self) -> BufferFlushResponse:
        async with self._memory_client() as client:
            if not isinstance(client, BufferFlushCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_BUFFER_FLUSH_UNAVAILABLE_DETAIL)
            _ = await client.flush()
        return BufferFlushResponse(flushed=True, status=await self.buffer_status())

    async def shutdown(self) -> None:
        await self._extraction_queue.drain(
            drain_window_seconds=_EXTRACTION_DRAIN_TIMEOUT_SECONDS,
        )
        if self._app_settings.gnosis_write_mode != "buffered":
            return
        async with self._memory_client() as client:
            if not isinstance(client, BufferPendingCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_BUFFER_WAIT_UNAVAILABLE_DETAIL)
            _ = await client.wait_for_pending()

    async def _buffer_write_error_count(self) -> int:
        async with self._memory_client() as client:
            if not isinstance(client, BufferErrorCapableMemoryClient):
                return 0
            return len(client.write_errors)

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        return DiagnosticsResponse(
            tenant_id=self._app_settings.gnosis_tenant_id,
            config=DiagnosticsConfig(
                neo4j_uri=self._app_settings.neo4j_uri,
                neo4j_username=self._app_settings.neo4j_username,
                litellm_base_url=self._app_settings.litellm_base_url,
                gnosis_llm=self._app_settings.gnosis_llm,
                gnosis_embedding=self._app_settings.gnosis_embedding,
                gnosis_embedding_dimensions=(
                    self._app_settings.gnosis_embedding_dimensions
                ),
                gnosis_audit_read=self._app_settings.gnosis_audit_read,
                gnosis_conversation_ttl_days=(
                    self._app_settings.gnosis_conversation_ttl_days
                ),
                gnosis_write_mode=self._app_settings.gnosis_write_mode,
                gnosis_max_pending=self._app_settings.gnosis_max_pending,
                gnosis_fact_deduplication_enabled=(
                    self._app_settings.gnosis_fact_deduplication_enabled
                ),
                gnosis_trace_embedding_enabled=(
                    self._app_settings.gnosis_trace_embedding_enabled
                ),
                gnosis_extract_entities_enabled=(
                    self._app_settings.gnosis_extract_entities_enabled
                ),
                gnosis_extract_relations_enabled=(
                    self._app_settings.gnosis_extract_relations_enabled
                ),
                gnosis_extraction_preview_enabled=(
                    self._app_settings.gnosis_extraction_preview_enabled
                ),
                gnosis_extraction_batch_size=(
                    self._app_settings.gnosis_extraction_batch_size
                ),
                gnosis_extraction_max_concurrency=(
                    self._app_settings.gnosis_extraction_max_concurrency
                ),
                gnosis_extraction_chunk_size=(
                    self._app_settings.gnosis_extraction_chunk_size
                ),
                gnosis_extraction_chunk_overlap=(
                    self._app_settings.gnosis_extraction_chunk_overlap
                ),
                gnosis_fact_extraction_enabled=(
                    self._app_settings.gnosis_fact_extraction_enabled
                ),
                gnosis_fact_extraction_model=(
                    self._app_settings.gnosis_fact_extraction_model
                ),
                gnosis_fact_extraction_context_turns=(
                    self._app_settings.gnosis_fact_extraction_context_turns
                ),
                gnosis_fact_extraction_mode=(
                    self._app_settings.gnosis_fact_extraction_mode
                ),
                gnosis_fact_extraction_max_concurrency=(
                    self._app_settings.gnosis_fact_extraction_max_concurrency
                ),
                gnosis_fact_extraction_max_pending=(
                    self._app_settings.gnosis_fact_extraction_max_pending
                ),
                gnosis_ocr_enabled=self._app_settings.gnosis_ocr_enabled,
                gnosis_ocr_model=self._app_settings.gnosis_ocr_model,
                gnosis_ocr_max_image_bytes=(
                    self._app_settings.gnosis_ocr_max_image_bytes
                ),
                gnosis_rustfs_enabled=self._app_settings.gnosis_rustfs_enabled,
                gnosis_rustfs_bucket=self._app_settings.gnosis_rustfs_bucket,
                gnosis_rustfs_prefix=self._app_settings.gnosis_rustfs_prefix,
                gnosis_rustfs_endpoint=self._app_settings.gnosis_rustfs_endpoint,
                gnosis_rustfs_retention_days=(
                    self._app_settings.gnosis_rustfs_retention_days
                ),
                gnosis_prompt_entities_enabled=(
                    self._app_settings.gnosis_prompt_entities_enabled
                ),
                gnosis_prompt_preferences_enabled=(
                    self._app_settings.gnosis_prompt_preferences_enabled
                ),
                gnosis_prompt_reasoning_enabled=(
                    self._app_settings.gnosis_prompt_reasoning_enabled
                ),
                gnosis_consolidation_schedule_enabled=(
                    self._app_settings.gnosis_consolidation_schedule_enabled
                ),
            ),
            backend=readiness,
            extraction_queue=self._extraction_queue.status(
                mode=self._app_settings.gnosis_fact_extraction_mode,
            ),
        )

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        combined = await self.get_memory_context(_legacy_context_request(request))
        return _legacy_context_response(combined)

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        sections: list[MemoryContextSection] = []
        long_term_facts = LongTermFactsContext()
        decision = await self._route_decision(request.query)
        async with self._memory_client() as client:
            if request.include_short_term:
                try:
                    short_term = await client.short_term.get_context(
                        request.query,
                        session_id=_session_id(request.scope),
                        max_messages=request.max_items,
                        metadata_filters=_scope_metadata(request.scope),
                    )
                except (ValueError, ValidationError) as error:
                    _LOGGER.warning("short-term memory context skipped: %s", error)
                else:
                    _append_context_section(sections, "short_term", short_term)

            if request.include_long_term:
                long_term_facts = await self._get_long_term_facts_context(
                    request,
                    client,
                    decision,
                )
                _append_context_section(
                    sections,
                    "long_term_facts",
                    long_term_facts.context,
                )

                if _long_term_enrichment_enabled(self._app_settings):
                    long_term = await client.long_term.get_context(
                        request.query,
                        max_items=request.max_items,
                    )
                    _append_context_section(
                        sections,
                        "long_term_preferences_entities",
                        _redacted_text(long_term),
                    )

            if (
                request.include_reasoning
                and self._app_settings.gnosis_prompt_reasoning_enabled
            ):
                reasoning = await client.reasoning.get_context(
                    request.query,
                    max_traces=request.max_items,
                )
                _append_context_section(
                    sections,
                    "reasoning",
                    _safe_reasoning_context(reasoning),
                )

        if request.include_graph:
            graph = await self.get_graph_context(
                GraphContextRequest(
                    scope=request.scope,
                    query=request.query,
                    limit=request.graph_limit,
                ),
            )
            graph = _dedupe_graph_context(graph, long_term_facts.markers)
            if graph.context:
                sections.append(
                    MemoryContextSection(
                        source="graph",
                        content=graph.context,
                        facts=graph.facts,
                    ),
                )

        if (
            request.include_long_term
            and decision.route == "aggregative"
            and self._app_settings.gnosis_community_graph_enabled
        ):
            community_text = await self._get_community_context(request.scope)
            _append_context_section(sections, "community", community_text)

        sufficiency = await self._assess_sufficiency(
            request.query, sections, decision=decision
        )
        sections = await self._rewrite_and_expand(request, sections, sufficiency)
        sections = self._with_abstention_instruction(sections, decision)
        return MemoryContextResponse(sections=sections, sufficiency=sufficiency)

    async def _route_decision(self, query: str) -> RouteDecision:
        """Resolve the effective read-path feature set for one query.

        The globally configured flags while GNOSIS_ADAPTIVE_ROUTING_ENABLED is
        off or the query is empty (byte-identical unrouted behavior). With the
        flag on, one cheap structured-output LLM call classifies the query and
        the route's measured-best feature set replaces the global toggles for
        this request; any classifier failure degrades to the global flags with
        a structured warning, so routing can never fail a read.
        """
        unrouted = RouteDecision.from_settings(self._app_settings)
        if not self._app_settings.gnosis_adaptive_routing_enabled or not query:
            return unrouted
        try:
            verdict = await self._query_router.classify(query)
        except (RuntimeError, OSError, OpenAIError, ValidationError) as error:
            _LOGGER.warning(
                "query routing failed; using globally configured flags",
                extra={"error_type": type(error).__name__},
            )
            return unrouted
        if verdict is None:
            _LOGGER.warning(
                "query router returned no verdict; using globally configured flags",
            )
            return unrouted
        return RouteDecision.for_route(verdict.route, self._app_settings)

    def _with_abstention_instruction(
        self,
        sections: list[MemoryContextSection],
        decision: RouteDecision,
    ) -> list[MemoryContextSection]:
        """Prepend the standing reading instruction as a leading section.

        Two prompt-only reading aids share this seam, both added as a leading
        section so existing section parsing stays intact and both no-ops
        (byte-identical output) when off or when no memory content was
        assembled:

        * the decision's ``chain_of_note`` - Chain-of-Note read-then-reason
          (arXiv 2311.09210): note each memory's relevance first, then answer
          from the relevant ones or abstain. Subsumes the abstention line, so
          it takes precedence when the effective decision also asks for the
          bare abstention instruction. Carried on the route decision because
          it is route-aware: stacked globally with routing it cost temporal
          8.9 points (Run 14 - the note step parrots hybrid's relative-dated
          raw turns), so routed requests skip it on the temporal route.
        * the decision's ``abstention_prompt`` - the bare grounding
          instruction (AbstentionBench, arXiv 2506.09038), globally via
          GNOSIS_ABSTENTION_PROMPT_ENABLED or routed for unanswerable-risk
          queries.
        """
        if not sections:
            return sections
        if decision.chain_of_note:
            content = self._chain_of_note_instruction(decision)
        elif decision.abstention_prompt:
            content = _ABSTENTION_INSTRUCTION
        else:
            return sections
        instruction = MemoryContextSection(
            source="instructions",
            content=content,
        )
        return [instruction, *sections]

    def _chain_of_note_instruction(self, decision: RouteDecision) -> str:
        """Assemble the route's Chain-of-Note instruction from its clauses.

        With both widening flags off this returns the exact Run 18 CoN v3
        instruction byte-for-byte. GNOSIS_CON_SPECULATIVE_INFERENCE_ENABLED
        swaps the likelihood carve-out for the wider speculative-judgment
        carve-out (LOCOMO open-domain misses are "Would X ...?" questions
        that never say "likely"). GNOSIS_CON_ENUMERATION_ENABLED appends the
        exhaustive-enumeration clause on multi-hop/aggregative routed reads
        only (Run 19: enumeration misses are a reader problem - full gold
        coverage in context still answered with one item).
        """
        inference_clause = (
            _CON_SPECULATIVE_CLAUSE
            if self._app_settings.gnosis_con_speculative_inference_enabled
            else _CON_LIKELIHOOD_CLAUSE
        )
        enumeration_clause = ""
        if (
            self._app_settings.gnosis_con_enumeration_enabled
            and decision.route in _ENUMERATION_CLAUSE_ROUTES
        ):
            enumeration_clause = _CON_ENUMERATION_CLAUSE
        recency_clause = (
            _CON_RECENCY_CLAUSE
            if self._app_settings.gnosis_con_recency_preference_enabled
            and decision.route not in _RECENCY_CLAUSE_EXCLUDED_ROUTES
            else ""
        )
        abstention_clause = (
            _CON_ABSTENTION_CLAUSE
            if self._app_settings.gnosis_con_abstention_enabled
            else ""
        )
        recommendation_clause = (
            _CON_RECOMMENDATION_CLAUSE
            if self._app_settings.gnosis_con_recommendation_enabled
            else ""
        )
        return (
            f"{_CHAIN_OF_NOTE_BASE}{inference_clause}{recency_clause}"
            f"{enumeration_clause}{abstention_clause}{recommendation_clause}"
        )

    async def _assess_sufficiency(
        self,
        query: str,
        sections: list[MemoryContextSection],
        *,
        decision: RouteDecision | None = None,
    ) -> SufficiencyAssessment | None:
        """Judge whether the assembled context can answer the query.

        One structured-output LLM call behind GNOSIS_SUFFICIENCY_CHECK_ENABLED;
        absent (``None``) while the flag is off or the query is empty. Any
        failure degrades to ``assessed=False`` so the check never blocks the
        context response.
        """
        enabled = (
            decision.sufficiency_check_enabled
            if decision is not None
            else self._app_settings.gnosis_sufficiency_check_enabled
        )
        if not enabled or not query:
            return None
        context = "\n\n".join(section.content for section in sections)
        try:
            verdict = await self._sufficiency_assessor.assess(query, context)
        except (RuntimeError, OSError, OpenAIError) as error:
            _LOGGER.warning(
                "sufficiency check failed; reporting not assessed",
                extra={"error_type": type(error).__name__},
            )
            return SufficiencyAssessment(assessed=False)
        if verdict is None:
            return SufficiencyAssessment(assessed=False)
        return SufficiencyAssessment(
            assessed=True,
            sufficient=verdict.sufficient,
            reason=bounded_reason(verdict.reason),
        )

    async def rebuild_communities(
        self,
        request: CommunityRebuildRequest,
    ) -> CommunityRebuildResponse:
        """Detect entity communities via BFS WCC and persist LLM summaries.

        Reads the full entity + relation graph for the scope, runs weakly-
        connected-components in Python (no Neo4j GDS required), skips
        components below the configured minimum size, summarizes each via one
        LLM call, and writes Community nodes + MEMBER_OF edges idempotently
        via MERGE so repeat calls are safe. Any per-community failure (LLM or
        write) is logged and skipped; the rebuild still returns a count of the
        communities that succeeded.
        """
        scope = request.scope
        tenant_id = scope.tenant_id
        user_id = scope.user_id
        scope_params: CypherParameters = {
            "tenant_id": tenant_id,
            "user_id": user_id,
        }

        async with self._memory_client() as client:
            entity_rows, relation_rows = await asyncio.gather(
                client.query.cypher(_FETCH_ENTITIES_CYPHER, scope_params),
                client.query.cypher(_FETCH_RELATIONS_CYPHER, scope_params),
            )

        entities: list[str] = []
        name_map: dict[str, str] = {}
        for row in entity_rows:
            normalized = row.get("normalized")
            if not isinstance(normalized, str):
                continue
            entities.append(normalized)
            name_val = row.get("name")
            name_map[normalized] = name_val if isinstance(name_val, str) else normalized
        edges: list[tuple[str, str]] = []
        for row in relation_rows:
            head = row.get("head_normalized")
            tail = row.get("tail_normalized")
            if isinstance(head, str) and isinstance(tail, str):
                edges.append((head, tail))

        components = weakly_connected_components(entities, edges)
        min_size = self._app_settings.gnosis_community_min_entities
        components = [c for c in components if len(c) >= min_size]

        _model = self._app_settings.gnosis_llm
        _base_url = self._app_settings.litellm_base_url
        _api_key = self._app_settings.litellm_api_key

        async def _build_one(members: list[str]) -> CommunityRecord | None:
            fact_params: CypherParameters = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "entity_normals": cast("list[JsonValue]", members),
                "limit": _COMMUNITY_FACT_CAP,
            }
            try:
                async with self._memory_client() as client:
                    fact_rows = await client.query.cypher(
                        _FETCH_ENTITY_FACTS_CYPHER, fact_params
                    )
            except (RuntimeError, OSError, Neo4jError) as err:
                _LOGGER.warning(
                    "community fact fetch failed; skipping community",
                    extra={"error_type": type(err).__name__},
                )
                return None
            fact_snippets: list[str] = []
            for fr in fact_rows:
                content = fr.get("content")
                if isinstance(content, str) and content:
                    fact_snippets.append(content)
            entity_names = [name_map.get(m, m) for m in members]
            summary = await summarize_community(
                entity_names=entity_names,
                fact_snippets=fact_snippets,
                model=_model,
                base_url=_base_url,
                api_key=_api_key,
            )
            if summary is None:
                return None
            return CommunityRecord(
                community_id=community_id_for(tenant_id, user_id, members),
                normalized_members=members,
                summary=summary,
                member_count=len(members),
            )

        records_raw = await asyncio.gather(*[_build_one(c) for c in components])
        valid_records: list[CommunityRecord] = [r for r in records_raw if r is not None]

        await self._persist_community_records(valid_records, tenant_id, user_id)
        return CommunityRebuildResponse(
            tenant_id=tenant_id,
            user_id=user_id,
            communities_built=len(valid_records),
            entities_processed=len(entities),
        )

    async def _persist_community_records(
        self,
        records: list[CommunityRecord],
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Write community MERGE + MEMBER_OF statements to Neo4j idempotently."""
        if not records:
            return
        try:
            async with self._memory_client() as client:
                graph_write = _graph_write_query(client)
                _ = await graph_write.execute_write(_CREATE_COMMUNITY_INDEX_CYPHER, {})
                for record in records:
                    for cypher, params in community_write_statements(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        community=record,
                    ):
                        _ = await graph_write.execute_write(cypher, params)
        except (RuntimeError, OSError, Neo4jError) as err:
            _LOGGER.warning(
                "community write failed; communities may be partially persisted",
                extra={"error_type": type(err).__name__},
            )

    async def _get_community_context(
        self,
        scope: MemoryScope,
    ) -> str:
        """Read persisted community summaries for the scope and format them.

        Returns an empty string when community graph is disabled, no
        communities exist yet, or the read fails - callers treat an empty
        string as a no-op section.
        """
        if not self._app_settings.gnosis_community_graph_enabled:
            return ""
        limit = self._app_settings.gnosis_community_context_limit
        params: CypherParameters = {
            "tenant_id": scope.tenant_id,
            "user_id": scope.user_id,
            "limit": limit,
        }
        try:
            async with self._memory_client() as client:
                rows = await client.query.cypher(
                    _FETCH_COMMUNITY_SUMMARIES_CYPHER, params
                )
        except (RuntimeError, OSError, Neo4jError) as err:
            _LOGGER.warning(
                "community context read failed; skipping community section",
                extra={"error_type": type(err).__name__},
            )
            return ""
        if not rows:
            return ""
        lines: list[str] = []
        for row in rows:
            summary = row.get("summary")
            if isinstance(summary, str) and summary:
                lines.append(f"- {summary}")
        if not lines:
            return ""
        return "Entity community summaries:\n" + "\n".join(lines)

    async def _rewrite_and_expand(
        self,
        request: MemoryContextRequest,
        sections: list[MemoryContextSection],
        sufficiency: SufficiencyAssessment | None,
    ) -> list[MemoryContextSection]:
        """Expand context via multi-query rewrite when the sufficiency check fails.

        On sufficiency failure, generates complementary retrieval queries via
        the configured rewrite model, runs dense retrieval for each, dedupes
        against the already-retrieved fact IDs, and appends novel facts as an
        additional ``query_rewrite`` section. A no-op (returns ``sections``
        unchanged) when:
        - GNOSIS_QUERY_REWRITE_ENABLED is off
        - the sufficiency check was not run (``None``) or assessed sufficient
        - the rewriter returns no alternative queries
        - all rewrite candidates were already retrieved
        """
        if not self._app_settings.gnosis_query_rewrite_enabled:
            return sections
        if sufficiency is None or sufficiency.sufficient:
            return sections
        if not request.query:
            return sections

        existing_context = "\n\n".join(s.content for s in sections)
        reason = sufficiency.reason if sufficiency.assessed else None

        rewrite_result = await self._query_rewriter.rewrite(
            original_query=request.query,
            retrieved_context=existing_context,
            insufficiency_reason=reason,
        )
        if rewrite_result is None or not rewrite_result.queries:
            return sections

        metadata = _scope_metadata(request.scope)
        decision = RouteDecision.from_settings(self._app_settings)

        rewrite_facts: list[JsonObject] = []
        async with self._memory_client() as client:
            for alt_query in rewrite_result.queries:
                try:
                    alt_facts = await self._query_ranked_facts(
                        client, alt_query, metadata, decision
                    )
                    rewrite_facts.extend(alt_facts)
                except (RuntimeError, OSError, Neo4jError) as err:
                    _LOGGER.warning(
                        "rewrite retrieval failed for alternative query",
                        extra={"error_type": type(err).__name__},
                    )

        # Collect IDs already in sections via fact objects - use content as
        # dedup key since section text is already serialized.
        seen_content: set[str] = {s.content for s in sections}
        novel_lines: list[str] = []
        for fact in rewrite_facts:
            line = _fact_context_line(fact)
            if line not in seen_content:
                novel_lines.append(line)
                seen_content.add(line)

        if not novel_lines:
            return sections

        rewrite_text = "\n".join(novel_lines)
        return [
            *sections,
            MemoryContextSection(source="query_rewrite", content=rewrite_text),
        ]

    async def _reranked_facts(
        self,
        query: str,
        facts: list[JsonObject],
        route: QueryRoute | None = None,
    ) -> list[JsonObject]:
        """Reorder fused candidates by query relevance before the budget cut.

        One structured-output LLM call behind GNOSIS_RERANK_ENABLED; a no-op
        while the flag is off, the query is empty, or there is nothing to
        reorder. Any failure degrades to the input order so reranking never
        drops a candidate or blocks a context read.

        Route-aware: temporal and unanswerable_risk routes skip reranking.
        Temporal BM25+dense ordering is already measured-best (Run L-1:
        reranker -15.8pp on temporal); unanswerable_risk relies on retrieval
        exhaustion for correct abstention, not relevance reordering.
        """
        if (
            not self._app_settings.gnosis_rerank_enabled
            or not query
            or len(facts) < MIN_RERANK_CANDIDATES
            or route in ("temporal", "unanswerable_risk")
        ):
            return facts
        cap = rerank_candidate_cap(self._app_settings)
        candidates = [_fact_context_line(fact) for fact in facts[:cap]]
        try:
            result = await self._reranker.rerank(query, candidates)
        except (RuntimeError, OSError, OpenAIError, ValidationError) as error:
            _LOGGER.warning(
                "rerank failed; keeping retrieval order",
                extra={"error_type": type(error).__name__},
            )
            return facts
        if result is None:
            return facts
        return apply_rerank(facts, result.order, cap)

    async def _get_long_term_facts_context(
        self,
        request: MemoryContextRequest,
        client: MemoryClientContext,
        decision: RouteDecision,
    ) -> "LongTermFactsContext":
        """Render scoped long-term facts with the same read reach as search.

        Reads narrow by the cross-session scope (long-term memory spans
        sessions, so session_id never prunes recall). When the request has a
        query, candidates come from the same embedding-similarity search that
        /v1/memories/search uses so relevance decides which facts fit the
        ``max_items`` budget; recency ordering is the fallback when similarity
        search has nothing to rank (no query or no embedder). With
        GNOSIS_RECALL_FILTER_ENABLED on, one LLM call then screens the top
        candidates against the query before the item budget applies. The
        ``decision`` carries the effective per-request read-path feature set
        (the global flags unrouted, or the route's measured-best set).
        """
        metadata = _scope_metadata(request.scope)
        facts, graph_facts, traversal_facts = await asyncio.gather(
            self._query_ranked_facts(client, request.query, metadata, decision),
            self._graphqa_fused_facts(request, decision),
            self._traversal_facts(client, request, metadata, decision),
        )
        if not facts:
            facts = await _query_recent_facts(client, metadata)
        # knowledge_update route: guarantee the most recently ingested facts
        # are in the candidate pool. Updates about a changing fact rank low on
        # embedding similarity (the update is phrased in a different context
        # than the question) — injecting by ingest recency ensures the newest
        # value is always a candidate before the budget cut. Deduplication by
        # fact id prevents double-counting facts already in the dense ranking.
        if decision.recency_injection_enabled:
            recent = await _query_recent_facts(client, metadata)
            existing_ids = {
                fact_id for f in facts if isinstance(fact_id := f.get("id"), str)
            }
            injected = [
                {**rf, "recency_injected": True}
                for rf in recent[:_RECENCY_INJECTION_LIMIT]
                if rf.get("id") not in existing_ids
            ]
            if injected:
                _LOGGER.info(
                    "knowledge_update recency injection",
                    extra={"injected": len(injected), "dense": len(facts)},
                )
                facts = [*facts, *injected]
        # The directed bridge hop reads hop-1's dense evidence, so it runs
        # after the parallel retrieval legs, not among them.
        bridge_facts = await self._bridge_traversal_facts(
            client,
            request,
            metadata,
            decision,
            facts,
        )
        facts = _fuse_graph_facts(
            facts,
            [*graph_facts, *traversal_facts, *bridge_facts],
        )
        facts = await self._recall_filtered_facts(request.query, facts)
        facts = self._superseded_facts(facts, decision=decision)
        facts = await self._reranked_facts(request.query, facts, route=decision.route)
        facts = _cut_with_graph_reserve(
            facts,
            request.max_items * decision.budget_multiplier,
        )
        if not facts:
            return LongTermFactsContext()
        expansion = await self._verbatim_expansion(client, facts, metadata, decision)
        lines = ["### Long-Term Facts"]
        for fact in facts:
            lines.append(_fact_context_line(fact))
            fact_id = fact.get("id")
            if isinstance(fact_id, str):
                lines.extend(
                    f"  quote: {_redacted_text(quote)}"
                    for quote in expansion.get(fact_id, ())
                )
        return LongTermFactsContext(
            context="\n".join(lines),
            markers=_fact_markers(facts),
        )

    async def _verbatim_expansion(
        self,
        client: MemoryClientContext,
        facts: list[JsonObject],
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
    ) -> dict[str, list[str]]:
        """Map each top extracted fact to its source verbatim turn text(s).

        Frontier technique (EverMemOS facts->episodes; True Memory verbatim):
        rank on the compact extracted fact for precision, then surface the
        linked raw turn(s) for nuance. A no-op empty mapping while the
        effective decision leaves expansion off (byte-identical output).
        Only the highest-ranked GNOSIS_FACT_VERBATIM_EXPANSION_MAX extracted
        facts expand; a verbatim turn already present in the ranked facts (or
        already attached to an earlier fact) is never double-rendered; the
        scoped batch lookup re-checks scope on every fetched row so a source
        id can never leak a cross-scope turn. Any lookup failure or empty
        source set degrades to the compact fact alone.
        """
        if not decision.verbatim_expansion:
            return {}
        targets = _verbatim_expansion_targets(
            facts,
            cap=self._app_settings.gnosis_fact_verbatim_expansion_max,
        )
        wanted_ids: set[str] = {
            source_id for _, source_ids in targets for source_id in source_ids
        }
        if not wanted_ids:
            return {}
        verbatim = await self._fetch_verbatim_turns(
            client,
            sorted(wanted_ids),
            scope_metadata,
        )
        if not verbatim:
            return {}
        rendered: set[str] = set()
        expansion: dict[str, list[str]] = {}
        for fact_id, source_ids in targets:
            quotes: list[str] = []
            for source_id in source_ids:
                if source_id not in verbatim or source_id in rendered:
                    continue
                rendered.add(source_id)
                quotes.append(verbatim[source_id])
            if quotes:
                expansion[fact_id] = quotes
        if expansion:
            _LOGGER.info(
                "facts-to-verbatim expansion attached source turns",
                extra={"expanded": len(expansion), "candidates": len(facts)},
            )
        return expansion

    async def _fetch_verbatim_turns(
        self,
        client: MemoryClientContext,
        memory_ids: list[str],
        scope_metadata: Mapping[str, JsonValue],
    ) -> dict[str, str]:
        """Batch-fetch source verbatim turns by id, re-checking scope.

        One parameterized Cypher lookup narrowed by the scope metadata
        fragments; every returned row is scope-re-checked with the same
        predicate the ranked read uses, so a cross-scope turn is dropped and
        never rendered. A lookup failure degrades to no expansion.
        """
        try:
            rows = await client.query.cypher(
                LOOKUP_MEMORIES_BY_IDS_CYPHER,
                {
                    "memory_ids": list(memory_ids),
                    "scope_fragments": _metadata_fragments(scope_metadata),
                },
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "verbatim expansion lookup failed; keeping compact facts",
                extra={"error_type": type(error).__name__},
            )
            return {}
        verbatim: dict[str, str] = {}
        for row in rows:
            memory = stored_memory_from_row(row)
            if memory is None:
                continue
            fact = _fact_from_memory(memory)
            if not _fact_matches_scope(fact, scope_metadata):
                continue
            verbatim[memory.memory_id] = memory.content
        return verbatim

    async def add_memories(self, request: MemoryAddRequest) -> MemoryAddResponse:
        _require_memory_add_mode(request)
        metadata = _write_metadata(request.scope, request.metadata, None)
        results: list[MemoryAddResult] = []
        async with self._memory_client() as client:
            if request.content is not None:
                results.append(
                    await self._add_memory_fact(
                        client,
                        request.scope,
                        predicate=VERBATIM_MEMORY_PREDICATE,
                        content=_redacted_text(request.content),
                        metadata=metadata,
                    ),
                )
            else:
                results.extend(
                    await self._add_turn_memories(client, request, metadata),
                )
        return MemoryAddResponse(results=results)

    async def _add_turn_memories(
        self,
        client: MemoryClientContext,
        request: MemoryAddRequest,
        metadata: JsonObject,
    ) -> list[MemoryAddResult]:
        policy = _extraction_policy(
            extract_entities=None,
            extract_relations=None,
            settings=self._app_settings,
        )
        extraction_enabled = self._app_settings.gnosis_fact_extraction_enabled
        extract_inline = extraction_enabled and not self._background_extraction()
        context_turns: list[ConversationTurn] = []
        if extract_inline:
            # Fetched before the verbatim writes so the just-added turns
            # cannot leak into their own extraction context.
            context_turns = await self._recent_session_turns(client, request.scope)
        results: list[MemoryAddResult] = []
        for message in request.messages:
            _ = await client.short_term.add_message(
                session_id=_session_id(request.scope),
                role=message.role,
                content=message.content,
                user_identifier=_user_identifier(request.scope),
                metadata=_scope_metadata(request.scope),
                extract_entities=policy.extract_entities,
                extract_relations=policy.extract_relations,
            )
            results.append(
                await self._add_memory_fact(
                    client,
                    request.scope,
                    predicate=f"{TURN_MEMORY_PREDICATE_PREFIX}{message.role}",
                    content=message.content,
                    metadata=metadata,
                ),
            )
        new_turns = [
            ConversationTurn(speaker=message.role, content=message.content)
            for message in request.messages
        ]
        if extract_inline:
            results.extend(
                await self._extracted_unit_results(
                    client,
                    request.scope,
                    caller_metadata=request.metadata,
                    context_turns=context_turns,
                    new_turns=new_turns,
                    source_memory_ids=[result.memory_id for result in results],
                ),
            )
        elif extraction_enabled:
            # Background mode: the response carries only the verbatim
            # results; extraction runs on the queue after the request.
            self._enqueue_background_extraction(
                request.scope,
                caller_metadata=request.metadata,
                new_turns=new_turns,
                source_memory_ids=[result.memory_id for result in results],
            )
        return results

    async def _add_memory_fact(
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        predicate: str,
        content: str,
        metadata: JsonObject,
    ) -> MemoryAddResult:
        raw_record = await client.long_term.add_fact(
            _user_identifier(scope),
            predicate,
            content,
            metadata=metadata,
            generate_embedding=True,
        )
        memory = stored_memory_from_sdk(raw_record)
        event: MemoryAddEvent = "ADD"
        if memory is not None:
            event = memory_add_event(memory)
        else:
            memory = await _lookup_latest_memory(
                client,
                scope,
                predicate=predicate,
                content=content,
            )
        if memory is None:
            raise BackendCapabilityUnavailable(_MEMORY_ID_UNAVAILABLE_DETAIL)
        return MemoryAddResult(
            memory_id=memory.memory_id,
            content=_redacted_text(memory.content),
            event=event,
            metadata=public_memory_metadata(memory),
        )

    async def _recent_session_turns(
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        exclude_memory_ids: frozenset[str] = _NO_EXCLUDED_MEMORY_IDS,
    ) -> list[ConversationTurn]:
        """Read the extraction context window: recent turns of this session.

        Returns the last ``GNOSIS_FACT_EXTRACTION_CONTEXT_TURNS`` verbatim
        ``said_*`` facts for the scope's session in chronological order. In
        sync mode the window is read before the verbatim writes; in
        background mode the turns being extracted are already stored when
        the job runs, so ``exclude_memory_ids`` (their fact ids) filters
        them out - a pair is never its own context. Any read failure
        degrades to an empty context window - extraction still runs, it
        just resolves references less well - because extraction may never
        fail an add.
        """
        limit = self._app_settings.gnosis_fact_extraction_context_turns
        if limit <= 0:
            return []
        try:
            rows = await client.query.cypher(
                RECENT_TURN_MEMORIES_CYPHER,
                {
                    "subject": _user_identifier(scope),
                    "predicate_prefix": TURN_MEMORY_PREDICATE_PREFIX,
                    "scope_fragments": session_read_fragments(scope),
                    # Over-fetch by the excluded rows so exclusion never
                    # shrinks the effective window.
                    "limit": limit + len(exclude_memory_ids),
                },
            )
        except (RuntimeError, OSError, Neo4jError) as error:
            _LOGGER.warning(
                "fact extraction context read failed; extracting without context",
                extra={"error_type": type(error).__name__},
            )
            return []
        recent = [
            memory
            for row in rows
            if (memory := stored_memory_from_row(row)) is not None
            and memory.memory_id not in exclude_memory_ids
            and memory_matches_scope(memory, scope)
        ]
        turns = [
            ConversationTurn(
                speaker=memory.predicate.removeprefix(TURN_MEMORY_PREDICATE_PREFIX),
                content=memory.content,
            )
            for memory in recent[:limit]
        ]
        turns.reverse()
        return turns

    def _background_extraction(self) -> bool:
        return self._app_settings.gnosis_fact_extraction_mode == "background"

    def _enqueue_background_extraction(
        self,
        scope: MemoryScope,
        *,
        caller_metadata: JsonObject,
        new_turns: list[ConversationTurn],
        source_memory_ids: list[str],
    ) -> None:
        """Queue extraction for turns whose verbatim facts already landed.

        The job opens its own memory client and re-reads the context window
        at processing time, excluding the just-written verbatim facts. On
        queue overflow the extraction is dropped with a structured warning -
        never backpressure for message ingestion.
        """

        async def job() -> None:
            async with self._memory_client() as client:
                context_turns = await self._recent_session_turns(
                    client,
                    scope,
                    exclude_memory_ids=frozenset(source_memory_ids),
                )
                _ = await self._extracted_unit_results(
                    client,
                    scope,
                    caller_metadata=caller_metadata,
                    context_turns=context_turns,
                    new_turns=new_turns,
                    source_memory_ids=source_memory_ids,
                )

        _ = self._extraction_queue.submit(
            job,
            source_memory_ids=source_memory_ids,
        )

    async def _extracted_unit_results(  # noqa: PLR0913 - One argument per prompt input.
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        caller_metadata: JsonObject,
        context_turns: list[ConversationTurn],
        new_turns: list[ConversationTurn],
        source_memory_ids: list[str],
    ) -> list[MemoryAddResult]:
        """Extract memory units from the new turns and write them as facts.

        Strictly additive: extraction and per-unit write failures log a
        structured warning and leave the verbatim add untouched.
        """
        conversation_date = _conversation_date(caller_metadata)
        units = await extract_memory_units(
            self._fact_extractor,
            conversation_date=conversation_date,
            context_turns=context_turns,
            new_turns=new_turns,
        )
        results: list[MemoryAddResult] = []
        for unit in units:
            try:
                results.append(
                    await self._add_extracted_fact(
                        client,
                        scope,
                        unit=unit,
                        caller_metadata=caller_metadata,
                        source_memory_ids=source_memory_ids,
                        conversation_date=conversation_date,
                    ),
                )
            except (
                RuntimeError,
                OSError,
                Neo4jError,
                OpenAIError,
                ValidationError,
                BackendCapabilityUnavailable,
            ) as error:
                _LOGGER.warning(
                    "extracted fact write failed; keeping verbatim add",
                    extra={"error_type": type(error).__name__},
                )
        return results

    async def _add_extracted_fact(  # noqa: PLR0913, C901 - One argument per fact field; write-time supersession branch.
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        *,
        unit: MemoryUnit,
        caller_metadata: JsonObject,
        source_memory_ids: list[str],
        conversation_date: str | None = None,
    ) -> MemoryAddResult:
        """Write one extracted unit as an ordinary long-term ``Fact`` node.

        The write is a direct parameterized CREATE so it bypasses the SDK's
        write-time dedup, which can silently swallow a near-duplicate add
        into an existing fact - unacceptable for distinct dated events.
        """
        extraction_metadata: JsonObject = {
            "extracted": True,
            "extraction_version": EXTRACTION_VERSION,
            "extraction_model": _fact_extraction_model(self._app_settings),
            "entities": list(unit.entities),
            "source_turn_ids": list(unit.source_turn_ids),
        }
        if unit.event_date is not None:
            extraction_metadata["event_date"] = unit.event_date
        # Store the conversation date as the observation anchor so temporal
        # queries see when this fact was true, not when gnosis ingested it.
        # event_date wins (more specific); this is the fallback for ongoing-state
        # facts like "has been doing X for N months" where no specific event date
        # exists. Without this, _fact_date() falls through to created_at (today),
        # which is wrong and causes temporal hallucinations in the answer model.
        if conversation_date:
            extraction_metadata["date"] = conversation_date
        if unit.temporal_state not in ("unknown", None):
            extraction_metadata["temporal_state"] = unit.temporal_state
        if unit.supersedes_hint is not None:
            extraction_metadata["supersedes_hint"] = unit.supersedes_hint
        # Compute relation-class slots for precise supersession. Only singleton
        # relations (works_at, lives_in, married_to, …) occupy a named slot so
        # that a newer "Alice works at NVIDIA" displaces "Alice works at Google"
        # without touching unrelated facts. Additive relations (likes, prefers,
        # has_hobby) are intentionally excluded: they can have multiple concurrent
        # values and a shared slot would cause silent data loss.
        _rslots: list[JsonValue] = []
        if unit.temporal_state in ("starts", "ongoing", "ends"):
            for _rel in unit_relations(unit):
                _head = _rel.head.strip().casefold()
                _cls = "_".join(
                    p for p in _rel.relation.strip().casefold().split() if p
                )
                if _head and _cls and is_singleton_relation_class(_cls):
                    _rslots.append(f"{_head}:{_cls}")
            if _rslots:
                extraction_metadata["relation_slots"] = _rslots
        metadata = _write_metadata(scope, caller_metadata | extraction_metadata, None)
        # Provenance ids are gateway-generated fact UUIDs, added after
        # redaction because the opaque-value secret pattern matches UUIDs.
        metadata["source_memory_ids"] = list(source_memory_ids)
        memory_id = str(uuid4())
        content = _redacted_text(unit.text)
        embedding = await _memory_embedding(client, content)
        _ = await _graph_write_query(client).execute_write(
            CREATE_MEMORY_CYPHER,
            {
                "memory_id": memory_id,
                "subject": _user_identifier(scope),
                "predicate": EXTRACTED_FACT_PREDICATE,
                "object": content,
                "embedding": embedding,
                "tenant_id": scope.tenant_id,
                "user_id": scope.user_id,
                "metadata": json.dumps(metadata),
            },
        )
        await self._materialize_entity_graph(client, scope, memory_id, unit)
        # Write-time supersession: mark same-slot older facts as superseded and
        # link them with a SUPERSEDES edge from this new fact. This structural
        # fix ensures that when the knowledge_update route filters by
        # valid_to IS NULL, stale facts are excluded from the vector search
        # instead of outranking the fresh value in embedding space (L-31).
        if _rslots:
            try:
                _ = await _graph_write_query(client).execute_write(
                    WRITE_TIME_SUPERSEDE_CYPHER,
                    {
                        "new_fact_id": memory_id,
                        "scope_fragments": scope_read_fragments(scope),
                        "slot_fragments": [json.dumps(s) for s in _rslots],
                    },
                )
            except (RuntimeError, OSError, Neo4jError) as _supersede_err:
                _LOGGER.warning(
                    "write-time supersession failed; read-time still active",
                    extra={"error_type": type(_supersede_err).__name__},
                )
        stored = StoredMemory(
            memory_id=memory_id,
            subject=_user_identifier(scope),
            predicate=EXTRACTED_FACT_PREDICATE,
            content=content,
            metadata=metadata,
            created_at=None,
            updated_at=None,
        )
        return MemoryAddResult(
            memory_id=memory_id,
            content=content,
            event="ADD",
            metadata=public_memory_metadata(stored),
        )

    async def _materialize_entity_graph(
        self,
        client: MemoryClientContext,
        scope: MemoryScope,
        fact_id: str,
        unit: MemoryUnit,
    ) -> None:
        """Materialize the entity graph for a just-written extracted fact.

        Behind GNOSIS_ENTITY_GRAPH_ENABLED: MERGE a scope-keyed ``:Entity`` per
        named entity, link the fact to each with ``:MENTIONS``, and connect the
        entities with directed ``:RELATES`` edges from the unit's extracted
        triples (HippoRAG-2 / Graphiti). A no-op while the flag is off, so the
        extracted-fact write is byte-identical to before. The fact already
        landed, so any failure here degrades to "no graph materialized" with a
        structured warning and never fails the add or drops the fact.
        """
        if not self._app_settings.gnosis_entity_graph_enabled:
            return
        statements = entity_graph_statements(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            fact_id=fact_id,
            entities=unit.entities,
            relations=[
                RelationTriple(
                    head=relation.head,
                    relation=relation.relation,
                    tail=relation.tail,
                )
                for relation in unit_relations(unit)
            ],
            event_date=unit.event_date,
        )
        if not statements:
            return
        try:
            await self._ensure_entity_graph_schema(client)
            graph_write = _graph_write_query(client)
            for cypher, parameters in statements:
                _ = await graph_write.execute_write(cypher, parameters)
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "entity graph materialization failed; fact kept without graph",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": scope.tenant_id,
                },
            )

    async def _ensure_entity_graph_schema(self, client: MemoryClientContext) -> None:
        """Create the entity scope-key range index if absent.

        Idempotent ``CREATE INDEX ... IF NOT EXISTS`` through the same graph
        write handle the fact writes use, once per backend instance; a failed
        attempt retries on the next entity write.
        """
        if self._entity_graph_schema_ready:
            return
        _ = await _graph_write_query(client).execute_write(
            CREATE_ENTITY_SCOPE_INDEX_CYPHER,
            {},
        )
        self._entity_graph_schema_ready = True

    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        filters = _parsed_memory_filters(request.filters)
        decision = await self._route_decision(request.query)
        async with self._memory_client() as client:
            dense = await self._dense_memory_candidates(
                client,
                request.query,
                scope_read_fragments(request.scope),
                filter_superseded=decision.filter_superseded,
            )
            candidates = await self._hybrid_memory_candidates(
                client,
                request.query,
                scope_read_fragments(request.scope),
                dense,
                decision,
                filter_superseded=decision.filter_superseded,
            )
        budget = self._search_match_budget(request)
        matches: list[StoredMemory] = []
        for memory in candidates:
            if not memory_matches_scope(memory, request.scope):
                continue
            if not matches_filters(filters, memory_filter_fields(memory)):
                continue
            if not _meets_min_score(memory, request.min_score):
                continue
            matches.append(memory)
            if len(matches) == budget:
                break
        matches = await self._recall_filtered_matches(request, matches)
        matches = self._superseded_matches(matches)
        return MemorySearchResponse(
            results=[
                memory_record(memory, include_score=True)
                for memory in matches[: request.limit]
            ],
        )

    def _superseded_facts(
        self,
        facts: list[JsonObject],
        *,
        decision: RouteDecision | None = None,
    ) -> list[JsonObject]:
        """Drop same-slot older facts from the ranked context candidates."""
        enabled = (
            decision.supersession_enabled
            if decision is not None
            else self._app_settings.gnosis_read_supersession_enabled
        )
        if not enabled:
            return facts
        kept, dropped = drop_superseded(facts, _fact_freshness)
        _log_supersession(dropped, len(facts), surface="context")
        return kept

    def _superseded_matches(
        self,
        matches: list[StoredMemory],
    ) -> list[StoredMemory]:
        """Drop same-slot older facts from the ranked search matches."""
        if not self._app_settings.gnosis_read_supersession_enabled:
            return matches
        kept, dropped = drop_superseded(matches, _memory_freshness)
        _log_supersession(dropped, len(matches), surface="search")
        return kept

    async def _graphqa_fused_facts(
        self,
        request: MemoryContextRequest,
        decision: RouteDecision,
    ) -> list[JsonObject]:
        """Run the planned graph-QA route as a parallel fusion candidate leg.

        Frontier technique (Mnemis dual-route, frontier-2026 T3): the existing
        LLM-planned, validated, scope-safe, read-only graph-QA path runs on
        every context query in parallel with dense long-term retrieval; its
        derived nodes join the long-term candidate set before ranking is cut so
        multi-hop traversal facts survive the item budget. A no-op empty list
        while GNOSIS_GRAPHQA_ENABLED is off or the effective decision leaves
        fusion off (byte-identical dense-only output) or with no query. The
        route is bounded by GNOSIS_GRAPHQA_FUSION_TIMEOUT_SECONDS and any
        planner/execution failure (LLM, Neo4j, validation rejection, timeout)
        degrades to dense-only with a structured warning - the context request
        never fails on the graph leg.

        Distinct from the per-request ``include_graph`` flag, which renders a
        separate graph section rather than fusing into the ranked facts.
        """
        if (
            not self._app_settings.gnosis_graphqa_enabled
            or not decision.graphqa_fusion
            or not request.query
        ):
            return []
        try:
            graph = await asyncio.wait_for(
                self._graph_store.get_context(
                    GraphContextRequest(
                        scope=request.scope,
                        query=request.query,
                        limit=request.graph_limit,
                    ),
                ),
                timeout=self._app_settings.gnosis_graphqa_fusion_timeout_seconds,
            )
        except (
            TimeoutError,
            OpenAIError,
            Neo4jError,
            OSError,
            RuntimeError,
            ValidationError,
        ) as error:
            _LOGGER.warning(
                "graph-QA fusion route failed; assembling dense-only context",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": request.scope.tenant_id,
                    "guild_id": request.scope.guild_id,
                    "channel_id": request.scope.channel_id,
                },
            )
            return []
        return _graph_facts_to_candidates(graph.facts)

    async def _traversal_facts(
        self,
        client: MemoryClientContext,
        request: MemoryContextRequest,
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
    ) -> list[JsonObject]:
        """Entity-anchored graph traversal candidates for context fusion (T1).

        Pins the query's entity mentions as ``:Entity`` seed nodes, expands
        1-2 ``RELATES`` hops to reach bridge entities the query never names,
        and follows edge provenance back to the dated extracted facts - all
        in one fixed parameterized Cypher read, zero extra LLM calls. The
        provenance facts join the candidate pool tagged as graph-derived so
        they hold the reserved graph slots of the item budget. A no-op empty
        list while the effective decision leaves traversal off or the query
        pins nothing; any read failure degrades to dense-only with a
        structured warning - the context request never fails on this leg.
        """
        if not decision.graph_traversal or not request.query:
            return []
        seeds = query_seed_candidates(request.query)
        if not seeds:
            return []
        try:
            rows = await client.query.cypher(
                ENTITY_TRAVERSAL_CYPHER,
                traversal_parameters(
                    tenant_id=request.scope.tenant_id,
                    user_id=request.scope.user_id,
                    seeds=seeds,
                    scope_fragments=_metadata_fragments(scope_metadata),
                    limit=request.graph_limit,
                ),
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "entity traversal failed; assembling dense-only context",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": request.scope.tenant_id,
                },
            )
            return []
        candidates: list[JsonObject] = []
        for row in rows:
            memory = stored_memory_from_row(row)
            if memory is None:
                continue
            fact = _fact_from_memory(memory)
            if not _fact_matches_scope(fact, scope_metadata):
                continue
            fact["graphqa"] = True
            candidates.append(fact)
        if candidates:
            _LOGGER.info(
                "entity traversal fused provenance facts",
                extra={"count": len(candidates), "seeds": len(seeds)},
            )
        return candidates

    async def _bridge_traversal_facts(
        self,
        client: MemoryClientContext,
        request: MemoryContextRequest,
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
        dense_facts: list[JsonObject],
    ) -> list[JsonObject]:
        """Directed bridge-hop candidates for context fusion (T1-directed).

        One LLM call reads the query plus hop-1's dense evidence and names
        the bridge entities the question needs but never names; a fixed
        Cypher then fetches the dated extracted facts MENTIONing those
        entities - hop 2's evidence, unreachable by any ranking of the query
        text. The bridge facts join the pool tagged graph-derived so they
        hold the reserved graph slots. Entities the query itself names are
        dropped from the bridge list (they are hop 1, already dense-covered).
        A no-op empty list while the effective decision leaves the hop off,
        the query or evidence is empty, or the namer finds no bridge; any
        namer or read failure degrades to dense-only with a structured
        warning - the context request never fails on this leg.
        """
        if not decision.bridge_traversal or not request.query or not dense_facts:
            return []
        evidence = [_fact_context_line(fact) for fact in dense_facts]
        try:
            reply = await self._bridge_namer.name_bridges(request.query, evidence)
        except (RuntimeError, OSError, OpenAIError) as error:
            _LOGGER.warning(
                "bridge namer failed; assembling dense-only context",
                extra={"error_type": type(error).__name__},
            )
            return []
        query_mentions = set(query_seed_candidates(request.query))
        bridges = [
            normalized
            for name in parse_bridge_names(reply)
            if (normalized := normalize_entity_name(name))
            and normalized not in query_mentions
        ]
        if not bridges:
            return []
        try:
            rows = await client.query.cypher(
                BRIDGE_MENTION_CYPHER,
                bridge_parameters(
                    tenant_id=request.scope.tenant_id,
                    user_id=request.scope.user_id,
                    bridges=bridges,
                    scope_fragments=_metadata_fragments(scope_metadata),
                    limit=request.graph_limit,
                ),
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "bridge traversal read failed; assembling dense-only context",
                extra={
                    "error_type": type(error).__name__,
                    "tenant_id": request.scope.tenant_id,
                },
            )
            return []
        candidates: list[JsonObject] = []
        for row in rows:
            memory = stored_memory_from_row(row)
            if memory is None:
                continue
            fact = _fact_from_memory(memory)
            if not _fact_matches_scope(fact, scope_metadata):
                continue
            fact["graphqa"] = True
            candidates.append(fact)
        if candidates:
            _LOGGER.info(
                "bridge traversal fused mention facts",
                extra={"count": len(candidates), "bridges": len(bridges)},
            )
        return candidates

    async def _query_ranked_facts(
        self,
        client: MemoryClientContext,
        query: str,
        scope_metadata: Mapping[str, JsonValue],
        decision: RouteDecision,
    ) -> list[JsonObject]:
        """Long-term fact candidates ranked by embedding similarity.

        Reuses the /v1/memories/search candidate path (SDK ``search_facts``
        plus the same hybrid lexical fusion when the effective decision has
        hybrid retrieval on) so context and search rank identical
        stored data the same way, then re-checks scope on the deserialized
        records. Nothing ranked (no query, no embedder, no lexical hits)
        tells the caller to fall back to recency ordering.
        """
        if not query:
            return []
        fs = decision.filter_superseded
        dense = await self._dense_memory_candidates(
            client,
            query,
            _metadata_fragments(scope_metadata),
            filter_superseded=fs,
        )
        candidates = await self._hybrid_memory_candidates(
            client,
            query,
            _metadata_fragments(scope_metadata),
            dense,
            decision,
            filter_superseded=fs,
        )
        return [
            fact
            for memory in candidates
            if _fact_matches_scope(fact := _fact_from_memory(memory), scope_metadata)
        ]

    async def _dense_memory_candidates(
        self,
        client: MemoryClientContext,
        query: str,
        scope_fragments: list[JsonValue],
        filter_superseded: bool = False,
    ) -> list[StoredMemory]:
        """Embedding-similarity candidates for one scope, best score first.

        With GNOSIS_SCOPED_DENSE_RETRIEVAL_ENABLED off (default) this is the
        SDK's global search_facts ranking, byte-identical to the historical
        contract. On, the candidates come from a scope-narrowed vector query
        instead, so other scopes' facts can never crowd this scope out of the
        candidate pool (multi-user stores with near-duplicate content). Any
        failure on the scoped path - no embedder, embedding call failure, or
        the vector query itself - degrades to the SDK ranking with a warning
        rather than failing the read.

        When filter_superseded is True (knowledge_update route), the Cypher
        excludes facts where valid_to IS NOT NULL so structurally superseded
        facts cannot crowd out the current value in the dense top-20.
        """
        if not self._app_settings.gnosis_scoped_dense_retrieval_enabled:
            return await self._sdk_dense_candidates(client, query)
        try:
            embedding = _required_query_embedding(
                await _memory_embedding(client, query),
            )
            rows = await client.query.cypher(
                SCOPED_DENSE_MEMORY_SEARCH_CYPHER,
                {
                    "embedding": embedding,
                    "vector_pool": self._app_settings.gnosis_dense_scope_pool,
                    "scope_fragments": scope_fragments,
                    "candidate_limit": _MEMORY_SEARCH_CANDIDATE_LIMIT,
                    "filter_superseded": filter_superseded,
                },
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "scoped dense search failed; degrading to global dense retrieval",
                extra={"error_type": type(error).__name__},
            )
            return await self._sdk_dense_candidates(client, query)
        return [
            memory
            for row in rows
            if (memory := scored_stored_memory_from_row(row)) is not None
        ]

    @staticmethod
    async def _sdk_dense_candidates(
        client: MemoryClientContext,
        query: str,
    ) -> list[StoredMemory]:
        raw_records = await client.long_term.search_facts(
            query,
            limit=_MEMORY_SEARCH_CANDIDATE_LIMIT,
        )
        return stored_memories_from_sdk(raw_records)

    async def _hybrid_memory_candidates(  # noqa: PLR0913 - One knob per retrieval leg.
        self,
        client: MemoryClientContext,
        query: str,
        scope_fragments: list[JsonValue],
        dense: list[StoredMemory],
        decision: RouteDecision,
        filter_superseded: bool = False,
    ) -> list[StoredMemory]:
        """Fuse the dense ranking with BM25 lexical candidates via RRF.

        A no-op passthrough of the dense ranking while the effective decision
        leaves hybrid retrieval off (safe default) and whenever the
        lexical leg has nothing to add, so the dense-only behavior is
        byte-identical with hybrid off.
        """
        if not decision.hybrid_retrieval:
            return dense
        lexical = await self._lexical_memory_candidates(
            client,
            query,
            scope_fragments,
            filter_superseded=filter_superseded,
        )
        if not lexical:
            return dense
        return fuse_memory_rankings(
            dense,
            lexical,
            lexical_weight=self._app_settings.gnosis_rrf_lexical_weight,
        )

    async def _lexical_memory_candidates(
        self,
        client: MemoryClientContext,
        query: str,
        scope_fragments: list[JsonValue],
        filter_superseded: bool = False,
    ) -> list[StoredMemory]:
        """BM25 full-text candidates over Fact content, best score first.

        The Lucene query string is sanitized so user input can never inject
        Lucene operators, and any full-text failure (index bootstrap or
        query) degrades to an empty lexical leg with a structured warning -
        the read never fails because of the lexical path.

        When filter_superseded is True, the Cypher excludes structurally
        superseded facts (valid_to IS NOT NULL) from the BM25 candidate pool.
        """
        lucene_query = sanitize_lucene_query(query)
        if not lucene_query:
            return []
        try:
            await self._ensure_fulltext_index(client)
            rows = await client.query.cypher(
                LEXICAL_MEMORY_SEARCH_CYPHER,
                {
                    "query": lucene_query,
                    "scope_fragments": scope_fragments,
                    "candidate_limit": _MEMORY_SEARCH_CANDIDATE_LIMIT,
                    "filter_superseded": filter_superseded,
                },
            )
        except (
            RuntimeError,
            OSError,
            Neo4jError,
            BackendCapabilityUnavailable,
        ) as error:
            _LOGGER.warning(
                "lexical memory search failed; degrading to dense-only retrieval",
                extra={"error_type": type(error).__name__},
            )
            return []
        return [
            lexical_stored_memory(memory)
            for row in rows
            if (memory := stored_memory_from_row(row)) is not None
        ]

    async def _ensure_fulltext_index(self, client: MemoryClientContext) -> None:
        """Create the gateway-owned Fact full-text index if absent.

        The SDK owns the rest of the Fact schema, so this idempotent
        ``CREATE FULLTEXT INDEX ... IF NOT EXISTS`` goes through the same
        graph write handle the direct Fact writes use, once per backend
        instance; a failed attempt retries on the next lexical read.
        """
        if self._fulltext_index_ready:
            return
        _ = await _graph_write_query(client).execute_write(
            CREATE_FACT_OBJECT_FULLTEXT_INDEX_CYPHER,
            {},
        )
        self._fulltext_index_ready = True

    async def filter_recalled_memories(
        self,
        query: str,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Screen already-scoped memory records with the recall filter.

        The federated search route uses this over the merged local+remote
        result set; the filter only removes or keeps records, so the
        shareable-only scope of remote results is untouched. A no-op while
        GNOSIS_RECALL_FILTER_ENABLED is off.
        """
        if not self._app_settings.gnosis_recall_filter_enabled or not query:
            return list(records)
        return await keep_relevant_candidates(
            self._recall_filter,
            query=query,
            items=records,
            render=_memory_record_line,
            max_candidates=self._app_settings.gnosis_recall_filter_candidates,
        )

    async def _recall_filtered_facts(
        self,
        query: str,
        facts: list[JsonObject],
    ) -> list[JsonObject]:
        """Screen ranked long-term fact candidates for context assembly."""
        if not self._app_settings.gnosis_recall_filter_enabled or not query:
            return facts
        return await keep_relevant_candidates(
            self._recall_filter,
            query=query,
            items=facts,
            render=_fact_context_line,
            max_candidates=self._app_settings.gnosis_recall_filter_candidates,
        )

    def _search_recall_filter_active(self, request: MemorySearchRequest) -> bool:
        """Whether this search call runs the recall filter in the backend.

        Federated searches (``peers`` named) are filtered once over the
        merged local+remote result set by the route instead, keeping the
        budget at one LLM call per request.
        """
        return self._app_settings.gnosis_recall_filter_enabled and not request.peers

    def _search_match_budget(self, request: MemorySearchRequest) -> int:
        if not self._search_recall_filter_active(request):
            return request.limit
        return max(
            request.limit,
            self._app_settings.gnosis_recall_filter_candidates,
        )

    async def _recall_filtered_matches(
        self,
        request: MemorySearchRequest,
        matches: list[StoredMemory],
    ) -> list[StoredMemory]:
        if not self._search_recall_filter_active(request):
            return matches
        return await keep_relevant_candidates(
            self._recall_filter,
            query=request.query,
            items=matches,
            render=_stored_memory_line,
            max_candidates=self._app_settings.gnosis_recall_filter_candidates,
        )

    async def list_memories(self, request: MemoryListRequest) -> MemoryListResponse:
        filters = _parsed_memory_filters(request.filters)
        narrowing = build_cypher_filter(filters)
        parameters: dict[str, JsonValue] = {
            "scope_fragments": scope_read_fragments(request.scope),
            "scan_limit": _MEMORY_LIST_SCAN_LIMIT,
        }
        parameters.update(narrowing.parameters)
        async with self._memory_client() as client:
            rows = await client.query.cypher(
                list_memories_cypher(narrowing.fragment),
                parameters,
            )
        memories = [
            memory
            for row in rows
            if (memory := stored_memory_from_row(row)) is not None
            and memory_matches_scope(memory, request.scope)
            and matches_filters(filters, memory_filter_fields(memory))
        ]
        start = (request.page - 1) * request.page_size
        return MemoryListResponse(
            results=[
                memory_record(memory, include_score=False)
                for memory in memories[start : start + request.page_size]
            ],
            total=len(memories),
            page=request.page,
            page_size=request.page_size,
        )

    async def update_memory(
        self,
        memory_id: str,
        request: MemoryUpdateRequest,
    ) -> MemoryUpdateResponse:
        if request.content is None and request.metadata is None:
            raise BackendRequestError(_MEMORY_UPDATE_FIELDS_DETAIL)
        async with self._memory_client() as client:
            memory = await _require_scoped_memory(client, memory_id, request.scope)
            content = None
            embedding = None
            if request.content is not None:
                content = _redacted_text(request.content)
                embedding = await _memory_embedding(client, content)
            metadata = None
            if request.metadata is not None:
                metadata = merged_memory_metadata(memory, request.metadata)
            parameters: dict[str, JsonValue] = {
                "memory_id": memory.memory_id,
                "content": content,
                "metadata": metadata,
                "embedding": embedding,
            }
            rows = await _graph_write_query(client).execute_write(
                UPDATE_MEMORY_CYPHER,
                parameters,
            )
        _LOGGER.info(
            "memory update applied",
            extra=_memory_edit_audit(memory.memory_id, request.scope),
        )
        return MemoryUpdateResponse(
            memory_id=memory.memory_id,
            content=_updated_memory_content(rows, content or memory.content),
        )

    async def delete_memory(
        self,
        memory_id: str,
        request: MemoryDeleteRequest,
    ) -> MemoryDeleteResponse:
        async with self._memory_client() as client:
            memory = await _require_scoped_memory(client, memory_id, request.scope)
            _ = await _graph_write_query(client).execute_write(
                DELETE_MEMORY_CYPHER,
                {"memory_id": memory.memory_id},
            )
        _LOGGER.info(
            "memory delete applied",
            extra=_memory_edit_audit(memory.memory_id, request.scope),
        )
        return MemoryDeleteResponse(memory_id=memory.memory_id)

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        result = await self._graph_store.ingest_event(event)
        await self._event_fact_promoter.promote_for_result(
            event,
            result,
            self._memory_client(),
        )
        return result

    async def ingest_events(
        self,
        request: ClientEventBatchRequest,
    ) -> ClientEventBatchResponse:
        results = [await self.ingest_event(event) for event in request.events]
        return ClientEventBatchResponse(results=results)

    async def get_graph_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse:
        return await self._graph_store.get_context(request)

    async def get_sdk_stats(self, request: SdkStatsRequest) -> SdkStatsResponse:
        async with self._memory_client() as client:
            if not isinstance(client, StatsCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_STATS_UNAVAILABLE_DETAIL)
            stats = await client.get_stats()
        return SdkStatsResponse(
            scope=request.scope,
            stats=_redacted_object(_json_object(stats)),
        )

    async def get_dedup_stats(
        self,
        request: DedupStatsRequest,
    ) -> DedupStatsResponse:
        async with self._memory_client() as client:
            if not isinstance(client.long_term, DedupCapableLongTermMemory):
                raise BackendCapabilityUnavailable(_DEDUP_UNAVAILABLE_DETAIL)
            stats = await client.long_term.get_deduplication_stats()
        return DedupStatsResponse(
            scope=request.scope,
            stats=_redacted_object(_json_object(_dedup_stats_payload(stats))),
        )

    async def find_dedup_candidates(
        self,
        request: DedupCandidateRequest,
    ) -> DedupCandidateResponse:
        async with self._memory_client() as client:
            if not isinstance(client.long_term, DedupCapableLongTermMemory):
                raise BackendCapabilityUnavailable(_DEDUP_UNAVAILABLE_DETAIL)
            raw_candidates = await client.long_term.find_potential_duplicates(
                limit=request.limit,
            )
        candidates = [
            _dedup_candidate(source, target, similarity)
            for source, target, similarity in raw_candidates
        ]
        snapshot_hash = _dedup_snapshot_hash(request.scope, candidates)
        expires_at = datetime.now(UTC) + _CONSOLIDATION_TOKEN_TTL
        for candidate in candidates:
            self._dedup_candidates[candidate.candidate_id] = DedupCandidateState(
                candidate_id=candidate.candidate_id,
                version=candidate.version,
                scope=request.scope,
                source_id=UUID(candidate.source.id),
                target_id=UUID(candidate.target.id),
                graph_snapshot_hash=snapshot_hash,
            )
        return DedupCandidateResponse(
            scope=request.scope,
            candidates=[
                candidate.model_copy(
                    update={
                        "reject_dry_run_token": _dedup_token(
                            self._app_settings,
                            DedupTokenClaims(
                                scope=request.scope,
                                candidate_id=candidate.candidate_id,
                                candidate_version=candidate.version,
                                graph_snapshot_hash=snapshot_hash,
                                operation="reject",
                                expires_at=expires_at,
                            ),
                        ),
                        "merge_dry_run_token": _dedup_token(
                            self._app_settings,
                            DedupTokenClaims(
                                scope=request.scope,
                                candidate_id=candidate.candidate_id,
                                candidate_version=candidate.version,
                                graph_snapshot_hash=snapshot_hash,
                                operation="merge",
                                expires_at=expires_at,
                            ),
                        ),
                    },
                )
                for candidate in candidates
            ],
            graph_snapshot_hash=snapshot_hash,
            expires_at=expires_at.isoformat(),
        )

    async def apply_dedup_candidate(
        self,
        request: DedupApplyRequest,
    ) -> DedupApplyResponse:
        if not request.apply:
            raise BackendRequestError(_DEDUP_APPLY_REQUIRED_DETAIL)
        fingerprint = request.model_dump_json(exclude={"dry_run_token"})
        if (record := self._dedup_idempotency.get(request.idempotency_key)) is not None:
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise BackendRequestError(_DEDUP_IDEMPOTENCY_DETAIL)
            return record.response
        state = _require_current_dedup_candidate(
            request,
            self._dedup_candidates.get(request.candidate_id),
        )
        _require_dedup_token(self._app_settings, request)
        async with self._memory_client() as client:
            if not isinstance(client.long_term, DedupCapableLongTermMemory):
                raise BackendCapabilityUnavailable(_DEDUP_UNAVAILABLE_DETAIL)
            result = await _apply_dedup_operation(client.long_term, request, state)
        response = DedupApplyResponse(
            scope=request.scope,
            operation=request.operation,
            candidate_id=request.candidate_id,
            candidate_version=request.candidate_version,
            applied=True,
            result=_redacted_object(result),
            audit=request.audit,
        )
        self._dedup_idempotency[request.idempotency_key] = DedupIdempotencyRecord(
            request_fingerprint=fingerprint,
            response=response,
        )
        return response

    async def dry_run_consolidation(
        self,
        request: ConsolidationDryRunRequest,
    ) -> ConsolidationDryRunResponse:
        async with self._memory_client() as client:
            if not isinstance(client, ConsolidationCapableMemoryClient):
                raise BackendCapabilityUnavailable(_CONSOLIDATION_UNAVAILABLE_DETAIL)
            report = await _run_consolidation_operation(
                client.consolidation,
                request,
                dry_run=True,
            )
        report_payload = _safe_consolidation_report(report)
        fingerprint = _consolidation_request_fingerprint(request)
        graph_snapshot_hash = _hash_json(
            {
                "scope": _json_object(request.scope.model_dump(mode="json")),
                "operation": request.operation,
                "request_fingerprint": fingerprint,
                "report": report_payload,
            },
        )
        expires_at = datetime.now(UTC) + _DEDUP_TOKEN_TTL
        self._consolidation_dry_runs[graph_snapshot_hash] = ConsolidationDryRunState(
            scope=request.scope,
            operation=request.operation,
            graph_snapshot_hash=graph_snapshot_hash,
            request_fingerprint=fingerprint,
        )
        return ConsolidationDryRunResponse(
            scope=request.scope,
            operation=request.operation,
            dry_run=True,
            report=report_payload,
            graph_snapshot_hash=graph_snapshot_hash,
            dry_run_token=_consolidation_token(
                self._app_settings,
                ConsolidationTokenClaims(
                    scope=request.scope,
                    operation=request.operation,
                    graph_snapshot_hash=graph_snapshot_hash,
                    request_fingerprint=fingerprint,
                    expires_at=expires_at,
                ),
            ),
            expires_at=expires_at.isoformat(),
        )

    async def apply_consolidation(
        self,
        request: ConsolidationApplyRequest,
    ) -> ConsolidationApplyResponse:
        if not request.apply:
            raise BackendRequestError(_CONSOLIDATION_APPLY_REQUIRED_DETAIL)
        fingerprint = _consolidation_apply_fingerprint(request)
        if (
            record := self._consolidation_idempotency.get(request.idempotency_key)
        ) is not None:
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise BackendRequestError(_CONSOLIDATION_IDEMPOTENCY_DETAIL)
            return record.response
        state = _require_current_consolidation_dry_run(
            request,
            self._consolidation_dry_runs.get(request.graph_snapshot_hash),
        )
        _require_consolidation_token(self._app_settings, request, state)
        async with self._memory_client() as client:
            if not isinstance(client, ConsolidationCapableMemoryClient):
                raise BackendCapabilityUnavailable(_CONSOLIDATION_UNAVAILABLE_DETAIL)
            result = await _run_consolidation_operation(
                client.consolidation,
                request,
                dry_run=False,
            )
        response = ConsolidationApplyResponse(
            scope=request.scope,
            operation=request.operation,
            applied=True,
            result=_safe_consolidation_report(result),
            audit=request.audit,
        )
        self._consolidation_idempotency[request.idempotency_key] = (
            ConsolidationIdempotencyRecord(
                request_fingerprint=fingerprint,
                response=response,
            )
        )
        return response

    async def export_graph(self, request: GraphExportRequest) -> GraphExportResponse:
        async with self._memory_client() as client:
            if not isinstance(client, GraphCapableMemoryClient):
                raise BackendCapabilityUnavailable(_SDK_GRAPH_UNAVAILABLE_DETAIL)
            graph = await client.get_graph(
                memory_types=request.memory_types,
                session_id=request.scope.session_id,
                include_embeddings=False,
                limit=request.limit,
            )
        return _graph_export_response(request, graph)

    async def search_entities(
        self,
        request: EntitySearchRequest,
    ) -> EntitySearchResponse:
        filters = _scoped_filters(request.scope, request.metadata)
        async with self._memory_client() as client:
            raw_records = await client.long_term.search_entities(
                request.query,
                limit=request.limit,
            )
        records = _ENTITY_RECORDS_ADAPTER.validate_python(raw_records)
        return EntitySearchResponse(
            entities=[
                _redacted_entity(record)
                for record in records
                if _record_matches_filters(record.metadata, filters)
            ],
        )

    async def search_facts(self, request: FactSearchRequest) -> FactSearchResponse:
        filters = _scoped_filters(request.scope, request.metadata)
        async with self._memory_client() as client:
            raw_records = await client.long_term.search_facts(
                request.query,
                limit=request.limit,
            )
        records = _FACT_RECORDS_ADAPTER.validate_python(raw_records)
        return FactSearchResponse(
            facts=[
                _redacted_fact(record)
                for record in records
                if _record_matches_filters(record.metadata, filters)
            ],
        )

    async def search_preferences(
        self,
        request: PreferenceSearchRequest,
    ) -> PreferenceSearchResponse:
        filters = _scoped_filters(request.scope, request.metadata)
        async with self._memory_client() as client:
            raw_records = await client.long_term.search_preferences(
                request.query,
                category=request.category,
                limit=request.limit,
            )
        records = _PREFERENCE_RECORDS_ADAPTER.validate_python(raw_records)
        return PreferenceSearchResponse(
            preferences=[
                _redacted_preference(record)
                for record in records
                if _record_matches_filters(record.metadata, filters)
            ],
        )

    async def add_entity(self, request: EntityWriteRequest) -> EntityRecord:
        async with self._memory_client() as client:
            raw_record = await client.long_term.add_entity(
                request.name,
                request.type,
                subtype=request.subtype,
                description=_redacted_optional_text(request.description),
                aliases=request.aliases,
                attributes=_redacted_object(request.attributes),
                resolve=request.resolve,
                generate_embedding=request.generate_embedding,
                deduplicate=request.deduplicate,
                geocode=False,
                enrich=False,
                metadata=_write_metadata(
                    request.scope,
                    request.metadata,
                    request.provenance,
                ),
            )
        record = _ENTITY_RECORD_ADAPTER.validate_python(raw_record)
        return _redacted_entity(record)

    async def add_fact(self, request: FactWriteRequest) -> FactRecord:
        async with self._memory_client() as client:
            raw_record = await client.long_term.add_fact(
                _redacted_text(request.subject),
                _redacted_text(request.predicate),
                _redacted_text(request.object),
                confidence=request.confidence,
                generate_embedding=request.generate_embedding,
                metadata=_write_metadata(
                    request.scope,
                    request.metadata,
                    request.provenance,
                ),
            )
        record = _FACT_RECORD_ADAPTER.validate_python(raw_record)
        return _redacted_fact(record)

    async def add_preference(self, request: PreferenceWriteRequest) -> PreferenceRecord:
        async with self._memory_client() as client:
            raw_record = await client.long_term.add_preference(
                request.category,
                _redacted_text(request.preference),
                context=_redacted_optional_text(request.context),
                confidence=request.confidence,
                generate_embedding=request.generate_embedding,
                metadata=_write_metadata(
                    request.scope,
                    request.metadata,
                    request.provenance,
                ),
                user_identifier=(
                    request.user_identifier or _user_identifier(request.scope)
                ),
            )
        record = _PREFERENCE_RECORD_ADAPTER.validate_python(raw_record)
        return _redacted_preference(record)

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        await self._graph_store.require_available()
        return await self._skill_registry.list_skills(request)

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        await self._graph_store.require_available()
        return await self._skill_registry.propose_skill(proposal)

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        await self._graph_store.require_available()
        return await self._skill_registry.record_skill_usage(usage)

    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse:
        async with self._memory_client() as client:
            trace = await client.reasoning.start_trace(
                request.session_id,
                _redacted_text(request.task),
                generate_embedding=True,
                metadata=_reasoning_write_metadata(request.scope, request.metadata),
                triggered_by_message_id=request.triggered_by_message_id,
                user_identifier=(
                    request.user_identifier or _user_identifier(request.scope)
                ),
            )
        return ReasoningTraceStartResponse(
            trace_id=str(trace.id),
            session_id=trace.session_id,
            task=trace.task,
        )

    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse:
        async with self._memory_client() as client:
            step = await client.reasoning.add_step(
                UUID(request.trace_id),
                thought=None,
                action=_redacted_optional_text(request.action),
                observation=_redacted_optional_text(request.observation),
                generate_embedding=True,
                metadata=_reasoning_write_metadata(request.scope, request.metadata),
            )
        return ReasoningStepResponse(
            step_id=str(step.id),
            trace_id=str(step.trace_id),
            step_number=step.step_number,
        )

    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse:
        async with self._memory_client() as client:
            tool_call = await client.reasoning.record_tool_call(
                UUID(request.step_id),
                request.tool_name,
                _redacted_object(request.arguments),
                result=redact_secrets(request.result),
                status=ToolCallStatus(request.status),
                duration_ms=request.duration_ms,
                error=_redacted_optional_text(request.error),
                message_id=request.message_id,
                touched_entities=[
                    EntityRef(id=entity.id, name=entity.name, type=entity.type)
                    for entity in request.touched_entities
                ],
            )
        return ReasoningToolCallResponse(
            tool_call_id=str(tool_call.id),
            trace_id=request.trace_id,
            step_id=request.step_id,
        )

    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse:
        async with self._memory_client() as client:
            trace = await client.reasoning.complete_trace(
                UUID(request.trace_id),
                outcome=_redacted_optional_text(request.outcome),
                success=request.success,
                generate_step_embeddings=False,
            )
        completed_at = None
        if trace.completed_at is not None:
            completed_at = trace.completed_at.isoformat()
        return ReasoningTraceCompleteResponse(
            trace_id=str(trace.id),
            success=trace.success,
            outcome=trace.outcome,
            completed_at=completed_at,
        )

    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse:
        async with self._memory_client() as client:
            context = await client.reasoning.get_context(
                request.query,
                max_traces=request.max_items,
            )
        redacted_context = _redacted_text(context)
        if not redacted_context:
            redacted_context = "No similar reasoning traces found."
        return ReasoningContextResponse(context=redacted_context, traces=[])

    async def list_reasoning_traces(
        self,
        request: ReasoningTraceListRequest,
    ) -> ReasoningTraceListResponse:
        async with self._memory_client() as client:
            try:
                traces = await client.reasoning.list_traces(
                    session_id=request.session_id or request.scope.session_id,
                    success_only=request.success_only,
                    limit=request.limit,
                    offset=request.offset,
                )
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        return ReasoningTraceListResponse(
            scope=request.scope,
            traces=_scoped_reasoning_traces(request.scope, traces),
        )

    async def get_reasoning_trace(
        self,
        request: ReasoningTraceDetailRequest,
    ) -> ReasoningTraceDetailResponse:
        async with self._memory_client() as client:
            try:
                trace = await _get_reasoning_trace(client.reasoning, request)
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        if trace is None or not _reasoning_trace_matches_scope(trace, request.scope):
            return ReasoningTraceDetailResponse(scope=request.scope)
        return ReasoningTraceDetailResponse(
            scope=request.scope,
            trace=_reasoning_trace_summary(trace),
            steps=[_reasoning_step_record(step) for step in trace.steps],
        )

    async def find_similar_reasoning_traces(
        self,
        request: ReasoningSimilarTracesRequest,
    ) -> ReasoningSimilarTracesResponse:
        async with self._memory_client() as client:
            try:
                traces = await client.reasoning.get_similar_traces(
                    _redacted_text(request.task),
                    limit=request.limit,
                    success_only=request.success_only,
                    threshold=request.threshold,
                )
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        return ReasoningSimilarTracesResponse(
            scope=request.scope,
            traces=_scoped_reasoning_traces(request.scope, traces),
        )

    async def search_reasoning_steps(
        self,
        request: ReasoningStepSearchRequest,
    ) -> ReasoningStepSearchResponse:
        async with self._memory_client() as client:
            try:
                steps = await client.reasoning.search_steps(
                    _redacted_text(request.query),
                    limit=request.limit,
                    success_only=request.success_only,
                    threshold=request.threshold,
                )
            except AttributeError as error:
                raise BackendCapabilityUnavailable(
                    _REASONING_READ_UNAVAILABLE_DETAIL,
                ) from error
        return ReasoningStepSearchResponse(
            scope=request.scope,
            steps=[
                _reasoning_step_record(step)
                for step in steps
                if _reasoning_step_matches_scope(step, request.scope)
            ],
        )

    async def get_reasoning_tool_stats(
        self,
        request: ReasoningToolStatsRequest,
    ) -> ReasoningToolStatsResponse:
        _ = request.tool_name
        return ReasoningToolStatsResponse(
            scope=request.scope,
            tools=[],
        )

    def _memory_client(self) -> MemoryClientContext:
        if self._memory_client_factory is not None:
            return self._memory_client_factory(self._settings)
        return _memory_client_context(MemoryClient(self._settings))


def _parsed_memory_filters(filters: JsonObject | None) -> MemoryFilter | None:
    if filters is None:
        return None
    try:
        return parse_filters(filters)
    except FilterValidationError as error:
        raise BackendRequestError(error.detail) from error


def _meets_min_score(memory: StoredMemory, min_score: float | None) -> bool:
    if min_score is None:
        return True
    score = memory_score(memory)
    return score is not None and score >= min_score


async def _lookup_latest_memory(
    client: MemoryClientContext,
    scope: MemoryScope,
    *,
    predicate: str,
    content: str,
) -> StoredMemory | None:
    rows = await client.query.cypher(
        LOOKUP_LATEST_MEMORY_CYPHER,
        {
            "subject": _user_identifier(scope),
            "predicate": predicate,
            "object": content,
            "scope_fragments": scope_read_fragments(scope),
        },
    )
    for row in rows:
        memory = stored_memory_from_row(row)
        if memory is not None and memory_matches_scope(memory, scope):
            return memory
    return None


async def _require_scoped_memory(
    client: MemoryClientContext,
    memory_id: str,
    scope: MemoryScope,
) -> StoredMemory:
    rows = await client.query.cypher(
        LOOKUP_MEMORY_CYPHER,
        {"memory_id": memory_id},
    )
    for row in rows:
        memory = stored_memory_from_row(row)
        if memory is not None and memory_matches_scope(memory, scope):
            return memory
    raise MemoryNotFoundError


def _required_query_embedding(
    embedding: list[JsonValue] | None,
) -> list[JsonValue]:
    if embedding is None:
        raise BackendCapabilityUnavailable(_QUERY_EMBEDDER_UNAVAILABLE_DETAIL)
    return embedding


def _updated_memory_content(rows: list[JsonObject], fallback: str) -> str:
    for row in rows:
        value = row.get("object")
        if isinstance(value, str) and value:
            return _redacted_text(value)
    return _redacted_text(fallback)


def _buffer_readiness_status(write_errors: int) -> Literal["ready", "degraded"]:
    if write_errors == 0:
        return "ready"
    return "degraded"
