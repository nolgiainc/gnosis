import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from os import environ
from typing import TYPE_CHECKING, Self, cast
from uuid import UUID

import pytest
from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.memory.long_term import EntityType
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import (
    ReasoningStepWithContext,
    ToolCall,
    ToolCallStatus,
    ToolStats,
)
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.schema.models import EntityRef

environ["GNOSIS_TOKEN"] = "test-token"
environ["GNOSIS_READ_OPERATOR_TOKEN"] = "read-operator-token"
environ["GNOSIS_EXPORT_OPERATOR_TOKEN"] = "export-operator-token"
environ["GNOSIS_WRITE_OPERATOR_TOKEN"] = "write-operator-token"
environ["GNOSIS_ADMIN_OPERATOR_TOKEN"] = "admin-operator-token"
environ["NEO4J_URI"] = "bolt://neo4j.local:7687"
environ["NEO4J_PASSWORD"] = "test-password"
environ["LITELLM_BASE_URL"] = "http://litellm.local/v1"
environ["LITELLM_API_KEY"] = "test-litellm-key"

from pydantic import TypeAdapter, ValidationError

from gnosis.backend import (
    BackendCapabilityUnavailable,
    BackendRequestError,
    Neo4jAgentMemoryBackend,
    build_direct_graph_store,
    litellm_embedding_model,
)
from gnosis.graph_probe import DirectNeo4jProbe, GraphPersistenceUnavailableError
from gnosis.graph_store import (
    DirectNeo4jGraphStore,
    InMemoryGraphExecutor,
    Neo4jGraphExecutor,
)
from gnosis.models import (
    BackendReadiness,
    BufferFlushResponse,
    BufferStatus,
    ClientEvent,
    ClientEventActor,
    ClientEventBatchRequest,
    ClientEventBatchResponse,
    ClientEventSubject,
    ClientEventType,
    ConsolidationApplyRequest,
    ConsolidationApplyResponse,
    ConsolidationDryRunRequest,
    ConsolidationDryRunResponse,
    ContextRequest,
    ContextResponse,
    DedupApplyRequest,
    DedupApplyResponse,
    DedupCandidate,
    DedupCandidateRequest,
    DedupCandidateResponse,
    DedupOperationName,
    DedupOperatorAudit,
    DedupStatsRequest,
    DedupStatsResponse,
    DiagnosticsConfig,
    DiagnosticsResponse,
    DiscordEventContext,
    EntityMessageLinkRequest,
    EntityRecord,
    EntitySearchRequest,
    EntitySearchResponse,
    EntityWriteRequest,
    EventIngestResult,
    EventIngestStatus,
    ExtractionCandidate,
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
    GraphExportNode,
    GraphExportRelationship,
    GraphExportRequest,
    GraphExportResponse,
    JsonObject,
    JsonValue,
    MemoryAddRequest,
    MemoryAddResponse,
    MemoryContextRequest,
    MemoryContextResponse,
    MemoryContextSection,
    MemoryDeleteRequest,
    MemoryDeleteResponse,
    MemoryListRequest,
    MemoryListResponse,
    MemoryProvenance,
    MemoryScope,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchUnavailable,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
    MemoryVisibility,
    MessageRole,
    MessageWriteRequest,
    MessageWriteResponse,
    OcrImageReference,
    PreferenceRecord,
    PreferenceSearchRequest,
    PreferenceSearchResponse,
    PreferenceWriteRequest,
    RawTextDocument,
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
    RustFSSourceReference,
    SdkStatsRequest,
    SdkStatsResponse,
    SkillListRequest,
    SkillListResponse,
    SkillProposal,
    SkillUsage,
    SourceClient,
)
from gnosis.settings import Settings

if TYPE_CHECKING:
    from gnosis.backend import LongTermMemory, MemoryBackend, MemoryClientContext
    from gnosis.graph_query_qa import LiteLLMGraphQueryPlanner


_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def test_litellm_embedding_model_when_embedding_alias_is_bare() -> None:
    # Given: self-hosted config uses a bare LiteLLM proxy alias for memory embeddings.
    model = "local-qwen3-embedding-0.6b"

    # When: the model is prepared for the LiteLLM SDK provider.
    sdk_model = litellm_embedding_model(model)

    # Then: only the SDK-facing model is provider-qualified.
    assert model == "local-qwen3-embedding-0.6b"
    assert sdk_model == "openai/local-qwen3-embedding-0.6b"


def test_litellm_embedding_model_when_embedding_alias_is_qualified() -> None:
    # Given: a caller already supplied a LiteLLM provider-qualified embedding model.
    model = "openai/local-qwen3-embedding-0.6b"

    # When: the model is prepared for the LiteLLM SDK provider.
    sdk_model = litellm_embedding_model(model)

    # Then: the configured provider prefix is preserved without double-prefixing.
    assert sdk_model == "openai/local-qwen3-embedding-0.6b"


def test_backend_default_graph_store_has_no_planner_when_graphqa_disabled() -> None:
    # Given: default settings - LLM-planned graph QA is opt-in.
    settings = Settings(
        gnosis_llm="openai/gpt-5.5",
        litellm_base_url="http://litellm.test/v1",
        litellm_api_key="test-graph-key",
    )

    # When: the backend builds its default direct graph store.
    store = build_direct_graph_store(settings)

    # Then: no LLM Cypher planner is wired, so graph context only runs the
    # fixed parameterised queries.
    executor = cast("Neo4jGraphExecutor", store.executor)
    assert settings.gnosis_graphqa_enabled is False
    assert executor.graph_query_planner is None


def test_backend_default_graph_store_uses_gnosis_graph_query_planner() -> None:
    # Given: gnosis has its own LiteLLM graph QA settings and graph QA enabled.
    settings = Settings(
        gnosis_llm="openai/gpt-5.5",
        litellm_base_url="http://litellm.test/v1",
        litellm_api_key="test-graph-key",
        gnosis_graphqa_enabled=True,
    )

    # When: the backend builds its default direct graph store.
    store = build_direct_graph_store(settings)

    # Then: graph context can plan safe dynamic Cypher through gnosis LiteLLM envs.
    executor = cast("Neo4jGraphExecutor", store.executor)
    planner = cast("LiteLLMGraphQueryPlanner", executor.graph_query_planner)
    assert planner.model == "openai/gpt-5.5"
    assert planner.base_url == "http://litellm.test/v1"


@pytest.mark.anyio
async def test_build_memory_settings_keeps_safe_memory_feature_defaults() -> None:
    # Given: the default gateway settings.
    memory_client_factory = CapturingMemoryClientFactory(RecordingMemoryClient())
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=memory_client_factory,
        graph_store=RecordingGraphStore(),
    )

    # When: the backend opens an SDK client context for a write.
    _ = await backend.add_message(_message_write_request())

    # Then: current supported SDK flags are wired without enabling new behavior.
    memory_settings = memory_client_factory.settings[0]
    assert memory_settings.memory.multi_tenant is True
    assert memory_settings.memory.write_mode == "sync"
    assert memory_settings.memory.max_pending == 200
    assert memory_settings.memory.conversation_ttl_days is None
    assert memory_settings.memory.audit_read is False
    assert memory_settings.memory.fact_deduplication_enabled is True
    assert memory_settings.memory.trace_embedding_enabled is True


@pytest.mark.anyio
async def test_build_memory_settings_applies_supported_memory_feature_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: supported SDK memory policy fields are overridden by environment.
    monkeypatch.setenv("GNOSIS_WRITE_MODE", "buffered")
    monkeypatch.setenv("GNOSIS_MAX_PENDING", "7")
    monkeypatch.setenv("GNOSIS_CONVERSATION_TTL_DAYS", "30")
    monkeypatch.setenv("GNOSIS_AUDIT_READ", "true")
    monkeypatch.setenv("GNOSIS_FACT_DEDUPLICATION_ENABLED", "false")
    monkeypatch.setenv("GNOSIS_TRACE_EMBEDDING_ENABLED", "false")
    memory_client_factory = CapturingMemoryClientFactory(RecordingMemoryClient())
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=memory_client_factory,
        graph_store=RecordingGraphStore(),
    )

    # When: the backend opens an SDK client context for a write.
    _ = await backend.add_message(_message_write_request())

    # Then: only supported MemoryConfig fields receive the overrides.
    memory_settings = memory_client_factory.settings[0]
    assert memory_settings.memory.write_mode == "buffered"
    assert memory_settings.memory.max_pending == 7
    assert memory_settings.memory.conversation_ttl_days == 30
    assert memory_settings.memory.audit_read is True
    assert memory_settings.memory.fact_deduplication_enabled is False
    assert memory_settings.memory.trace_embedding_enabled is False


@pytest.mark.anyio
async def test_buffer_status_reflects_buffered_policy_and_write_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: buffered writes are enabled and the SDK exposes write errors.
    monkeypatch.setenv("GNOSIS_WRITE_MODE", "buffered")
    monkeypatch.setenv("GNOSIS_MAX_PENDING", "7")
    fake_client = RecordingMemoryClient(write_errors=[{"message": "failed"}])
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: readiness asks the backend for buffer state.
    status = await backend.buffer_status()
    readiness = await backend.readiness()

    # Then: only counters and safe policy values are exposed.
    assert status == BufferStatus(
        write_mode="buffered",
        max_pending=7,
        pending_writes=None,
        write_errors=1,
        status="degraded",
    )
    assert readiness.buffer_status == "degraded"


@pytest.mark.anyio
async def test_flush_buffer_calls_sdk_flush_and_returns_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: buffered writes are enabled and the SDK exposes flush.
    monkeypatch.setenv("GNOSIS_WRITE_MODE", "buffered")
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: an admin flushes the write buffer.
    response = await backend.flush_buffer()

    # Then: the SDK flush runs once and the returned status is ready.
    assert response == BufferFlushResponse(
        flushed=True,
        status=BufferStatus(
            write_mode="buffered",
            max_pending=200,
            pending_writes=None,
            write_errors=0,
            status="ready",
        ),
    )
    assert fake_client.flush_calls == 1


@pytest.mark.anyio
async def test_shutdown_flushes_pending_buffered_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: buffered writes are enabled and the SDK exposes wait_for_pending.
    monkeypatch.setenv("GNOSIS_WRITE_MODE", "buffered")
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: FastAPI lifespan shutdown closes the backend.
    await backend.shutdown()

    # Then: pending writes are drained exactly once.
    assert fake_client.wait_for_pending_calls == 1


@pytest.mark.anyio
async def test_add_message_keeps_extraction_disabled_when_policy_is_disabled() -> None:
    # Given: a request opts into entity extraction while global policy is disabled.
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: the message is written through the SDK adapter.
    _ = await backend.add_message(_message_write_request(extract_entities=True))

    # Then: the SDK call receives safe extraction flags.
    assert fake_client.short_term.messages == [
        ShortTermMessageWrite(extract_entities=False, extract_relations=False),
    ]


@pytest.mark.anyio
async def test_add_message_enables_entity_extraction_when_policy_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: both the service policy and request opt into entity extraction.
    monkeypatch.setenv("GNOSIS_EXTRACT_ENTITIES_ENABLED", "true")
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: the message is written through the SDK adapter.
    _ = await backend.add_message(_message_write_request(extract_entities=True))

    # Then: only the requested and globally allowed SDK flag is enabled.
    assert fake_client.short_term.messages == [
        ShortTermMessageWrite(extract_entities=True, extract_relations=False),
    ]


@pytest.mark.anyio
async def test_add_message_rejects_relation_only_extraction() -> None:
    # Given: a request asks for relation extraction without entity extraction.
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(RecordingMemoryClient()),
        graph_store=RecordingGraphStore(),
    )

    # When / Then: relation-only extraction is rejected before SDK write.
    with pytest.raises(BackendRequestError, match="requires entity extraction"):
        _ = await backend.add_message(_message_write_request(extract_relations=True))


@pytest.mark.anyio
async def test_preview_extraction_returns_candidates_without_durable_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: preview is enabled with raw text input.
    monkeypatch.setenv("GNOSIS_EXTRACT_ENTITIES_ENABLED", "true")
    monkeypatch.setenv("GNOSIS_EXTRACTION_PREVIEW_ENABLED", "true")
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: a dry-run extraction preview is requested.
    response = await backend.preview_extraction(
        ExtractionPreviewRequest(
            scope=_scope(),
            content="Cartman remembers the MegaCorp plan.",
            raw_text_documents=[
                RawTextDocument(source_id="doc-1", text="Kenny prefers concise plans."),
            ],
            extract_entities=True,
        ),
    )

    # Then: preview data is returned and no SDK memory write occurs.
    assert response.extract_entities is True
    assert response.extract_relations is False
    assert response.metrics.documents == 2
    assert response.provenance.source_ids == ["message.content", "doc-1"]
    assert [candidate.source_id for candidate in response.candidates] == [
        "message.content",
        "doc-1",
    ]
    assert fake_client.short_term.messages == []
    assert fake_client.long_term.facts == []


@pytest.mark.anyio
async def test_preview_extraction_rejects_ocr_when_policy_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: preview is enabled but OCR policy stays disabled.
    monkeypatch.setenv("GNOSIS_EXTRACTION_PREVIEW_ENABLED", "true")
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(RecordingMemoryClient()),
        graph_store=RecordingGraphStore(),
    )

    # When / Then: OCR references fail safely without media bytes.
    with pytest.raises(BackendRequestError, match="OCR preview is disabled"):
        _ = await backend.preview_extraction(
            ExtractionPreviewRequest(
                scope=_scope(),
                ocr_image_references=[_ocr_reference()],
            ),
        )


@pytest.mark.anyio
async def test_preview_extraction_accepts_ocr_when_policy_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: preview and OCR are explicitly enabled.
    monkeypatch.setenv("GNOSIS_EXTRACTION_PREVIEW_ENABLED", "true")
    monkeypatch.setenv("GNOSIS_OCR_ENABLED", "true")
    monkeypatch.setenv("GNOSIS_OCR_MODEL", "openai/unlimited-ocr")
    monkeypatch.setenv("GNOSIS_OCR_MAX_IMAGE_BYTES", "1024")
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(RecordingMemoryClient()),
        graph_store=RecordingGraphStore(),
    )

    # When: an image reference is previewed.
    response = await backend.preview_extraction(
        ExtractionPreviewRequest(
            scope=_scope(),
            ocr_image_references=[_ocr_reference()],
        ),
    )

    # Then: the response exposes only derived placeholder text and source metadata.
    assert response.metrics.ocr_images == 1
    assert response.candidates[0].kind == "ocr_text"
    assert response.candidates[0].source_id == "image-1"


@pytest.mark.anyio
async def test_preview_extraction_enforces_rustfs_bucket_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: RustFS preview is limited to one bucket and object prefix.
    monkeypatch.setenv("GNOSIS_EXTRACTION_PREVIEW_ENABLED", "true")
    monkeypatch.setenv("GNOSIS_RUSTFS_ENABLED", "true")
    monkeypatch.setenv("GNOSIS_RUSTFS_BUCKET", "memory-private")
    monkeypatch.setenv("GNOSIS_RUSTFS_PREFIX", "gnosis/")
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(RecordingMemoryClient()),
        graph_store=RecordingGraphStore(),
    )

    # When: an allowed RustFS reference is previewed.
    response = await backend.preview_extraction(
        ExtractionPreviewRequest(
            scope=_scope(),
            rustfs_source_references=[_rustfs_reference()],
        ),
    )

    # Then: only the reference is echoed in provenance.
    assert response.metrics.rustfs_objects == 1
    assert response.provenance.rustfs_objects == [_rustfs_reference()]

    # When / Then: an out-of-prefix object is rejected.
    with pytest.raises(BackendRequestError, match="outside service policy"):
        _ = await backend.preview_extraction(
            ExtractionPreviewRequest(
                scope=_scope(),
                rustfs_source_references=[
                    _rustfs_reference(object_key="public/leak.png"),
                ],
            ),
        )


@pytest.mark.anyio
async def test_backend_protocol_ingests_event_batch() -> None:
    # Given: a fake backend implementing the full persistence protocol.
    backend: MemoryBackend = RecordingBackend()
    event = _client_event()

    # When: a typed batch is ingested through the protocol seam.
    response = await backend.ingest_events(ClientEventBatchRequest(events=[event]))

    # Then: fake implementations can satisfy the seam without Neo4j.
    assert response == ClientEventBatchResponse(
        results=[
            EventIngestResult(
                event_id="discord-message-999",
                status=EventIngestStatus.ACCEPTED,
            ),
        ],
    )


@pytest.mark.anyio
async def test_long_term_fake_supports_protocol() -> None:
    # Given: a fake SDK client at the expanded long-term seam.
    client = RecordingMemoryClient()
    client_context: MemoryClientContext = client
    _ = client_context

    # When: supported entity, fact, preference, and provenance methods are called.
    entity = await client.long_term.add_entity(
        "Cartman",
        "PERSON",
        metadata={"tenant_id": "nolgia"},
    )
    fact = await client.long_term.add_fact(
        "Cartman",
        "prefers",
        "snacks",
        metadata={"tenant_id": "nolgia"},
        generate_embedding=True,
    )
    preference = await client.long_term.add_preference(
        "style",
        "concise answers",
        metadata={"tenant_id": "nolgia"},
        user_identifier="nolgia:cartman",
    )
    entity_linked = await client.long_term.link_entity_to_message(
        UUID("00000000-0000-0000-0000-000000000001"),
        "message-1",
        confidence=0.9,
    )

    # Then: fakes satisfy supported SDK methods without unsupported gap shims.
    assert [
        item.name for item in await client.long_term.search_entities("cartman")
    ] == ["Cartman"]
    assert [item.subject for item in await client.long_term.search_facts("snacks")] == [
        "Cartman",
    ]
    assert [
        item.category for item in await client.long_term.search_preferences("concise")
    ] == ["style"]
    assert [
        item.preference
        for item in await client.long_term.get_preferences_for("nolgia:cartman")
    ] == ["concise answers"]
    assert [
        item.object for item in await client.long_term.get_facts_about("Cartman")
    ] == ["snacks"]
    assert entity.name == "Cartman"
    assert fact.confidence == 1.0
    assert preference.user_identifier == "nolgia:cartman"
    assert entity_linked is True


def test_long_term_protocol_assignment_requires_expanded_methods() -> None:
    # Given: a fake long-term client that implements the confirmed SDK seam.
    long_term: LongTermMemory = RecordingLongTermMemory()

    # Then: missing expanded seam methods would fail basedpyright.
    assert isinstance(long_term, RecordingLongTermMemory)


def test_long_term_operation_contracts_are_json_safe_and_limited() -> None:
    # Given: typed API-facing contracts for long-term operations.
    scope = _scope()
    provenance = MemoryProvenance(source="message", source_id="message-1")

    # When: requests and responses are constructed at the gateway boundary.
    request = EntitySearchRequest(scope=scope, query="cartman", limit=25)
    response = EntitySearchResponse(
        entities=[
            EntityRecord(
                id="entity-1",
                name="Cartman",
                type="PERSON",
                metadata={"tenant_id": "nolgia"},
                provenance=provenance,
            ),
        ],
        unavailable=[MemorySearchUnavailable(capability="relationships")],
    )
    fact_write = FactWriteRequest(
        scope=scope,
        subject="Cartman",
        predicate="prefers",
        object="snacks",
        confidence=0.7,
        provenance=provenance,
        metadata={"source": "test"},
    )
    preference_write = PreferenceWriteRequest(
        scope=scope,
        category="style",
        preference="concise answers",
        confidence=0.9,
        provenance=provenance,
        metadata={"audience": ["agents"]},
    )

    # Then: limits, scope, provenance, confidence, and JSON-safe metadata are explicit.
    assert request.limit == 25
    assert response.entities[0].metadata == {"tenant_id": "nolgia"}
    assert fact_write.provenance == provenance
    assert preference_write.metadata == {"audience": ["agents"]}
    assert FactSearchRequest(scope=scope, query="snacks").limit == 10
    assert PreferenceSearchRequest(scope=scope, query="concise").limit == 10
    assert FactSearchResponse(
        facts=[FactRecord(subject="a", predicate="b", object="c")],
    )
    assert PreferenceSearchResponse(
        preferences=[PreferenceRecord(category="style", preference="concise")],
    )
    assert EntityWriteRequest(scope=scope, name="Kenny", type="PERSON").resolve is True
    assert (
        EntityMessageLinkRequest(
            scope=scope,
            entity_id="entity-1",
            message_id="message-1",
        ).confidence
        == 1.0
    )

    # When / Then: non-JSON metadata is rejected before API return.
    with pytest.raises(ValidationError):
        _ = EntityRecord.model_validate(
            {"name": "Cartman", "type": "PERSON", "metadata": {"unsafe": object()}},
        )

    with pytest.raises(ValidationError):
        _ = EntitySearchRequest(scope=scope, query="cartman", limit=101)


@pytest.mark.anyio
async def test_direct_neo4j_probe_failure_degrades_to_clear_error() -> None:
    # Given: direct graph persistence is unavailable at startup.
    probe = DirectNeo4jProbe(driver_factory=FailingDriverFactory())

    # When / Then: the seam fails explicitly instead of silently no-oping.
    with pytest.raises(GraphPersistenceUnavailableError) as error:
        await probe.require_available()

    assert "Neo4j structured graph persistence is unavailable" in str(error.value)


@pytest.mark.anyio
async def test_backend_promotes_accepted_event_to_embedded_long_term_fact() -> None:
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=DirectNeo4jGraphStore(executor=InMemoryGraphExecutor()),
    )
    event = _client_event()

    result = await backend.ingest_event(event)

    assert result.status == EventIngestStatus.ACCEPTED
    assert fake_client.long_term.facts == [
        LongTermFactWrite(
            subject="tenant:nolgia:message:message-999",
            predicate="discord.message_created",
            obj="message message-999: remember this",
            metadata={
                "agent_id": "nolgia-agent",
                "channel_id": "456",
                "event_id": "discord-message-999",
                "event_type": "message_created",
                "guild_id": "123",
                "idempotency_key": "discord:message:message-999:create",
                "session_id": "guild:123:channel:456",
                "tenant_id": "nolgia",
                "user_id": "789",
                "visibility": "channel",
            },
            generate_embedding=True,
        ),
    ]


@pytest.mark.anyio
async def test_backend_repairs_duplicate_event_graph_without_promoting_fact() -> None:
    fake_client = RecordingMemoryClient()
    executor = InMemoryGraphExecutor()
    store = DirectNeo4jGraphStore(executor=executor)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=store,
    )
    event = _client_event()
    _ = await backend.ingest_event(event)
    executor.clear_current_nodes_for_test()

    duplicate = await backend.ingest_event(event)

    assert duplicate.status == EventIngestStatus.DUPLICATE
    assert len(fake_client.long_term.facts) == 1
    assert executor.semantic_node_ids_for_test() == {
        "tenant:nolgia:agent:nolgia-agent",
        "tenant:nolgia:channel:456",
        "tenant:nolgia:client:discord",
        "tenant:nolgia:guild:123",
        "tenant:nolgia:message:message-999",
        "tenant:nolgia:tenant:nolgia",
        "tenant:nolgia:user:789",
    }


@pytest.mark.anyio
async def test_backend_retries_fact_promotion_after_initial_failure() -> None:
    # Given: graph persistence accepts an event before long-term fact promotion fails.
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(failed_writes_remaining=1),
    )
    executor = InMemoryGraphExecutor()
    store = DirectNeo4jGraphStore(executor=executor)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=store,
    )
    event = _client_event()

    # When: the caller retries the same event after the promotion failure.
    with pytest.raises(PromotionFailureError):
        _ = await backend.ingest_event(event)
    retry = await backend.ingest_event(event)

    # Then: the graph stays idempotent while the missing fact is promoted once.
    assert retry.status == EventIngestStatus.DUPLICATE
    assert executor.event_count == 1
    assert len(fake_client.long_term.facts) == 1
    assert fake_client.long_term.facts[0].generate_embedding is True


@pytest.mark.anyio
async def test_memory_context_combines_labeled_sections_in_order() -> None:
    # Given: a backend with every prompt enrichment flag explicitly enabled.
    fake_client = RecordingMemoryClient(
        short_term=RecordingShortTermMemory(context="recent chat"),
        long_term=RecordingLongTermMemory(context="### User Preferences\n- concise"),
        reasoning=RecordingReasoningMemory(context="### Similar Past Tasks\n- replied"),
    )
    graph_store = RecordingGraphStore(context="graph summary")
    backend = Neo4jAgentMemoryBackend(
        Settings(
            gnosis_prompt_entities_enabled=True,
            gnosis_prompt_preferences_enabled=True,
            gnosis_prompt_reasoning_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: combined memory context is requested with graph enabled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="what matters?",
            max_items=4,
            graph_limit=3,
        ),
    )

    # Then: sections are labeled in the required order and scoped graph is used.
    assert response == MemoryContextResponse(
        sections=[
            MemoryContextSection(source="short_term", content="recent chat"),
            MemoryContextSection(
                source="long_term_preferences_entities",
                content="### User Preferences\n- concise",
            ),
            MemoryContextSection(
                source="reasoning",
                content="### Similar Past Tasks\n- replied",
            ),
            MemoryContextSection(
                source="graph",
                content="graph summary",
                facts=[{"kind": "graph"}],
            ),
        ],
    )
    assert fake_client.short_term.context_queries == ["what matters?"]
    assert fake_client.long_term.context_queries == ["what matters?"]
    assert fake_client.reasoning.context_queries == ["what matters?"]
    assert graph_store.context_requests == [
        GraphContextRequest(scope=_scope(), query="what matters?", limit=3),
    ]


@pytest.mark.anyio
async def test_memory_context_omits_empty_sections_and_disabled_graph() -> None:
    # Given: only long-term context has content and graph recall is disabled.
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(context="### Relevant Entities\n- Cartman"),
    )
    graph_store = RecordingGraphStore(context="graph summary")
    backend = Neo4jAgentMemoryBackend(
        Settings(gnosis_prompt_entities_enabled=True),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: combined memory context is requested without graph context.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="what matters?",
            include_graph=False,
        ),
    )

    # Then: empty sections are omitted and graph storage is not queried.
    assert response == MemoryContextResponse(
        sections=[
            MemoryContextSection(
                source="long_term_preferences_entities",
                content="### Relevant Entities\n- Cartman",
            ),
        ],
    )
    assert graph_store.context_requests == []


@pytest.mark.anyio
async def test_memory_context_uses_short_term_and_facts_by_default() -> None:
    # Given: SDK long-term and reasoning contexts exist while prompt flags are disabled.
    fact = _fact_row(
        subject="tenant:nolgia:message:message-999",
        predicate="discord.message_created",
        object_value="message message-999: default fact",
        metadata=_scope_metadata(_scope()),
    )
    fake_client = RecordingMemoryClient(
        short_term=RecordingShortTermMemory(context="recent chat"),
        long_term=RecordingLongTermMemory(context="### User Preferences\n- concise"),
        reasoning=RecordingReasoningMemory(context="### Similar Past Tasks\n- hidden"),
        query=RecordingCypherQuery(rows=[{"f": fact}]),
    )
    graph_store = RecordingGraphStore(context="graph summary")
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: combined memory context is requested with default feature flags.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="what matters?",
            include_graph=False,
        ),
    )

    # Then: short-term and scoped facts are routed without opt-in enrichments.
    assert [section.source for section in response.sections] == [
        "short_term",
        "long_term_facts",
    ]
    assert "recent chat" in response.sections[0].content
    assert "message message-999: default fact" in response.sections[1].content
    assert fake_client.long_term.context_queries == []
    assert fake_client.reasoning.context_queries == []


@pytest.mark.anyio
async def test_memory_context_preserves_graph_when_short_term_recall_fails() -> None:
    # Given: short-term SDK recall fails while graph context has ranked facts.
    fake_client = RecordingMemoryClient(
        short_term=RecordingShortTermMemory(
            context_error=ValueError("badly formed hexadecimal UUID string"),
        ),
    )
    graph_store = RecordingGraphStore(
        context="BlackDave active channel #1: general-chat",
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=graph_store,
    )

    # When: combined memory context includes short-term and graph recall.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="@BlackDave top 5 most active channels??",
            include_long_term=False,
            include_reasoning=False,
        ),
    )

    # Then: the failing short-term section is omitted without suppressing graph facts.
    assert response == MemoryContextResponse(
        sections=[
            MemoryContextSection(
                source="graph",
                content="BlackDave active channel #1: general-chat",
                facts=[{"kind": "graph"}],
            ),
        ],
    )
    assert fake_client.short_term.context_queries == [
        "@BlackDave top 5 most active channels??",
    ]
    assert graph_store.context_requests == [
        GraphContextRequest(
            scope=_scope(),
            query="@BlackDave top 5 most active channels??",
            limit=8,
        ),
    ]


@pytest.mark.anyio
async def test_memory_context_redacts_reasoning_when_prompt_flag_enabled() -> None:
    # Given: reasoning recall returns safe summaries mixed with inert secret-like text.
    fake_client = RecordingMemoryClient(
        reasoning=RecordingReasoningMemory(
            context=(
                "### Similar Past Tasks\n"
                "- used tool\n"
                "- thought: private internal reasoning\n"
                "- token=test-secret-token"
            ),
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(gnosis_prompt_reasoning_enabled=True),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: only reasoning memory is requested for prompt context.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="how did we solve this?",
            include_short_term=False,
            include_long_term=False,
            include_graph=False,
        ),
    )

    # Then: reasoning is separate from long-term facts and secret-like text is redacted.
    assert response.sections == [
        MemoryContextSection(
            source="reasoning",
            content="### Similar Past Tasks\n- used tool\n- token=[REDACTED]",
        ),
    ]
    assert fake_client.reasoning.context_queries == ["how did we solve this?"]


@pytest.mark.anyio
async def test_reasoning_trace_list_filters_scope_and_redacts_sensitive() -> None:
    # Given: the SDK returns scoped and cross-tenant traces with sensitive metadata.
    scoped_trace_id = UUID("00000000-0000-0000-0000-000000000001")
    other_trace_id = UUID("00000000-0000-0000-0000-000000000002")
    fake_client = RecordingMemoryClient(
        reasoning=RecordingReasoningMemory(
            traces=[
                _sdk_reasoning_trace(
                    scoped_trace_id,
                    metadata=_scope_metadata(_scope())
                    | {"safe": "kept", "token": "memory-token-sentinel"},
                    task="answer with API_KEY=plain-value",
                    success=True,
                ),
                _sdk_reasoning_trace(
                    other_trace_id,
                    metadata=_scope_metadata(_scope()) | {"tenant_id": "evil-corp"},
                    task="other tenant",
                    success=True,
                ),
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: reasoning traces are listed for the configured scope.
    response = await backend.list_reasoning_traces(
        ReasoningTraceListRequest(scope=_scope(), success_only=True, limit=5, offset=0),
    )

    # Then: only scoped summaries are returned and unsafe fields are not exposed.
    assert [trace.trace_id for trace in response.traces] == [str(scoped_trace_id)]
    assert response.traces[0].task == "answer with API_KEY=[REDACTED]"
    assert response.traces[0].metadata == {"safe": "kept"}
    assert fake_client.reasoning.list_trace_calls == [
        {
            "session_id": _scope().session_id,
            "success_only": True,
            "since": None,
            "until": None,
            "limit": 5,
            "offset": 0,
            "order_by": "started_at",
            "order_dir": "desc",
        },
    ]


@pytest.mark.anyio
async def test_reasoning_lifecycle_writes_scope_metadata() -> None:
    # Given: lifecycle requests include user metadata that tries to override scope.
    trace_id = UUID("00000000-0000-0000-0000-000000000009")
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: trace and step lifecycle writes run through the backend.
    _ = await backend.start_reasoning_trace(
        ReasoningTraceStartRequest(
            scope=_scope(),
            session_id=_scope().session_id,
            task="discord reply",
            metadata={"tenant_id": "evil-corp", "safe": "kept"},
        ),
    )
    _ = await backend.add_reasoning_step(
        ReasoningStepRequest(
            scope=_scope(),
            trace_id=str(trace_id),
            action="lookup",
            metadata={"tenant_id": "evil-corp", "safe": "step"},
        ),
    )

    # Then: scope metadata is persisted and wins over caller-supplied metadata.
    assert fake_client.reasoning.trace_start_metadata == [
        _scope_metadata(_scope()) | {"safe": "kept"},
    ]
    assert fake_client.reasoning.step_metadata == [
        _scope_metadata(_scope()) | {"safe": "step"},
    ]


@pytest.mark.anyio
async def test_reasoning_trace_detail_returns_public_steps_only() -> None:
    # Given: SDK trace detail includes hidden thought, embeddings, and tool secrets.
    trace_id = UUID("00000000-0000-0000-0000-000000000003")
    step_id = UUID("00000000-0000-0000-0000-000000000004")
    tool_call = ToolCall(
        id=UUID("00000000-0000-0000-0000-000000000005"),
        created_at=datetime(2026, 6, 29, tzinfo=UTC),
        step_id=step_id,
        tool_name="memory.get_context",
        arguments={"query": "lookup", "api_key": "plain-value"},
        result={"token": "memory-token-sentinel", "safe": "kept"},
        status=ToolCallStatus.SUCCESS,
        duration_ms=12,
        metadata={"chain_of_thought": "hidden", "safe": "tool"},
    )
    step = SdkReasoningStep(
        id=step_id,
        created_at=datetime(2026, 6, 29, tzinfo=UTC),
        trace_id=trace_id,
        step_number=1,
        thought="private chain of thought",
        action="get memory",
        observation="returned API_KEY=plain-value",
        embedding=[0.1],
        tool_calls=[tool_call],
        metadata={"thought": "hidden", "safe": "step"},
    )
    fake_client = RecordingMemoryClient(
        reasoning=RecordingReasoningMemory(
            traces=[
                _sdk_reasoning_trace(
                    trace_id,
                    metadata=_scope_metadata(_scope()),
                    task="answer user",
                    steps=[step],
                ),
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: a trace detail read includes steps.
    response = await backend.get_reasoning_trace(
        ReasoningTraceDetailRequest(scope=_scope(), trace_id=str(trace_id)),
    )

    # Then: only public step/tool fields are returned and sensitive fields are redacted.
    assert response.trace is not None
    assert response.trace.trace_id == str(trace_id)
    assert response.steps[0].action == "get memory"
    assert response.steps[0].observation == "returned API_KEY=[REDACTED]"
    assert response.steps[0].metadata == {"safe": "step"}
    assert response.steps[0].tool_calls == [
        {
            "tool_call_id": str(tool_call.id),
            "step_id": str(step_id),
            "tool_name": "memory.get_context",
            "arguments": {"query": "lookup", "api_key": "[REDACTED]"},
            "result": {"safe": "kept"},
            "status": "success",
            "duration_ms": 12,
            "error": None,
            "metadata": {"safe": "tool"},
        },
    ]
    assert fake_client.reasoning.detail_with_steps_trace_ids == [trace_id]


@pytest.mark.anyio
async def test_reasoning_step_search_returns_api_created_scoped_steps() -> None:
    # Given: a step shaped like an API-created write has full scope metadata.
    trace_id = UUID("00000000-0000-0000-0000-00000000000a")
    step = SdkReasoningStep(
        id=UUID("00000000-0000-0000-0000-00000000000b"),
        created_at=datetime(2026, 6, 29, tzinfo=UTC),
        trace_id=trace_id,
        step_number=1,
        action="lookup",
        metadata=_scope_metadata(_scope()) | {"safe": "step"},
    )
    fake_client = RecordingMemoryClient(
        reasoning=RecordingReasoningMemory(
            steps=[
                ReasoningStepWithContext(
                    step=step,
                    similarity=0.91,
                    parent_task="discord reply",
                ),
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: step search runs for the same scope.
    response = await backend.search_reasoning_steps(
        ReasoningStepSearchRequest(scope=_scope(), query="lookup", limit=4),
    )

    # Then: the scoped step remains visible through scoped read/search APIs.
    assert [record.step_id for record in response.steps] == [str(step.id)]
    assert response.steps[0].metadata == _scope_metadata(_scope()) | {"safe": "step"}


@pytest.mark.anyio
async def test_reasoning_similarity_step_search_and_tool_stats_use_sdk_reads() -> None:
    # Given: SDK reasoning read methods return scoped traces, steps, and tool stats.
    trace_id = UUID("00000000-0000-0000-0000-000000000006")
    step = SdkReasoningStep(
        id=UUID("00000000-0000-0000-0000-000000000007"),
        created_at=datetime(2026, 6, 29, tzinfo=UTC),
        trace_id=trace_id,
        step_number=1,
        action="lookup",
        metadata=_scope_metadata(_scope()),
    )
    metadata_less_step = SdkReasoningStep(
        id=UUID("00000000-0000-0000-0000-000000000008"),
        created_at=datetime(2026, 6, 29, tzinfo=UTC),
        trace_id=trace_id,
        step_number=2,
        action="unscoped lookup",
        metadata={},
    )
    fake_client = RecordingMemoryClient(
        reasoning=RecordingReasoningMemory(
            traces=[
                _sdk_reasoning_trace(
                    trace_id,
                    metadata=_scope_metadata(_scope()),
                    task="discord reply",
                    success=True,
                ),
            ],
            steps=[
                ReasoningStepWithContext(
                    step=step,
                    similarity=0.91,
                    parent_task="discord reply",
                ),
                ReasoningStepWithContext(
                    step=metadata_less_step,
                    similarity=0.9,
                    parent_task="other tenant task",
                ),
            ],
            tool_stats=[
                ToolStats(
                    name="memory.get_context",
                    description="reads memory",
                    total_calls=3,
                    successful_calls=2,
                    failed_calls=1,
                    success_rate=2 / 3,
                    avg_duration_ms=10.5,
                ),
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: operator read surfaces query similar traces, steps, and tool stats.
    similar = await backend.find_similar_reasoning_traces(
        ReasoningSimilarTracesRequest(scope=_scope(), task="discord reply", limit=3),
    )
    search = await backend.search_reasoning_steps(
        ReasoningStepSearchRequest(scope=_scope(), query="lookup", limit=4),
    )
    stats = await backend.get_reasoning_tool_stats(
        ReasoningToolStatsRequest(scope=_scope(), tool_name="memory.get_context"),
    )

    # Then: trace and step reads are scoped, while global SDK tool stats do not leak.
    assert [trace.trace_id for trace in similar.traces] == [str(trace_id)]
    assert [record.step_id for record in search.steps] == [str(step.id)]
    assert stats.tools == []
    assert fake_client.reasoning.similar_trace_calls[0]["task"] == "discord reply"
    assert fake_client.reasoning.step_search_calls[0]["query"] == "lookup"
    assert fake_client.reasoning.tool_stats_names == []


def test_settings_require_explicit_operator_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: deploy-time operator tokens are absent from the environment.
    monkeypatch.delenv("GNOSIS_READ_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("GNOSIS_EXPORT_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("GNOSIS_WRITE_OPERATOR_TOKEN", raising=False)
    monkeypatch.delenv("GNOSIS_ADMIN_OPERATOR_TOKEN", raising=False)

    # When / Then: settings refuse predictable built-in operator credentials.
    with pytest.raises(ValidationError):
        _ = Settings()


@pytest.mark.anyio
async def test_reasoning_trace_list_reports_capability_unavailable() -> None:
    # Given: an older SDK reasoning memory lacks read methods.
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MissingCapabilityClientFactory(),
        graph_store=RecordingGraphStore(),
    )

    # When / Then: the backend returns a typed capability error instead of faking data.
    request = ReasoningTraceListRequest(scope=_scope())
    with pytest.raises(BackendCapabilityUnavailable, match="reasoning read"):
        _ = await backend.list_reasoning_traces(request)


@pytest.mark.anyio
async def test_sdk_stats_uses_sdk_capability_and_redacts_secrets() -> None:
    # Given: the SDK stats capability returns operational counters and secrets.
    fake_client = RecordingMemoryClient(
        stats={"nodes": 3, "token": "memory-token-sentinel"},
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: scoped stats are requested through the backend adapter.
    response = await backend.get_sdk_stats(SdkStatsRequest(scope=_scope()))

    # Then: stats are redacted and tied to the requested scope.
    assert response == SdkStatsResponse(
        scope=_scope(),
        stats={"nodes": 3, "token": "[REDACTED]"},
    )
    assert fake_client.stats_calls == 1


@pytest.mark.anyio
async def test_graph_export_uses_scoped_sdk_graph_parameters() -> None:
    # Given: the SDK graph capability returns nodes, relationships, and metadata.
    sdk_graph = RecordingSdkGraph(
        nodes=[
            RecordingSdkNode(
                id="message-1",
                labels=["Message"],
                properties={"tenant_id": "nolgia", "api_key": "sk-secret"},
            ),
        ],
        relationships=[
            RecordingSdkRelationship(
                id="rel-1",
                type="MENTIONS",
                from_node="message-1",
                to_node="entity-1",
                properties={"token": "memory-token-sentinel"},
            ),
        ],
        metadata={"limit": 2, "secret": "hidden"},
    )
    fake_client = RecordingMemoryClient(graph=sdk_graph)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: graph export is requested with explicit memory types and limit.
    response = await backend.export_graph(
        GraphExportRequest(scope=_scope(), memory_types=["long_term"], limit=2),
    )

    # Then: the adapter forwards scoped SDK filters and returns a redacted typed graph.
    assert fake_client.graph_calls == [
        GraphCall(
            memory_types=["long_term"],
            session_id="guild:123:channel:456",
            include_embeddings=False,
            limit=2,
        ),
    ]
    assert response == GraphExportResponse(
        scope=_scope(),
        nodes=[
            GraphExportNode(
                id="message-1",
                labels=["Message"],
                properties={"tenant_id": "nolgia", "api_key": "[REDACTED]"},
            ),
        ],
        relationships=[
            GraphExportRelationship(
                id="rel-1",
                type="MENTIONS",
                from_node="message-1",
                to_node="entity-1",
                properties={"token": "[REDACTED]"},
            ),
        ],
        metadata={"limit": 2, "secret": "[REDACTED]"},
    )


@pytest.mark.anyio
async def test_scoped_long_term_search_filters_and_redacts_sdk_records() -> None:
    # Given: SDK search returns mixed-scope records with sensitive metadata.
    scoped_metadata = _json_object(
        {
            "tenant_id": "nolgia",
            "space_id": "discord",
            "agent_id": "nolgia-agent",
            "session_id": "guild:123:channel:456",
            "user_id": "789",
            "visibility": "channel",
            "guild_id": "123",
            "channel_id": "456",
            "topic": "snacks",
            "api_key": "sk-secret",
        },
    )
    other_metadata = _json_object(scoped_metadata | {"tenant_id": "evil-corp"})
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(
            entities=[
                EntityRecord(name="Cartman", type="PERSON", metadata=scoped_metadata),
                EntityRecord(name="Butters", type="PERSON", metadata=other_metadata),
            ],
            fact_records=[
                FactRecord(
                    subject="Cartman",
                    predicate="prefers",
                    object="snacks",
                    metadata=scoped_metadata,
                ),
                FactRecord(
                    subject="Butters",
                    predicate="prefers",
                    object="chaos",
                    metadata=other_metadata,
                ),
            ],
            preferences=[
                PreferenceRecord(
                    category="style",
                    preference="concise answers",
                    metadata=scoped_metadata,
                ),
                PreferenceRecord(
                    category="style",
                    preference="cross tenant",
                    metadata=other_metadata,
                ),
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: each scoped search includes an additional metadata filter.
    entities = await backend.search_entities(
        EntitySearchRequest(
            scope=_scope(),
            query="cartman",
            metadata={"topic": "snacks"},
        ),
    )
    facts = await backend.search_facts(
        FactSearchRequest(scope=_scope(), query="snacks", metadata={"topic": "snacks"}),
    )
    preferences = await backend.search_preferences(
        PreferenceSearchRequest(
            scope=_scope(),
            query="concise",
            category="style",
            metadata={"topic": "snacks"},
        ),
    )

    # Then: only same-scope records are returned and secret metadata is redacted.
    assert [entity.name for entity in entities.entities] == ["Cartman"]
    assert entities.entities[0].metadata["api_key"] == "[REDACTED]"
    assert [fact.subject for fact in facts.facts] == ["Cartman"]
    assert facts.facts[0].metadata["api_key"] == "[REDACTED]"
    assert [preference.preference for preference in preferences.preferences] == [
        "concise answers",
    ]
    assert preferences.preferences[0].metadata["api_key"] == "[REDACTED]"


@pytest.mark.anyio
async def test_scoped_long_term_writes_attach_provenance_and_redact_metadata() -> None:
    # Given: direct long-term writes include metadata and provenance.
    fake_client = RecordingMemoryClient()
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )
    provenance = MemoryProvenance(source="message", source_id="message-1")

    # When: entity, fact, and preference writes pass through the adapter.
    entity = await backend.add_entity(
        EntityWriteRequest(
            scope=_scope(),
            name="Cartman",
            type="PERSON",
            metadata={"api_key": "sk-secret"},
            provenance=provenance,
        ),
    )
    fact = await backend.add_fact(
        FactWriteRequest(
            scope=_scope(),
            subject="Cartman",
            predicate="prefers",
            object="snacks",
            metadata={"token": "memory-token-sentinel"},
            provenance=provenance,
            generate_embedding=False,
        ),
    )
    preference = await backend.add_preference(
        PreferenceWriteRequest(
            scope=_scope(),
            category="style",
            preference="concise answers",
            provenance=provenance,
        ),
    )

    # Then: SDK metadata is scope-bound, redacted, and provenance-aware.
    assert entity.metadata["tenant_id"] == "nolgia"
    assert entity.metadata["api_key"] == "[REDACTED]"
    assert entity.metadata["source"] == "message"
    assert fake_client.long_term.facts == [
        LongTermFactWrite(
            subject="Cartman",
            predicate="prefers",
            obj="snacks",
            metadata={
                "agent_id": "nolgia-agent",
                "channel_id": "456",
                "guild_id": "123",
                "session_id": "guild:123:channel:456",
                "source": "message",
                "source_id": "message-1",
                "space_id": "discord",
                "tenant_id": "nolgia",
                "token": "[REDACTED]",
                "user_id": "789",
                "visibility": "channel",
            },
            generate_embedding=False,
        ),
    ]
    assert fact.metadata["token"] == "[REDACTED]"
    assert preference.metadata["tenant_id"] == "nolgia"
    assert preference.user_identifier == "nolgia:discord:channel:nolgia-agent:789"


@pytest.mark.anyio
async def test_sdk_stats_reports_unavailable_when_sdk_lacks_capability() -> None:
    # Given: an SDK client without the newer stats capability.
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MissingCapabilityClientFactory(),
        graph_store=RecordingGraphStore(),
    )

    # When / Then: the backend reports capability absence instead of faking success.
    with pytest.raises(BackendCapabilityUnavailable, match="SDK stats are unavailable"):
        _ = await backend.get_sdk_stats(SdkStatsRequest(scope=_scope()))


@pytest.mark.anyio
async def test_graph_export_reports_unavailable_when_sdk_lacks_capability() -> None:
    # Given: an SDK client without the newer graph export capability.
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MissingCapabilityClientFactory(),
        graph_store=RecordingGraphStore(),
    )

    # When / Then: the backend reports capability absence instead of raw fallback reads.
    with pytest.raises(
        BackendCapabilityUnavailable,
        match="SDK graph export is unavailable",
    ):
        _ = await backend.export_graph(GraphExportRequest(scope=_scope(), limit=2))


@pytest.mark.anyio
async def test_dedup_stats_uses_sdk_capability_and_redacts_secrets() -> None:
    # Given: the SDK dedup stats capability returns counters with accidental secrets.
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(
            dedup_stats={"pending_reviews": 2, "token": "memory-token-sentinel"},
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: scoped dedup stats are requested through the backend adapter.
    response = await backend.get_dedup_stats(DedupStatsRequest(scope=_scope()))

    # Then: stats are redacted and tied to the requested scope.
    assert response == DedupStatsResponse(
        scope=_scope(),
        stats={"pending_reviews": 2, "token": "[REDACTED]"},
    )
    assert fake_client.long_term.dedup_stats_calls == 1


@pytest.mark.anyio
async def test_dedup_candidates_return_redacted_dry_run_tokens() -> None:
    # Given: the SDK returns one duplicate candidate with secret metadata.
    source = _dedup_entity(
        "00000000-0000-0000-0000-000000000001",
        metadata={"api_key": "sk-secret"},
    )
    target = _dedup_entity("00000000-0000-0000-0000-000000000002")
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(
            duplicate_candidates=[(source, target, 0.94)],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: a candidate review is requested.
    response = await backend.find_dedup_candidates(
        DedupCandidateRequest(scope=_scope(), limit=5),
    )

    # Then: candidate snapshots are redacted and operation-bound tokens are issued.
    candidate = response.candidates[0]
    assert fake_client.long_term.duplicate_limits == [5]
    assert candidate.source.metadata == {"api_key": "[REDACTED]"}
    assert candidate.reject_dry_run_token != candidate.merge_dry_run_token
    assert response.graph_snapshot_hash != ""


@pytest.mark.anyio
async def test_dedup_reject_requires_dry_run_and_is_idempotent() -> None:
    # Given: a dry-run candidate response has issued a reject token.
    source = _dedup_entity("00000000-0000-0000-0000-000000000001")
    target = _dedup_entity("00000000-0000-0000-0000-000000000002")
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(
            duplicate_candidates=[(source, target, 0.94)],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )
    candidate_response = await backend.find_dedup_candidates(
        DedupCandidateRequest(scope=_scope()),
    )
    candidate = candidate_response.candidates[0]
    request = _dedup_apply_request(
        candidate_response,
        candidate,
        operation="reject",
        dry_run_token=candidate.reject_dry_run_token,
    )

    # When: the reject operation is applied twice with the same idempotency key.
    first = await backend.apply_dedup_candidate(request)
    second = await backend.apply_dedup_candidate(request)

    # Then: the SDK is called once with confirm=false and the same response is replayed.
    assert first == second
    assert fake_client.long_term.review_calls == [
        DedupReviewCall(
            source_id=UUID("00000000-0000-0000-0000-000000000001"),
            target_id=UUID("00000000-0000-0000-0000-000000000002"),
            confirm=False,
        ),
    ]


@pytest.mark.anyio
async def test_dedup_merge_rejects_wrong_token_and_applies_merge_token() -> None:
    # Given: a dry-run candidate has separate reject and merge tokens.
    source = _dedup_entity("00000000-0000-0000-0000-000000000001")
    target = _dedup_entity("00000000-0000-0000-0000-000000000002")
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(
            duplicate_candidates=[(source, target, 0.94)],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )
    candidate_response = await backend.find_dedup_candidates(
        DedupCandidateRequest(scope=_scope()),
    )
    candidate = candidate_response.candidates[0]

    # When / Then: a reject token cannot authorize merge.
    with pytest.raises(BackendRequestError, match="dry-run token"):
        _ = await backend.apply_dedup_candidate(
            _dedup_apply_request(
                candidate_response,
                candidate,
                operation="merge",
                dry_run_token=candidate.reject_dry_run_token,
            ),
        )

    # When: the merge token is applied.
    response = await backend.apply_dedup_candidate(
        _dedup_apply_request(
            candidate_response,
            candidate,
            operation="merge",
            dry_run_token=candidate.merge_dry_run_token,
            idempotency_key="dedup-merge-1",
        ),
    )

    # Then: the SDK merge operation receives the candidate pair.
    assert response.result == {"merged": True}
    assert fake_client.long_term.merge_calls == [
        DedupMergeCall(
            source_id=UUID("00000000-0000-0000-0000-000000000001"),
            target_id=UUID("00000000-0000-0000-0000-000000000002"),
        ),
    ]


@pytest.mark.anyio
async def test_dedup_apply_rejects_false_apply_and_stale_candidate() -> None:
    # Given: a candidate dry-run has been issued.
    source = _dedup_entity("00000000-0000-0000-0000-000000000001")
    target = _dedup_entity("00000000-0000-0000-0000-000000000002")
    fake_client = RecordingMemoryClient(
        long_term=RecordingLongTermMemory(
            duplicate_candidates=[(source, target, 0.94)],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )
    candidate_response = await backend.find_dedup_candidates(
        DedupCandidateRequest(scope=_scope()),
    )
    candidate = candidate_response.candidates[0]

    # When / Then: explicit apply=true and the same snapshot are required.
    with pytest.raises(BackendRequestError, match="apply=true"):
        _ = await backend.apply_dedup_candidate(
            _dedup_apply_request(
                candidate_response,
                candidate,
                operation="reject",
                dry_run_token=candidate.reject_dry_run_token,
                apply=False,
            ),
        )
    with pytest.raises(BackendRequestError, match="stale"):
        _ = await backend.apply_dedup_candidate(
            _dedup_apply_request(
                candidate_response,
                candidate,
                operation="reject",
                dry_run_token=candidate.reject_dry_run_token,
                graph_snapshot_hash="stale",
            ),
        )


@pytest.mark.anyio
async def test_dedup_reports_unavailable_when_sdk_lacks_capability() -> None:
    # Given: an SDK client without deduplication methods.
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MissingCapabilityClientFactory(),
        graph_store=RecordingGraphStore(),
    )

    # When / Then: dedup stats report capability absence instead of faking success.
    with pytest.raises(
        BackendCapabilityUnavailable,
        match="SDK deduplication is unavailable",
    ):
        _ = await backend.get_dedup_stats(DedupStatsRequest(scope=_scope()))


@pytest.mark.anyio
async def test_consolidation_dry_run_redacts_report_and_issues_token() -> None:
    # Given: the SDK consolidation report contains accidental unsafe fields.
    consolidation = RecordingConsolidationMemory(
        reports={
            "dedupe_entities": {
                "kind": "dedupe_entities",
                "dry_run": True,
                "candidates": [
                    {
                        "kind": "duplicate_entity",
                        "description": "Cartman duplicate",
                        "payload": {
                            "api_key": "sk-1234567890abcdef",
                            "cypher": "MATCH (n) RETURN n",
                        },
                    },
                ],
                "raw_prompt": "hidden reasoning",
            },
        },
    )
    fake_client = RecordingMemoryClient(consolidation=consolidation)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )

    # When: an operator requests a read-only consolidation dry-run.
    response = await backend.dry_run_consolidation(
        ConsolidationDryRunRequest(
            scope=_scope(),
            operation="dedupe_entities",
            similarity_threshold=0.91,
            max_pairs=25,
        ),
    )

    # Then: the SDK receives dry_run=true and unsafe report fields are removed.
    assert response.scope == _scope()
    assert response.operation == "dedupe_entities"
    assert response.dry_run is True
    assert response.report == {
        "kind": "dedupe_entities",
        "dry_run": True,
        "candidates": [
            {
                "kind": "duplicate_entity",
                "description": "Cartman duplicate",
                "payload": {"api_key": "[REDACTED]"},
            },
        ],
    }
    assert response.graph_snapshot_hash != ""
    assert response.dry_run_token != ""
    assert consolidation.calls == [
        ConsolidationCall(
            operation="dedupe_entities",
            dry_run=True,
            params={"similarity_threshold": 0.91, "max_pairs": 25},
        ),
    ]


@pytest.mark.anyio
async def test_consolidation_apply_requires_current_dry_run_and_is_idempotent() -> None:
    # Given: a consolidation dry-run has issued an operation-bound token.
    consolidation = RecordingConsolidationMemory(
        reports={
            "summarize_long_traces": {
                "kind": "summarize_long_traces",
                "dry_run": True,
                "candidates": [{"kind": "trace_summary", "description": "long trace"}],
            },
        },
    )
    fake_client = RecordingMemoryClient(consolidation=consolidation)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )
    dry_run = await backend.dry_run_consolidation(
        ConsolidationDryRunRequest(
            scope=_scope(),
            operation="summarize_long_traces",
            min_steps=4,
            max_traces=9,
        ),
    )
    request = _consolidation_apply_request(dry_run, min_steps=4, max_traces=9)

    # When: the same apply request is replayed with the same idempotency key.
    first = await backend.apply_consolidation(request)
    second = await backend.apply_consolidation(request)

    # Then: the SDK mutating operation runs once and the saved response is replayed.
    assert first == second
    assert first.applied is True
    assert first.audit == request.audit
    assert consolidation.calls == [
        ConsolidationCall(
            operation="summarize_long_traces",
            dry_run=True,
            params={"min_steps": 4, "max_traces": 9},
        ),
        ConsolidationCall(
            operation="summarize_long_traces",
            dry_run=False,
            params={"min_steps": 4, "max_traces": 9},
        ),
    ]


@pytest.mark.anyio
async def test_consolidation_apply_rejects_invalid_apply_requests() -> None:
    # Given: a dry-run token is tied to one operation and parameter set.
    consolidation = RecordingConsolidationMemory()
    fake_client = RecordingMemoryClient(consolidation=consolidation)
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MemoryClientFactory(fake_client),
        graph_store=RecordingGraphStore(),
    )
    dry_run = await backend.dry_run_consolidation(
        ConsolidationDryRunRequest(
            scope=_scope(),
            operation="archive_expired_conversations",
            ttl_days=30,
        ),
    )

    # When / Then: explicit apply, valid token, and unchanged params are required.
    with pytest.raises(BackendRequestError, match="apply=true"):
        _ = await backend.apply_consolidation(
            _consolidation_apply_request(dry_run, apply=False, ttl_days=30),
        )
    with pytest.raises(BackendRequestError, match="dry-run token"):
        _ = await backend.apply_consolidation(
            _consolidation_apply_request(
                dry_run,
                dry_run_token="wrong-token",  # noqa: S106
                ttl_days=30,
            ),
        )
    with pytest.raises(BackendRequestError, match="stale"):
        _ = await backend.apply_consolidation(
            _consolidation_apply_request(dry_run, ttl_days=31),
        )


@pytest.mark.anyio
async def test_consolidation_reports_unavailable_when_sdk_lacks_capability() -> None:
    # Given: an SDK client without the consolidation capability.
    backend = Neo4jAgentMemoryBackend(
        Settings(),
        memory_client_factory=MissingCapabilityClientFactory(),
        graph_store=RecordingGraphStore(),
    )

    # When / Then: consolidation dry-run reports capability absence.
    with pytest.raises(
        BackendCapabilityUnavailable,
        match="SDK consolidation is unavailable",
    ):
        _ = await backend.dry_run_consolidation(
            ConsolidationDryRunRequest(
                scope=_scope(),
                operation="dedupe_entities",
            ),
        )


@dataclass(slots=True)
class RecordingBackend:
    events: list[ClientEvent] = field(default_factory=list)

    async def add_message(self, request: MessageWriteRequest) -> MessageWriteResponse:
        _ = request
        return MessageWriteResponse(accepted=True)

    async def preview_extraction(
        self,
        request: ExtractionPreviewRequest,
    ) -> ExtractionPreviewResponse:
        _ = request
        return ExtractionPreviewResponse(
            candidates=[
                ExtractionCandidate(
                    kind="text_chunk",
                    text="preview",
                    source_id="message.content",
                    confidence=1.0,
                ),
            ],
            metrics=ExtractionPreviewMetrics(
                documents=1,
                chunks=1,
                ocr_images=0,
                rustfs_objects=0,
                batch_size=25,
                max_concurrency=1,
            ),
            provenance=ExtractionPreviewProvenance(
                source_ids=["message.content"],
            ),
            extract_entities=False,
            extract_relations=False,
        )

    async def get_context(self, request: ContextRequest) -> ContextResponse:
        _ = request
        return ContextResponse(context="")

    async def get_memory_context(
        self,
        request: MemoryContextRequest,
    ) -> MemoryContextResponse:
        _ = request
        return MemoryContextResponse()

    async def add_memories(self, request: MemoryAddRequest) -> MemoryAddResponse:
        _ = request
        return MemoryAddResponse()

    async def search_memories(
        self,
        request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        _ = request
        return MemorySearchResponse()

    async def list_memories(self, request: MemoryListRequest) -> MemoryListResponse:
        return MemoryListResponse(
            total=0,
            page=request.page,
            page_size=request.page_size,
        )

    async def update_memory(
        self,
        memory_id: str,
        request: MemoryUpdateRequest,
    ) -> MemoryUpdateResponse:
        _ = request
        return MemoryUpdateResponse(memory_id=memory_id, content="updated")

    async def delete_memory(
        self,
        memory_id: str,
        request: MemoryDeleteRequest,
    ) -> MemoryDeleteResponse:
        _ = request
        return MemoryDeleteResponse(memory_id=memory_id)

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        self.events.append(event)
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

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
        _ = request
        return GraphContextResponse(context="fake graph context")

    async def get_sdk_stats(self, request: SdkStatsRequest) -> SdkStatsResponse:
        return SdkStatsResponse(scope=request.scope, stats={"nodes": 0})

    async def get_dedup_stats(
        self,
        request: DedupStatsRequest,
    ) -> DedupStatsResponse:
        return DedupStatsResponse(scope=request.scope, stats={})

    async def find_dedup_candidates(
        self,
        request: DedupCandidateRequest,
    ) -> DedupCandidateResponse:
        return DedupCandidateResponse(
            scope=request.scope,
            graph_snapshot_hash="snapshot-placeholder",
            expires_at="2026-06-29T00:15:00+00:00",
        )

    async def apply_dedup_candidate(
        self,
        request: DedupApplyRequest,
    ) -> DedupApplyResponse:
        return DedupApplyResponse(
            scope=request.scope,
            operation=request.operation,
            candidate_id=request.candidate_id,
            candidate_version=request.candidate_version,
            applied=True,
            audit=request.audit,
        )

    async def dry_run_consolidation(
        self,
        request: ConsolidationDryRunRequest,
    ) -> ConsolidationDryRunResponse:
        return ConsolidationDryRunResponse(
            scope=request.scope,
            operation=request.operation,
            dry_run=True,
            graph_snapshot_hash="snapshot-placeholder",
            dry_run_token="consolidation-token",  # noqa: S106
            expires_at="2026-06-29T00:15:00+00:00",
        )

    async def apply_consolidation(
        self,
        request: ConsolidationApplyRequest,
    ) -> ConsolidationApplyResponse:
        return ConsolidationApplyResponse(
            scope=request.scope,
            operation=request.operation,
            applied=True,
            audit=request.audit,
        )

    async def export_graph(self, request: GraphExportRequest) -> GraphExportResponse:
        return GraphExportResponse(scope=request.scope)

    async def search_entities(
        self,
        request: EntitySearchRequest,
    ) -> EntitySearchResponse:
        _ = request
        return EntitySearchResponse()

    async def search_facts(self, request: FactSearchRequest) -> FactSearchResponse:
        _ = request
        return FactSearchResponse()

    async def search_preferences(
        self,
        request: PreferenceSearchRequest,
    ) -> PreferenceSearchResponse:
        _ = request
        return PreferenceSearchResponse()

    async def add_entity(self, request: EntityWriteRequest) -> EntityRecord:
        return EntityRecord(name=request.name, type=request.type)

    async def add_fact(self, request: FactWriteRequest) -> FactRecord:
        return FactRecord(
            subject=request.subject,
            predicate=request.predicate,
            object=request.object,
        )

    async def add_preference(
        self,
        request: PreferenceWriteRequest,
    ) -> PreferenceRecord:
        return PreferenceRecord(
            category=request.category,
            preference=request.preference,
        )

    async def list_skills(self, request: SkillListRequest) -> SkillListResponse:
        _ = request
        return SkillListResponse()

    async def propose_skill(self, proposal: SkillProposal) -> SkillProposal:
        return proposal

    async def record_skill_usage(self, usage: SkillUsage) -> EventIngestResult:
        return EventIngestResult(
            event_id=usage.skill_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def start_reasoning_trace(
        self,
        request: ReasoningTraceStartRequest,
    ) -> ReasoningTraceStartResponse:
        _ = request
        return ReasoningTraceStartResponse(
            trace_id="trace-placeholder",
            session_id="session-placeholder",
            task="task-placeholder",
        )

    async def add_reasoning_step(
        self,
        request: ReasoningStepRequest,
    ) -> ReasoningStepResponse:
        _ = request
        return ReasoningStepResponse(
            step_id="step-placeholder",
            trace_id="trace-placeholder",
            step_number=1,
        )

    async def record_reasoning_tool_call(
        self,
        request: ReasoningToolCallRequest,
    ) -> ReasoningToolCallResponse:
        _ = request
        return ReasoningToolCallResponse(
            tool_call_id="tool-call-placeholder",
            trace_id="trace-placeholder",
            step_id="step-placeholder",
        )

    async def complete_reasoning_trace(
        self,
        request: ReasoningTraceCompleteRequest,
    ) -> ReasoningTraceCompleteResponse:
        _ = request
        return ReasoningTraceCompleteResponse(trace_id="trace-placeholder")

    async def get_reasoning_context(
        self,
        request: ReasoningContextRequest,
    ) -> ReasoningContextResponse:
        _ = request
        return ReasoningContextResponse(context="reasoning context")

    async def list_reasoning_traces(
        self,
        request: ReasoningTraceListRequest,
    ) -> ReasoningTraceListResponse:
        return ReasoningTraceListResponse(scope=request.scope)

    async def get_reasoning_trace(
        self,
        request: ReasoningTraceDetailRequest,
    ) -> ReasoningTraceDetailResponse:
        return ReasoningTraceDetailResponse(scope=request.scope)

    async def find_similar_reasoning_traces(
        self,
        request: ReasoningSimilarTracesRequest,
    ) -> ReasoningSimilarTracesResponse:
        return ReasoningSimilarTracesResponse(scope=request.scope)

    async def search_reasoning_steps(
        self,
        request: ReasoningStepSearchRequest,
    ) -> ReasoningStepSearchResponse:
        return ReasoningStepSearchResponse(scope=request.scope)

    async def get_reasoning_tool_stats(
        self,
        request: ReasoningToolStatsRequest,
    ) -> ReasoningToolStatsResponse:
        return ReasoningToolStatsResponse(scope=request.scope)

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def buffer_status(self) -> BufferStatus:
        return BufferStatus(
            write_mode="sync",
            max_pending=200,
            pending_writes=None,
            write_errors=0,
            status="ready",
        )

    async def flush_buffer(self) -> BufferFlushResponse:
        return BufferFlushResponse(flushed=True, status=await self.buffer_status())

    async def shutdown(self) -> None:
        return None

    def diagnostics(self, readiness: BackendReadiness) -> DiagnosticsResponse:
        settings = Settings()
        return DiagnosticsResponse(
            tenant_id=settings.gnosis_tenant_id,
            config=DiagnosticsConfig(
                neo4j_uri=settings.neo4j_uri,
                neo4j_username=settings.neo4j_username,
                litellm_base_url=settings.litellm_base_url,
                gnosis_llm=settings.gnosis_llm,
                gnosis_embedding=settings.gnosis_embedding,
                gnosis_embedding_dimensions=settings.gnosis_embedding_dimensions,
                gnosis_audit_read=settings.gnosis_audit_read,
                gnosis_conversation_ttl_days=settings.gnosis_conversation_ttl_days,
                gnosis_write_mode=settings.gnosis_write_mode,
                gnosis_max_pending=settings.gnosis_max_pending,
                gnosis_fact_deduplication_enabled=(
                    settings.gnosis_fact_deduplication_enabled
                ),
                gnosis_trace_embedding_enabled=settings.gnosis_trace_embedding_enabled,
                gnosis_extract_entities_enabled=(
                    settings.gnosis_extract_entities_enabled
                ),
                gnosis_extract_relations_enabled=(
                    settings.gnosis_extract_relations_enabled
                ),
                gnosis_extraction_preview_enabled=(
                    settings.gnosis_extraction_preview_enabled
                ),
                gnosis_extraction_batch_size=settings.gnosis_extraction_batch_size,
                gnosis_extraction_max_concurrency=(
                    settings.gnosis_extraction_max_concurrency
                ),
                gnosis_extraction_chunk_size=settings.gnosis_extraction_chunk_size,
                gnosis_extraction_chunk_overlap=(
                    settings.gnosis_extraction_chunk_overlap
                ),
                gnosis_fact_extraction_enabled=(
                    settings.gnosis_fact_extraction_enabled
                ),
                gnosis_fact_extraction_model=settings.gnosis_fact_extraction_model,
                gnosis_fact_extraction_context_turns=(
                    settings.gnosis_fact_extraction_context_turns
                ),
                gnosis_fact_extraction_mode=settings.gnosis_fact_extraction_mode,
                gnosis_fact_extraction_max_concurrency=(
                    settings.gnosis_fact_extraction_max_concurrency
                ),
                gnosis_fact_extraction_max_pending=(
                    settings.gnosis_fact_extraction_max_pending
                ),
                gnosis_ocr_enabled=settings.gnosis_ocr_enabled,
                gnosis_ocr_model=settings.gnosis_ocr_model,
                gnosis_ocr_max_image_bytes=settings.gnosis_ocr_max_image_bytes,
                gnosis_rustfs_enabled=settings.gnosis_rustfs_enabled,
                gnosis_rustfs_bucket=settings.gnosis_rustfs_bucket,
                gnosis_rustfs_prefix=settings.gnosis_rustfs_prefix,
                gnosis_rustfs_endpoint=settings.gnosis_rustfs_endpoint,
                gnosis_rustfs_retention_days=settings.gnosis_rustfs_retention_days,
                gnosis_prompt_entities_enabled=settings.gnosis_prompt_entities_enabled,
                gnosis_prompt_preferences_enabled=(
                    settings.gnosis_prompt_preferences_enabled
                ),
                gnosis_prompt_reasoning_enabled=settings.gnosis_prompt_reasoning_enabled,
                gnosis_consolidation_schedule_enabled=(
                    settings.gnosis_consolidation_schedule_enabled
                ),
            ),
            backend=readiness,
        )


@dataclass(frozen=True, slots=True)
class LongTermFactWrite:
    subject: str
    predicate: str
    obj: str
    metadata: JsonObject
    generate_embedding: bool


@dataclass(frozen=True, slots=True)
class ShortTermMessageWrite:
    extract_entities: bool
    extract_relations: bool


@dataclass(frozen=True, slots=True)
class DedupReviewCall:
    source_id: UUID
    target_id: UUID
    confirm: bool


@dataclass(frozen=True, slots=True)
class DedupMergeCall:
    source_id: UUID
    target_id: UUID


@dataclass(frozen=True, slots=True)
class ConsolidationCall:
    operation: str
    dry_run: bool
    params: JsonObject


@dataclass(slots=True)
class RecordingConsolidationMemory:
    reports: dict[str, JsonObject] = field(default_factory=dict)
    calls: list[ConsolidationCall] = field(default_factory=list)

    async def archive_expired_conversations(
        self,
        *,
        ttl_days: int | None = None,
        dry_run: bool = True,
    ) -> JsonObject:
        self.calls.append(
            ConsolidationCall(
                operation="archive_expired_conversations",
                dry_run=dry_run,
                params={"ttl_days": ttl_days},
            ),
        )
        return self.reports.get(
            "archive_expired_conversations",
            {"kind": "archive_expired_conversations", "dry_run": dry_run},
        )

    async def dedupe_entities(
        self,
        *,
        similarity_threshold: float = 0.95,
        max_pairs: int = 10000,
        dry_run: bool = True,
    ) -> JsonObject:
        self.calls.append(
            ConsolidationCall(
                operation="dedupe_entities",
                dry_run=dry_run,
                params={
                    "similarity_threshold": similarity_threshold,
                    "max_pairs": max_pairs,
                },
            ),
        )
        return self.reports.get(
            "dedupe_entities",
            {"kind": "dedupe_entities", "dry_run": dry_run},
        )

    async def detect_superseded_preferences(
        self,
        *,
        user_identifier: str | None = None,
        similarity_threshold: float = 0.92,
        dry_run: bool = True,
    ) -> JsonObject:
        self.calls.append(
            ConsolidationCall(
                operation="detect_superseded_preferences",
                dry_run=dry_run,
                params={
                    "user_identifier": user_identifier,
                    "similarity_threshold": similarity_threshold,
                },
            ),
        )
        return self.reports.get(
            "detect_superseded_preferences",
            {"kind": "detect_superseded_preferences", "dry_run": dry_run},
        )

    async def summarize_long_traces(
        self,
        *,
        min_steps: int = 20,
        max_traces: int = 1000,
        dry_run: bool = True,
    ) -> JsonObject:
        self.calls.append(
            ConsolidationCall(
                operation="summarize_long_traces",
                dry_run=dry_run,
                params={"min_steps": min_steps, "max_traces": max_traces},
            ),
        )
        return self.reports.get(
            "summarize_long_traces",
            {"kind": "summarize_long_traces", "dry_run": dry_run},
        )


@dataclass(slots=True)
class MissingDedupLongTermMemory:
    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[EntityRecord]:
        _ = (query, entity_types, limit, threshold)
        return []

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[FactRecord]:
        _ = (query, limit, threshold)
        return []

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[PreferenceRecord]:
        _ = (query, category, limit, threshold)
        return []

    async def add_entity(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        name: str,
        entity_type: EntityType | str,
        *,
        subtype: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: JsonObject | None = None,
        resolve: bool = True,
        generate_embedding: bool = True,
        deduplicate: bool = True,
        geocode: bool = True,
        enrich: bool = True,
        coordinates: tuple[float, float] | None = None,
        metadata: JsonObject | None = None,
    ) -> EntityRecord:
        _ = (resolve, generate_embedding, deduplicate, geocode, enrich, coordinates)
        return EntityRecord(
            name=name,
            type=str(entity_type),
            subtype=subtype,
            description=description,
            aliases=aliases or [],
            attributes=attributes or {},
            metadata=metadata or {},
        )

    async def add_fact(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
    ) -> FactRecord:
        _ = (valid_from, valid_until, generate_embedding)
        return FactRecord(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            metadata=metadata or {},
        )

    async def add_preference(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        category: str,
        preference: str,
        *,
        context: str | None = None,
        confidence: float = 1.0,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
        user_identifier: str | None = None,
        applies_to: object | None = None,
    ) -> PreferenceRecord:
        _ = (generate_embedding, applies_to)
        return PreferenceRecord(
            category=category,
            preference=preference,
            context=context,
            confidence=confidence,
            user_identifier=user_identifier,
            metadata=metadata or {},
        )

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = (query, max_items)
        return ""


@dataclass(slots=True)
class RecordingLongTermMemory:
    entities: list[EntityRecord] = field(default_factory=list)
    facts: list[LongTermFactWrite] = field(default_factory=list)
    fact_records: list[FactRecord] = field(default_factory=list)
    preferences: list[PreferenceRecord] = field(default_factory=list)
    message_links: list[EntityMessageLinkRequest] = field(default_factory=list)
    extractor_links: list[str] = field(default_factory=list)
    failed_writes_remaining: int = 0
    context: str = ""
    context_queries: list[str] = field(default_factory=list)
    dedup_stats: JsonObject = field(default_factory=dict)
    dedup_stats_calls: int = 0
    duplicate_candidates: list[tuple[EntityRecord, EntityRecord, float]] = field(
        default_factory=list,
    )
    duplicate_limits: list[int] = field(default_factory=list)
    review_calls: list[DedupReviewCall] = field(default_factory=list)
    merge_calls: list[DedupMergeCall] = field(default_factory=list)

    async def get_deduplication_stats(self) -> JsonObject:
        self.dedup_stats_calls += 1
        return self.dedup_stats

    async def find_potential_duplicates(
        self,
        *,
        limit: int = 100,
    ) -> list[tuple[EntityRecord, EntityRecord, float]]:
        self.duplicate_limits.append(limit)
        return self.duplicate_candidates[:limit]

    async def review_duplicate(
        self,
        source_id: UUID,
        target_id: UUID,
        *,
        confirm: bool,
    ) -> bool:
        self.review_calls.append(
            DedupReviewCall(
                source_id=source_id,
                target_id=target_id,
                confirm=confirm,
            ),
        )
        return True

    async def merge_duplicate_entities(
        self,
        source_id: UUID,
        target_id: UUID,
    ) -> tuple[EntityRecord, EntityRecord] | None:
        self.merge_calls.append(
            DedupMergeCall(source_id=source_id, target_id=target_id),
        )
        candidates = self.duplicate_candidates
        for candidate_source, candidate_target, _similarity in candidates:
            if candidate_source.id == str(source_id) and candidate_target.id == str(
                target_id
            ):
                return (candidate_source, candidate_target)
        return None

    async def search_entities(
        self,
        query: str,
        *,
        entity_types: list[EntityType | str] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[EntityRecord]:
        _ = (query, entity_types, threshold)
        return self.entities[:limit]

    async def search_facts(
        self,
        query: str,
        *,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[FactRecord]:
        _ = (query, threshold)
        return self.fact_records[:limit]

    async def search_preferences(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[PreferenceRecord]:
        _ = (query, threshold)
        return [
            preference
            for preference in self.preferences
            if category is None or preference.category == category
        ][:limit]

    async def add_entity(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        name: str,
        entity_type: EntityType | str,
        *,
        subtype: str | None = None,
        description: str | None = None,
        aliases: list[str] | None = None,
        attributes: JsonObject | None = None,
        resolve: bool = True,
        generate_embedding: bool = True,
        deduplicate: bool = True,
        geocode: bool = True,
        enrich: bool = True,
        coordinates: tuple[float, float] | None = None,
        metadata: JsonObject | None = None,
    ) -> EntityRecord:
        _ = (resolve, generate_embedding, deduplicate, geocode, enrich, coordinates)
        entity = EntityRecord(
            id=f"entity-{len(self.entities) + 1}",
            name=name,
            type=str(entity_type),
            subtype=subtype,
            description=description,
            aliases=aliases or [],
            attributes=attributes or {},
            metadata=metadata or {},
        )
        self.entities.append(entity)
        return entity

    async def add_fact(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
    ) -> FactRecord:
        _ = (valid_from, valid_until)
        if self.failed_writes_remaining > 0:
            self.failed_writes_remaining -= 1
            raise PromotionFailureError
        fact_metadata = metadata or {}
        self.facts.append(
            LongTermFactWrite(
                subject=subject,
                predicate=predicate,
                obj=obj,
                metadata=fact_metadata,
                generate_embedding=generate_embedding,
            ),
        )
        fact = FactRecord(
            id=f"fact-{len(self.fact_records) + 1}",
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            metadata=fact_metadata,
        )
        self.fact_records.append(fact)
        return fact

    async def add_preference(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        category: str,
        preference: str,
        *,
        context: str | None = None,
        confidence: float = 1.0,
        generate_embedding: bool = True,
        metadata: JsonObject | None = None,
        user_identifier: str | None = None,
        applies_to: object | None = None,
    ) -> PreferenceRecord:
        _ = (generate_embedding, applies_to)
        preference_record = PreferenceRecord(
            id=f"preference-{len(self.preferences) + 1}",
            category=category,
            preference=preference,
            context=context,
            confidence=confidence,
            user_identifier=user_identifier,
            metadata=metadata or {},
        )
        self.preferences.append(preference_record)
        return preference_record

    async def get_preferences_for(
        self,
        user_identifier: str,
        *,
        applies_to: object | None = None,
        active_only: bool = True,
        as_of: datetime | None = None,
    ) -> list[PreferenceRecord]:
        _ = (applies_to, active_only, as_of)
        return [
            preference
            for preference in self.preferences
            if preference.user_identifier == user_identifier
        ]

    async def get_facts_about(
        self,
        subject: str,
        *,
        limit: int = 100,
    ) -> list[FactRecord]:
        return [fact for fact in self.fact_records if fact.subject == subject][:limit]

    async def link_entity_to_message(  # noqa: PLR0913 - Mirrors SDK API.
        self,
        entity: EntityRecord | UUID,
        message_id: UUID | str,
        *,
        confidence: float = 1.0,
        start_pos: int | None = None,
        end_pos: int | None = None,
        context: str | None = None,
    ) -> bool:
        _ = (start_pos, end_pos)
        self.message_links.append(
            EntityMessageLinkRequest(
                scope=_scope(),
                entity_id=str(entity),
                message_id=str(message_id),
                confidence=confidence,
                context=context,
            ),
        )
        return True

    async def link_entity_to_extractor(
        self,
        entity: EntityRecord | UUID,
        extractor_name: str,
        *,
        confidence: float = 1.0,
        extraction_time_ms: float | None = None,
    ) -> bool:
        _ = (entity, confidence, extraction_time_ms)
        self.extractor_links.append(extractor_name)
        return True

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = max_items
        self.context_queries.append(query)
        return self.context


class PromotionFailureError(Exception):
    pass


@dataclass(slots=True)
class RecordingShortTermMemory:
    context: str = ""
    context_error: ValueError | ValidationError | None = None
    messages: list[ShortTermMessageWrite] = field(default_factory=list)
    context_queries: list[str] = field(default_factory=list)

    async def add_message(  # noqa: PLR0913
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        user_identifier: str,
        metadata: dict[str, str],
        extract_entities: bool,
        extract_relations: bool,
    ) -> None:
        _ = (session_id, role, content, user_identifier, metadata)
        self.messages.append(
            ShortTermMessageWrite(
                extract_entities=extract_entities,
                extract_relations=extract_relations,
            ),
        )

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str:
        _ = (session_id, max_messages, metadata_filters)
        self.context_queries.append(query)
        if self.context_error is not None:
            raise self.context_error
        return self.context


@dataclass(slots=True)
class RecordingReasoningMemory:
    context: str = ""
    context_queries: list[str] = field(default_factory=list)
    traces: list[SdkReasoningTrace] = field(default_factory=list)
    steps: list[ReasoningStepWithContext] = field(default_factory=list)
    tool_stats: list[ToolStats] = field(default_factory=list)
    list_trace_calls: list[dict[str, object]] = field(default_factory=list)
    detail_trace_ids: list[UUID | str] = field(default_factory=list)
    detail_with_steps_trace_ids: list[UUID] = field(default_factory=list)
    similar_trace_calls: list[dict[str, object]] = field(default_factory=list)
    step_search_calls: list[dict[str, object]] = field(default_factory=list)
    tool_stats_names: list[str | None] = field(default_factory=list)
    trace_start_metadata: list[JsonObject | None] = field(default_factory=list)
    step_metadata: list[JsonObject | None] = field(default_factory=list)

    async def get_context(self, query: str, *, max_traces: int) -> str:
        _ = max_traces
        self.context_queries.append(query)
        return self.context

    async def start_trace(  # noqa: PLR0913
        self,
        session_id: str,
        task: str,
        *,
        generate_embedding: bool,
        metadata: JsonObject | None,
        triggered_by_message_id: str | None,
        user_identifier: str,
    ) -> SdkReasoningTrace:
        _ = (generate_embedding, triggered_by_message_id, user_identifier)
        self.trace_start_metadata.append(metadata)
        return SdkReasoningTrace(
            created_at=datetime(2026, 6, 29, tzinfo=UTC),
            started_at=datetime(2026, 6, 29, tzinfo=UTC),
            session_id=session_id,
            task=task,
        )

    async def add_step(  # noqa: PLR0913
        self,
        trace_id: UUID,
        *,
        thought: None,
        action: str | None,
        observation: str | None,
        generate_embedding: bool,
        metadata: JsonObject | None,
    ) -> SdkReasoningStep:
        _ = (thought, action, observation, generate_embedding)
        self.step_metadata.append(metadata)
        return SdkReasoningStep(
            created_at=datetime(2026, 6, 29, tzinfo=UTC),
            trace_id=trace_id,
            step_number=1,
        )

    async def record_tool_call(  # noqa: PLR0913
        self,
        step_id: UUID,
        tool_name: str,
        arguments: JsonObject,
        *,
        result: JsonValue | None,
        status: ToolCallStatus,
        duration_ms: int | None,
        error: str | None,
        message_id: str | None,
        touched_entities: list[EntityRef],
    ) -> ToolCall:
        _ = (result, message_id, touched_entities)
        return ToolCall(
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            duration_ms=duration_ms,
            error=error,
        )

    async def complete_trace(
        self,
        trace_id: UUID,
        *,
        outcome: str | None,
        success: bool | None,
        generate_step_embeddings: bool,
    ) -> SdkReasoningTrace:
        _ = generate_step_embeddings
        return SdkReasoningTrace(
            id=trace_id,
            session_id="session-placeholder",
            task="task-placeholder",
            outcome=outcome,
            success=success,
        )

    async def list_traces(  # noqa: PLR0913 - Mirrors neo4j-agent-memory SDK API.
        self,
        *,
        session_id: str | None = None,
        success_only: bool | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "started_at",
        order_dir: str = "desc",
    ) -> list[SdkReasoningTrace]:
        self.list_trace_calls.append(
            {
                "session_id": session_id,
                "success_only": success_only,
                "since": since,
                "until": until,
                "limit": limit,
                "offset": offset,
                "order_by": order_by,
                "order_dir": order_dir,
            },
        )
        return self.traces[offset : offset + limit]

    async def get_trace(self, trace_id: UUID | str) -> SdkReasoningTrace | None:
        self.detail_trace_ids.append(trace_id)
        return _trace_by_id(self.traces, trace_id)

    async def get_trace_with_steps(self, trace_id: UUID) -> SdkReasoningTrace | None:
        self.detail_with_steps_trace_ids.append(trace_id)
        return _trace_by_id(self.traces, trace_id)

    async def get_similar_traces(
        self,
        task: str,
        *,
        limit: int = 5,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[SdkReasoningTrace]:
        self.similar_trace_calls.append(
            {
                "task": task,
                "limit": limit,
                "success_only": success_only,
                "threshold": threshold,
            },
        )
        return self.traces[:limit]

    async def search_steps(
        self,
        query: str,
        *,
        limit: int = 10,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[object]:
        self.step_search_calls.append(
            {
                "query": query,
                "limit": limit,
                "success_only": success_only,
                "threshold": threshold,
            },
        )
        return list(self.steps[:limit])

    async def get_tool_stats(self, tool_name: str | None = None) -> list[ToolStats]:
        self.tool_stats_names.append(tool_name)
        if tool_name is None:
            return self.tool_stats
        return [
            tool_stat for tool_stat in self.tool_stats if tool_stat.name == tool_name
        ]


@dataclass(frozen=True, slots=True)
class MissingReasoningReadMemory:
    async def get_context(self, query: str, *, max_traces: int) -> str:
        _ = (query, max_traces)
        return "reasoning context"

    async def start_trace(  # noqa: PLR0913
        self,
        session_id: str,
        task: str,
        *,
        generate_embedding: bool,
        metadata: JsonObject | None,
        triggered_by_message_id: str | None,
        user_identifier: str,
    ) -> SdkReasoningTrace:
        _ = (generate_embedding, metadata, triggered_by_message_id, user_identifier)
        return SdkReasoningTrace(
            created_at=datetime(2026, 6, 29, tzinfo=UTC),
            started_at=datetime(2026, 6, 29, tzinfo=UTC),
            session_id=session_id,
            task=task,
        )

    async def add_step(self, trace_id: UUID, **kwargs: object) -> SdkReasoningStep:
        _ = kwargs
        return SdkReasoningStep(trace_id=trace_id, step_number=1)

    async def record_tool_call(
        self,
        step_id: UUID,
        tool_name: str,
        arguments: JsonObject,
        **kwargs: object,
    ) -> ToolCall:
        _ = kwargs
        return ToolCall(step_id=step_id, tool_name=tool_name, arguments=arguments)

    async def complete_trace(
        self,
        trace_id: UUID,
        **kwargs: object,
    ) -> SdkReasoningTrace:
        _ = kwargs
        return SdkReasoningTrace(
            id=trace_id,
            session_id=_scope().session_id,
            task="task",
        )


@dataclass(slots=True)
class RecordingCypherQuery:
    rows: list[JsonObject] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        _ = (query, params)
        return self.rows


@dataclass(frozen=True, slots=True)
class GraphCall:
    memory_types: list[str]
    session_id: str
    include_embeddings: bool
    limit: int


@dataclass(frozen=True, slots=True)
class RecordingSdkNode:
    id: str
    labels: list[str]
    properties: JsonObject


@dataclass(frozen=True, slots=True)
class RecordingSdkRelationship:
    id: str
    type: str
    from_node: str
    to_node: str
    properties: JsonObject


@dataclass(frozen=True, slots=True)
class RecordingSdkGraph:
    nodes: list[RecordingSdkNode] = field(default_factory=list)
    relationships: list[RecordingSdkRelationship] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class RecordingMemoryClient:
    short_term: RecordingShortTermMemory = field(
        default_factory=RecordingShortTermMemory,
    )
    long_term: RecordingLongTermMemory = field(default_factory=RecordingLongTermMemory)
    reasoning: RecordingReasoningMemory = field(
        default_factory=RecordingReasoningMemory,
    )
    query: RecordingCypherQuery = field(
        default_factory=RecordingCypherQuery,
    )
    stats: JsonObject = field(default_factory=dict)
    graph: RecordingSdkGraph = field(default_factory=RecordingSdkGraph)
    write_errors: list[JsonObject] = field(default_factory=list)
    consolidation: RecordingConsolidationMemory = field(
        default_factory=RecordingConsolidationMemory,
    )
    stats_calls: int = 0
    flush_calls: int = 0
    wait_for_pending_calls: int = 0
    graph_calls: list[GraphCall] = field(default_factory=list)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)

    async def get_stats(self) -> JsonObject:
        self.stats_calls += 1
        return self.stats

    async def flush(self) -> None:
        self.flush_calls += 1

    async def wait_for_pending(self) -> None:
        self.wait_for_pending_calls += 1

    async def get_graph(
        self,
        *,
        memory_types: list[str],
        session_id: str,
        include_embeddings: bool,
        limit: int,
    ) -> RecordingSdkGraph:
        self.graph_calls.append(
            GraphCall(
                memory_types=memory_types,
                session_id=session_id,
                include_embeddings=include_embeddings,
                limit=limit,
            ),
        )
        return self.graph


@dataclass(slots=True)
class MissingCapabilityMemoryClient:
    short_term: RecordingShortTermMemory = field(
        default_factory=RecordingShortTermMemory,
    )
    long_term: MissingDedupLongTermMemory = field(
        default_factory=MissingDedupLongTermMemory,
    )
    reasoning: MissingReasoningReadMemory = field(
        default_factory=MissingReasoningReadMemory,
    )
    query: RecordingCypherQuery = field(
        default_factory=RecordingCypherQuery,
    )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)


@dataclass(frozen=True, slots=True)
class MemoryClientFactory:
    client: RecordingMemoryClient

    def __call__(self, settings: MemorySettings) -> "MemoryClientContext":
        _ = settings
        return cast("MemoryClientContext", self.client)


@dataclass(frozen=True, slots=True)
class MissingCapabilityClientFactory:
    def __call__(self, settings: MemorySettings) -> "MemoryClientContext":
        _ = settings
        client = cast("object", MissingCapabilityMemoryClient())
        return cast("MemoryClientContext", client)


@dataclass(slots=True)
class CapturingMemoryClientFactory:
    client: RecordingMemoryClient
    settings: list[MemorySettings] = field(default_factory=list)

    def __call__(self, settings: MemorySettings) -> "MemoryClientContext":
        self.settings.append(settings)
        return cast("MemoryClientContext", self.client)


@dataclass(slots=True)
class RecordingGraphStore:
    context: str = ""
    context_requests: list[GraphContextRequest] = field(default_factory=list)

    async def require_available(self) -> None:
        return None

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        return EventIngestResult(
            event_id=event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        self.context_requests.append(request)
        return GraphContextResponse(context=self.context, facts=[{"kind": "graph"}])


@dataclass(frozen=True, slots=True)
class FailingDriverFactory:
    def __call__(self) -> "FailingDriver":
        return FailingDriver()


@dataclass(frozen=True, slots=True)
class FailingDriver:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)

    async def verify_connectivity(self) -> None:
        reason = "connection refused"
        raise OSError(reason)


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="nolgia",
        space_id="discord",
        agent_id="nolgia-agent",
        session_id="guild:123:channel:456",
        user_id="789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="123",
        channel_id="456",
    )


def _json_object(value: object) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(value)


def _scope_metadata(scope: MemoryScope) -> JsonObject:
    return _JSON_OBJECT_ADAPTER.validate_python(
        {
            "tenant_id": scope.tenant_id,
            "space_id": scope.space_id,
            "agent_id": scope.agent_id,
            "session_id": scope.session_id,
            "user_id": scope.user_id,
            "visibility": scope.visibility.value,
            "guild_id": scope.guild_id or "",
            "channel_id": scope.channel_id or "",
        }
    )


def _trace_by_id(
    traces: list[SdkReasoningTrace],
    trace_id: UUID | str,
) -> SdkReasoningTrace | None:
    trace_id_text = str(trace_id)
    for trace in traces:
        if str(trace.id) == trace_id_text:
            return trace
    return None


def _sdk_reasoning_trace(
    trace_id: UUID,
    *,
    metadata: JsonObject,
    task: str,
    steps: list[SdkReasoningStep] | None = None,
    success: bool | None = None,
) -> SdkReasoningTrace:
    return SdkReasoningTrace(
        id=trace_id,
        created_at=datetime(2026, 6, 29, tzinfo=UTC),
        session_id=_scope().session_id,
        task=task,
        steps=steps or [],
        success=success,
        started_at=datetime(2026, 6, 29, tzinfo=UTC),
        metadata=metadata,
    )


def _fact_row(
    *,
    subject: str,
    predicate: str,
    object_value: str,
    metadata: JsonObject,
) -> JsonObject:
    return {
        "id": subject,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "confidence": 1.0,
        "created_at": None,
        "metadata": json.dumps(metadata),
    }


def _message_write_request(
    *,
    extract_entities: bool | None = None,
    extract_relations: bool | None = None,
) -> MessageWriteRequest:
    return MessageWriteRequest(
        scope=_scope(),
        role=MessageRole.USER,
        content="remember this",
        extract_entities=extract_entities,
        extract_relations=extract_relations,
    )


def _ocr_reference() -> OcrImageReference:
    return OcrImageReference(
        source_id="image-1",
        media_type="image/png",
        size_bytes=512,
        checksum_sha256="image-checksum",
    )


def _rustfs_reference(
    *,
    object_key: str = "gnosis/image-1.png",
) -> RustFSSourceReference:
    return RustFSSourceReference(
        bucket="memory-private",
        object_key=object_key,
        content_type="image/png",
        size_bytes=512,
        checksum_sha256="image-checksum",
    )


def _dedup_entity(
    entity_id: str,
    *,
    name: str = "Cartman",
    metadata: JsonObject | None = None,
) -> EntityRecord:
    return EntityRecord(
        id=entity_id,
        name=name,
        type="PERSON",
        metadata=metadata or {},
    )


def _dedup_apply_request(  # noqa: PLR0913
    response: DedupCandidateResponse,
    candidate: DedupCandidate,
    *,
    operation: DedupOperationName,
    dry_run_token: str,
    apply: bool = True,
    graph_snapshot_hash: str | None = None,
    idempotency_key: str = "dedup-apply-1",
) -> DedupApplyRequest:
    return DedupApplyRequest(
        scope=response.scope,
        apply=apply,
        operation=operation,
        candidate_id=candidate.candidate_id,
        candidate_version=candidate.version,
        graph_snapshot_hash=graph_snapshot_hash or response.graph_snapshot_hash,
        dry_run_token=dry_run_token,
        idempotency_key=idempotency_key,
        audit=DedupOperatorAudit(operator_id="admin-1", reason="reviewed"),
    )


def _consolidation_apply_request(  # noqa: PLR0913
    response: ConsolidationDryRunResponse,
    *,
    apply: bool = True,
    dry_run_token: str | None = None,
    idempotency_key: str = "consolidation-apply-1",
    ttl_days: int | None = None,
    similarity_threshold: float | None = None,
    max_pairs: int | None = None,
    user_identifier: str | None = None,
    min_steps: int | None = None,
    max_traces: int | None = None,
) -> ConsolidationApplyRequest:
    return ConsolidationApplyRequest(
        scope=response.scope,
        apply=apply,
        operation=response.operation,
        graph_snapshot_hash=response.graph_snapshot_hash,
        dry_run_token=dry_run_token or response.dry_run_token,
        idempotency_key=idempotency_key,
        audit=DedupOperatorAudit(operator_id="admin-1", reason="reviewed"),
        ttl_days=ttl_days,
        similarity_threshold=similarity_threshold,
        max_pairs=max_pairs,
        user_identifier=user_identifier,
        min_steps=min_steps,
        max_traces=max_traces,
    )


def _client_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="nolgia",
        source_client=SourceClient.DISCORD,
        agent_id="nolgia-agent",
        event_id="discord-message-999",
        event_type=ClientEventType.MESSAGE_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="discord:message:message-999:create",
        scope=_scope(),
        actor=ClientEventActor(id="789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(id="message-999", type="message"),
        payload={"content": "remember this", "payload_version": 1},
        discord=DiscordEventContext(
            guild_id="123",
            channel_id="456",
            message_id="message-999",
        ),
    )
