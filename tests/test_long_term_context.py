import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Self, cast
from uuid import UUID

import pytest
from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.memory.long_term import EntityType
from neo4j_agent_memory.memory.reasoning import ReasoningStep as SdkReasoningStep
from neo4j_agent_memory.memory.reasoning import ReasoningTrace as SdkReasoningTrace
from neo4j_agent_memory.memory.reasoning import ToolCall, ToolCallStatus, ToolStats
from neo4j_agent_memory.schema.models import EntityRef
from openai import OpenAIError

from gnosis.backend import Neo4jAgentMemoryBackend
from gnosis.models import (
    BackendReadiness,
    EntityRecord,
    EventIngestResult,
    EventIngestStatus,
    FactRecord,
    GraphContextRequest,
    GraphContextResponse,
    JsonObject,
    JsonValue,
    MemoryContextRequest,
    MemoryContextSection,
    MemoryScope,
    MemorySearchRequest,
    MemoryVisibility,
    PreferenceRecord,
)
from gnosis.query_router import QueryRoute, RouteVerdict
from gnosis.reranker import RerankResult
from gnosis.settings import Settings
from gnosis.sufficiency import SufficiencyVerdict

if TYPE_CHECKING:
    from gnosis.backend import MemoryClientContext


@pytest.mark.anyio
async def test_combined_context_includes_scoped_facts_preferences_entities() -> None:
    # Given: one locally stored fact plus separate upstream long-term context.
    fact = _fact_row(
        subject="tenant:nolgia:message:one",
        predicate="discord.message_created",
        object_value="message one mentions the library schedule",
        metadata=_scope_metadata(_scope()) | {"event_id": "event-one"},
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(rows=[{"f": fact}]),
        long_term=RecordingLongTermMemory(
            context=(
                "### User Preferences\n- concise updates\n\n"
                "### Relevant Entities\n- library"
            ),
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_prompt_entities_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: combined memory context is requested for the matching scope.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=_scope(),
            query="what should I remember?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=5,
        ),
    )

    # Then: facts are formatted once and preferences/entities remain separate.
    assert response.sections == [
        MemoryContextSection(
            source="long_term_facts",
            content=(
                "### Long-Term Facts\n"
                "- tenant:nolgia:message:one discord.message_created: "
                "message one mentions the library schedule"
            ),
        ),
        MemoryContextSection(
            source="long_term_preferences_entities",
            content=(
                "### User Preferences\n- concise updates\n\n"
                "### Relevant Entities\n- library"
            ),
        ),
    ]
    assert response.sections[0].content.count("tenant:nolgia:message:one") == 1
    assert "tenant:nolgia:message:one" not in response.sections[1].content
    assert client.long_term.context_queries == ["what should I remember?"]
    assert client.query.cypher_calls[0].parameters == {
        "candidate_limit": 100,
        "metadata_fragments": [
            '"tenant_id": "nolgia"',
            '"agent_id": "nolgia-agent"',
            '"user_id": "789"',
            '"visibility": "channel"',
            '"guild_id": "123"',
            '"channel_id": "456"',
        ],
    }


@dataclass
class _ReversingReranker:
    """Reverses candidate order, to prove reranking wires into the read path."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[str],
    ) -> RerankResult | None:
        _ = query
        return RerankResult(order=list(reversed(range(len(candidates)))))


@pytest.mark.anyio
async def test_rerank_reorders_rendered_long_term_facts() -> None:
    # Given: three scoped facts retrieved in alpha, beta, gamma order; rerank on.
    scope = _scope()
    rows: list[JsonObject] = [
        {
            "f": _fact_row(
                subject=f"tenant:nolgia:message:{name}",
                predicate="fact",
                object_value=f"note {name}",
                metadata=_scope_metadata(scope),
                created_at="2026-06-28T00:00:00Z",
            ),
        }
        for name in ("alpha", "beta", "gamma")
    ]
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_rerank_enabled=True),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(query=RecordingQuery(rows=rows)),
        ),
        graph_store=RecordingGraphStore(),
        reranker=_ReversingReranker(),
    )

    # When: context is read with a query.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="which notes?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=5,
        ),
    )

    # Then: the rendered facts appear in reranked (reversed) order.
    content = response.sections[0].content
    assert (
        content.index("note gamma")
        < content.index("note beta")
        < content.index("note alpha")
    )


@pytest.mark.anyio
async def test_fact_context_does_not_cross_tenant_or_channel_scope() -> None:
    # Given: facts for the requested channel plus facts from other scopes.
    requested_scope = _scope()
    other_tenant_scope = _scope(tenant_id="other-tenant")
    other_channel_scope = _scope(channel_id="999")
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:visible",
                        predicate="discord.message_created",
                        object_value="visible channel note",
                        metadata=_scope_metadata(requested_scope),
                    ),
                },
                {
                    "f": _fact_row(
                        subject="tenant:other:message:hidden",
                        predicate="discord.message_created",
                        object_value="other tenant note",
                        metadata=_scope_metadata(other_tenant_scope),
                    ),
                },
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:wrong-channel",
                        predicate="discord.message_created",
                        object_value="wrong channel note",
                        metadata=_scope_metadata(other_channel_scope),
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: combined memory context is requested for the original channel.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=requested_scope,
            query="channel recall",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: only the same-tenant same-channel fact appears.
    assert len(response.sections) == 1
    content = response.sections[0].content
    assert "tenant:nolgia:message:visible" in content
    assert "tenant:other:message:hidden" not in content
    assert "tenant:nolgia:message:wrong-channel" not in content
    assert "other tenant note" not in content
    assert "wrong channel note" not in content


@pytest.mark.anyio
async def test_fact_context_dates_prefer_stored_session_date() -> None:
    # Given: one fact tagged with a stored session date and one without any tag.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:tagged",
                        predicate="said_user",
                        object_value="we went hiking yesterday",
                        metadata=(
                            _scope_metadata(scope) | {"session_date": "7 May 2023"}
                        ),
                        created_at="2026-06-27T01:02:03Z",
                    ),
                },
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:untagged",
                        predicate="said_assistant",
                        object_value="hiking sounds great",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T09:10:11.123000000Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: combined memory context is requested.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when did they go hiking?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: each fact is one dated line anchored on stored metadata first.
    content = response.sections[0].content
    assert "- [7 May 2023] we went hiking yesterday" in content
    assert "- [2026-06-28] hiking sounds great" in content
    assert "[2026-06-27]" not in content
    assert "subject:" not in content
    assert "provenance:" not in content


@pytest.mark.anyio
async def test_fact_context_spans_sessions_and_matches_search_item_count() -> None:
    # Given: the same five turn facts stored under five different sessions.
    scope = _scope()
    rows: list[JsonObject] = []
    search_results: list[FactRecord] = []
    for index in range(5):
        subject = f"tenant:nolgia:message:{index}"
        content_value = f"turn {index} of the conversation"
        metadata = _scope_metadata(scope) | {"session_id": f"session-{index}"}
        rows.append(
            {
                "f": _fact_row(
                    subject=subject,
                    predicate="said_user",
                    object_value=content_value,
                    metadata=metadata,
                    created_at=f"2023-05-0{index + 1}T00:00:00Z",
                ),
            },
        )
        search_results.append(
            FactRecord(
                id=subject,
                subject=subject,
                predicate="said_user",
                object=content_value,
                metadata=cast("JsonObject", dict(metadata)),
            ),
        )
    client = RecordingMemoryClient(
        query=RecordingQuery(rows=rows),
        long_term=RecordingLongTermMemory(search_results=search_results),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: search and context read the same data with the same item budget.
    search = await backend.search_memories(
        MemorySearchRequest(scope=scope, query="conversation", limit=5),
    )
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="conversation",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=5,
        ),
    )

    # Then: context renders exactly as many items as search returns.
    content = response.sections[0].content
    assert len(search.results) == 5
    assert content.count("\n- ") == len(search.results)


@pytest.mark.anyio
async def test_fact_context_truncates_to_max_items_after_scope_filter() -> None:
    # Given: more scoped candidate facts than the requested item budget.
    scope = _scope()
    rows: list[JsonObject] = [
        {
            "f": _fact_row(
                subject=f"tenant:nolgia:message:{index}",
                predicate="said_user",
                object_value=f"note {index}",
                metadata=_scope_metadata(scope),
                created_at=f"2023-05-0{index + 1}T00:00:00Z",
            ),
        }
        for index in range(3)
    ]
    client = RecordingMemoryClient(query=RecordingQuery(rows=rows))
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: combined memory context is requested with max_items below supply.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="notes",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=2,
        ),
    )

    # Then: the item budget bounds the rendered facts after scope filtering.
    assert response.sections[0].content.count("\n- ") == 2


@pytest.mark.anyio
async def test_fact_context_ranks_relevance_over_recency_when_query_present() -> None:
    # Given: similarity search ranks an old relevant fact above a recent decoy
    # while the recency read would surface the decoy first.
    scope = _scope()
    relevant_old = _fact_record(
        subject="tenant:nolgia:message:relevant",
        object_value="Maria adopted a golden retriever named Biscuit",
        metadata=_scope_metadata(scope) | {"session_date": "7 May 2023"},
    )
    decoy_recent = _fact_record(
        subject="tenant:nolgia:message:decoy",
        object_value="the weather was rainy this morning",
        metadata=_scope_metadata(scope) | {"session_date": "28 June 2026"},
    )
    off_tenant = _fact_record(
        subject="tenant:other:message:leak",
        object_value="other tenant secret",
        metadata=_scope_metadata(_scope(tenant_id="other-tenant")),
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:decoy",
                        predicate="said_user",
                        object_value="the weather was rainy this morning",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
        long_term=RecordingLongTermMemory(
            search_results=[off_tenant, relevant_old, decoy_recent],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is requested with a query and a one-item budget.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="what pet did Maria adopt?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=1,
        ),
    )

    # Then: the relevant old fact wins the budget over the recent decoy, the
    # off-scope candidate never renders, and the recency read is skipped.
    content = response.sections[0].content
    assert "- [7 May 2023] Maria adopted a golden retriever named Biscuit" in content
    assert "rainy" not in content
    assert "other tenant secret" not in content
    assert client.long_term.search_queries == ["what pet did Maria adopt?"]
    assert client.query.cypher_calls == []


@pytest.mark.anyio
async def test_fact_context_keeps_recency_order_without_similarity_ranking() -> None:
    # Given: similarity search has no embedder and returns nothing, while the
    # recency read yields scoped facts newest first.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:new",
                        predicate="said_user",
                        object_value="newest note",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:old",
                        predicate="said_user",
                        object_value="older note",
                        metadata=_scope_metadata(scope),
                        created_at="2023-05-07T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is requested with a query anyway.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="notes",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the recency fallback preserves newest-first ordering.
    content = response.sections[0].content
    assert content.index("newest note") < content.index("older note")


@pytest.mark.anyio
async def test_fact_context_renders_signal_predicates_with_subject() -> None:
    # Given: one conversational turn fact and one typed knowledge fact.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:turn",
                        predicate="said_user",
                        object_value="I moved to Lisbon last spring",
                        metadata=_scope_metadata(scope) | {"date": "7 May 2023"},
                    ),
                },
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:event",
                        predicate="discord.message_created",
                        object_value="message event: schedule posted",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: combined memory context is requested.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="where does the user live?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: turn facts render as bare dated content while typed facts keep
    # their subject and predicate, with no provenance block anywhere.
    content = response.sections[0].content
    assert "- [7 May 2023] I moved to Lisbon last spring" in content
    assert (
        "- [2026-06-28] tenant:nolgia:message:event discord.message_created: "
        "message event: schedule posted"
    ) in content
    assert "provenance:" not in content
    assert "subject:" not in content


def _extracted_fact(  # noqa: PLR0913 - Test builder mirrors the stored fact shape.
    *,
    memory_id: str,
    subject: str,
    object_value: str,
    entities: list[str],
    event_date: str,
    scope: MemoryScope,
) -> FactRecord:
    metadata: JsonObject = dict(_scope_metadata(scope))
    metadata["entities"] = cast("JsonValue", entities)
    metadata["event_date"] = event_date
    metadata["extracted"] = True
    return FactRecord(
        id=memory_id,
        subject=subject,
        predicate="fact",
        object=object_value,
        metadata=metadata,
    )


@dataclass(slots=True)
class RecordingSufficiencyAssessor:
    verdict: SufficiencyVerdict | Exception
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def assess(
        self,
        query: str,
        context: str,
    ) -> SufficiencyVerdict | None:
        self.calls.append((query, context))
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


@dataclass(slots=True)
class RecordingQueryRouter:
    verdict: RouteVerdict | Exception | None = None
    queries: list[str] = field(default_factory=list)

    async def classify(self, query: str) -> RouteVerdict | None:
        self.queries.append(query)
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return self.verdict


@pytest.mark.anyio
async def test_read_supersession_keeps_only_newest_same_slot_fact() -> None:
    # Given: two extracted facts in the same slot (same user + first entity),
    # the newer one carrying a later event_date.
    scope = _scope()
    older = _extracted_fact(
        memory_id="fact-old",
        subject="user:789",
        object_value="Maria's dog is Rex",
        entities=["Maria"],
        event_date="2024-01-01",
        scope=scope,
    )
    newer = _extracted_fact(
        memory_id="fact-new",
        subject="user:789",
        object_value="Maria's dog is Biscuit",
        entities=["Maria"],
        event_date="2024-06-01",
        scope=scope,
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(),
        long_term=RecordingLongTermMemory(search_results=[older, newer]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_read_supersession_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled with supersession enabled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="what is Maria's dog?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: only the newest fact in the slot survives.
    content = response.sections[0].content
    assert "Biscuit" in content
    assert "Rex" not in content


@pytest.mark.anyio
async def test_read_supersession_off_is_byte_identical() -> None:
    # Given: the same same-slot pair with the flag left at its default.
    scope = _scope()
    older = _extracted_fact(
        memory_id="fact-old",
        subject="user:789",
        object_value="Maria's dog is Rex",
        entities=["Maria"],
        event_date="2024-01-01",
        scope=scope,
    )
    newer = _extracted_fact(
        memory_id="fact-new",
        subject="user:789",
        object_value="Maria's dog is Biscuit",
        entities=["Maria"],
        event_date="2024-06-01",
        scope=scope,
    )

    off = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[older, newer]),
            ),
        ),
        graph_store=RecordingGraphStore(),
    )
    request = MemoryContextRequest(
        scope=scope,
        query="what is Maria's dog?",
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
    )

    # When/Then: both older and newer render when supersession is off.
    response = await off.get_memory_context(request)
    content = response.sections[0].content
    assert "Biscuit" in content
    assert "Rex" in content


@pytest.mark.anyio
async def test_read_supersession_keeps_distinct_slot_facts() -> None:
    # Given: two extracted facts about different entities (different slots).
    scope = _scope()
    dog = _extracted_fact(
        memory_id="fact-dog",
        subject="user:789",
        object_value="Maria's dog is Biscuit",
        entities=["Maria"],
        event_date="2024-06-01",
        scope=scope,
    )
    city = _extracted_fact(
        memory_id="fact-city",
        subject="user:789",
        object_value="Alice lives in Lisbon",
        entities=["Alice"],
        event_date="2024-01-01",
        scope=scope,
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(),
        long_term=RecordingLongTermMemory(search_results=[dog, city]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_read_supersession_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled with supersession enabled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="tell me about Maria and Alice",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: independent facts are both kept - never over-superseded.
    content = response.sections[0].content
    assert "Biscuit" in content
    assert "Lisbon" in content


@pytest.mark.anyio
async def test_read_supersession_applies_to_search() -> None:
    # Given: two same-slot extracted facts returned by fact search.
    scope = _scope()
    older = _extracted_fact(
        memory_id="fact-old",
        subject="user:789",
        object_value="Maria's dog is Rex",
        entities=["Maria"],
        event_date="2024-01-01",
        scope=scope,
    )
    newer = _extracted_fact(
        memory_id="fact-new",
        subject="user:789",
        object_value="Maria's dog is Biscuit",
        entities=["Maria"],
        event_date="2024-06-01",
        scope=scope,
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(),
        long_term=RecordingLongTermMemory(search_results=[older, newer]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_read_supersession_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: search runs with supersession enabled.
    response = await backend.search_memories(
        MemorySearchRequest(scope=scope, query="Maria's dog", limit=5),
    )

    # Then: only the newest fact in the slot is returned.
    assert [result.memory_id for result in response.results] == ["fact-new"]


@pytest.mark.anyio
async def test_abstention_instruction_prepended_when_enabled() -> None:
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_abstention_prompt_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when does the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    assert response.sections[0].source == "instructions"
    assert response.sections[0].content == (
        "Answer only from the memories below; if they do not contain the "
        "answer, say you don't know."
    )


@pytest.mark.anyio
async def test_chain_of_note_instruction_prepended_when_enabled() -> None:
    # Given: Chain-of-Note on and one stored memory to read.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_chain_of_note_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when does the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the note-taking instruction leads and the memory still renders.
    assert response.sections[0].source == "instructions"
    assert "take notes on each memory" in response.sections[0].content
    assert "say you don't know" in response.sections[0].content
    assert "the library opens at nine" in response.sections[1].content


@pytest.mark.anyio
async def test_routed_temporal_query_reads_without_chain_of_note() -> None:
    # Given: Chain-of-Note AND adaptive routing on, and the classifier tags
    # the query temporal (whose hybrid retrieval texture the note step
    # measurably harms - Run 14: temporal 92.2 -> 83.3).
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opened on 2023-05-07",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    router = RecordingQueryRouter(verdict=RouteVerdict(route="temporal"))
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_chain_of_note_enabled=True,
            gnosis_adaptive_routing_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
        query_router=router,
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when did the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: no reading instruction renders on the temporal route even though
    # the Chain-of-Note flag is globally on; the memory still renders.
    assert all(section.source != "instructions" for section in response.sections)
    assert "the library opened on 2023-05-07" in response.sections[0].content


@pytest.mark.anyio
async def test_routed_multi_hop_query_keeps_chain_of_note() -> None:
    # Given: Chain-of-Note AND adaptive routing on with a multi-hop verdict.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    router = RecordingQueryRouter(verdict=RouteVerdict(route="multi_hop"))
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_chain_of_note_enabled=True,
            gnosis_adaptive_routing_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
        query_router=router,
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="which street is the library where Alice works on?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the note-taking instruction still leads on the multi-hop route.
    assert response.sections[0].source == "instructions"
    assert "take notes on each memory" in response.sections[0].content


@pytest.mark.anyio
async def test_chain_of_note_unchanged_while_widening_flags_off() -> None:
    # Given: Chain-of-Note on with both widening flags at their defaults.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_chain_of_note_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when does the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the instruction is the exact Run 18 CoN v3 text - the likelihood
    # carve-out, no speculative widening, no enumeration clause.
    content = response.sections[0].content
    likelihood_tail = (
        "Only when the question itself asks what is likely or probable, infer "
        "the most plausible answer from the relevant memories instead of "
        "saying you don't know."
    )
    assert content.endswith(likelihood_tail)
    assert "speculative judgment" not in content
    assert "enumerate every distinct real-world item" not in content


@pytest.mark.anyio
async def test_speculative_inference_flag_widens_the_carve_out() -> None:
    # Given: the speculative-inference widening on top of Chain-of-Note.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="fact",
                        object_value="user enjoys classical music",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_chain_of_note_enabled=True,
            gnosis_con_speculative_inference_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="would the user enjoy a Vivaldi concert?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the speculative carve-out replaces the likelihood clause, and the
    # never-invent-specifics guard stays.
    content = response.sections[0].content
    assert "speculative judgment about a person" in content
    assert "Still never invent specific facts" in content
    assert "Only when the question itself asks what is likely" not in content


@pytest.mark.anyio
async def test_enumeration_clause_applies_on_aggregative_route_only() -> None:
    # Given: the enumeration widening with routing on; two identical
    # backends differing only in the routed verdict.
    scope = _scope()

    def build(route: str) -> Neo4jAgentMemoryBackend:
        client = RecordingMemoryClient(
            query=RecordingQuery(
                rows=[
                    {
                        "f": _fact_row(
                            subject="tenant:nolgia:message:one",
                            predicate="fact",
                            object_value="user made peach cobbler",
                            metadata=_scope_metadata(scope),
                            created_at="2026-06-28T00:00:00Z",
                        ),
                    },
                ],
            ),
        )
        return Neo4jAgentMemoryBackend(
            _settings(
                gnosis_chain_of_note_enabled=True,
                gnosis_adaptive_routing_enabled=True,
                gnosis_con_enumeration_enabled=True,
            ),
            memory_client_factory=MemoryClientFactory(client),
            graph_store=RecordingGraphStore(),
            query_router=RecordingQueryRouter(
                verdict=RouteVerdict(route=route),  # pyright: ignore[reportArgumentType]
            ),
        )

    # When: the same query reads via the aggregative and single-hop routes.
    request = MemoryContextRequest(
        scope=scope,
        query="what desserts has the user made?",
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
    )
    aggregative = await build("aggregative").get_memory_context(request)
    single_hop = await build("single_hop").get_memory_context(request)

    # Then: only the aggregative route reads with the enumeration clause.
    assert "enumerate every distinct real-world item" in aggregative.sections[0].content
    assert "count unique items or events" in aggregative.sections[0].content
    assert (
        "enumerate every distinct real-world item" not in single_hop.sections[0].content
    )


@pytest.mark.anyio
async def test_chain_of_note_takes_precedence_over_abstention_prompt() -> None:
    # Given: both prompt-only reading aids enabled at once.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_chain_of_note_enabled=True,
            gnosis_abstention_prompt_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when does the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: exactly one leading instruction renders - the Chain-of-Note one
    # (it subsumes the bare abstention line).
    instructions = [s for s in response.sections if s.source == "instructions"]
    assert len(instructions) == 1
    assert "take notes on each memory" in instructions[0].content


@pytest.mark.anyio
async def test_chain_of_note_absent_when_no_memory_content() -> None:
    # Given: Chain-of-Note on but nothing retrieved for the query.
    scope = _scope()
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_chain_of_note_enabled=True),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(query=RecordingQuery()),
        ),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled with an empty store.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when does the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: no instruction-only response is fabricated.
    assert all(section.source != "instructions" for section in response.sections)


@pytest.mark.anyio
async def test_abstention_instruction_absent_when_disabled() -> None:
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when does the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    assert all(section.source != "instructions" for section in response.sections)
    assert response.sufficiency is None


@pytest.mark.anyio
async def test_routing_disabled_never_calls_classifier() -> None:
    # Given: adaptive routing off (default) with a recorded router injected.
    scope = _scope()
    router = RecordingQueryRouter(verdict=RouteVerdict(route="multi_hop"))
    backend = Neo4jAgentMemoryBackend(
        _settings(),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(query=RecordingQuery()),
        ),
        graph_store=RecordingGraphStore(),
        query_router=router,
    )

    # When: context and search both run with a query present.
    _ = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="where does Alice work?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )
    _ = await backend.search_memories(
        MemorySearchRequest(scope=scope, query="where does Alice work?", limit=5),
    )

    # Then: the classifier is never consulted while the flag is off.
    assert router.queries == []


@pytest.mark.anyio
async def test_routing_multi_hop_enables_graph_fusion_despite_global_off() -> None:
    # Given: routing on, the classifier tags the query multi-hop, and the
    # global graph-QA fusion flag is OFF.
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    graph_store = FusionGraphStore(
        facts=[
            _graph_fact(
                node_id="graph-node-1",
                summary="The city library is on Elm Street",
            ),
        ],
    )
    router = RecordingQueryRouter(verdict=RouteVerdict(route="multi_hop"))
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_adaptive_routing_enabled=True,
            gnosis_graphqa_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[dense]),
            ),
        ),
        graph_store=graph_store,
        query_router=router,
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="which street is the library where Alice works on?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the graph traversal leg ran for this query and its node rendered.
    assert router.queries == ["which street is the library where Alice works on?"]
    content = response.sections[0].content
    assert "The city library is on Elm Street" in content


@pytest.mark.anyio
async def test_routed_multi_hop_reads_with_the_expanded_item_budget() -> None:
    # Given: routing on with a 2x multi-hop budget multiplier and more
    # ranked dense candidates than the request budget.
    scope = _scope()
    dense = [
        _fact_record(
            subject=f"user:789:fact:{index}",
            predicate="fact",
            object_value=f"enumeration item number {index}",
            metadata=_scope_metadata(scope),
        )
        for index in range(10)
    ]
    settings = _settings(
        gnosis_adaptive_routing_enabled=True,
        gnosis_coverage_budget_multiplier=2,
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(),
        long_term=RecordingLongTermMemory(search_results=dense),
    )

    async def assemble(route: QueryRoute) -> str:
        backend = Neo4jAgentMemoryBackend(
            settings,
            memory_client_factory=MemoryClientFactory(client),
            graph_store=RecordingGraphStore(),
            query_router=RecordingQueryRouter(verdict=RouteVerdict(route=route)),
        )
        response = await backend.get_memory_context(
            MemoryContextRequest(
                scope=scope,
                query="what activities has Alice done?",
                include_short_term=False,
                include_reasoning=False,
                include_graph=False,
                max_items=4,
            ),
        )
        return response.sections[-1].content

    # When/Then: the multi-hop route renders double the request budget
    # (coverage is its measured gap) while single-hop keeps the budget as-is.
    assert (await assemble("multi_hop")).count("\n- ") == 8
    assert (await assemble("single_hop")).count("\n- ") == 4


@pytest.mark.anyio
async def test_routing_single_hop_suppresses_globally_enabled_features() -> None:
    # Given: routing on with graph fusion AND the abstention prompt globally
    # enabled, but the classifier tags the query single-hop.
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    graph_store = FusionGraphStore(
        facts=[_graph_fact(node_id="graph-node-1", summary="library on Elm")],
    )
    router = RecordingQueryRouter(verdict=RouteVerdict(route="single_hop"))
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_adaptive_routing_enabled=True,
            gnosis_graphqa_enabled=True,
            gnosis_graphqa_fusion_enabled=True,
            gnosis_abstention_prompt_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[dense]),
            ),
        ),
        graph_store=graph_store,
        query_router=router,
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="where does Alice work?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the routed decision replaces the global flags - no graph route,
    # no abstention instruction - and the dense fact still renders.
    assert graph_store.calls == 0
    assert all(section.source != "instructions" for section in response.sections)
    assert "Alice works at the city library" in response.sections[0].content


@pytest.mark.anyio
async def test_routing_unanswerable_risk_prepends_abstention_instruction() -> None:
    # Given: routing on, global abstention prompt OFF, and a query the
    # classifier tags unanswerable-risk.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    router = RecordingQueryRouter(verdict=RouteVerdict(route="unanswerable_risk"))
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_adaptive_routing_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
        query_router=router,
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="what toothpaste does the librarian's dentist recommend?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the abstention standing instruction leads the sections.
    assert response.sections[0].source == "instructions"


@pytest.mark.anyio
async def test_routing_classifier_failure_falls_back_to_global_flags(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: routing on, the classifier raises, and the global abstention
    # prompt is ON.
    scope = _scope()
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                {
                    "f": _fact_row(
                        subject="tenant:nolgia:message:one",
                        predicate="said_user",
                        object_value="the library opens at nine",
                        metadata=_scope_metadata(scope),
                        created_at="2026-06-28T00:00:00Z",
                    ),
                },
            ],
        ),
    )
    router = RecordingQueryRouter(verdict=OpenAIError("router down"))
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_adaptive_routing_enabled=True,
            gnosis_abstention_prompt_enabled=True,
        ),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
        query_router=router,
    )

    # When: context is assembled despite the classifier failure.
    with caplog.at_level(logging.WARNING):
        response = await backend.get_memory_context(
            MemoryContextRequest(
                scope=scope,
                query="when does the library open?",
                include_short_term=False,
                include_reasoning=False,
                include_graph=False,
            ),
        )

    # Then: the read succeeds under the global flags with a warning logged.
    assert response.sections[0].source == "instructions"
    assert "query routing failed" in caplog.text


@pytest.mark.anyio
async def test_sufficiency_block_present_when_enabled() -> None:
    scope = _scope()
    client = RecordingMemoryClient(query=RecordingQuery())
    assessor = RecordingSufficiencyAssessor(
        verdict=SufficiencyVerdict(sufficient=True, reason="  the answer is present  "),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_sufficiency_check_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
        sufficiency_assessor=assessor,
    )

    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="anything?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    assert response.sufficiency is not None
    assert response.sufficiency.assessed is True
    assert response.sufficiency.sufficient is True
    assert response.sufficiency.reason == "the answer is present"
    assert len(assessor.calls) == 1


@pytest.mark.anyio
async def test_sufficiency_llm_failure_degrades_to_not_assessed() -> None:
    scope = _scope()
    client = RecordingMemoryClient(query=RecordingQuery())
    assessor = RecordingSufficiencyAssessor(verdict=RuntimeError("boom"))
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_sufficiency_check_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
        sufficiency_assessor=assessor,
    )

    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="anything?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    assert response.sufficiency is not None
    assert response.sufficiency.assessed is False
    assert response.sufficiency.sufficient is False
    assert response.sufficiency.reason is None


def _extracted_fact_with_sources(  # noqa: PLR0913 - Test builder mirrors stored shape.
    *,
    memory_id: str,
    object_value: str,
    entities: list[str],
    event_date: str,
    source_memory_ids: list[str],
    scope: MemoryScope,
) -> FactRecord:
    fact = _extracted_fact(
        memory_id=memory_id,
        subject="user:789",
        object_value=object_value,
        entities=entities,
        event_date=event_date,
        scope=scope,
    )
    fact.metadata["source_memory_ids"] = cast("JsonValue", source_memory_ids)
    return fact


def _verbatim_row(
    *,
    memory_id: str,
    object_value: str,
    metadata: dict[str, str],
) -> JsonObject:
    return {
        "id": memory_id,
        "subject": "user:789",
        "predicate": "said_user",
        "object": object_value,
        "metadata": json.dumps(metadata),
        "created_at": "2023-05-07T00:00:00Z",
        "updated_at": None,
    }


@dataclass(slots=True)
class FailingQuery:
    cypher_calls: list["CypherCall"] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.cypher_calls.append(CypherCall(statement=query, parameters=params or {}))
        msg = "verbatim lookup boom"
        raise RuntimeError(msg)


@pytest.mark.anyio
async def test_verbatim_expansion_attaches_source_turn_when_enabled() -> None:
    # Given: one ranked extracted fact linking to a stored verbatim turn.
    scope = _scope()
    extracted = _extracted_fact_with_sources(
        memory_id="fact-1",
        object_value="Maria adopted a dog",
        entities=["Maria"],
        event_date="2023-05-07",
        source_memory_ids=["turn-1"],
        scope=scope,
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                _verbatim_row(
                    memory_id="turn-1",
                    object_value="I finally adopted a golden retriever named Biscuit!",
                    metadata=_scope_metadata(scope),
                ),
            ],
        ),
        long_term=RecordingLongTermMemory(search_results=[extracted]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_fact_verbatim_expansion_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled with the expansion flag on.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="what pet did Maria adopt?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the compact fact renders followed by an indented verbatim quote.
    content = response.sections[0].content
    assert "user:789 fact: Maria adopted a dog" in content
    assert "  quote: I finally adopted a golden retriever named Biscuit!" in content


@pytest.mark.anyio
async def test_verbatim_expansion_respects_cap() -> None:
    # Given: two ranked extracted facts but a one-fact expansion budget.
    scope = _scope()
    first = _extracted_fact_with_sources(
        memory_id="fact-1",
        object_value="Maria adopted a dog",
        entities=["Maria"],
        event_date="2023-05-07",
        source_memory_ids=["turn-1"],
        scope=scope,
    )
    second = _extracted_fact_with_sources(
        memory_id="fact-2",
        object_value="Alice moved to Lisbon",
        entities=["Alice"],
        event_date="2023-06-01",
        source_memory_ids=["turn-2"],
        scope=scope,
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                _verbatim_row(
                    memory_id="turn-1",
                    object_value="verbatim about the dog",
                    metadata=_scope_metadata(scope),
                ),
                _verbatim_row(
                    memory_id="turn-2",
                    object_value="verbatim about Lisbon",
                    metadata=_scope_metadata(scope),
                ),
            ],
        ),
        long_term=RecordingLongTermMemory(search_results=[first, second]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_fact_verbatim_expansion_enabled=True,
            gnosis_fact_verbatim_expansion_max=1,
        ),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="tell me about Maria and Alice",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: only the top-ranked fact is expanded; the second stays compact.
    content = response.sections[0].content
    assert "  quote: verbatim about the dog" in content
    assert "verbatim about Lisbon" not in content
    # Only the requested (novel) source id is looked up.
    assert client.query.cypher_calls[0].parameters["memory_ids"] == ["turn-1"]


@pytest.mark.anyio
async def test_verbatim_expansion_skips_already_present_turn() -> None:
    # Given: the source verbatim turn is itself independently a ranked fact.
    scope = _scope()
    extracted = _extracted_fact_with_sources(
        memory_id="fact-1",
        object_value="Maria adopted a dog",
        entities=["Maria"],
        event_date="2023-05-07",
        source_memory_ids=["turn-1"],
        scope=scope,
    )
    present_turn = _fact_record(
        subject="turn-1",
        object_value="I adopted a golden retriever",
        metadata=_scope_metadata(scope),
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(),
        long_term=RecordingLongTermMemory(search_results=[extracted, present_turn]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_fact_verbatim_expansion_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="what pet did Maria adopt?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: no quote line is emitted and no verbatim lookup runs.
    content = response.sections[0].content
    assert "quote:" not in content
    assert client.query.cypher_calls == []


@pytest.mark.anyio
async def test_verbatim_expansion_drops_cross_scope_turn() -> None:
    # Given: the lookup surfaces a verbatim row from another tenant.
    scope = _scope()
    extracted = _extracted_fact_with_sources(
        memory_id="fact-1",
        object_value="Maria adopted a dog",
        entities=["Maria"],
        event_date="2023-05-07",
        source_memory_ids=["turn-1"],
        scope=scope,
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(
            rows=[
                _verbatim_row(
                    memory_id="turn-1",
                    object_value="cross-tenant secret turn",
                    metadata=_scope_metadata(_scope(tenant_id="other-tenant")),
                ),
            ],
        ),
        long_term=RecordingLongTermMemory(search_results=[extracted]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_fact_verbatim_expansion_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="what pet did Maria adopt?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the cross-scope turn never leaks into the prompt.
    content = response.sections[0].content
    assert "cross-tenant secret turn" not in content
    assert "quote:" not in content


@pytest.mark.anyio
async def test_verbatim_expansion_off_is_byte_identical() -> None:
    # Given: an extracted fact with a linked turn, flag left at its default.
    scope = _scope()
    extracted = _extracted_fact_with_sources(
        memory_id="fact-1",
        object_value="Maria adopted a dog",
        entities=["Maria"],
        event_date="2023-05-07",
        source_memory_ids=["turn-1"],
        scope=scope,
    )

    def build(*, enabled: bool) -> Neo4jAgentMemoryBackend:
        return Neo4jAgentMemoryBackend(
            _settings(gnosis_fact_verbatim_expansion_enabled=enabled),
            memory_client_factory=MemoryClientFactory(
                RecordingMemoryClient(
                    query=RecordingQuery(
                        rows=[
                            _verbatim_row(
                                memory_id="turn-1",
                                object_value="raw turn text",
                                metadata=_scope_metadata(scope),
                            ),
                        ],
                    ),
                    long_term=RecordingLongTermMemory(search_results=[extracted]),
                ),
            ),
            graph_store=RecordingGraphStore(),
        )

    request = MemoryContextRequest(
        scope=scope,
        query="what pet did Maria adopt?",
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
    )

    # When/Then: the disabled output has no quote and no lookup runs.
    off = await build(enabled=False).get_memory_context(request)
    content = off.sections[0].content
    assert "quote:" not in content
    assert "raw turn text" not in content


@pytest.mark.anyio
async def test_verbatim_expansion_lookup_failure_degrades_to_compact() -> None:
    # Given: the verbatim lookup raises.
    scope = _scope()
    extracted = _extracted_fact_with_sources(
        memory_id="fact-1",
        object_value="Maria adopted a dog",
        entities=["Maria"],
        event_date="2023-05-07",
        source_memory_ids=["turn-1"],
        scope=scope,
    )
    client = RecordingMemoryClient(
        query=cast("RecordingQuery", cast("object", FailingQuery())),
        long_term=RecordingLongTermMemory(search_results=[extracted]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_fact_verbatim_expansion_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="what pet did Maria adopt?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the compact fact still renders; no quote line is attached.
    content = response.sections[0].content
    assert "user:789 fact: Maria adopted a dog" in content
    assert "quote:" not in content


@pytest.mark.anyio
async def test_verbatim_expansion_ignores_non_extracted_facts() -> None:
    # Given: only a verbatim turn fact (no extracted units) is ranked.
    scope = _scope()
    turn = _fact_record(
        subject="tenant:nolgia:message:one",
        object_value="the library opens at nine",
        metadata=_scope_metadata(scope),
    )
    client = RecordingMemoryClient(
        query=RecordingQuery(),
        long_term=RecordingLongTermMemory(search_results=[turn]),
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_fact_verbatim_expansion_enabled=True),
        memory_client_factory=MemoryClientFactory(client),
        graph_store=RecordingGraphStore(),
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="when does the library open?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: no expansion is attempted for non-extracted facts.
    content = response.sections[0].content
    assert "quote:" not in content
    assert client.query.cypher_calls == []


@pytest.mark.anyio
async def test_graphqa_fusion_adds_graph_nodes_to_facts_when_enabled() -> None:
    # Given: dense retrieval finds one fact; the graph route yields another.
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    graph_store = FusionGraphStore(
        facts=[
            _graph_fact(
                node_id="graph-node-1",
                summary="The city library is on Elm Street",
            ),
        ],
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_graphqa_enabled=True, gnosis_graphqa_fusion_enabled=True),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[dense]),
            ),
        ),
        graph_store=graph_store,
    )

    # When: context is assembled with a query present.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="where does Alice work?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: both the dense fact and the fused graph node render as facts.
    content = response.sections[0].content
    assert graph_store.calls == 1
    assert "Alice works at the city library" in content
    assert "- The city library is on Elm Street" in content


@pytest.mark.anyio
async def test_graphqa_fusion_dedupes_node_already_in_dense_results() -> None:
    # Given: the graph route returns a node sharing the dense fact's memory id.
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    graph_store = FusionGraphStore(
        facts=[
            _graph_fact(
                node_id="user:789",
                summary="Alice works at the city library",
            ),
        ],
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_graphqa_enabled=True, gnosis_graphqa_fusion_enabled=True),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[dense]),
            ),
        ),
        graph_store=graph_store,
    )

    # When: context is assembled.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="where does Alice work?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the shared node is not double-added.
    content = response.sections[0].content
    assert content.count("city library") == 1


@pytest.mark.anyio
async def test_graphqa_fusion_survives_item_budget_over_full_dense_pool() -> None:
    # Given: dense retrieval fills the candidate pool past the item budget
    # (the production shape: 100 candidates, max_items 8) and the graph route
    # yields one traversal fact ranked after all of them.
    scope = _scope()
    dense = [
        _fact_record(
            subject=f"user:{index}",
            predicate="fact",
            object_value=f"dense filler fact number {index}",
            metadata=_scope_metadata(scope),
        )
        for index in range(30)
    ]
    graph_store = FusionGraphStore(
        facts=[
            _graph_fact(
                node_id="graph-node-1",
                summary="The city library is on Elm Street",
            ),
        ],
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_graphqa_enabled=True, gnosis_graphqa_fusion_enabled=True),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=dense),
            ),
        ),
        graph_store=graph_store,
    )

    # When: context is assembled with a budget smaller than the dense pool.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="which street is the library on?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
            max_items=8,
        ),
    )

    # Then: the graph fact holds a reserved slot instead of being cut, the
    # budget is respected, and dense candidates keep the remaining slots in
    # ranking order.
    content = response.sections[0].content
    assert "- The city library is on Elm Street" in content
    fact_lines = [line for line in content.splitlines() if line.startswith("- ")]
    assert len(fact_lines) == 8
    assert "dense filler fact number 0" in content
    assert "dense filler fact number 6" in content
    assert "dense filler fact number 7" not in content


@pytest.mark.anyio
async def test_graphqa_fusion_planner_failure_degrades_to_dense_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the graph route raises (planner / execution failure).
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    graph_store = FusionGraphStore(error=OpenAIError("planner down"))
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_graphqa_enabled=True, gnosis_graphqa_fusion_enabled=True),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[dense]),
            ),
        ),
        graph_store=graph_store,
    )

    # When: context is assembled.
    with caplog.at_level(logging.WARNING):
        response = await backend.get_memory_context(
            MemoryContextRequest(
                scope=scope,
                query="where does Alice work?",
                include_short_term=False,
                include_reasoning=False,
                include_graph=False,
            ),
        )

    # Then: the dense fact still renders and a structured warning is logged.
    content = response.sections[0].content
    assert "Alice works at the city library" in content
    assert "graph-QA fusion route failed" in caplog.text


@pytest.mark.anyio
async def test_graphqa_fusion_timeout_degrades_to_dense_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: the graph route stalls past the fusion timeout budget.
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    graph_store = FusionGraphStore(
        facts=[_graph_fact(node_id="graph-node-1", summary="unreachable")],
        delay=0.5,
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(
            gnosis_graphqa_enabled=True,
            gnosis_graphqa_fusion_enabled=True,
            gnosis_graphqa_fusion_timeout_seconds=0.01,
        ),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[dense]),
            ),
        ),
        graph_store=graph_store,
    )

    # When: context is assembled.
    with caplog.at_level(logging.WARNING):
        response = await backend.get_memory_context(
            MemoryContextRequest(
                scope=scope,
                query="where does Alice work?",
                include_short_term=False,
                include_reasoning=False,
                include_graph=False,
            ),
        )

    # Then: dense-only context returns and the slow node never renders.
    content = response.sections[0].content
    assert "Alice works at the city library" in content
    assert "unreachable" not in content
    assert "graph-QA fusion route failed" in caplog.text


@pytest.mark.anyio
async def test_graphqa_fusion_is_noop_while_graphqa_disabled() -> None:
    # Given: fusion is on but LLM-planned graph QA (GNOSIS_GRAPHQA_ENABLED)
    # is off, and the graph route has nodes.
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    graph_store = FusionGraphStore(
        facts=[_graph_fact(node_id="graph-node-1", summary="library on Elm")],
    )
    backend = Neo4jAgentMemoryBackend(
        _settings(gnosis_graphqa_fusion_enabled=True),
        memory_client_factory=MemoryClientFactory(
            RecordingMemoryClient(
                query=RecordingQuery(),
                long_term=RecordingLongTermMemory(search_results=[dense]),
            ),
        ),
        graph_store=graph_store,
    )

    # When: context is assembled with a query present.
    response = await backend.get_memory_context(
        MemoryContextRequest(
            scope=scope,
            query="where does Alice work?",
            include_short_term=False,
            include_reasoning=False,
            include_graph=False,
        ),
    )

    # Then: the graph route never runs and the output is dense-only.
    assert graph_store.calls == 0
    assert "library on Elm" not in response.sections[0].content


@pytest.mark.anyio
async def test_graphqa_fusion_off_is_byte_identical() -> None:
    # Given: identical inputs, fusion flag toggled; the graph route has nodes.
    scope = _scope()
    dense = _fact_record(
        subject="user:789",
        predicate="fact",
        object_value="Alice works at the city library",
        metadata=_scope_metadata(scope),
    )
    request = MemoryContextRequest(
        scope=scope,
        query="where does Alice work?",
        include_short_term=False,
        include_reasoning=False,
        include_graph=False,
    )

    def build(*, enabled: bool) -> tuple[Neo4jAgentMemoryBackend, FusionGraphStore]:
        store = FusionGraphStore(
            facts=[_graph_fact(node_id="graph-node-1", summary="library on Elm")],
        )
        backend = Neo4jAgentMemoryBackend(
            _settings(
                gnosis_graphqa_enabled=True, gnosis_graphqa_fusion_enabled=enabled
            ),
            memory_client_factory=MemoryClientFactory(
                RecordingMemoryClient(
                    query=RecordingQuery(),
                    long_term=RecordingLongTermMemory(search_results=[dense]),
                ),
            ),
            graph_store=store,
        )
        return backend, store

    # When: assembled with the flag off vs on.
    off_backend, off_store = build(enabled=False)
    off = await off_backend.get_memory_context(request)
    on_backend, _ = build(enabled=True)
    on = await on_backend.get_memory_context(request)

    # Then: the disabled run never calls the graph route and drops its nodes.
    assert off_store.calls == 0
    assert "library on Elm" not in off.sections[0].content
    assert "library on Elm" in on.sections[0].content


@dataclass(frozen=True, slots=True)
class CypherCall:
    statement: str
    parameters: dict[str, JsonValue]


@dataclass(slots=True)
class RecordingQuery:
    rows: list[JsonObject] = field(default_factory=list)
    cypher_calls: list[CypherCall] = field(default_factory=list)

    async def cypher(
        self,
        query: str,
        params: dict[str, JsonValue] | None = None,
    ) -> list[JsonObject]:
        self.cypher_calls.append(
            CypherCall(statement=query, parameters=params or {}),
        )
        return self.rows


@dataclass(slots=True)
class RecordingLongTermMemory:
    context: str = ""
    context_queries: list[str] = field(default_factory=list)
    search_results: list[FactRecord] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)

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
        _ = (limit, threshold)
        self.search_queries.append(query)
        return list(self.search_results)

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
        _ = (
            name,
            entity_type,
            subtype,
            description,
            aliases,
            attributes,
            resolve,
            generate_embedding,
            deduplicate,
            geocode,
            enrich,
            coordinates,
            metadata,
        )
        return EntityRecord(name=name, type=str(entity_type))

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
        _ = (
            subject,
            predicate,
            obj,
            confidence,
            valid_from,
            valid_until,
            metadata,
            generate_embedding,
        )
        return FactRecord(subject=subject, predicate=predicate, object=obj)

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
        _ = (
            category,
            preference,
            context,
            confidence,
            generate_embedding,
            metadata,
            user_identifier,
            applies_to,
        )
        return PreferenceRecord(category=category, preference=preference)

    async def get_preferences_for(
        self,
        user_identifier: str,
        *,
        applies_to: object | None = None,
        active_only: bool = True,
        as_of: datetime | None = None,
    ) -> list[PreferenceRecord]:
        _ = (user_identifier, applies_to, active_only, as_of)
        return []

    async def get_facts_about(
        self,
        subject: str,
        *,
        limit: int = 100,
    ) -> list[FactRecord]:
        _ = (subject, limit)
        return []

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
        _ = (entity, message_id, confidence, start_pos, end_pos, context)
        return True

    async def link_entity_to_extractor(
        self,
        entity: EntityRecord | UUID,
        extractor_name: str,
        *,
        confidence: float = 1.0,
        extraction_time_ms: float | None = None,
    ) -> bool:
        _ = (entity, extractor_name, confidence, extraction_time_ms)
        return True

    async def get_context(self, query: str, *, max_items: int) -> str:
        _ = max_items
        self.context_queries.append(query)
        return self.context


@dataclass(slots=True)
class RecordingShortTermMemory:
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
        _ = (
            session_id,
            role,
            content,
            user_identifier,
            metadata,
            extract_entities,
            extract_relations,
        )

    async def get_context(
        self,
        query: str,
        *,
        session_id: str,
        max_messages: int,
        metadata_filters: dict[str, str],
    ) -> str:
        _ = (query, session_id, max_messages, metadata_filters)
        return ""


@dataclass(slots=True)
class RecordingReasoningMemory:
    async def get_context(self, query: str, *, max_traces: int) -> str:
        _ = (query, max_traces)
        return ""

    async def list_traces(
        self,
        *,
        session_id: str | None = None,
        limit: int = 10,
        success_only: bool | None = None,
    ) -> list[SdkReasoningTrace]:
        _ = (session_id, limit, success_only)
        return []

    async def get_trace(self, trace_id: UUID | str) -> SdkReasoningTrace | None:
        _ = trace_id
        return None

    async def get_trace_with_steps(
        self,
        trace_id: UUID | str,
    ) -> SdkReasoningTrace | None:
        _ = trace_id
        return None

    async def get_similar_traces(
        self,
        task: str,
        *,
        limit: int = 5,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[SdkReasoningTrace]:
        _ = (task, limit, success_only, threshold)
        return []

    async def search_steps(
        self,
        query: str,
        *,
        limit: int = 10,
        success_only: bool = True,
        threshold: float = 0.7,
    ) -> list[object]:
        _ = (query, limit, success_only, threshold)
        return []

    async def get_tool_stats(self, tool_name: str | None = None) -> list[ToolStats]:
        _ = tool_name
        return []

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
        return SdkReasoningTrace(session_id=session_id, task=task)

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
        _ = (thought, action, observation, generate_embedding, metadata)
        return SdkReasoningStep(trace_id=trace_id, step_number=1)

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


@dataclass(slots=True)
class RecordingMemoryClient:
    query: RecordingQuery
    long_term: RecordingLongTermMemory = field(default_factory=RecordingLongTermMemory)
    short_term: RecordingShortTermMemory = field(
        default_factory=RecordingShortTermMemory,
    )
    reasoning: RecordingReasoningMemory = field(
        default_factory=RecordingReasoningMemory,
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
        client = cast("object", self.client)
        return cast("MemoryClientContext", client)


@dataclass(slots=True)
class RecordingGraphStore:
    async def require_available(self) -> None:
        return None

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: object) -> EventIngestResult:
        _ = event
        return EventIngestResult(
            event_id="event-placeholder",
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        _ = request
        return GraphContextResponse(context="")


@dataclass(slots=True)
class FusionGraphStore:
    """Graph store stub whose planned route returns known nodes (or fails)."""

    facts: list[JsonObject] = field(default_factory=list)
    error: Exception | None = None
    delay: float = 0.0
    calls: int = 0

    async def require_available(self) -> None:
        return None

    async def readiness(self) -> BackendReadiness:
        return BackendReadiness(graph="ready", schema="ready")

    async def ingest_event(self, event: object) -> EventIngestResult:
        _ = event
        return EventIngestResult(
            event_id="event-placeholder",
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        _ = request
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return GraphContextResponse(
            context="\n".join(str(fact["summary"]) for fact in self.facts),
            facts=list(self.facts),
        )


def _graph_fact(
    *,
    node_id: str,
    summary: str,
    node_type: str = "user",
    deleted: bool = False,
) -> JsonObject:
    return {
        "id": node_id,
        "type": node_type,
        "scope": "channel",
        "summary": summary,
        "deleted": deleted,
    }


def _settings(
    *,
    gnosis_prompt_entities_enabled: bool = False,
    **overrides: JsonValue,
) -> Settings:
    settings_values: JsonObject = {
        "gnosis_token": "value",
        "gnosis_tenant_id": "nolgia",
        "neo4j_uri": "bolt://neo4j.local:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "value",
        "litellm_base_url": "http://litellm.local/v1",
        "litellm_api_key": "value",
        "gnosis_llm": "openai/gemma4",
        "gnosis_embedding": "local-qwen3-embedding-0.6b",
        "gnosis_embedding_dimensions": 1024,
        "gnosis_prompt_entities_enabled": gnosis_prompt_entities_enabled,
        **overrides,
    }
    return Settings.model_validate(settings_values)


def _scope(
    *,
    tenant_id: str = "nolgia",
    channel_id: str = "456",
) -> MemoryScope:
    return MemoryScope(
        tenant_id=tenant_id,
        space_id="discord",
        agent_id="nolgia-agent",
        session_id=f"guild:123:channel:{channel_id}",
        user_id="789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="123",
        channel_id=channel_id,
    )


def _scope_metadata(scope: MemoryScope) -> dict[str, str]:
    return {
        "tenant_id": scope.tenant_id,
        "agent_id": scope.agent_id,
        "session_id": scope.session_id,
        "user_id": scope.user_id,
        "visibility": scope.visibility.value,
        "guild_id": scope.guild_id or "",
        "channel_id": scope.channel_id or "",
    }


def _fact_record(
    *,
    subject: str,
    object_value: str,
    metadata: dict[str, str],
    predicate: str = "said_user",
) -> FactRecord:
    return FactRecord(
        id=subject,
        subject=subject,
        predicate=predicate,
        object=object_value,
        metadata=cast("JsonObject", dict(metadata)),
    )


def _fact_row(
    *,
    subject: str,
    predicate: str,
    object_value: str,
    metadata: dict[str, str],
    created_at: str | None = None,
) -> JsonObject:
    return {
        "id": subject,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "confidence": 1.0,
        "created_at": created_at,
        "metadata": json.dumps(metadata),
    }
