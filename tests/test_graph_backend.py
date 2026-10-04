from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import pytest

from gnosis.graph_cypher import (
    upsert_parameters,
)
from gnosis.graph_events import plan_event
from gnosis.graph_schema import GRAPH_SCHEMA_CYPHER, graph_vector_schema_cypher
from gnosis.graph_store import (
    DirectNeo4jGraphStore,
    InMemoryGraphExecutor,
    Neo4jGraphExecutor,
)
from gnosis.models import (
    ClientEvent,
    ClientEventActor,
    ClientEventSubject,
    ClientEventType,
    DiscordEventContext,
    EventIngestStatus,
    GraphContextRequest,
    JsonValue,
    MemoryScope,
    MemoryVisibility,
    SourceClient,
)


@pytest.mark.anyio
async def test_discord_message_upsert_and_context_retrieval() -> None:
    # Given: a direct graph store receives a Discord channel message event.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    event = _message_event(
        _MessageEventValues(content="remember the plasma conduit"),
    )

    # When: the event is ingested and graph context is requested in the same scope.
    result = await store.ingest_event(event)
    context = await store.get_context(
        GraphContextRequest(scope=event.scope, query="plasma", limit=4),
    )

    # Then: current graph state and structured facts are visible without Neo4j.
    assert result.status == EventIngestStatus.ACCEPTED
    assert context.context == "message message-999: remember the plasma conduit"
    assert context.facts == [
        {
            "id": "tenant:nolgia:message:message-999",
            "type": "message",
            "scope": "channel",
            "summary": "message message-999: remember the plasma conduit",
            "deleted": False,
        },
    ]


@pytest.mark.anyio
async def test_graph_context_does_not_cross_channel_scope() -> None:
    # Given: two channel-scoped events in the same tenant and guild.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    first = _message_event(
        _MessageEventValues(
            message_id="message-1",
            channel_id="channel-1",
            content="alpha",
        ),
    )
    second = _message_event(
        _MessageEventValues(
            message_id="message-2",
            channel_id="channel-2",
            content="beta",
        ),
    )
    _ = await store.ingest_event(first)
    _ = await store.ingest_event(second)

    # When: graph context is requested from the first channel scope.
    context = await store.get_context(
        GraphContextRequest(scope=first.scope, query="channel", limit=10),
    )

    # Then: facts from sibling channels never leak into the response.
    assert context.context == "message message-1: alpha"
    assert [fact["id"] for fact in context.facts] == [
        "tenant:nolgia:message:message-1",
    ]


@pytest.mark.anyio
async def test_graph_context_ranks_top_active_channels_for_queried_user() -> None:
    # Given: guild-scoped Discord messages from multiple users across channels.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    for event in (
        _channel_event(
            event_id="channel_created:channel-1",
            idempotency_key="channel_created:channel-1",
            channel_id="channel-1",
            name="general",
        ),
        _channel_event(
            event_id="channel_created:channel-2",
            idempotency_key="channel_created:channel-2",
            channel_id="channel-2",
            name="bot-lab",
        ),
        _message_event(
            _MessageEventValues(
                event_id="message_created:message-1",
                idempotency_key="message_created:message-1",
                message_id="message-1",
                channel_id="channel-1",
                actor_id="black-dave",
                actor_display_name="BlackDave",
            ),
        ),
        _message_event(
            _MessageEventValues(
                event_id="message_created:message-2",
                idempotency_key="message_created:message-2",
                message_id="message-2",
                channel_id="channel-1",
                actor_id="black-dave",
                actor_display_name="BlackDave",
            ),
        ),
        _message_event(
            _MessageEventValues(
                event_id="message_created:message-3",
                idempotency_key="message_created:message-3",
                message_id="message-3",
                channel_id="channel-2",
                actor_id="black-dave",
                actor_display_name="BlackDave",
            ),
        ),
        _message_event(
            _MessageEventValues(
                event_id="message_created:message-4",
                idempotency_key="message_created:message-4",
                message_id="message-4",
                channel_id="channel-1",
                actor_id="cartman",
                actor_display_name="cartman",
            ),
        ),
    ):
        _ = await store.ingest_event(event)

    # When: nolgia-agent asks for the mentioned user's top channels at guild scope.
    context = await store.get_context(
        GraphContextRequest(
            scope=_guild_scope(),
            query="@BlackDave top 5 most active channels??",
            limit=5,
        ),
    )

    # Then: gnosis returns an aggregate answer instead of recent raw messages.
    assert context.context == (
        "BlackDave active channel #1: general (2 messages)\n"
        "BlackDave active channel #2: bot-lab (1 message)"
    )
    assert context.facts == [
        {
            "id": "aggregate:nolgia:guild-123:black-dave:channel-1",
            "type": "channel_activity",
            "scope": "guild",
            "summary": "BlackDave active channel #1: general (2 messages)",
            "deleted": False,
            "rank": 1,
            "user_id": "black-dave",
            "user_display_name": "BlackDave",
            "channel_id": "channel-1",
            "channel_name": "general",
            "message_count": 2,
        },
        {
            "id": "aggregate:nolgia:guild-123:black-dave:channel-2",
            "type": "channel_activity",
            "scope": "guild",
            "summary": "BlackDave active channel #2: bot-lab (1 message)",
            "deleted": False,
            "rank": 2,
            "user_id": "black-dave",
            "user_display_name": "BlackDave",
            "channel_id": "channel-2",
            "channel_name": "bot-lab",
            "message_count": 1,
        },
    ]


@pytest.mark.anyio
async def test_duplicate_idempotency_key_is_noop_duplicate() -> None:
    # Given: the graph store has already accepted an event idempotency key.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    event = _message_event()
    accepted = await store.ingest_event(event)

    # When: the same event is delivered again.
    duplicate = await store.ingest_event(event)

    # Then: the duplicate is reported without changing graph state or history.
    assert accepted.status == EventIngestStatus.ACCEPTED
    assert duplicate.status == EventIngestStatus.DUPLICATE
    assert duplicate.reason == "idempotency key already ingested"
    assert await store.event_count() == 1


@pytest.mark.anyio
async def test_message_delete_creates_tombstone() -> None:
    # Given: a Discord message exists in graph state.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    created = _message_event(_MessageEventValues(content="remove me"))
    deleted = _message_event(
        _MessageEventValues(
            event_id="message_deleted:message-999",
            event_type=ClientEventType.MESSAGE_DELETED,
            idempotency_key="message_deleted:message-999",
            content="",
        ),
    )
    _ = await store.ingest_event(created)

    # When: a delete event is ingested.
    result = await store.ingest_event(deleted)
    context = await store.get_context(
        GraphContextRequest(scope=created.scope, query="message", limit=4),
    )

    # Then: current state is tombstoned while both events remain in history.
    assert result.status == EventIngestStatus.ACCEPTED
    assert context.facts == [
        {
            "id": "tenant:nolgia:message:message-999",
            "type": "message",
            "scope": "channel",
            "summary": "message message-999: deleted",
            "deleted": True,
        },
    ]
    assert await store.event_count() == 2


@pytest.mark.anyio
async def test_channel_rename_updates_current_state_preserving_history() -> None:
    # Given: a channel was created with its original name.
    store = DirectNeo4jGraphStore(executor=InMemoryGraphExecutor())
    created = _channel_event(name="general")
    renamed = _channel_event(
        event_id="channel_updated:channel-456",
        event_type=ClientEventType.CHANNEL_UPDATED,
        idempotency_key="channel_updated:channel-456",
        name="operations",
    )
    _ = await store.ingest_event(created)

    # When: the channel rename arrives.
    result = await store.ingest_event(renamed)
    context = await store.get_context(
        GraphContextRequest(scope=created.scope, query="operations", limit=4),
    )

    # Then: the current channel fact reflects the rename and both events remain.
    assert result.status == EventIngestStatus.ACCEPTED
    assert context.context == "channel channel-456: operations"
    assert context.facts[0]["summary"] == "channel channel-456: operations"
    assert await store.event_count() == 2


def test_upsert_parameters_serializes_message_payload_for_neo4j() -> None:
    # Given: a Discord backfill message carries structured payload metadata.
    event = _message_event().model_copy(
        update={
            "payload": {
                "message_id": "message-999",
                "channel_id": "channel-456",
                "guild_id": "guild-123",
                "content": "remember this",
                "attachment_count": 2,
                "source_marker": "backfill",
            },
        },
    )

    # When: the event is converted to Neo4j upsert parameters.
    parameters = upsert_parameters(plan_event(event))

    # Then: the payload property is a deterministic scalar, not a Neo4j MAP.
    assert parameters["payload"] == (
        '{"attachment_count":2,"channel_id":"channel-456","content":"remember this",'
        '"guild_id":"guild-123","message_id":"message-999","source_marker":"backfill"}'
    )
    assert not isinstance(parameters["payload"], dict)


def test_graph_schema_cypher_declares_constraints_and_indexes() -> None:
    # Given: graph operations rely on uniqueness and query indexes.
    statements = "\n".join(GRAPH_SCHEMA_CYPHER)

    # When / Then: schema bootstrap repairs existing duplicates before constraints.
    assert GRAPH_SCHEMA_CYPHER[0].find("collect(e) AS events") >= 0
    assert GRAPH_SCHEMA_CYPHER[0].find("MERGE (keep)-[:AFFECTS]->(target)") >= 0
    assert GRAPH_SCHEMA_CYPHER[0].find("DETACH DELETE duplicate") >= 0
    assert (
        GRAPH_SCHEMA_CYPHER[1].find(
            "CREATE CONSTRAINT event_idempotency IF NOT EXISTS",
        )
        >= 0
    )
    assert GRAPH_SCHEMA_CYPHER[2].find("collect(n) AS nodes") >= 0
    assert GRAPH_SCHEMA_CYPHER[2].find("MERGE (source)-[:AFFECTS]->(keep)") >= 0
    assert GRAPH_SCHEMA_CYPHER[2].find("DETACH DELETE duplicate") >= 0
    assert (
        GRAPH_SCHEMA_CYPHER[3].find(
            "CREATE CONSTRAINT graph_node_id IF NOT EXISTS",
        )
        >= 0
    )
    assert "CREATE CONSTRAINT event_idempotency IF NOT EXISTS" in statements
    assert "CREATE CONSTRAINT graph_node_id IF NOT EXISTS" in statements
    assert "MATCH (n:GraphNode {type: 'message'})" in statements
    assert "SET n:Message" in statements
    assert "MERGE (u)-[:AUTHORED]->(m)" in statements
    assert "MERGE (m)-[:IN_CHANNEL]->(ch)" in statements
    assert "CREATE INDEX graph_node_scope IF NOT EXISTS" in statements


def test_graph_vector_schema_cypher_uses_configured_dimensions() -> None:
    # Given: embedding dimensions can vary with the selected embedding model.
    statement = graph_vector_schema_cypher(768)

    # When / Then: the Neo4j vector index matches runtime configuration.
    assert "CREATE VECTOR INDEX graph_node_embedding IF NOT EXISTS" in statement
    assert "`vector.dimensions`: 768" in statement


@pytest.mark.anyio
async def test_neo4j_executor_bootstraps_schema_once_before_operations() -> None:
    # Given: a Neo4j executor connected to a recording driver.
    driver = RecordingCypherDriver()
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=768,
    )
    event = plan_event(_message_event())

    # When: multiple graph operations run through the executor.
    _ = await executor.upsert_event(event)
    _ = await executor.get_context(
        GraphContextRequest(scope=event.event.scope, query="message", limit=4),
    )

    # Then: schema bootstrap runs idempotently once before operation Cypher.
    assert driver.queries[: len(GRAPH_SCHEMA_CYPHER)] == list(GRAPH_SCHEMA_CYPHER)
    assert "collect(e) AS events" in driver.queries[0]
    assert "CREATE CONSTRAINT event_idempotency IF NOT EXISTS" in driver.queries[1]
    assert "collect(n) AS nodes" in driver.queries[2]
    assert "CREATE CONSTRAINT graph_node_id IF NOT EXISTS" in driver.queries[3]
    assert driver.queries.count(GRAPH_SCHEMA_CYPHER[0]) == 1
    assert driver.queries[len(GRAPH_SCHEMA_CYPHER)] == graph_vector_schema_cypher(768)
    assert driver.queries[len(GRAPH_SCHEMA_CYPHER) + 1] != GRAPH_SCHEMA_CYPHER[0]


@pytest.mark.anyio
async def test_neo4j_executor_readiness_reports_unavailable_on_failure() -> None:
    # Given: schema bootstrap cannot reach Neo4j.
    executor = Neo4jGraphExecutor(
        driver_factory=FailingCypherDriverFactory(),
        embedding_dimensions=1024,
    )

    # When: readiness is checked for Kubernetes.
    readiness = await executor.readiness()

    # Then: readiness degrades to a safe unavailable status instead of leaking errors.
    assert readiness.graph == "unavailable"
    assert readiness.schema_status == "unavailable"


@pytest.mark.anyio
async def test_in_memory_executor_records_schema_bootstrap_before_operations() -> None:
    # Given: the in-memory executor models the production schema lifecycle.
    executor = InMemoryGraphExecutor()
    store = DirectNeo4jGraphStore(executor=executor)

    # When: graph ingestion occurs.
    _ = await store.ingest_event(_message_event())

    # Then: tests can assert schema readiness without a Neo4j instance.
    assert executor.schema_bootstrap_count_for_test() == 1


def _scope(*, channel_id: str = "channel-456") -> MemoryScope:
    return MemoryScope(
        tenant_id="nolgia",
        space_id="discord",
        agent_id="nolgia-agent",
        session_id=f"guild:guild-123:channel:{channel_id}",
        user_id="user-789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="guild-123",
        channel_id=channel_id,
    )


def _guild_scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="nolgia",
        space_id="discord",
        agent_id="nolgia-agent",
        session_id="guild:guild-123",
        user_id="user-789",
        visibility=MemoryVisibility.GUILD,
        guild_id="guild-123",
    )


@dataclass(frozen=True, slots=True)
class _MessageEventValues:
    event_id: str = "message_created:message-999"
    event_type: ClientEventType = ClientEventType.MESSAGE_CREATED
    idempotency_key: str = "message_created:message-999"
    message_id: str = "message-999"
    channel_id: str = "channel-456"
    content: str = "remember this"
    actor_id: str = "user-789"
    actor_display_name: str = "cartman"


def _message_event(
    values: _MessageEventValues | None = None,
) -> ClientEvent:
    event_values = values or _MessageEventValues()
    return ClientEvent(
        tenant_id="nolgia",
        source_client=SourceClient.DISCORD,
        agent_id="nolgia-agent",
        event_id=event_values.event_id,
        event_type=event_values.event_type,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key=event_values.idempotency_key,
        scope=_scope(channel_id=event_values.channel_id),
        actor=ClientEventActor(
            id=event_values.actor_id,
            display_name=event_values.actor_display_name,
            is_bot=False,
        ),
        subject=ClientEventSubject(
            id=event_values.message_id,
            type="message",
            parent_id=event_values.channel_id,
        ),
        payload={
            "message_id": event_values.message_id,
            "channel_id": event_values.channel_id,
            "guild_id": "guild-123",
            "content": event_values.content,
        },
        discord=DiscordEventContext(
            guild_id="guild-123",
            channel_id=event_values.channel_id,
            message_id=event_values.message_id,
        ),
    )


def _channel_event(
    *,
    event_id: str = "channel_created:channel-456",
    event_type: ClientEventType = ClientEventType.CHANNEL_CREATED,
    idempotency_key: str = "channel_created:channel-456",
    channel_id: str = "channel-456",
    name: str,
) -> ClientEvent:
    return ClientEvent(
        tenant_id="nolgia",
        source_client=SourceClient.DISCORD,
        agent_id="nolgia-agent",
        event_id=event_id,
        event_type=event_type,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key=idempotency_key,
        scope=_scope(channel_id=channel_id),
        actor=ClientEventActor(id="system", display_name="discord", is_bot=True),
        subject=ClientEventSubject(id=channel_id, type="channel"),
        payload={
            "channel_id": channel_id,
            "guild_id": "guild-123",
            "name": name,
        },
        discord=DiscordEventContext(guild_id="guild-123", channel_id=channel_id),
    )


@dataclass(slots=True)
class RecordingCypherDriver:
    queries: list[str] = field(default_factory=list)

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, JsonValue],
    ) -> Sequence[dict[str, JsonValue]]:
        _ = parameters
        self.queries.append(query)
        return [{"duplicate": False}]

    async def execute_read_query(
        self,
        query: str,
        parameters: dict[str, JsonValue],
    ) -> Sequence[dict[str, JsonValue]]:
        return await self.execute_query(query, parameters)

    async def verify_connectivity(self) -> None:
        self.queries.append("verify_connectivity")

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
class RecordingDriverFactory:
    driver: RecordingCypherDriver

    def __call__(self) -> RecordingCypherDriver:
        return self.driver


@dataclass(frozen=True, slots=True)
class FailingCypherDriverFactory:
    def __call__(self) -> "FailingCypherDriver":
        return FailingCypherDriver()


@dataclass(frozen=True, slots=True)
class FailingCypherDriver:
    async def execute_query(
        self,
        query: str,
        parameters: dict[str, JsonValue],
    ) -> Sequence[dict[str, JsonValue]]:
        _ = (query, parameters)
        reason = "connection refused"
        raise OSError(reason)

    async def execute_read_query(
        self,
        query: str,
        parameters: dict[str, JsonValue],
    ) -> Sequence[dict[str, JsonValue]]:
        return await self.execute_query(query, parameters)

    async def verify_connectivity(self) -> None:
        reason = "connection refused"
        raise OSError(reason)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)
