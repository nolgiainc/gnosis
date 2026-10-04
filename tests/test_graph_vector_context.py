from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import pytest

from gnosis.graph_cypher import SEMANTIC_CONTEXT_CYPHER, TOP_ACTIVE_CHANNELS_CYPHER
from gnosis.graph_events import plan_event
from gnosis.graph_store import Neo4jGraphExecutor
from gnosis.graph_types import CypherParameters
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
async def test_neo4j_executor_uses_semantic_context_with_embeddings() -> None:
    driver = RecordingCypherDriver()
    embedding_provider = StaticEmbeddingProvider(vector=[0.1, 0.2, 0.3])
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=3,
        embedding_provider=embedding_provider,
    )
    request = GraphContextRequest(scope=_scope(), query="plasma conduit", limit=4)

    _ = await executor.get_context(request)

    assert driver.queries[-1] == SEMANTIC_CONTEXT_CYPHER
    assert driver.parameters[-1]["query_embedding"] == [0.1, 0.2, 0.3]
    assert driver.parameters[-1]["vector_limit"] == 16
    assert embedding_provider.embedded_texts == ["plasma conduit"]


@pytest.mark.anyio
async def test_neo4j_executor_uses_activity_aggregate_before_embeddings() -> None:
    # Given: a Neo4j executor has embeddings configured for normal context recall.
    driver = RecordingCypherDriver()
    embedding_provider = StaticEmbeddingProvider(vector=[0.1, 0.2, 0.3])
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=3,
        embedding_provider=embedding_provider,
    )
    request = GraphContextRequest(
        scope=_scope(),
        query="@BlackDave top 5 most active channels??",
        limit=5,
    )

    # When: the query requests a Discord channel activity aggregate.
    _ = await executor.get_context(request)

    # Then: the aggregate Cypher runs without spending an embedding request.
    assert driver.queries[-1] == TOP_ACTIVE_CHANNELS_CYPHER
    assert driver.parameters[-1]["user_token"] == "blackdave"
    assert driver.parameters[-1]["limit"] == 5
    assert embedding_provider.embedded_texts == []


def test_top_active_channels_cypher_uses_graph_guild_membership() -> None:
    # Given: channel properties can be poisoned by Discord channel/guild ID collisions.
    # When: gnosis ranks channel activity for a guild.
    # Then: the aggregate can recover channels through graph guild membership.
    assert "OPTIONAL MATCH (ch)-[:IN_GUILD]->(guild:Guild" in TOP_ACTIVE_CHANNELS_CYPHER
    assert "$guild_id IN guild_ids" in TOP_ACTIVE_CHANNELS_CYPHER


@pytest.mark.anyio
async def test_neo4j_executor_embeds_graph_node_on_upsert() -> None:
    driver = RecordingCypherDriver()
    embedding_provider = StaticEmbeddingProvider(vector=[0.4, 0.5, 0.6])
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=3,
        embedding_provider=embedding_provider,
    )

    result = await executor.upsert_event(plan_event(_message_event()))

    assert result.status == EventIngestStatus.ACCEPTED
    assert driver.parameters[-1]["node_embedding"] == [0.4, 0.5, 0.6]
    assert embedding_provider.embedded_texts == ["message message-999: remember this"]


def _scope() -> MemoryScope:
    return MemoryScope(
        tenant_id="nolgia",
        space_id="discord",
        agent_id="nolgia-agent",
        session_id="guild:guild-123:channel:channel-456",
        user_id="user-789",
        visibility=MemoryVisibility.CHANNEL,
        guild_id="guild-123",
        channel_id="channel-456",
    )


def _message_event() -> ClientEvent:
    return ClientEvent(
        tenant_id="nolgia",
        source_client=SourceClient.DISCORD,
        agent_id="nolgia-agent",
        event_id="message_created:message-999",
        event_type=ClientEventType.MESSAGE_CREATED,
        occurred_at="2026-06-27T01:02:03Z",
        observed_at="2026-06-27T01:02:04Z",
        idempotency_key="message_created:message-999",
        scope=_scope(),
        actor=ClientEventActor(id="user-789", display_name="cartman", is_bot=False),
        subject=ClientEventSubject(
            id="message-999",
            type="message",
            parent_id="channel-456",
        ),
        payload={
            "message_id": "message-999",
            "channel_id": "channel-456",
            "guild_id": "guild-123",
            "content": "remember this",
        },
        discord=DiscordEventContext(
            guild_id="guild-123",
            channel_id="channel-456",
            message_id="message-999",
        ),
    )


@dataclass(slots=True)
class RecordingCypherDriver:
    queries: list[str] = field(default_factory=list)
    parameters: list[CypherParameters] = field(default_factory=list)

    async def execute_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> Sequence[dict[str, JsonValue]]:
        self.queries.append(query)
        self.parameters.append(parameters)
        return [{"duplicate": False}]

    async def execute_read_query(
        self,
        query: str,
        parameters: CypherParameters,
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
class StaticEmbeddingProvider:
    vector: list[float]
    embedded_texts: list[str] = field(default_factory=list)

    async def embed_one(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return self.vector
