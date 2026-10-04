import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self, override

import pytest

from gnosis.graph_query_qa import GraphQueryPlan, GraphQueryPlanner
from gnosis.graph_store import Neo4jGraphExecutor
from gnosis.graph_types import CypherParameters
from gnosis.models import GraphContextRequest, JsonValue, MemoryScope, MemoryVisibility


@pytest.mark.anyio
async def test_dynamic_graph_query_runs_after_activity_aggregate_miss() -> None:
    # Given: the deterministic activity query finds no rows, and a planner can answer.
    driver = RecordingCypherDriver(rows_by_query=[[], [_graph_row()]])
    planner = StaticGraphQueryPlanner(
        plan=GraphQueryPlan(
            cypher="""
            MATCH (ch:Channel {tenant_id: $tenant_id})
            WHERE ch.guild_id = $guild_id AND ch.channel_id = $channel_id
            RETURN ch.id AS id, ch.tenant_id AS tenant_id, 'graph_query' AS type,
              coalesce(ch.name, ch.channel_id) AS summary, false AS deleted
            ORDER BY summary ASC
            LIMIT $limit
            """,
            parameters={},
            answer_kind="channels_by_guild",
        ),
    )
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=3,
        graph_query_planner=planner,
    )
    request = GraphContextRequest(
        scope=_scope(),
        query="@cartman top 5 active channels and list channels if none",
        limit=5,
    )

    # When: graph context is requested.
    nodes = await executor.get_context(request)

    # Then: gnosis tries deterministic Cypher first, then executes the safe plan.
    assert len(nodes) == 1
    assert nodes[0].summary == "general-chat"
    assert planner.requests == [request]
    assert driver.read_queries == [planner.plan.cypher if planner.plan else ""]
    assert driver.parameters[-1]["tenant_id"] == "nolgia"
    assert driver.parameters[-1]["guild_id"] == "guild-123"


@pytest.mark.anyio
async def test_dynamic_graph_query_falls_back_when_planner_fails() -> None:
    # Given: the planner fails, but basic graph context can still answer.
    driver = RecordingCypherDriver(rows_by_query=[[_graph_row()]])
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=3,
        graph_query_planner=FailingGraphQueryPlanner(),
    )
    request = GraphContextRequest(scope=_scope(), query="Which channel?", limit=5)

    # When: graph context is requested.
    nodes = await executor.get_context(request)

    # Then: the planner failure is non-fatal and fallback context is returned.
    assert len(nodes) == 1
    assert nodes[0].summary == "general-chat"


@pytest.mark.anyio
async def test_dynamic_graph_query_falls_back_when_rows_have_bad_shape() -> None:
    # Given: the planned query returns a row missing the required graph shape.
    driver = RecordingCypherDriver(rows_by_query=[[{"id": "bad"}], [_graph_row()]])
    planner = StaticGraphQueryPlanner(
        plan=GraphQueryPlan(
            cypher="""
            MATCH (ch:Channel {tenant_id: $tenant_id, agent_id: $agent_id})
            WHERE ch.guild_id = $guild_id AND ch.channel_id = $channel_id
            RETURN ch.id AS id, ch.tenant_id AS tenant_id, 'graph_query' AS type,
              ch.name AS summary, false AS deleted
            LIMIT $limit
            """,
            parameters={},
            answer_kind="channels_by_guild",
        ),
    )
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=3,
        graph_query_planner=planner,
    )
    request = GraphContextRequest(scope=_scope(), query="Which channel?", limit=5)

    # When: graph context is requested.
    nodes = await executor.get_context(request)

    # Then: malformed dynamic rows do not suppress fallback context.
    assert len(nodes) == 1
    assert nodes[0].summary == "general-chat"


@pytest.mark.anyio
async def test_dynamic_graph_query_drops_rows_outside_caller_tenant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a validated plan whose execution (e.g. through a validator gap)
    # returns rows from another tenant, rows without a tenant column, and a
    # row whose secondary node belongs to another tenant.
    foreign = {**_graph_row(), "tenant_id": "other-tenant", "summary": "secret-a"}
    untagged = {k: v for k, v in _graph_row().items() if k != "tenant_id"}
    untagged["summary"] = "secret-b"
    mixed = {**_graph_row(), "e_tenant_id": "other-tenant", "summary": "secret-c"}
    driver = RecordingCypherDriver(
        rows_by_query=[[_graph_row(), foreign, untagged, mixed]],
    )
    planner = StaticGraphQueryPlanner(
        plan=GraphQueryPlan(
            cypher="""
            MATCH (ch:Channel {tenant_id: $tenant_id, agent_id: $agent_id})
            WHERE ch.guild_id = $guild_id AND ch.channel_id = $channel_id
            RETURN ch.id AS id, ch.tenant_id AS tenant_id, 'graph_query' AS type,
              ch.name AS summary, false AS deleted
            LIMIT $limit
            """,
            parameters={},
            answer_kind="channels_by_guild",
        ),
    )
    executor = Neo4jGraphExecutor(
        driver_factory=RecordingDriverFactory(driver),
        embedding_dimensions=3,
        graph_query_planner=planner,
    )
    request = GraphContextRequest(scope=_scope(), query="Which channel?", limit=5)

    # When: graph context is requested.
    with caplog.at_level(logging.WARNING, logger="gnosis.graph_query_execution"):
        nodes = await executor.get_context(request)

    # Then: the planned query ran read-only, only the caller-tenant row is
    # stamped with the caller scope, and the drop is logged as counts only.
    assert len(driver.read_queries) == 1
    assert [node.summary for node in nodes] == ["general-chat"]
    dropped = [
        record
        for record in caplog.records
        if record.getMessage() == "graph QA dropped rows outside the caller tenant"
    ]
    assert len(dropped) == 1
    assert dropped[0].__dict__["dropped_count"] == 3
    assert dropped[0].__dict__["kept_count"] == 1
    assert "secret" not in caplog.text


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


def _graph_row() -> dict[str, JsonValue]:
    return {
        "id": "tenant:nolgia:channel:channel-456",
        "type": "graph_query",
        "summary": "general-chat",
        "deleted": False,
        "tenant_id": "nolgia",
    }


@dataclass(slots=True)
class RecordingCypherDriver:
    rows_by_query: list[Sequence[dict[str, JsonValue]]]
    queries: list[str] = field(default_factory=list)
    parameters: list[CypherParameters] = field(default_factory=list)
    read_queries: list[str] = field(default_factory=list)

    async def execute_read_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> Sequence[dict[str, JsonValue]]:
        self.read_queries.append(query)
        return await self.execute_query(query, parameters)

    async def execute_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> Sequence[dict[str, JsonValue]]:
        self.queries.append(query)
        self.parameters.append(parameters)
        if (
            "graph_query" not in query
            and "channel_activity" not in query
            and "RETURN n.id AS id" not in query
        ):
            return []
        if self.rows_by_query:
            return self.rows_by_query.pop(0)
        return []

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


@dataclass(slots=True)
class StaticGraphQueryPlanner(GraphQueryPlanner):
    plan: GraphQueryPlan | None
    requests: list[GraphContextRequest] = field(default_factory=list)

    @override
    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        self.requests.append(request)
        return self.plan


@dataclass(frozen=True, slots=True)
class FailingGraphQueryPlanner(GraphQueryPlanner):
    @override
    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        _ = request
        reason = "planner failed"
        raise RuntimeError(reason)
