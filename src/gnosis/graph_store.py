import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, Self

from neo4j.exceptions import Neo4jError

from gnosis.graph_cypher import (
    CONTEXT_CYPHER,
    SEMANTIC_CONTEXT_CYPHER,
    TOP_ACTIVE_CHANNELS_CYPHER,
    UPSERT_EVENT_CYPHER,
    CypherParameters,
    context_parameters,
    is_duplicate_result,
    is_top_active_channels_request,
    top_active_channel_parameters,
    upsert_parameters,
)
from gnosis.graph_events import (
    GraphNode,
    PlannedGraphEvent,
    fact_from_node,
    node_from_row,
    plan_event,
)
from gnosis.graph_memory_store import InMemoryGraphExecutor
from gnosis.graph_query_execution import plan_graph_query, rows_to_graph_nodes
from gnosis.graph_query_qa import GraphQueryPlanner
from gnosis.graph_query_validation import (
    GraphQueryValidationError,
    SafeGraphQueryValidator,
)
from gnosis.graph_schema import GRAPH_SCHEMA_CYPHER, graph_vector_schema_cypher
from gnosis.graph_types import vector_parameter
from gnosis.models import (
    BackendReadiness,
    ClientEvent,
    EventIngestResult,
    EventIngestStatus,
    GraphContextRequest,
    GraphContextResponse,
    JsonValue,
)

_LOGGER = logging.getLogger(__name__)
__all__ = [
    "CypherDriver",
    "CypherDriverFactory",
    "DirectNeo4jGraphStore",
    "GraphExecutor",
    "InMemoryGraphExecutor",
    "Neo4jGraphExecutor",
    "TextEmbeddingProvider",
]


class CypherDriver(Protocol):
    async def execute_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> Sequence[dict[str, JsonValue]]: ...
    async def execute_read_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> Sequence[dict[str, JsonValue]]: ...
    async def verify_connectivity(self) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


class CypherDriverFactory(Protocol):
    def __call__(self) -> CypherDriver: ...


class TextEmbeddingProvider(Protocol):
    async def embed_one(self, text: str) -> list[float]: ...


class GraphExecutor(Protocol):
    async def require_available(self) -> None: ...
    async def readiness(self) -> BackendReadiness: ...
    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult: ...
    async def get_context(
        self,
        request: GraphContextRequest,
    ) -> Sequence[GraphNode]: ...


@dataclass(slots=True)
class Neo4jGraphExecutor:
    driver_factory: CypherDriverFactory
    embedding_dimensions: int
    embedding_provider: TextEmbeddingProvider | None = None
    graph_query_planner: GraphQueryPlanner | None = None
    _schema_bootstrapped: bool = False

    async def require_available(self) -> None:
        await self._bootstrap_schema()
        async with self.driver_factory() as driver:
            await driver.verify_connectivity()

    async def readiness(self) -> BackendReadiness:
        try:
            await self.require_available()
        except (Neo4jError, OSError):
            return BackendReadiness(graph="unavailable", schema="unavailable")
        return BackendReadiness(graph="ready", schema="ready")

    async def upsert_event(self, event: PlannedGraphEvent) -> EventIngestResult:
        await self._bootstrap_schema()
        parameters = upsert_parameters(event)
        if self.embedding_provider is not None:
            parameters["node_embedding"] = vector_parameter(
                await self.embedding_provider.embed_one(event.node.summary),
            )
        async with self.driver_factory() as driver:
            rows = await driver.execute_query(
                UPSERT_EVENT_CYPHER,
                parameters,
            )
        if is_duplicate_result(rows):
            return EventIngestResult(
                event_id=event.event.event_id,
                status=EventIngestStatus.DUPLICATE,
                reason="idempotency key already ingested",
            )
        return EventIngestResult(
            event_id=event.event.event_id,
            status=EventIngestStatus.ACCEPTED,
        )

    async def get_context(self, request: GraphContextRequest) -> Sequence[GraphNode]:
        await self._bootstrap_schema()
        if is_top_active_channels_request(request):
            async with self.driver_factory() as driver:
                rows = await driver.execute_query(
                    TOP_ACTIVE_CHANNELS_CYPHER,
                    top_active_channel_parameters(request),
                )
            if rows:
                return tuple(node_from_row(row, request.scope) for row in rows)
        if self.graph_query_planner is not None:
            planned_nodes = await self._get_planned_graph_context(request)
            if planned_nodes:
                return planned_nodes
        query = CONTEXT_CYPHER
        parameters = context_parameters(request)
        if self.embedding_provider is not None:
            query = SEMANTIC_CONTEXT_CYPHER
            parameters = context_parameters(
                request,
                await self.embedding_provider.embed_one(request.query),
            )
        async with self.driver_factory() as driver:
            rows = await driver.execute_query(
                query,
                parameters,
            )
        return tuple(node_from_row(row, request.scope) for row in rows)

    async def _get_planned_graph_context(
        self,
        request: GraphContextRequest,
    ) -> Sequence[GraphNode]:
        if self.graph_query_planner is None:
            return ()
        plan = await plan_graph_query(self.graph_query_planner, request)
        if plan is None:
            return ()
        try:
            validated = SafeGraphQueryValidator().validate(plan, request)
        except GraphQueryValidationError as error:
            _LOGGER.info(
                "graph QA generated query rejected",
                extra={
                    "answer_kind": plan.answer_kind,
                    "reason": error.reason,
                    "tenant_id": request.scope.tenant_id,
                    "guild_id": request.scope.guild_id,
                    "channel_id": request.scope.channel_id,
                },
            )
            return ()
        try:
            # Planned Cypher is LLM-generated: run it in a READ transaction so
            # the server refuses any write the validator might have missed.
            async with self.driver_factory() as driver:
                rows = await driver.execute_read_query(
                    validated.cypher,
                    validated.parameters,
                )
        except (Neo4jError, OSError) as error:
            _LOGGER.info(
                "graph QA query failed",
                extra={
                    "answer_kind": validated.answer_kind,
                    "error_type": type(error).__name__,
                    "tenant_id": request.scope.tenant_id,
                    "guild_id": request.scope.guild_id,
                    "channel_id": request.scope.channel_id,
                },
            )
            return ()
        _LOGGER.info(
            "graph QA query executed",
            extra={
                "answer_kind": validated.answer_kind,
                "row_count": len(rows),
                "tenant_id": request.scope.tenant_id,
                "guild_id": request.scope.guild_id,
                "channel_id": request.scope.channel_id,
            },
        )
        return rows_to_graph_nodes(rows, request, validated)

    async def _bootstrap_schema(self) -> None:
        if self._schema_bootstrapped:
            return
        async with self.driver_factory() as driver:
            for statement in GRAPH_SCHEMA_CYPHER:
                _ = await driver.execute_query(statement, {})
            _ = await driver.execute_query(
                graph_vector_schema_cypher(self.embedding_dimensions),
                {},
            )
        self._schema_bootstrapped = True


@dataclass(frozen=True, slots=True)
class DirectNeo4jGraphStore:
    executor: GraphExecutor

    async def require_available(self) -> None:
        await self.executor.require_available()

    async def readiness(self) -> BackendReadiness:
        return await self.executor.readiness()

    async def ingest_event(self, event: ClientEvent) -> EventIngestResult:
        await self.require_available()
        return await self.executor.upsert_event(plan_event(event))

    async def get_context(self, request: GraphContextRequest) -> GraphContextResponse:
        await self.require_available()
        nodes = await self.executor.get_context(request)
        return GraphContextResponse(
            context="\n".join(node.summary for node in nodes),
            facts=[fact_from_node(node) for node in nodes],
        )

    async def event_count(self) -> int:
        if isinstance(self.executor, InMemoryGraphExecutor):
            return self.executor.event_count
        return 0
