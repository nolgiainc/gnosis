from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self, override

from neo4j import RoutingControl
from neo4j.exceptions import Neo4jError

from gnosis.models import (
    BackendReadiness,
    ClientEvent,
    EventIngestResult,
    GraphContextRequest,
    GraphContextResponse,
    JsonValue,
)
from gnosis.settings import Settings

type CypherParameters = dict[str, JsonValue]
type CypherRow = dict[str, JsonValue]

if TYPE_CHECKING:

    class AsyncGraphDatabaseType(Protocol):
        def driver(
            self,
            uri: str,
            *,
            auth: tuple[str, str],
        ) -> "AsyncNeo4jRawDriver": ...

    AsyncGraphDatabase: AsyncGraphDatabaseType
else:
    from neo4j import AsyncGraphDatabase


class AsyncNeo4jDriver(Protocol):
    async def verify_connectivity(self) -> None: ...
    async def execute_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> list[CypherRow]: ...
    async def execute_read_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> list[CypherRow]: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


class ConnectivityNeo4jDriver(Protocol):
    async def verify_connectivity(self) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None: ...


class AsyncNeo4jRawDriver(Protocol):
    async def verify_connectivity(self) -> None: ...
    async def execute_query(
        self,
        query_: str,
        parameters_: CypherParameters,
        routing_: RoutingControl = ...,
    ) -> tuple[list[CypherRow], object, object]: ...
    async def close(self) -> None: ...


class DirectNeo4jDriverFactory(Protocol):
    def __call__(self) -> AsyncNeo4jDriver: ...


class ConnectivityNeo4jDriverFactory(Protocol):
    def __call__(self) -> ConnectivityNeo4jDriver: ...


class StructuredGraphStore(Protocol):
    async def require_available(self) -> None: ...
    async def readiness(self) -> BackendReadiness: ...
    async def ingest_event(self, event: ClientEvent) -> EventIngestResult: ...
    async def get_context(
        self,
        request: GraphContextRequest,
    ) -> GraphContextResponse: ...


@dataclass(frozen=True, slots=True)
class GraphPersistenceUnavailableError(RuntimeError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Neo4j structured graph persistence is unavailable: {self.reason}"


@dataclass(frozen=True, slots=True)
class DirectNeo4jProbe:
    driver_factory: ConnectivityNeo4jDriverFactory

    async def require_available(self) -> None:
        try:
            async with self.driver_factory() as driver:
                await driver.verify_connectivity()
        except (Neo4jError, OSError) as error:
            raise GraphPersistenceUnavailableError(str(error)) from error


@dataclass(frozen=True, slots=True)
class DirectNeo4jDriverContext:
    driver: AsyncNeo4jRawDriver

    async def verify_connectivity(self) -> None:
        await self.driver.verify_connectivity()

    async def execute_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> list[CypherRow]:
        records, _, _ = await self.driver.execute_query(query, parameters)
        return records

    async def execute_read_query(
        self,
        query: str,
        parameters: CypherParameters,
    ) -> list[CypherRow]:
        """Run ``query`` in a READ-access transaction.

        The server enforces the access mode, so a write clause fails with
        ``Neo.ClientError.Statement.AccessMode`` instead of executing. Records
        are copied into plain dicts: a ``neo4j.Record`` is a tuple, so ``in``
        tests its values and ``keys()`` returns a list, which breaks key-based
        row checks downstream.
        """
        records, _, _ = await self.driver.execute_query(
            query,
            parameters,
            routing_=RoutingControl.READ,
        )
        return [dict(record) for record in records]

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)
        await self.driver.close()


def direct_neo4j_driver_factory(settings: Settings) -> DirectNeo4jDriverFactory:
    def create_driver() -> DirectNeo4jDriverContext:
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        return DirectNeo4jDriverContext(driver=driver)

    return create_driver
