import logging
from collections.abc import Sequence
from typing import Final

from neo4j.exceptions import Neo4jError
from openai import OpenAIError
from pydantic import ValidationError

from gnosis.graph_events import GraphNode, node_from_row
from gnosis.graph_query_qa import GraphQueryPlan, GraphQueryPlanner, ValidatedGraphQuery
from gnosis.graph_types import CypherParameters
from gnosis.models import GraphContextRequest, JsonValue

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_GRAPH_QUERY_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {"id", "type", "summary", "deleted"},
)
_TENANT_COLUMN: Final[str] = "tenant_id"


async def plan_graph_query(
    planner: GraphQueryPlanner,
    request: GraphContextRequest,
) -> GraphQueryPlan | None:
    try:
        return await planner.plan_query(request)
    except (RuntimeError, OSError, Neo4jError, OpenAIError, ValidationError) as error:
        # ValidationError: the planner LLM sometimes ignores the structured-output
        # contract and returns prose or fenced JSON, which `.parse()` rejects.
        # Graph QA is a best-effort enhancement; a planner miss must degrade to
        # no graph context, never fail the caller's request.
        _LOGGER.info(
            "graph QA planner failed",
            extra=_error_context(error, request),
        )
        return None


def rows_to_graph_nodes(
    rows: Sequence[dict[str, JsonValue]],
    request: GraphContextRequest,
    query: ValidatedGraphQuery,
) -> Sequence[GraphNode]:
    rows = _rows_in_caller_tenant(rows, request, query)
    if not _rows_have_graph_query_shape(rows):
        _LOGGER.info(
            "graph QA query returned invalid row shape",
            extra={
                "answer_kind": query.answer_kind,
                "row_count": len(rows),
                "tenant_id": request.scope.tenant_id,
                "guild_id": request.scope.guild_id,
                "channel_id": request.scope.channel_id,
            },
        )
        return ()
    return tuple(node_from_row(row, request.scope) for row in rows)


def _rows_in_caller_tenant(
    rows: Sequence[dict[str, JsonValue]],
    request: GraphContextRequest,
    query: ValidatedGraphQuery,
) -> Sequence[dict[str, JsonValue]]:
    """Drop planned rows that are not provably in the caller's tenant.

    The validator requires every planned query to project the tenant of each
    returned node (``tenant_id`` for the node supplying ``id``, a
    ``*tenant_id`` column for any other). Rows are later stamped with the
    caller's scope, so a row missing ``tenant_id`` or carrying any tenant
    column that differs from the caller is discarded here rather than trusted.
    Only counts are logged - never row data.
    """
    tenant_id = request.scope.tenant_id
    kept = tuple(row for row in rows if _row_in_tenant(row, tenant_id))
    dropped = len(rows) - len(kept)
    if dropped:
        _LOGGER.warning(
            "graph QA dropped rows outside the caller tenant",
            extra={
                "answer_kind": query.answer_kind,
                "dropped_count": dropped,
                "kept_count": len(kept),
                "tenant_id": tenant_id,
            },
        )
    return kept


def _row_in_tenant(row: dict[str, JsonValue], tenant_id: str) -> bool:
    if _TENANT_COLUMN not in row:
        return False
    return all(
        value == tenant_id for key, value in row.items() if key.endswith(_TENANT_COLUMN)
    )


def _rows_have_graph_query_shape(rows: Sequence[dict[str, JsonValue]]) -> bool:
    return all(row.keys() >= _GRAPH_QUERY_ROW_KEYS for row in rows)


def _error_context(
    error: RuntimeError | OSError | Neo4jError | OpenAIError | ValidationError,
    request: GraphContextRequest,
) -> CypherParameters:
    return {
        "error_type": type(error).__name__,
        "tenant_id": request.scope.tenant_id,
        "guild_id": request.scope.guild_id,
        "channel_id": request.scope.channel_id,
    }
