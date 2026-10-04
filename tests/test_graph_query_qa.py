import httpx
import pytest
from openai import APIConnectionError

from gnosis.graph_query_execution import plan_graph_query
from gnosis.graph_query_qa import GraphQueryPlan, proxy_model_name
from gnosis.graph_query_validation import (
    GraphQueryValidationError,
    SafeGraphQueryValidator,
)
from gnosis.models import GraphContextRequest, MemoryScope, MemoryVisibility


def test_validator_accepts_scoped_read_only_query() -> None:
    # Given: a generated query uses only approved labels and scope parameters.
    request = GraphContextRequest(scope=_scope(), query="Which roles exist?", limit=5)
    plan = GraphQueryPlan(
        cypher="""
        MATCH (r:Role {tenant_id: $tenant_id})
        WHERE r.guild_id = $guild_id
        RETURN r.id AS id, r.tenant_id AS tenant_id, 'graph_query' AS type,
          coalesce(r.name, r.role_id) AS summary, false AS deleted
        ORDER BY summary ASC
        LIMIT $limit
        """,
        parameters={},
        answer_kind="roles_by_guild",
    )

    # When: gnosis validates the query before execution.
    validated = SafeGraphQueryValidator().validate(plan, request)

    # Then: runtime scope parameters are injected and the generated query is kept.
    assert validated.cypher == plan.cypher
    assert validated.parameters["tenant_id"] == "nolgia"
    assert validated.parameters["guild_id"] == "guild-123"
    assert validated.parameters["limit"] == 5


@pytest.mark.parametrize(
    "cypher",
    [
        "MATCH (n) DETACH DELETE n RETURN n LIMIT $limit",
        "MERGE (n:Message {tenant_id: $tenant_id}) RETURN n LIMIT $limit",
        "CALL db.labels() YIELD label RETURN label LIMIT $limit",
        "MATCH (n:Message {tenant_id: 'nolgia'}) RETURN n LIMIT $limit",
        "MATCH (n:Message {tenant_id: $tenant_id}) RETURN n",
        "MATCH (n:Secret {tenant_id: $tenant_id}) RETURN n LIMIT $limit",
    ],
)
def test_validator_rejects_unsafe_generated_cypher(cypher: str) -> None:
    # Given: a generated query violates one graph QA safety rule.
    request = GraphContextRequest(scope=_scope(), query="anything", limit=5)
    plan = GraphQueryPlan(
        cypher=cypher,
        parameters={},
        answer_kind="unsafe",
    )

    # When / Then: validation blocks the query before Neo4j can execute it.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


@pytest.mark.parametrize(
    "cypher",
    [
        """
        MATCH (t:Tenant {tenant_id: $tenant_id})
        WITH t
        MATCH (m:Message)
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        WITH $tenant_id AS tenant_id
        MATCH (m:Message)
        WHERE tenant_id = $tenant_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:Message {tenant_id: $tenant_id})
        WHERE m.guild_id = $guild_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
    ],
)
def test_validator_rejects_scope_bypass_queries(cypher: str) -> None:
    # Given: generated Cypher mentions scope without applying full runtime scope.
    request = GraphContextRequest(scope=_scope(), query="messages", limit=5)
    plan = GraphQueryPlan(cypher=cypher, parameters={}, answer_kind="messages")

    # When / Then: validation blocks the query before Neo4j can execute it.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


@pytest.mark.parametrize(
    "cypher",
    [
        """
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        MATCH (m)-[r:SECRET_REL]->(u:User)
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:`Secret` {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, 'graph_query' AS type,
          m.`secret` AS summary, false AS deleted
        LIMIT $limit
        """,
        """
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, 'graph_query' AS type,
          properties(m) AS summary, false AS deleted
        LIMIT $limit
        """,
    ],
)
def test_validator_rejects_unsupported_schema_syntax(cypher: str) -> None:
    # Given: generated Cypher uses syntax outside the regex allowlist contract.
    request = GraphContextRequest(scope=_scope(), query="messages", limit=5)
    plan = GraphQueryPlan(cypher=cypher, parameters={}, answer_kind="messages")

    # When / Then: validation rejects the unsupported schema access.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


def test_validator_accepts_fully_scoped_message_query() -> None:
    # Given: a message query applies every runtime scope field to the read alias.
    request = GraphContextRequest(scope=_scope(), query="messages", limit=5)
    plan = GraphQueryPlan(
        cypher="""
        MATCH (m:Message {tenant_id: $tenant_id, agent_id: $agent_id})
        WHERE m.guild_id = $guild_id AND m.channel_id = $channel_id
        RETURN m.id AS id, m.tenant_id AS tenant_id, 'graph_query' AS type,
          m.summary AS summary, false AS deleted
        ORDER BY m.updated_at DESC
        LIMIT $limit
        """,
        parameters={},
        answer_kind="messages_by_channel",
    )

    # When: gnosis validates the generated query.
    validated = SafeGraphQueryValidator().validate(plan, request)

    # Then: the query survives with trusted runtime scope parameters.
    assert validated.parameters["channel_id"] == "channel-456"


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


def _memory_scope() -> MemoryScope:
    # The entity graph is per-user memory (LOCOMO-style), so its queries run
    # under a private-user scope with no guild/channel narrowing.
    return MemoryScope(
        tenant_id="nolgia",
        space_id="memory",
        agent_id="nolgia-agent",
        session_id="session-1",
        user_id="user-789",
        visibility=MemoryVisibility.PRIVATE_USER,
    )


def test_validator_accepts_scoped_entity_relates_traversal() -> None:
    # Given: a multi-hop entity-graph query scoped by tenant_id AND user_id on
    # every Entity alias, traversing RELATES for a bridge answer.
    request = GraphContextRequest(
        scope=_memory_scope(),
        query="Who founded the company Alice works at?",
        limit=5,
    )
    plan = GraphQueryPlan(
        cypher="""
        MATCH (a:Entity {tenant_id: $tenant_id, user_id: $user_id})
              -[:RELATES]->(company:Entity {tenant_id: $tenant_id, user_id: $user_id})
              -[r:RELATES]->(founder:Entity {tenant_id: $tenant_id, user_id: $user_id})
        WHERE a.name = $name
        RETURN founder.id AS id, founder.tenant_id AS tenant_id, 'entity' AS type,
          founder.name AS summary, false AS deleted
        LIMIT $limit
        """,
        parameters={"name": "Alice"},
        answer_kind="entity_bridge",
    )

    # When: gnosis validates the query.
    validated = SafeGraphQueryValidator().validate(plan, request)

    # Then: it is accepted with trusted runtime scope parameters injected.
    assert validated.parameters["tenant_id"] == "nolgia"
    assert validated.parameters["user_id"] == "user-789"
    assert validated.parameters["name"] == "Alice"


def test_validator_accepts_scoped_fact_mentions_traversal() -> None:
    # Given: a MENTIONS query fetching the facts naming an entity, with both
    # the Entity and Fact aliases scoped by tenant_id AND user_id.
    request = GraphContextRequest(
        scope=_memory_scope(),
        query="What do we know about Alice?",
        limit=5,
    )
    plan = GraphQueryPlan(
        cypher="""
        MATCH (e:Entity {tenant_id: $tenant_id, user_id: $user_id})
              <-[:MENTIONS]-(f:Fact {tenant_id: $tenant_id, user_id: $user_id})
        WHERE e.name = $name
        RETURN f.id AS id, f.tenant_id AS tenant_id, 'fact' AS type,
          f.object AS summary, false AS deleted
        LIMIT $limit
        """,
        parameters={"name": "Alice"},
        answer_kind="facts_about_entity",
    )

    # When / Then: the fully scoped traversal is accepted.
    validated = SafeGraphQueryValidator().validate(plan, request)
    assert validated.parameters["user_id"] == "user-789"


@pytest.mark.parametrize(
    "cypher",
    [
        # Entity alias scoped by tenant only - missing the mandatory user scope.
        """
        MATCH (a:Entity {tenant_id: $tenant_id})
        RETURN a.id AS id, 'entity' AS type, a.name AS summary, false AS deleted
        LIMIT $limit
        """,
        # Fact alias scoped by tenant only - a cross-user read of another
        # user's remembered facts.
        """
        MATCH (e:Entity {tenant_id: $tenant_id, user_id: $user_id})
              <-[:MENTIONS]-(f:Fact {tenant_id: $tenant_id})
        RETURN f.id AS id, 'fact' AS type, f.object AS summary, false AS deleted
        LIMIT $limit
        """,
        # A write against the entity graph.
        """
        MERGE (a:Entity {tenant_id: $tenant_id, user_id: $user_id})
        RETURN a.id AS id, 'entity' AS type, a.name AS summary, false AS deleted
        LIMIT $limit
        """,
    ],
)
def test_validator_rejects_unsafe_entity_graph_queries(cypher: str) -> None:
    # Given: an entity-graph query that under-scopes or writes.
    request = GraphContextRequest(scope=_memory_scope(), query="anything", limit=5)
    plan = GraphQueryPlan(cypher=cypher, parameters={}, answer_kind="unsafe")

    # When / Then: validation blocks the query before Neo4j can execute it.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


def test_proxy_model_name_strips_litellm_openai_prefix() -> None:
    # Given: the SDK-facing setting uses litellm provider-prefixed names.
    # Then: the raw OpenAI proxy client receives the bare model id.
    assert proxy_model_name("openai/gpt-5.5") == "gpt-5.5"
    assert proxy_model_name("openai/gemma4") == "gemma4"
    assert proxy_model_name("gpt-5.5") == "gpt-5.5"


class _ExplodingPlanner:
    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        del request
        raise APIConnectionError(request=httpx.Request("POST", "http://litellm.test"))


@pytest.mark.anyio
async def test_plan_graph_query_swallows_openai_errors() -> None:
    # Given: the upstream proxy rejects or drops the planner call.
    request = GraphContextRequest(scope=_scope(), query="Which roles exist?", limit=5)

    # When: gnosis plans a graph query.
    plan = await plan_graph_query(_ExplodingPlanner(), request)

    # Then: planner failures degrade to no graph context, never an error.
    assert plan is None


class _StructuredOutputViolatingPlanner:
    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        del request
        # The planner LLM returns prose + fenced JSON instead of the structured
        # contract; `.parse()` raises pydantic ValidationError.
        _ = GraphQueryPlan.model_validate_json("I don't have direct Neo4j access")
        return None


@pytest.mark.anyio
async def test_plan_graph_query_swallows_structured_output_violations() -> None:
    # Given: the planner LLM ignores the structured-output contract.
    request = GraphContextRequest(scope=_scope(), query="Which roles exist?", limit=5)

    # When: gnosis plans a graph query.
    plan = await plan_graph_query(_StructuredOutputViolatingPlanner(), request)

    # Then: it degrades to no graph context rather than 500-ing the caller.
    assert plan is None


_PINNED_FACT = "(f:Fact {tenant_id: $tenant_id, user_id: $user_id})"
_FACT_RETURN = """
RETURN f.id AS id, f.tenant_id AS tenant_id, 'fact' AS type,
  f.object AS summary, false AS deleted
LIMIT $limit
"""


@pytest.mark.parametrize(
    "cypher",
    [
        # Reported bypass 1: the tenant predicate OR-ed with a tautology.
        """
        MATCH (f:Fact)
        WHERE f.tenant_id = $tenant_id AND f.user_id = $user_id OR true
        """
        + _FACT_RETURN,
        f"MATCH {_PINNED_FACT} WHERE f.object = $name OR 1=1" + _FACT_RETURN,
        """
        MATCH (f:Fact)
        WHERE f.tenant_id = $tenant_id OR 1=1 AND f.user_id = $user_id
        """
        + _FACT_RETURN,
        f"MATCH {_PINNED_FACT} WHERE f.object = $name XOR true" + _FACT_RETURN,
        """
        MATCH (f:Fact {user_id: $user_id})
        WHERE NOT f.tenant_id <> $tenant_id
        """
        + _FACT_RETURN,
        # Tenant predicate only in WHERE, defeated without OR/NOT.
        """
        MATCH (f:Fact {user_id: $user_id})
        WHERE (f.tenant_id = $tenant_id) IS NULL
        """
        + _FACT_RETURN,
        """
        MATCH (f:Fact {user_id: $user_id})
        WHERE f.tenant_id = $tenant_id = false
        """
        + _FACT_RETURN,
        # Reported bypass 2: pin a Tenant node, then MATCH an unlabelled node.
        """
        MATCH (t:Tenant {tenant_id: $tenant_id})
        MATCH (f)
        RETURN f.id AS id, f.tenant_id AS tenant_id, 'node' AS type,
          f.summary AS summary, false AS deleted
        LIMIT $limit
        """,
        # Second MATCH on a labelled node with no tenant predicate.
        """
        MATCH (t:Tenant {tenant_id: $tenant_id})
        MATCH (f:Fact)
        """
        + _FACT_RETURN,
        # Unlabelled alias re-bound after WITH drops the pinned binding.
        f"MATCH {_PINNED_FACT} WITH 1 AS x MATCH (f)" + _FACT_RETURN,
        # Anonymous node in a traversal.
        f"MATCH {_PINNED_FACT}-[:MENTIONS]->()" + _FACT_RETURN,
        # Label expression instead of a single label.
        """
        MATCH (f:Fact|Entity {tenant_id: $tenant_id, user_id: $user_id})
        """
        + _FACT_RETURN,
        # Tenant pinned to something other than $tenant_id.
        "MATCH (f:Fact {tenant_id: $user_id, user_id: $user_id})" + _FACT_RETURN,
        "MATCH (f:Fact {tenant_id: f.user_id, user_id: $user_id})" + _FACT_RETURN,
        # Write clauses and procedures.
        f"MATCH {_PINNED_FACT} SET f.object = $name" + _FACT_RETURN,
        f"MATCH {_PINNED_FACT} DETACH DELETE f" + _FACT_RETURN,
        f"MATCH {_PINNED_FACT} FOREACH (x IN [1] | MERGE (f)-[:RELATES]->(f))"
        + _FACT_RETURN,
        "CALL db.labels() YIELD label RETURN label AS id LIMIT $limit",
        f"MATCH {_PINNED_FACT} CALL db.labels() YIELD label" + _FACT_RETURN,
        f"MATCH {_PINNED_FACT} WITH f, apoc.text.join([], '') AS j" + _FACT_RETURN,
        "USE system MATCH " + _PINNED_FACT + _FACT_RETURN,
        # Multiple statements and comments.
        f"MATCH {_PINNED_FACT}" + _FACT_RETURN + "; MATCH (n) RETURN n",
        f"MATCH {_PINNED_FACT} // (f:Fact {{tenant_id: $tenant_id}})" + _FACT_RETURN,
        # Unknown function and grouping parentheses.
        f"MATCH p = {_PINNED_FACT} WITH f, nodes(p) AS ns" + _FACT_RETURN,
        # Missing or literal-forged tenant column.
        f"""
        MATCH {_PINNED_FACT}
        RETURN f.id AS id, 'fact' AS type, f.object AS summary, false AS deleted
        LIMIT $limit
        """,
        f"""
        MATCH {_PINNED_FACT}
        RETURN f.id AS id, 'f.tenant_id AS tenant_id' AS type,
          f.object AS summary, false AS deleted
        LIMIT $limit
        """,
        # A second node's properties returned without its tenant column.
        f"""
        MATCH {_PINNED_FACT}-[:MENTIONS]->
          (e:Entity {{tenant_id: $tenant_id, user_id: $user_id}})
        RETURN f.id AS id, f.tenant_id AS tenant_id, 'fact' AS type,
          e.name AS summary, false AS deleted
        LIMIT $limit
        """,
        # Unterminated literal.
        f"MATCH {_PINNED_FACT} WHERE f.object = 'open" + _FACT_RETURN,
    ],
)
def test_validator_rejects_cross_tenant_bypass_queries(cypher: str) -> None:
    # Given: a planned query that could read outside the caller's tenant.
    request = GraphContextRequest(scope=_memory_scope(), query="anything", limit=5)
    plan = GraphQueryPlan(cypher=cypher, parameters={}, answer_kind="bypass")

    # When / Then: validation blocks the query before Neo4j can execute it.
    with pytest.raises(GraphQueryValidationError):
        _ = SafeGraphQueryValidator().validate(plan, request)


@pytest.mark.parametrize(
    "cypher",
    [
        # Boolean words inside a string literal are data, not operators.
        f"MATCH {_PINNED_FACT} WHERE f.object CONTAINS 'black or not white'"
        + _FACT_RETURN,
        # A second returned node projects its own tenant column.
        f"""
        MATCH {_PINNED_FACT}-[:MENTIONS]->
          (e:Entity {{tenant_id: $tenant_id, user_id: $user_id}})
        RETURN f.id AS id, f.tenant_id AS tenant_id, e.tenant_id AS e_tenant_id,
          'fact' AS type, e.name + ': ' + f.object AS summary, false AS deleted
        LIMIT $limit
        """,
    ],
)
def test_validator_accepts_pinned_queries_with_tenant_columns(cypher: str) -> None:
    # Given: every node pattern is tenant-pinned and every returned node
    # projects its tenant column.
    request = GraphContextRequest(scope=_memory_scope(), query="anything", limit=5)
    plan = GraphQueryPlan(cypher=cypher, parameters={}, answer_kind="facts")

    # When: gnosis validates the query.
    validated = SafeGraphQueryValidator().validate(plan, request)

    # Then: it is accepted unchanged.
    assert validated.cypher == cypher
