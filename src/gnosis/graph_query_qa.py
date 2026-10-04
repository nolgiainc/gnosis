import logging
import time
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, Field

from gnosis.graph_types import CypherParameters
from gnosis.models import GraphContextRequest

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_GRAPH_SCHEMA_GUIDE: Final[str] = """
Use this read-only Neo4j schema for gnosis graph QA.
Labels: Tenant, Agent, Client, Guild, Channel, Category, User, Bot, Role,
Message, Link, Attachment, Event, GraphNode, Entity, Fact.
Relationships: OWNS_AGENT, OWNS_CLIENT, USES_CLIENT, OWNS_GUILD, IN_GUILD,
IN_CATEGORY, OWNS_ROLE, HAS_ROLE, AUTHORED, IN_CHANNEL, LINKED_FROM,
ATTACHED_TO, AFFECTS, MENTIONS, RELATES.
Every node pattern, including every repeated reference to an alias, must be
written as (alias:Label {tenant_id: $tenant_id, ...}) with exactly one label
and the tenant pinned inside the property map - never (alias) or () alone and
never tenant_id in WHERE only. If a guild question has $guild_id, also scope by
guild_id = $guild_id or graph IN_GUILD membership. If a channel question has
$channel_id, scope by channel_id = $channel_id. Combine predicates with AND
only: OR, XOR, and NOT are rejected, as are comments, semicolons, and
parentheses other than node patterns and function calls.

Knowledge graph (use for who/what/relationship and multi-hop questions about a
user's remembered facts):
An (:Entity {name}) is a person, place, organization, or thing. A (:Fact
{object}) is a remembered dated statement; f.object is its text. Edges:
(:Fact)-[:MENTIONS]->(:Entity) links a fact to each entity it names, and
(:Entity)-[:RELATES {relation, event_date, fact_id}]->(:Entity) is a directed
relationship between two entities (r.relation is the verb phrase). Traverse
RELATES for multi-hop bridge questions; follow MENTIONS to fetch the facts
naming an entity. Match an entity by name with e.name = $name or a literal.
Every Entity and Fact node pattern MUST pin BOTH tenant_id: $tenant_id AND
user_id: $user_id in its property map.

Return rows with id, type, summary, deleted, and tenant_id, where id and
tenant_id come from the same node (n.id AS id, n.tenant_id AS tenant_id). If
the RETURN reads properties of any other node alias m, also return
m.tenant_id AS m_tenant_id. Use LIMIT $limit.
Never use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, LOAD, DROP, or procedures.
""".strip()


class GraphQueryPlan(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    cypher: str = Field(min_length=1)
    parameters: CypherParameters = Field(default_factory=dict)
    answer_kind: str = Field(min_length=1, max_length=64)


@dataclass(frozen=True, slots=True)
class ValidatedGraphQuery:
    cypher: str
    parameters: CypherParameters
    answer_kind: str


class GraphQueryPlanner(Protocol):
    async def plan_query(
        self,
        request: GraphContextRequest,
    ) -> GraphQueryPlan | None: ...


def proxy_model_name(model: str) -> str:
    """Normalize a litellm-library model name for an OpenAI-compatible proxy.

    The SDK's LiteLLMProvider wants provider-prefixed names such as
    ``openai/gpt-5.5``, but the proxy behind ``base_url`` registers bare
    model ids and rejects the prefixed form with a 400.
    """
    return model.removeprefix("openai/")


@dataclass(frozen=True, slots=True)
class LiteLLMGraphQueryPlanner:
    model: str
    base_url: str
    api_key: str

    async def plan_query(self, request: GraphContextRequest) -> GraphQueryPlan | None:
        start = time.perf_counter()
        async with AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            response = await client.beta.chat.completions.parse(
                messages=_messages(request),
                model=proxy_model_name(self.model),
                temperature=0,
                # Structured-output plans truncated at the old 700-token cap
                # raised LengthFinishReasonError and wasted the whole call.
                max_tokens=1500,
                response_format=GraphQueryPlan,
            )
        plan = response.choices[0].message.parsed
        if plan is None:
            _LOGGER.info(
                "graph QA planner returned no content",
                extra={"model": self.model},
            )
            return None
        _LOGGER.info(
            "graph QA planner produced plan",
            extra={
                "answer_kind": plan.answer_kind,
                "duration_ms": round((time.perf_counter() - start) * 1000),
                "model": self.model,
            },
        )
        return plan


def graph_schema_guide() -> str:
    return _GRAPH_SCHEMA_GUIDE


def _messages(request: GraphContextRequest) -> tuple[ChatCompletionMessageParam, ...]:
    scope = request.scope
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": _GRAPH_SCHEMA_GUIDE,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": (
            f"Question: {request.query}\n"
            f"tenant_id: {scope.tenant_id}\n"
            f"agent_id: {scope.agent_id}\n"
            f"guild_id: {scope.guild_id or ''}\n"
            f"channel_id: {scope.channel_id or ''}\n"
            f"user_id: {scope.user_id}\n"
            f"limit: {request.limit}"
        ),
    }
    return (system_message, user_message)
