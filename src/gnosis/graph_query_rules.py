import re
from re import Pattern
from typing import Final

LABEL_PATTERN: Final[Pattern[str]] = re.compile(r":([A-Za-z][A-Za-z0-9_]*)")
RELATIONSHIP_PATTERN: Final[Pattern[str]] = re.compile(
    r"\[[A-Za-z0-9_]*:([A-Z][A-Z0-9_]*)",
)
# Relationship bracket spans, blanked before node-label extraction so a
# relationship type (e.g. ``[:RELATES]``) is never mistaken for a node label -
# relationship types are checked separately against SAFE_RELATIONSHIPS.
RELATIONSHIP_BRACKET_PATTERN: Final[Pattern[str]] = re.compile(r"\[[^\]]*\]")
PROPERTY_PATTERN: Final[Pattern[str]] = re.compile(r"\.([A-Za-z][A-Za-z0-9_]*)")
KEYWORD_PATTERN: Final[Pattern[str]] = re.compile(r"\b[A-Z]+\b")
ALIAS_LABEL_PATTERN: Final[Pattern[str]] = re.compile(
    r"\((?P<alias>[A-Za-z][A-Za-z0-9_]*)\s*:(?P<label>[A-Za-z][A-Za-z0-9_]*)",
)
LIMIT_PATTERN: Final[Pattern[str]] = re.compile(r"\bLIMIT\s+\$limit\b", re.IGNORECASE)
SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\btenant_id\s*(?:[:=])\s*\$tenant_id\b",
    re.IGNORECASE,
)
GUILD_SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bguild_id\s*=\s*\$guild_id\b|\$guild_id\s+IN\b",
    re.IGNORECASE,
)
CHANNEL_SCOPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bchannel_id\s*=\s*\$channel_id\b",
    re.IGNORECASE,
)
RAW_SCOPE_LITERAL_PATTERN: Final[Pattern[str]] = re.compile(
    r"\b(?:tenant_id|guild_id|channel_id|user_id|agent_id)\s*(?:[:=])\s*['\"]",
    re.IGNORECASE,
)
UNSUPPORTED_SCHEMA_SYNTAX_PATTERN: Final[Pattern[str]] = re.compile(
    r"`|\[[^\]]*(?:\$[A-Za-z][A-Za-z0-9_]*|['\"][^'\"]+['\"])\]|\bproperties\s*\(",
    re.IGNORECASE,
)
RETURN_SHAPE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bRETURN\b(?=.*\bAS\s+id\b)(?=.*\bAS\s+type\b)(?=.*\bAS\s+summary\b)(?=.*\bAS\s+deleted\b)",
    re.IGNORECASE | re.DOTALL,
)
READ_PREFIX_PATTERN: Final[Pattern[str]] = re.compile(
    r"^\s*(?:MATCH|OPTIONAL\s+MATCH|WITH|CALL\s*\{)",
    re.IGNORECASE,
)
UNSAFE_PROCEDURE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bCALL\s+(?!\{)",
    re.IGNORECASE,
)
UNSAFE_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "ALTER",
        "CREATE",
        "DELETE",
        "DENY",
        "DETACH",
        "DROP",
        "FOREACH",
        "GRANT",
        "INSERT",
        "LOAD",
        "MERGE",
        "REMOVE",
        "REVOKE",
        "SET",
        "SHOW",
        "START",
        "STOP",
        "TERMINATE",
        "TRANSACTIONS",
        "USE",
    },
)
# Boolean operators that can weaken a conjunctive scope predicate
# (``f.tenant_id = $tenant_id OR true``, ``NOT f.tenant_id <> $tenant_id``).
# Matched outside string literals only.
BOOLEAN_OPERATOR_PATTERN: Final[Pattern[str]] = re.compile(
    r"\b(?:OR|XOR|NOT)\b",
    re.IGNORECASE,
)
# Procedure / function namespaces that reach outside the graph (APOC can run
# arbitrary Cypher as a *function*, without CALL).
UNSAFE_NAMESPACE_PATTERN: Final[Pattern[str]] = re.compile(
    r"\b(?:apoc|dbms|db|gds|genai|tx)\s*\.",
    re.IGNORECASE,
)
# A node pattern the planner may emit: exactly one label and an inline property
# map. Every node pattern must carry ``tenant_id: $tenant_id`` in that map; a
# map predicate is part of the match itself, so - unlike a WHERE predicate -
# it cannot be OR-ed, negated, or compared against false.
_IDENTIFIER: Final[str] = r"[A-Za-z][A-Za-z0-9_]*"
_NODE_HEAD: Final[str] = (
    rf"\(\s*(?P<alias>{_IDENTIFIER})?\s*:\s*(?P<label>{_IDENTIFIER})"
)
_NODE_MAP: Final[str] = r"\s*\{(?P<properties>[^{}]*)\}\s*\)"
NODE_PATTERN: Final[Pattern[str]] = re.compile(_NODE_HEAD + _NODE_MAP)
_NODE_MAP_VALUE: Final[str] = (
    rf"\${_IDENTIFIER}|'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?|true|false|null"
)
NODE_MAP_ENTRY_PATTERN: Final[Pattern[str]] = re.compile(
    rf"\s*(?P<key>{_IDENTIFIER})\s*:\s*(?P<value>{_NODE_MAP_VALUE})\s*",
    re.IGNORECASE,
)
PRECEDING_WORD_PATTERN: Final[Pattern[str]] = re.compile(
    r"(?P<word>[A-Za-z_][A-Za-z0-9_]*)\s*$",
)
# Scalar/aggregate functions the planner may call. Any other identifier
# directly before ``(`` is rejected (fail closed: unknown functions, APOC,
# path functions such as nodes()/relationships()).
ALLOWED_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "abs",
        "avg",
        "ceil",
        "coalesce",
        "collect",
        "count",
        "date",
        "datetime",
        "duration",
        "floor",
        "head",
        "last",
        "left",
        "localdatetime",
        "ltrim",
        "max",
        "min",
        "nullif",
        "replace",
        "reverse",
        "right",
        "round",
        "rtrim",
        "size",
        "split",
        "substring",
        "sum",
        "toboolean",
        "tofloat",
        "tointeger",
        "tolower",
        "tostring",
        "toupper",
        "trim",
        "type",
    },
)
# Clause keywords that may directly precede a node pattern's ``(``. Anything
# in this set must be followed by a fully pinned node pattern.
PATTERN_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "AND",
        "AS",
        "BY",
        "CASE",
        "CONTAINS",
        "DISTINCT",
        "ELSE",
        "END",
        "ENDS",
        "EXISTS",
        "IN",
        "IS",
        "MATCH",
        "OPTIONAL",
        "RETURN",
        "STARTS",
        "THEN",
        "UNION",
        "UNWIND",
        "WHEN",
        "WHERE",
        "WITH",
        "YIELD",
    },
)
RETURN_SEGMENT_PATTERN: Final[Pattern[str]] = re.compile(
    r"\bRETURN\b(?P<items>.*?)(?=\bUNION\b|\bORDER\s+BY\b|\bSKIP\b|\bLIMIT\b|$)",
    re.IGNORECASE | re.DOTALL,
)
ALIAS_PROPERTY_PATTERN: Final[Pattern[str]] = re.compile(
    r"\b(?P<alias>[A-Za-z][A-Za-z0-9_]*)\.[A-Za-z]",
)
ID_COLUMN_PATTERN: Final[Pattern[str]] = re.compile(
    r"\b(?P<alias>[A-Za-z][A-Za-z0-9_]*)\.id\s+AS\s+id\b",
    re.IGNORECASE,
)
SAFE_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Agent",
        "Attachment",
        "Bot",
        "Category",
        "Channel",
        "Client",
        "Entity",
        "Event",
        "Fact",
        "GraphNode",
        "Guild",
        "Link",
        "Message",
        "Role",
        "Tenant",
        "User",
    },
)
SAFE_RELATIONSHIPS: Final[frozenset[str]] = frozenset(
    {
        "AFFECTS",
        "ATTACHED_TO",
        "AUTHORED",
        "HAS_ROLE",
        "IN_CATEGORY",
        "IN_CHANNEL",
        "IN_GUILD",
        "LINKED_FROM",
        "MENTIONS",
        "OWNS_AGENT",
        "OWNS_CLIENT",
        "OWNS_GUILD",
        "OWNS_ROLE",
        "RELATES",
        "USES_CLIENT",
    },
)
SAFE_PROPERTIES: Final[frozenset[str]] = frozenset(
    {
        "agent_id",
        "category_id",
        "channel_id",
        "content",
        "deleted",
        "display_name",
        "event_date",
        "event_id",
        "event_type",
        "fact_id",
        "filename",
        "guild_id",
        "id",
        "is_bot",
        "kind",
        "message_id",
        "name",
        "object",
        "occurred_at",
        "payload",
        "predicate",
        "relation",
        "role_id",
        "subject",
        "summary",
        "tenant_id",
        "type",
        "updated_at",
        "url",
        "user_id",
        "user_type",
        "visibility",
    },
)
GUILD_SCOPED_LABELS: Final[frozenset[str]] = frozenset(
    {"Category", "Channel", "GraphNode", "Message", "Role"},
)
CHANNEL_SCOPED_LABELS: Final[frozenset[str]] = frozenset(
    {"Channel", "GraphNode", "Message"},
)
# Knowledge-graph nodes are per-user: every Entity and Fact alias must be
# scoped by user_id = $user_id (in addition to tenant_id) so entity traversal
# can never read another user's remembered facts within the same tenant.
USER_SCOPED_LABELS: Final[frozenset[str]] = frozenset({"Entity", "Fact"})


def alias_predicate_pattern(alias: str, property_name: str) -> Pattern[str]:
    return re.compile(
        rf"\b{re.escape(alias)}\.{property_name}\s*=\s*\${property_name}\b",
        re.IGNORECASE,
    )


def node_map_parameter_pattern(alias: str, property_name: str) -> Pattern[str]:
    return re.compile(
        rf"\({re.escape(alias)}\s*:[^)]*\{{[^}}]*\b{property_name}\s*:\s*\${property_name}\b",
        re.IGNORECASE | re.DOTALL,
    )


def tenant_column_pattern(alias: str, column: str) -> Pattern[str]:
    """``<alias>.tenant_id AS <column>`` inside a RETURN projection."""
    return re.compile(
        rf"\b{re.escape(alias)}\.tenant_id\s+AS\s+{column}\b",
        re.IGNORECASE,
    )
