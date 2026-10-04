from dataclasses import dataclass
from typing import Final, final

from pydantic import TypeAdapter

from gnosis import graph_query_rules as rules
from gnosis.graph_query_qa import GraphQueryPlan, ValidatedGraphQuery
from gnosis.graph_types import CypherParameters
from gnosis.models import GraphContextRequest, JsonValue

_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_WRITE_KEYWORD_REASON: Final[str] = "write keyword is not allowed"
_READ_PREFIX_REASON: Final[str] = "query must start with a read clause"
_PROCEDURE_REASON: Final[str] = "procedures are not allowed"
_RAW_SCOPE_REASON: Final[str] = "scope values must use parameters"
_LIMIT_REASON: Final[str] = "query must use LIMIT $limit"
_TENANT_SCOPE_REASON: Final[str] = "query must scope tenant_id with $tenant_id"
_USER_SCOPE_REASON: Final[str] = "query must scope user_id with $user_id"
_GUILD_SCOPE_REASON: Final[str] = "query must scope guild_id with $guild_id"
_CHANNEL_SCOPE_REASON: Final[str] = "channel queries must scope channel_id"
_AGENT_SCOPE_REASON: Final[str] = "query must scope agent_id with $agent_id"
_UNSUPPORTED_SCHEMA_REASON: Final[str] = "unsupported schema access syntax"
_RETURN_SHAPE_REASON: Final[str] = "query must return id, type, summary, and deleted"
_BOOLEAN_OPERATOR_REASON: Final[str] = "OR, XOR, and NOT are not allowed"
_MULTIPLE_STATEMENTS_REASON: Final[str] = "multiple statements are not allowed"
_COMMENT_REASON: Final[str] = "comments are not allowed"
_UNTERMINATED_LITERAL_REASON: Final[str] = "unterminated string literal"
_NAMESPACE_REASON: Final[str] = "procedure and function namespaces are not allowed"
_NODE_PATTERN_REASON: Final[str] = (
    "every node pattern must be (alias:Label {tenant_id: $tenant_id, ...})"
)
_NODE_USER_SCOPE_REASON: Final[str] = (
    "Entity and Fact node patterns must pin user_id: $user_id"
)
_TENANT_COLUMN_REASON: Final[str] = (
    "query must return <alias>.tenant_id for every returned node alias"
)
_QUOTES: Final[frozenset[str]] = frozenset({"'", '"'})


@final
class GraphQueryValidationError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason: str = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SafeGraphQueryValidator:
    def validate(
        self,
        plan: GraphQueryPlan,
        request: GraphContextRequest,
    ) -> ValidatedGraphQuery:
        cypher = plan.cypher.strip()
        _require_safe_cypher(cypher, request)
        return ValidatedGraphQuery(
            cypher=plan.cypher,
            parameters=_parameters(plan, request),
            answer_kind=plan.answer_kind,
        )


def _require_safe_cypher(raw_cypher: str, request: GraphContextRequest) -> None:
    # Structural checks run on a copy with string-literal contents blanked, so
    # text inside a literal can neither satisfy a required pattern nor hide a
    # forbidden one. Deny-list checks (write keywords, unknown tokens) still
    # run on the raw text, which only makes them stricter.
    cypher = _mask_literals(raw_cypher)
    _require_single_read_statement(raw_cypher, cypher)
    _require_safe_syntax(cypher)
    _require_pinned_node_patterns(cypher)
    _require_tenant_columns(cypher)
    _require_alias_scope(cypher, request)
    if (
        request.scope.guild_id is not None
        and rules.GUILD_SCOPE_PATTERN.search(cypher) is None
    ):
        raise GraphQueryValidationError(_GUILD_SCOPE_REASON)
    if (
        request.scope.channel_id is not None
        and "Channel" in _labels(cypher)
        and rules.CHANNEL_SCOPE_PATTERN.search(cypher) is None
        and rules.GUILD_SCOPE_PATTERN.search(cypher) is None
    ):
        raise GraphQueryValidationError(_CHANNEL_SCOPE_REASON)
    _require_known_tokens(_labels(raw_cypher), rules.SAFE_LABELS, "label")
    _require_known_tokens(
        _relationships(raw_cypher),
        rules.SAFE_RELATIONSHIPS,
        "relationship",
    )
    _require_known_tokens(
        _properties(raw_cypher),
        rules.SAFE_PROPERTIES,
        "property",
    )


def _require_single_read_statement(raw_cypher: str, cypher: str) -> None:
    upper_tokens = frozenset(rules.KEYWORD_PATTERN.findall(raw_cypher.upper()))
    if upper_tokens & rules.UNSAFE_KEYWORDS:
        raise GraphQueryValidationError(_WRITE_KEYWORD_REASON)
    if ";" in cypher:
        raise GraphQueryValidationError(_MULTIPLE_STATEMENTS_REASON)
    if "//" in cypher or "/*" in cypher:
        raise GraphQueryValidationError(_COMMENT_REASON)
    if rules.BOOLEAN_OPERATOR_PATTERN.search(cypher):
        raise GraphQueryValidationError(_BOOLEAN_OPERATOR_REASON)
    if rules.UNSAFE_NAMESPACE_PATTERN.search(cypher):
        raise GraphQueryValidationError(_NAMESPACE_REASON)


def _require_safe_syntax(cypher: str) -> None:
    if not rules.READ_PREFIX_PATTERN.search(cypher):
        raise GraphQueryValidationError(_READ_PREFIX_REASON)
    if rules.UNSAFE_PROCEDURE_PATTERN.search(cypher):
        raise GraphQueryValidationError(_PROCEDURE_REASON)
    if rules.RAW_SCOPE_LITERAL_PATTERN.search(cypher):
        raise GraphQueryValidationError(_RAW_SCOPE_REASON)
    if rules.UNSUPPORTED_SCHEMA_SYNTAX_PATTERN.search(cypher):
        raise GraphQueryValidationError(_UNSUPPORTED_SCHEMA_REASON)
    if rules.LIMIT_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_LIMIT_REASON)
    if rules.RETURN_SHAPE_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_RETURN_SHAPE_REASON)
    if rules.SCOPE_PATTERN.search(cypher) is None:
        raise GraphQueryValidationError(_TENANT_SCOPE_REASON)


def _mask_literals(cypher: str) -> str:
    """Return ``cypher`` with every string literal's contents blanked.

    Quotes are kept (``'abc'`` becomes ``'   '``) so literal-sensitive rules
    still see that a literal is present. Backslash escapes are honoured.
    """
    masked: list[str] = []
    quote: str | None = None
    escaped = False
    for char in cypher:
        if quote is None:
            masked.append(char)
            if char in _QUOTES:
                quote = char
            continue
        if escaped:
            escaped = False
            masked.append(" ")
        elif char == "\\":
            escaped = True
            masked.append(" ")
        elif char == quote:
            quote = None
            masked.append(char)
        else:
            masked.append(" ")
    if quote is not None:
        raise GraphQueryValidationError(_UNTERMINATED_LITERAL_REASON)
    return "".join(masked)


def _require_pinned_node_patterns(cypher: str) -> None:
    """Require every node pattern to be labelled and tenant-pinned in its map.

    Every ``(`` is either a call of an allow-listed function or the start of a
    node pattern. Node patterns must have exactly one label and an inline map
    of ``key: $param`` / literal entries containing ``tenant_id: $tenant_id``
    (and ``user_id: $user_id`` for per-user labels). Unlabelled re-references
    such as ``(m)``, anonymous ``()``, grouping parentheses, and label
    expressions are rejected, so no alias can ever bind an unpinned node.
    """
    for index, char in enumerate(cypher):
        if char != "(":
            continue
        preceding = rules.PRECEDING_WORD_PATTERN.search(cypher[:index])
        if preceding is not None:
            word = preceding.group("word")
            if word.lower() in rules.ALLOWED_FUNCTIONS:
                continue
            if word.upper() not in rules.PATTERN_KEYWORDS:
                reason = f"function is not allowed: {word}"
                raise GraphQueryValidationError(reason)
        node = rules.NODE_PATTERN.match(cypher, index)
        if node is None:
            raise GraphQueryValidationError(_NODE_PATTERN_REASON)
        entries = _node_map_entries(node.group("properties"))
        if entries.get("tenant_id") != "$tenant_id":
            raise GraphQueryValidationError(_NODE_PATTERN_REASON)
        if (
            node.group("label") in rules.USER_SCOPED_LABELS
            and entries.get("user_id") != "$user_id"
        ):
            raise GraphQueryValidationError(_NODE_USER_SCOPE_REASON)


def _node_map_entries(properties: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not properties.strip():
        return entries
    for raw_entry in properties.split(","):
        entry = rules.NODE_MAP_ENTRY_PATTERN.fullmatch(raw_entry)
        if entry is None:
            raise GraphQueryValidationError(_NODE_PATTERN_REASON)
        key = entry.group("key")
        if key in entries:
            raise GraphQueryValidationError(_NODE_PATTERN_REASON)
        entries[key] = entry.group("value")
    return entries


def _require_tenant_columns(cypher: str) -> None:
    """Require each RETURN to project the tenant of every returned node alias.

    The node supplying ``id`` must project ``<alias>.tenant_id AS tenant_id``;
    any other node alias whose properties are returned must project
    ``<alias>.tenant_id AS <name>_tenant_id``. Execution then drops rows whose
    tenant columns differ from the caller's tenant.
    """
    node_aliases = frozenset(
        alias
        for match in rules.NODE_PATTERN.finditer(cypher)
        if (alias := match.group("alias")) is not None
    )
    segments = [
        match.group("items") for match in rules.RETURN_SEGMENT_PATTERN.finditer(cypher)
    ]
    if not segments:
        raise GraphQueryValidationError(_RETURN_SHAPE_REASON)
    for segment in segments:
        id_column = rules.ID_COLUMN_PATTERN.search(segment)
        if id_column is None or id_column.group("alias") not in node_aliases:
            raise GraphQueryValidationError(_TENANT_COLUMN_REASON)
        if (
            rules.tenant_column_pattern(id_column.group("alias"), "tenant_id").search(
                segment,
            )
            is None
        ):
            raise GraphQueryValidationError(_TENANT_COLUMN_REASON)
        returned_aliases = {
            match.group("alias")
            for match in rules.ALIAS_PROPERTY_PATTERN.finditer(segment)
        } & node_aliases
        for alias in returned_aliases:
            column = rules.tenant_column_pattern(alias, r"[A-Za-z0-9_]*tenant_id")
            if column.search(segment) is None:
                raise GraphQueryValidationError(_TENANT_COLUMN_REASON)


def _require_alias_scope(cypher: str, request: GraphContextRequest) -> None:
    labels_by_alias = _labels_by_alias(cypher)
    for alias, label in labels_by_alias.items():
        if label == "Tenant":
            continue
        _require_alias_predicate(cypher, alias, "tenant_id", _TENANT_SCOPE_REASON)
        if label == "GraphNode":
            _require_alias_predicate(cypher, alias, "agent_id", _AGENT_SCOPE_REASON)
        if label in rules.USER_SCOPED_LABELS:
            _require_alias_predicate(cypher, alias, "user_id", _USER_SCOPE_REASON)
        if request.scope.guild_id is not None and label in rules.GUILD_SCOPED_LABELS:
            _require_alias_predicate(cypher, alias, "guild_id", _GUILD_SCOPE_REASON)
        if (
            request.scope.channel_id is not None
            and label in rules.CHANNEL_SCOPED_LABELS
        ):
            _require_alias_predicate(cypher, alias, "channel_id", _CHANNEL_SCOPE_REASON)


def _labels_by_alias(cypher: str) -> dict[str, str]:
    return {
        match.group("alias"): match.group("label")
        for match in rules.ALIAS_LABEL_PATTERN.finditer(cypher)
    }


def _require_alias_predicate(
    cypher: str,
    alias: str,
    property_name: str,
    reason: str,
) -> None:
    pattern = rules.alias_predicate_pattern(alias, property_name)
    if pattern.search(cypher) is None and not _node_map_has_parameter(
        cypher,
        alias,
        property_name,
    ):
        raise GraphQueryValidationError(reason)


def _node_map_has_parameter(cypher: str, alias: str, property_name: str) -> bool:
    pattern = rules.node_map_parameter_pattern(alias, property_name)
    return pattern.search(cypher) is not None


def _parameters(plan: GraphQueryPlan, request: GraphContextRequest) -> CypherParameters:
    parameters: CypherParameters = {
        key: _JSON_VALUE_ADAPTER.validate_python(value)
        for key, value in plan.parameters.items()
    }
    parameters["tenant_id"] = request.scope.tenant_id
    parameters["agent_id"] = request.scope.agent_id
    parameters["user_id"] = request.scope.user_id
    parameters["guild_id"] = request.scope.guild_id
    parameters["channel_id"] = request.scope.channel_id
    parameters["limit"] = request.limit
    return parameters


def _labels(cypher: str) -> frozenset[str]:
    # Blank relationship brackets first so a relationship type inside ``[...]``
    # is not extracted as a node label (it is validated as a relationship).
    node_cypher = rules.RELATIONSHIP_BRACKET_PATTERN.sub(" ", cypher)
    return frozenset(rules.LABEL_PATTERN.findall(node_cypher))


def _relationships(cypher: str) -> frozenset[str]:
    return frozenset(rules.RELATIONSHIP_PATTERN.findall(cypher))


def _properties(cypher: str) -> frozenset[str]:
    return frozenset(rules.PROPERTY_PATTERN.findall(cypher))


def _require_known_tokens(
    tokens: frozenset[str],
    allowed: frozenset[str],
    token_type: str,
) -> None:
    unknown = tokens - allowed
    if unknown:
        reason = f"unknown {token_type}: {sorted(unknown)[0]}"
        raise GraphQueryValidationError(reason)
