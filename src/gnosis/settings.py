from os import environ
from pathlib import Path
from typing import ClassVar, Literal, override

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# A YAML config file sets a deployment's baseline configuration. gnosis loads
# configs/default.yaml (the preferred / best-scoring config) automatically when
# GNOSIS_CONFIG_FILE is unset. Set GNOSIS_CONFIG_FILE to a path to load a
# different config, or to "" to opt out and use the safe minimal code defaults.
_CONFIG_FILE_ENV_VAR = "GNOSIS_CONFIG_FILE"
_DEFAULT_CONFIG_NAME = "default.yaml"


def _default_config_path() -> Path | None:
    """The shipped default config, if present, tried repo- then CWD-relative.

    Repo/editable layout puts it at ``<repo>/configs/default.yaml`` (three
    parents up from this file); the container image copies ``configs/`` next to
    the working directory, so ``./configs/default.yaml`` covers that too.
    """
    candidates = (
        Path(__file__).resolve().parents[2] / "configs" / _DEFAULT_CONFIG_NAME,
        Path.cwd() / "configs" / _DEFAULT_CONFIG_NAME,
    )
    return next((path for path in candidates if path.is_file()), None)


def _config_file_path() -> Path | None:
    """Resolve which YAML config to load, honoring GNOSIS_CONFIG_FILE.

    Unset -> the shipped default config (auto-load). Empty string -> opt out
    (no config file; code defaults). A path -> that file.
    """
    override = environ.get(_CONFIG_FILE_ENV_VAR)
    if override is not None:
        return Path(override) if override else None
    return _default_config_path()


type PeerDirection = Literal["both", "push", "pull"]

_PEER_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


def _required_operator_token(name: str) -> str:
    return environ.get(name, "")


class PeerConfig(BaseModel):
    """One remote gnosis deployment this instance may federate with."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=_PEER_NAME_PATTERN)
    base_url: str = Field(min_length=1)
    direction: PeerDirection = "both"
    remote_tenant_id: str = Field(min_length=1)

    @property
    def token_env_var(self) -> str:
        return f"GNOSIS_PEER_{self.name.upper().replace('-', '_')}_TOKEN"

    def allows_push(self) -> bool:
        return self.direction in {"both", "push"}

    def allows_pull(self) -> bool:
        return self.direction in {"both", "pull"}


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    @override
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Add the YAML config file as a low-priority settings source.

        Precedence, highest first: explicit init args, environment variables,
        the ``.env`` file, then the YAML config file (``configs/default.yaml``
        by default; see ``_config_file_path``), then field defaults. So the
        config file sets the baseline while individual env vars still override
        single keys.
        """
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        yaml_path = _config_file_path()
        if yaml_path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    gnosis_token: str = Field(default="", min_length=1)
    gnosis_read_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_READ_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_export_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_EXPORT_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_write_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_WRITE_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_admin_operator_token: str = Field(
        default_factory=lambda: _required_operator_token(
            "GNOSIS_ADMIN_OPERATOR_TOKEN",
        ),
        min_length=1,
    )
    gnosis_tenant_id: str = Field(default="nolgia", min_length=1)
    neo4j_uri: str = Field(default="", min_length=1)
    neo4j_username: str = Field(default="neo4j", min_length=1)
    neo4j_password: str = Field(default="", min_length=1)
    litellm_base_url: str = Field(default="", min_length=1)
    litellm_api_key: str = Field(default="", min_length=1)
    gnosis_llm: str = Field(default="openai/gemma4", min_length=1)
    gnosis_embedding: str = Field(default="local-qwen3-embedding-0.6b", min_length=1)
    gnosis_embedding_dimensions: int = Field(default=1024, gt=0)
    gnosis_audit_read: bool = False
    gnosis_conversation_ttl_days: int | None = Field(default=None, ge=1)
    gnosis_write_mode: Literal["sync", "buffered"] = "sync"
    gnosis_max_pending: int = Field(default=200, ge=1)
    gnosis_fact_deduplication_enabled: bool = True
    gnosis_trace_embedding_enabled: bool = True
    gnosis_extract_entities_enabled: bool = False
    gnosis_extract_relations_enabled: bool = False
    gnosis_extraction_preview_enabled: bool = False
    gnosis_extraction_batch_size: int = Field(default=25, ge=1)
    gnosis_extraction_max_concurrency: int = Field(default=1, ge=1)
    gnosis_extraction_chunk_size: int = Field(default=4000, ge=1)
    gnosis_extraction_chunk_overlap: int = Field(default=200, ge=0)
    gnosis_ocr_enabled: bool = False
    gnosis_ocr_model: str = ""
    gnosis_ocr_max_image_bytes: int = Field(default=0, ge=0)
    gnosis_rustfs_enabled: bool = False
    gnosis_rustfs_bucket: str = ""
    gnosis_rustfs_prefix: str = ""
    gnosis_rustfs_endpoint: str = ""
    gnosis_rustfs_retention_days: int | None = Field(default=None, ge=1)
    gnosis_recall_filter_enabled: bool = False
    gnosis_recall_filter_candidates: int = Field(default=30, ge=1)
    # Scope-narrowed dense retrieval. The SDK's search_facts ranks the fact
    # vector index globally and only the gateway's post-filter enforces scope,
    # so in a store holding many users (LongMemEval: one user per question
    # instance, with haystack sessions SHARED between instances) the global
    # top-k fills with other users' near-identical facts and the requesting
    # user's candidates get crowded out. When enabled, dense candidates come
    # from a scope-narrowed vector query instead (over-fetch the index by
    # gnosis_dense_scope_pool, filter to scope in-query, keep the top
    # candidate_limit). Off = byte-identical SDK dense path.
    gnosis_scoped_dense_retrieval_enabled: bool = False
    gnosis_dense_scope_pool: int = Field(default=4000, ge=100, le=100_000)
    gnosis_hybrid_retrieval_enabled: bool = False
    # Weight multiplier applied to the lexical (BM25) leg in RRF fusion.
    # Dense weight is always 1.0; raising this above 1.0 gives BM25 more
    # influence without changing the dense candidate ranking. Default 1.0 =
    # equal-weight RRF (original behavior).
    gnosis_rrf_lexical_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    # LLM-planned graph QA: the caller's free-text query is turned into Cypher
    # by GNOSIS_LLM and executed against Neo4j. Every tenant shares one Neo4j
    # instance and one credential, so the generated query is the only thing
    # between a prompt-injected query and another tenant's rows; the validator,
    # READ transaction, and per-row tenant filter are defense in depth, not a
    # reason to turn this on. Off = /v1/graph/context (and include_graph) use
    # only the fixed parameterised context queries, and graph-QA fusion is a
    # no-op. Default off.
    gnosis_graphqa_enabled: bool = False
    gnosis_graphqa_fusion_enabled: bool = False
    # The graph-QA planner is an LLM call that commonly takes ~10s on a
    # frontier model; a 5s budget timed out on every fusion request. 20s
    # leaves headroom while still bounding a stalled planner.
    gnosis_graphqa_fusion_timeout_seconds: float = Field(default=20.0, gt=0)
    gnosis_read_supersession_enabled: bool = False
    gnosis_entity_graph_enabled: bool = False
    gnosis_graph_traversal_enabled: bool = False
    gnosis_bridge_traversal_enabled: bool = False
    # Item-budget multiplier for coverage-hungry routed context reads.
    # LOCOMO miss analysis (Run 18, 2026-07-04): 27 of 41 multi-hop-category
    # misses are cross-session enumerations where the answer's facts rank
    # below the request budget cut - a retrieval *coverage* gap, not a
    # traversal gap - and the router classifies those enumerations as
    # aggregative (list/synthesis) or multi_hop. 1 = off (byte-identical);
    # with adaptive routing on, only those two routes read expanded.
    gnosis_coverage_budget_multiplier: int = Field(default=1, ge=1, le=5)
    gnosis_adaptive_routing_enabled: bool = False
    gnosis_routing_model: str = ""
    gnosis_sufficiency_check_enabled: bool = False
    gnosis_sufficiency_model: str = ""
    # Listwise LLM reranker over fused fact candidates, applied before the item
    # budget cut so it decides which candidates reach the prompt. Retrieval is
    # the long-haystack bottleneck (LongMemEval full-ctx 0.606 vs oracle 0.870);
    # a reranker is the lever common to the strongest 2026 systems. Default-off
    # (byte-identical read path); one extra structured-output call per query.
    gnosis_rerank_enabled: bool = False
    gnosis_rerank_model: str = ""
    gnosis_rerank_candidate_cap: int = Field(default=50, ge=1, le=200)
    gnosis_abstention_prompt_enabled: bool = False
    gnosis_chain_of_note_enabled: bool = False
    # CoN widenings, both default-off (byte-identical Run 18 instruction).
    # Speculative inference: widen the likelihood carve-out to speculative
    # judgment questions that never say "likely" (LOCOMO Run 18 open-domain
    # misses: 8/12 were abstentions on "Would X ...?" phrasings).
    gnosis_con_speculative_inference_enabled: bool = False
    # Exhaustive enumeration on multi-hop/aggregative routed reads (LOCOMO
    # Run 19: enumeration misses persist at full gold coverage - the reader
    # answers with one salient item; this instructs it to list all/count).
    gnosis_con_enumeration_enabled: bool = False
    # Recency preference clause: when two memories about the same fact give
    # different values, prefer the more recently-dated one. Targets LME_S
    # knowledge-update failures where multiple sessions contain different
    # counts and CoN picks the wrong one (LME_S L-12).
    gnosis_con_recency_preference_enabled: bool = False
    # Absence-implies-unknown clause: prevents two specific absence-of-evidence
    # errors that produce wrong answers on LME_S abstention questions.
    # (1) When no memory mentions an activity, models infer a count of zero
    # instead of saying "not enough information" (e.g., "zero egg tarts baked"
    # because baking was never mentioned). (2) When a question compares who
    # among multiple people did something first/most but only one party's data
    # exists, models answer from partial data instead of abstaining. LME_S
    # L-16 targets: 88432d0a_abs, 0ddfec37_abs, gpt4_fe651585_abs.
    gnosis_con_abstention_enabled: bool = False
    # Recommendation response clause (GNOSIS_CON_RECOMMENDATION_ENABLED).
    # LME_S L-19 targets: SSP questions where model describes preferences in
    # third-person instead of making concrete first/second-person recommendations.
    # `35a27287` ("Can you recommend cultural events?") and `a89d7624` ("Any
    # suggestions for Denver?") both fail because the model says "The user would
    # prefer..." rather than "You might enjoy X or Y."
    gnosis_con_recommendation_enabled: bool = False
    gnosis_fact_verbatim_expansion_enabled: bool = False
    gnosis_fact_verbatim_expansion_max: int = Field(default=5, ge=1)
    gnosis_fact_extraction_enabled: bool = False
    gnosis_fact_extraction_model: str = ""
    gnosis_fact_extraction_context_turns: int = Field(default=10, ge=0)
    gnosis_fact_extraction_mode: Literal["sync", "background"] = "sync"
    gnosis_fact_extraction_max_concurrency: int = Field(default=2, ge=1)
    gnosis_fact_extraction_max_pending: int = Field(default=200, ge=1)
    gnosis_prompt_entities_enabled: bool = False
    gnosis_prompt_preferences_enabled: bool = False
    gnosis_prompt_reasoning_enabled: bool = False
    gnosis_consolidation_schedule_enabled: bool = False
    # Community subgraph: detect connected-component clusters among :Entity nodes
    # per scope, generate LLM summaries, persist as :Community nodes. Closes the
    # open-domain gap (~30 pp vs pure entity retrieval, per Zep/Graphiti analysis).
    # Rebuilt via POST /v1/communities/rebuild or the consolidation schedule.
    # Read path adds community summaries to aggregative/open-domain-routed queries.
    gnosis_community_graph_enabled: bool = False
    gnosis_community_min_entities: int = Field(default=3, ge=2, le=50)
    gnosis_community_context_limit: int = Field(default=5, ge=1, le=20)
    # Multi-query rewrite: when the sufficiency check fires (context insufficient),
    # generate up to 3 complementary queries and RRF-fuse with the original results.
    # EverMemOS fires on 31% of queries; expected multi-hop and open-domain gains.
    # Requires gnosis_sufficiency_check_enabled=true to trigger.
    gnosis_query_rewrite_enabled: bool = False
    gnosis_query_rewrite_model: str = ""
    gnosis_memory_edit_enabled: bool = False
    gnosis_mcp_enabled: bool = False
    gnosis_mcp_agent_id: str = Field(default="mcp-client", min_length=1)
    gnosis_federation_token: str = ""
    gnosis_peers: list[PeerConfig] = Field(default_factory=list)

    @field_validator("gnosis_peers")
    @classmethod
    def _require_unique_peer_names(
        cls,
        peers: list[PeerConfig],
    ) -> list[PeerConfig]:
        seen: set[str] = set()
        for peer in peers:
            normalized = peer.name.casefold()
            if normalized in seen:
                detail = f"duplicate peer name: {peer.name}"
                raise ValueError(detail)
            seen.add(normalized)
        return peers


def load_settings() -> Settings:
    return Settings()
