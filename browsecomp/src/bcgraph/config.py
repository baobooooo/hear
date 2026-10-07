"""Validated configuration. Engine knobs belong to the serving process, not HTTP prompts."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EndpointConfig(StrictModel):
    base_url: str = "http://127.0.0.1:18400/v1"
    model: str = "GLM-4.7-Flash"
    engine: Literal["vllm", "sparse-vllm", "deepseek"] = "vllm"
    method: Literal["vanilla", "omnikv", "h2o", "snapkv"] = "vanilla"
    cache: Literal["prefix", "chain", "none"] = "prefix"
    chain_transport: Literal["delta", "full"] = "delta"
    # Set to a serving-process UUID/deployment ID. Handles are also scoped to this
    # client process, so an interrupted run never assumes that old GPU KV survived.
    engine_epoch: str = "manual-deployment-1"
    api_key_env: str = "HARNESS_API_KEY"
    timeout_seconds: float = Field(default=240, gt=0)
    max_context_tokens: int = Field(default=65536, gt=1024)
    max_inflight: int = Field(default=4, ge=1)
    temperature: float = Field(default=0, ge=0)
    top_p: float = Field(default=1, gt=0, le=1)
    enable_thinking: bool | None = False
    preserve_thinking: bool | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Allowed sampling fields only; cannot overwrite messages/cache routing.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    chain_missing_codes: list[str] = Field(
        default_factory=lambda: ["chain_not_found", "chain_gone", "chain_evicted", "unknown_chain", "expired_chain"]
    )
    max_safe_retries: int = Field(default=1, ge=0, le=3)
    max_cold_recoveries: int = Field(default=1, ge=0, le=2)
    # Local tokenizer must use exactly the serving model's tokenizer/template.
    tokenizer_path: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_format: Literal["hf_chat_template", "deepseek_v4"] = "hf_chat_template"
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def compatibility(self):
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if self.cache == "chain" and (self.engine != "sparse-vllm" or self.method not in {"h2o", "snapkv"}):
            raise ValueError("This harness enables chain only for sparse-vllm H2O/SnapKV")
        if self.method in {"h2o", "snapkv"} and self.cache == "prefix":
            raise ValueError("H2O/SnapKV cannot use this harness's radix/prefix path; select chain or none")
        if self.engine == "vllm" and self.method != "vanilla":
            raise ValueError("Select sparse-vllm to request an OmniKV/H2O/SnapKV endpoint")
        if self.engine == "deepseek" and (self.method != "vanilla" or self.chat_template_kwargs
                                          or self.preserve_thinking is not None):
            raise ValueError("DeepSeek API uses vanilla requests without local chat-template overrides")
        allowed = {"top_k", "presence_penalty", "repetition_penalty", "stop", "seed"}
        if set(self.extra_body) - allowed:
            raise ValueError(f"Unapproved extra_body keys: {sorted(set(self.extra_body) - allowed)}")
        if self.engine == "sparse-vllm" and "seed" in self.extra_body:
            raise ValueError("The inspected Sparse-vLLM chat schema does not accept seed; seed the server")
        return self


class RetrievalConfig(StrictModel):
    transport: Literal["http", "stdio", "fixture"] = "http"
    url: str = "http://127.0.0.1:8020/mcp/"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=120, gt=0)
    max_inflight: int = Field(default=8, ge=1)
    max_search_inflight: int = Field(default=6, ge=1)
    # top_k is enforced client-side; the server should be configured to match.
    top_k: int = Field(default=12, ge=1, le=100)
    cache_entries: int = Field(default=2048, ge=0)
    fixture_path: str | None = None

    @model_validator(mode="after")
    def parameters(self):
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio retrieval requires command and args")
        if self.transport == "fixture" and not self.fixture_path:
            raise ValueError("fixture retrieval requires fixture_path")
        return self


class WorkflowConfig(StrictModel):
    research_protocol: Literal["passages-v2", "legacy-evidence"] = "passages-v2"
    reader_mode: Literal["selector", "researcher"] = "selector"
    research_directions: bool = False
    research_direction_guards: bool = False
    reader_delivery_repair_tokens: int = Field(default=0, ge=0, le=4096)
    research_sync_interval: int = Field(default=0, ge=0, le=10)
    researcher_select_documents: bool = False
    document_catalog_tokens: int = Field(default=12000, ge=512)
    document_selection_output_tokens: int = Field(default=2048, ge=128)
    document_selection_repair_tokens: int = Field(default=1024, ge=128)
    expanded_research_reports: bool = False
    main_use_research_report: bool = True
    main_research_report_token_cap: int = Field(default=1200, ge=0)
    passage_chars: int = Field(default=1200, ge=256, le=4000)
    future_turn_overhead_tokens: int = Field(default=1024, ge=128)
    query_concurrency: int = Field(default=4, ge=1)
    max_cells_per_query: int = Field(default=1, ge=1, le=4)
    # Admission holds a slot for a whole cell, including retrieval gaps, not only
    # while its HTTP request is in flight. This bounds potential resident chains.
    max_live_cells: int = Field(default=4, ge=1)
    reader_priority_aging_seconds: float = Field(default=10, gt=0)
    max_reader_turns: int = Field(default=3, ge=1, le=10)
    max_searches_per_query: int = Field(default=10, ge=1, le=100)
    max_searches_per_fetch: int = Field(default=3, ge=1, le=8)
    first_documents: int = Field(default=9, ge=1)
    followup_documents: int = Field(default=4, ge=1)
    max_document_fetches_per_query: int = Field(default=30, ge=1)
    max_reopens: int = Field(default=1, ge=0, le=1)
    reopen_searches: int = Field(default=2, ge=1)
    reopen_turns: int = Field(default=1, ge=1, le=3)
    planner_output_tokens: int = Field(default=1400, ge=128)
    reader_first_output_tokens: int = Field(default=1536, ge=128)
    reader_followup_output_tokens: int = Field(default=1024, ge=128)
    final_output_tokens: int = Field(default=1400, ge=128)
    # Opt in per run, so archived configurations still reproduce their policy.
    decision_thinking: Literal["inherit", "phase", "off"] = "inherit"
    final_recovery_tokens: int = Field(default=0, ge=0)
    # Source budget, not a forced minimum. No padding to hit an H2O threshold.
    first_source_tokens: int = Field(default=22000, ge=256)
    followup_source_tokens: int = Field(default=5000, ge=128)
    max_source_tokens_per_document: int = Field(default=5000, ge=128)
    min_source_tokens: int = Field(default=80, ge=16)
    context_reserve_tokens: int = Field(default=512, ge=64)
    final_evidence_tokens: int = Field(default=14000, ge=1024)
    max_total_reader_output_tokens: int = Field(default=7000, ge=128)
    max_no_progress_turns: int = Field(default=2, ge=1)
    max_query_seconds: float = Field(default=900, gt=1)
    # Optional cold-cell routing is disabled for controlled ablations. A selected
    # backend is pinned to the cell for its whole lifetime, never switched mid-chain.
    cold_dense_below_tokens: int = Field(default=0, ge=0)
    reader_routing: Literal["off", "comfort", "balanced", "measured"] = "off"
    routing_h2o_min_prompt_tokens: int = Field(default=16384, ge=0)
    routing_cost_table: str | None = None
    routing_cost_table_sha256: str | None = None
    allow_approximate_tokenizer: bool = False
    store_raw_requests: bool = True
    require_full_coverage_for_final: bool = False
    # Conservative reproduces the existing behavior. Best-effort prioritizes
    # bounded research and a source-grounded best candidate, not fabricated answers.
    answer_policy: Literal["conservative", "best_effort"] = "conservative"

    @model_validator(mode="after")
    def budgets(self):
        if (self.research_sync_interval or self.researcher_select_documents or self.expanded_research_reports) and not self.research_direction_guards:
            raise ValueError('coordinated research requires research direction guards')
        if self.research_sync_interval and self.max_reopens:
            raise ValueError('synchronized research uses scheduled rounds instead of legacy reopens')
        if self.research_direction_guards and not self.research_directions:
            raise ValueError('research_direction_guards requires research_directions')
        if self.reader_delivery_repair_tokens and (not self.research_direction_guards or self.reader_delivery_repair_tokens < 128):
            raise ValueError('reader delivery repair requires direction guards and at least 128 tokens')
        if self.research_direction_guards:
            extra_rounds = self.max_reopens * self.reopen_turns
            required_docs = self.max_cells_per_query * (
                self.first_documents + (self.max_reader_turns - 1) * self.followup_documents
            ) + extra_rounds * self.followup_documents
            required_output = self.max_cells_per_query * (
                self.reader_first_output_tokens + (self.max_reader_turns - 1) * self.reader_followup_output_tokens
            ) + extra_rounds * self.reader_followup_output_tokens
            required_output += (self.max_cells_per_query * self.max_reader_turns + extra_rounds) * self.reader_delivery_repair_tokens
            if self.researcher_select_documents:
                required_output += (self.max_cells_per_query * self.max_reader_turns + extra_rounds) * (
                    self.document_selection_output_tokens + self.document_selection_repair_tokens)
            if self.max_document_fetches_per_query < required_docs:
                raise ValueError(f'direction document budget needs at least {required_docs} for all scheduled rounds')
            if self.max_total_reader_output_tokens < required_output:
                raise ValueError(f'direction output budget needs at least {required_output} for all scheduled rounds')
        if self.research_directions and (self.reader_mode != 'researcher' or self.research_protocol != 'passages-v2'):
            raise ValueError('research_directions requires passages-v2 researcher mode')
        if self.final_recovery_tokens and not 128 <= self.final_recovery_tokens <= self.final_output_tokens - 128:
            raise ValueError("final_recovery_tokens must leave at least 128 tokens for both requests")
        if self.answer_policy == "best_effort" and self.require_full_coverage_for_final:
            raise ValueError("best_effort allows evidence gaps; set require_full_coverage_for_final=false")
        if self.max_searches_per_query < self.max_cells_per_query:
            raise ValueError("Each cell needs at least one search slot")
        if self.max_document_fetches_per_query < self.max_cells_per_query:
            raise ValueError("Each cell needs a document-fetch budget")
        if self.max_total_reader_output_tokens < self.max_cells_per_query * 128:
            raise ValueError("Reader output budget too small for the requested cells")
        return self


class AppConfig(StrictModel):
    main: EndpointConfig = Field(default_factory=EndpointConfig)
    reader: EndpointConfig = Field(default_factory=EndpointConfig)
    reader_replicas: list[EndpointConfig] = Field(default_factory=list)
    dense_reader: EndpointConfig | None = None
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    run_label: str = "dense"
    # This metadata is recorded, not interpreted as configuration for the engine.
    engine_manifest: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def routing(self):
        if self.reader_replicas:
            if self.workflow.reader_routing != "off" or self.dense_reader is not None:
                raise ValueError("Reader replicas cannot be combined with dynamic reader routing")
            replica_fields = (
                "model", "engine", "method", "cache", "chain_transport",
                "timeout_seconds", "max_context_tokens", "max_inflight",
                "temperature", "top_p", "enable_thinking",
                "preserve_thinking", "extra_body", "chat_template_kwargs",
                "chain_missing_codes", "max_safe_retries", "max_cold_recoveries",
                "tokenizer_path", "tokenizer_revision", "tokenizer_format",
                "trust_remote_code",
            )
            for replica in self.reader_replicas:
                for field in replica_fields:
                    if getattr(replica, field) != getattr(self.reader, field):
                        raise ValueError(f"Reader replica differs in {field}")
            urls = [self.reader.base_url, *(replica.base_url for replica in self.reader_replicas)]
            if len(set(urls)) != len(urls):
                raise ValueError("Reader replica base_url values must be distinct")
        if self.workflow.reader_routing != "off":
            if self.workflow.reader_routing == "measured":
                if (not self.workflow.routing_cost_table or
                        not re.fullmatch(r'[0-9a-f]{64}', self.workflow.routing_cost_table_sha256 or '')):
                    raise ValueError('Measured routing requires a frozen cost table and SHA256')
            if self.workflow.cold_dense_below_tokens:
                raise ValueError("Dynamic routing cannot be combined with cold routing")
            if self.dense_reader is None:
                raise ValueError("Dynamic routing requires dense_reader")
            if self.workflow.reader_routing in {"comfort", "measured"} and self.reader.method != "h2o":
                raise ValueError("Comfort routing requires an H2O reader")
            for field in ("model", "max_context_tokens", "temperature", "top_p", "enable_thinking",
                          "preserve_thinking", "extra_body", "chat_template_kwargs", "tokenizer_path"):
                if getattr(self.reader, field) != getattr(self.dense_reader, field):
                    raise ValueError(f"Dynamic routing endpoints differ in {field}")
        if self.workflow.reader_mode == "researcher" and self.workflow.research_protocol != "passages-v2":
            raise ValueError("researcher requires passages-v2")
        if self.workflow.research_protocol == "passages-v2" and self.workflow.require_full_coverage_for_final:
            raise ValueError("passages-v2 checks provenance, not a formal coverage proof; require_full_coverage_for_final must be false")
        if self.main.cache == "chain":
            raise ValueError("Main uses stable full history + prefix caching, not a reader chain")
        if self.workflow.cold_dense_below_tokens and self.dense_reader is None:
            raise ValueError("cold_dense_below_tokens requires dense_reader")
        if self.dense_reader and (self.dense_reader.cache == "chain" or self.dense_reader.method != "vanilla"):
            raise ValueError("dense_reader must be a vanilla dense endpoint without chain mode")
        return self


def _expand(value: Any) -> Any:
    """Expand ${NAME} or ${NAME:-default}; fail on missing required variables."""
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ValueError(f"Missing environment variable: {name}")
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", replace, value)
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return AppConfig.model_validate(_expand(data))
