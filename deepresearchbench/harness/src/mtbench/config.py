from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


class RunConfig(BaseModel):
    instance_id: int = Field(ge=1, le=100)
    subagents: int = Field(ge=1, le=128)
    # This is an upper bound for full, untruncated pages collected per round.
    # The workflow stops adding pages once the round token target is reached.
    documents_per_subagent: int = Field(ge=1, le=200)
    report_rounds: int = Field(
        ge=1,
        le=10,
        validation_alias=AliasChoices("report_rounds", "max_report_rounds"),
    )
    max_document_chars: int = Field(default=20_000, ge=1_000, le=200_000)
    full_document_chars: int = Field(default=200_000, ge=1_000, le=500_000)
    round_input_token_targets: list[int] = Field(
        default_factory=lambda: [120_000, 150_000, 180_000],
        min_length=1,
        max_length=10,
    )
    research_output_tokens: int = Field(default=256, ge=32, le=8_192)
    planner_retries: int = Field(default=2, ge=0, le=5)
    json_retries: int = Field(default=2, ge=0, le=5)
    # None inherits main_model.max_tokens; stage overrides are not clamped to it.
    planner_max_tokens: int | None = Field(default=None, ge=128, le=65_536)
    review_max_tokens: int | None = Field(default=None, ge=128, le=65_536)
    research_target_chars: int = Field(default=0, ge=0, le=100_000)
    research_min_chars: int = Field(default=0, ge=0, le=100_000)
    research_max_chars: int = Field(default=0, ge=0, le=100_000)
    # Long-context main agent.  review_document_tokens feeds the reviewer the
    # sub-agents' own source text; writer_history_tokens feeds the writer every
    # round's memos instead of only the final ones.  Both are token budgets
    # (0 = off) held under main_context_token_cap.
    review_document_tokens: int = Field(default=0, ge=0, le=150_000)
    review_documents_per_agent: int = Field(default=5, ge=1, le=50)
    writer_history_tokens: int = Field(default=0, ge=0, le=150_000)
    writer_document_tokens: int = Field(default=0, ge=0, le=150_000)
    review_instruction_chars: int = Field(default=0, ge=0, le=20_000)
    review_analysis_chars: int = Field(default=0, ge=0, le=20_000)
    review_assessment_chars: int = Field(default=0, ge=0, le=20_000)
    main_context_token_cap: int = Field(default=150_000, ge=8_000, le=200_000)
    writer_target_chars: int = Field(default=0, ge=0, le=100_000)
    writer_min_chars: int = Field(default=0, ge=0, le=100_000)
    writer_max_chars: int = Field(default=0, ge=0, le=100_000)
    length_retries: int = Field(default=0, ge=0, le=5)
    judge: bool = False

    @model_validator(mode="after")
    def validate_length_targets(self) -> "RunConfig":
        if self.full_document_chars < self.max_document_chars:
            raise ValueError(
                "full_document_chars must be greater than or equal to max_document_chars"
            )
        for prefix in ("research", "writer"):
            target = getattr(self, f"{prefix}_target_chars")
            minimum = getattr(self, f"{prefix}_min_chars")
            maximum = getattr(self, f"{prefix}_max_chars")
            values = (target, minimum, maximum)
            if values == (0, 0, 0):
                continue
            if not (0 < minimum <= target <= maximum):
                raise ValueError(
                    f"{prefix} length target must satisfy 0 < min <= target <= max; "
                    f"got min={minimum}, target={target}, max={maximum}"
                )
        if len(self.round_input_token_targets) < self.report_rounds:
            raise ValueError(
                "round_input_token_targets must contain at least report_rounds entries"
            )
        if any(target < 1_024 for target in self.round_input_token_targets):
            raise ValueError("every round_input_token_targets entry must be >= 1024")
        return self


class ModelConfig(BaseModel):
    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=128, le=65_536)
    timeout_seconds: float = Field(default=600, gt=0)
    max_concurrency: int | None = Field(default=None, ge=1, le=128)
    continuation_mode: Literal["full_transcript", "chain_delta"] = "full_transcript"
    # GLM-4.7 chain-of-thought.  Reasoning arrives as reasoning_content and is
    # counted in completion_tokens, so it lengthens the decode without changing
    # the JSON contract carried in content.
    enable_thinking: bool = False


class RetrievalConfig(BaseModel):
    keys_file: Path
    search_url: str = "https://google.serper.dev/search"
    proxy_url: str | None = None
    # Serper is reachable directly from the H100 host and is faster that way;
    # the QEMU VM has no route of its own and must keep this true.
    search_via_proxy: bool = True
    results_per_search: int = Field(default=10, ge=1, le=20)
    max_search_pages: int = Field(default=3, ge=1, le=10)
    per_key_concurrency: int = Field(default=5, ge=1, le=5)
    fetch_concurrency: int = Field(default=16, ge=1, le=128)
    search_timeout_seconds: float = Field(default=20, gt=0)
    fetch_timeout_seconds: float = Field(default=20, gt=0)
    fetch_retries: int = Field(default=1, ge=0, le=5)


class ExperimentConfig(BaseModel):
    run: RunConfig
    main_model: ModelConfig
    researcher_model: ModelConfig
    retrieval: RetrievalConfig

    @model_validator(mode="after")
    def validate_distinct_roles(self) -> "ExperimentConfig":
        if not self.main_model.base_url.endswith("/v1"):
            raise ValueError("main_model.base_url must end in /v1")
        if not self.researcher_model.base_url.endswith("/v1"):
            raise ValueError("researcher_model.base_url must end in /v1")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        return ExperimentConfig.model_validate(tomllib.load(handle))
