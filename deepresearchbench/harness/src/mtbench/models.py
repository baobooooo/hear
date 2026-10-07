from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PlanTask(BaseModel):
    agent_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=300)
    objective: str = Field(min_length=1, max_length=2000)
    search_query: str = Field(min_length=1, max_length=500)


class PlanOutput(BaseModel):
    tasks: list[PlanTask]


class Document(BaseModel):
    url: str
    title: str
    text: str
    fetched_at: str
    elapsed_seconds: float


class ResearchReport(BaseModel):
    agent_id: str
    round_no: int
    task: PlanTask
    documents: list[Document]
    report: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    chain_id: str | None = None
    visible_chars: int = 0
    length_retry_count: int = 0
    length_target_met: bool | None = None
    chain_user_message: str = ""


class FeedbackItem(BaseModel):
    agent_id: str
    # The reviewer reasons in `analysis` before committing to `instruction`;
    # both grow with review_analysis_chars / review_instruction_chars.
    analysis: str = Field(default="", max_length=30000)
    instruction: str = Field(min_length=1, max_length=30000)
    document_id: str | None = Field(default=None, max_length=20)
    search_query: str | None = Field(default=None, max_length=500)


class ReviewDecision(BaseModel):
    # A long chain-of-thought answer sometimes loses its wrapper: the reply is a
    # bare feedback item, or a list of them, or an object without `decision`.
    # The content is still usable, so coerce those shapes instead of retrying;
    # the caller supplies the round's required decision when it is missing.
    decision: str = ""
    assessment: str = Field(default="", max_length=30000)

    @model_validator(mode="before")
    @classmethod
    def accept_unwrapped(cls, value: Any) -> Any:
        if isinstance(value, list):
            return {"feedback": value}
        if isinstance(value, dict) and "feedback" not in value and "agent_id" in value:
            return {"feedback": [value]}
        return value
    feedback: list[FeedbackItem] = Field(default_factory=list)
