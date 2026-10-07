"""Small model-output contracts. Only ordinary chat completion JSON is required."""
from __future__ import annotations

import json
import re
from typing import Any, Literal
from pydantic import Field, model_validator
from .config import StrictModel


class Constraint(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=1200)
    time_scope: str = "unspecified"
    required: bool = True
    answer_target: bool = False


class CellPlan(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    focus: str = Field(min_length=1, max_length=2000)
    constraint_ids: list[str] = Field(min_length=1, max_length=16)
    initial_queries: list[str] = Field(min_length=1, max_length=6)


class Plan(StrictModel):
    constraints: list[Constraint] = Field(min_length=1, max_length=16)
    cells: list[CellPlan] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def references(self):
        ids = {c.id for c in self.constraints}
        if len(ids) != len(self.constraints):
            raise ValueError("Duplicate constraint ids")
        if len({c.id for c in self.cells}) != len(self.cells):
            raise ValueError("Duplicate cell ids")
        if not any(c.answer_target and c.required for c in self.constraints):
            raise ValueError("At least one required answer_target constraint is needed")
        assigned: set[str] = set()
        for cell in self.cells:
            if set(cell.constraint_ids) - ids:
                raise ValueError("Cell refers to an unknown constraint")
            assigned.update(cell.constraint_ids)
        if {c.id for c in self.constraints if c.required} - assigned:
            raise ValueError("Required constraints not assigned to any cell")
        return self


class EvidenceProposal(StrictModel):
    candidate: str = Field(min_length=1, max_length=300)
    constraint_id: str = Field(min_length=1, max_length=40)
    time_scope: str = Field(default="unspecified", max_length=200)
    relation: Literal["SUPPORTS", "CONTRADICTS"] = "SUPPORTS"
    source_id: str = Field(min_length=1, max_length=80)
    quote: str = Field(min_length=4, max_length=2500)
    claim: str = Field(default="", max_length=1200)
    answer_value: str | None = Field(default=None, max_length=500)


class ReadMore(StrictModel):
    docid: str = Field(min_length=1, max_length=300)
    offset: int = Field(default=0, ge=0)


class ReaderReply(StrictModel):
    # Entries are validated individually so one malformed citation does not erase
    # the other correctly sourced evidence in the same model response.
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=32)
    retract_ids: list[str] = Field(default_factory=list, max_length=20)
    next_queries: list[str] = Field(default_factory=list, max_length=6)
    read_more: list[ReadMore] = Field(default_factory=list, max_length=8)
    summary: str = Field(default="", max_length=3000)

    @model_validator(mode="before")
    @classmethod
    def optional_metadata(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        # Legacy outputs are accepted without using these advisory fields.
        value.pop("complete", None)
        value.pop("confidence", None)
        for key in ("evidence", "retract_ids", "next_queries", "read_more"):
            if value.get(key, []) is None:
                value[key] = []
        if value.get("summary", "") is None:
            value["summary"] = ""
        return value


class FinalDecision(StrictModel):
    action: Literal["answer", "research", "unresolved"]
    candidate: str = Field(default="", max_length=300)
    exact_answer: str = Field(default="", max_length=500)
    explanation: str = Field(default="", max_length=4000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    reopen_cell_id: str | None = None
    next_queries: list[str] = Field(default_factory=list, max_length=4)
    read_more: list[ReadMore] = Field(default_factory=list, max_length=4)

    @model_validator(mode="before")
    @classmethod
    def optional_fields(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        value.pop("confidence", None)  # Not a planning/acceptance/calibration input.
        for key in ("candidate", "exact_answer", "explanation"):
            if value.get(key, "") is None:
                value[key] = ""
        for key in ("evidence_ids", "next_queries", "read_more"):
            if value.get(key, []) is None:
                value[key] = []
        return value

    @model_validator(mode="after")
    def action_fields(self):
        if self.action == "answer" and not self.exact_answer.strip():
            raise ValueError("action=answer requires a nonempty exact_answer")
        # A research/abstention response is not required to name a candidate.
        return self


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    # Reasoning is not treated as a final structured answer.
    return ""


def parse_json_object(text: str) -> dict[str, Any]:
    """Tolerate outer markdown/thinking, never modify stored assistant history.

    Do not salvage arbitrary nested objects from truncated JSON. More than one
    top-level JSON object is ambiguous and is rejected.
    """
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.S).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object in model content") from None
        try:
            obj, end = json.JSONDecoder().raw_decode(text, start)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed/truncated JSON: {exc.msg}") from exc
        tail = text[end:].strip()
        if "{" in tail or "}" in tail:
            raise ValueError("Multiple/ambiguous JSON objects")
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj


def parse_plan_output(text: str, *, max_cells: int | None = None) -> tuple[Plan, list[str]]:
    """Normalize unambiguous planner serialization only, never invent a plan.

    One surplus terminal brace after an already complete plan is removable.
    Multiple JSON objects, incomplete JSON, unknown IDs and ambiguous multi-cell
    assignments remain errors. All normalization is reported to the caller;
    the original assistant content must remain unchanged in model history.
    """
    if max_cells is not None and (not isinstance(max_cells, int) or isinstance(max_cells, bool) or not 1 <= max_cells <= 4):
        raise ValueError("max_cells must be between 1 and 4")
    notes: list[str] = []
    raw = re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.S).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    if not raw.startswith("{"):
        # Keep legacy support for a surrounding prose preamble; it cannot take
        # the special terminal-brace recovery route.
        obj = parse_json_object(text)
    else:
        try:
            obj, end = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed/truncated plan JSON: " + exc.msg) from exc
        if not isinstance(obj, dict):
            raise ValueError("Expected a plan JSON object")
        tail = raw[end:].strip()
        if tail:
            if tail == "}" and set(obj) == {"constraints", "cells"}:
                notes.append("discarded_one_redundant_terminal_brace")
            else:
                raise ValueError("Multiple/ambiguous JSON objects or unexpected plan tail")
    # A one-cell deployment has one unambiguous owner of every declared task.
    # Normalize the deployment topology BEFORE the assignment validator runs;
    # otherwise a missing assignment in a discarded multi-cell draft destroys
    # an otherwise usable plan. Never normalize malformed/unknown/duplicate IDs.
    raw_cells = obj.get("cells")
    if max_cells == 1 and isinstance(raw_cells, list) and 1 < len(raw_cells) <= 4:
        if not isinstance(obj.get("constraints"), list):
            # Let Pydantic report malformed schema fields, not a Python TypeError.
            return Plan.model_validate(obj), notes
        constraints = [Constraint.model_validate(c) for c in obj["constraints"]]
        cells = [CellPlan.model_validate(c) for c in raw_cells]
        known = {c.id for c in constraints}
        if len(known) != len(constraints):
            raise ValueError("Duplicate constraint ids")
        if len({c.id for c in cells}) != len(cells):
            raise ValueError("Duplicate cell ids")
        if any(set(c.constraint_ids) - known for c in cells):
            raise ValueError("Cell refers to an unknown constraint")
        queries: list[str] = []
        seen: set[str] = set()
        for cell in cells:
            for query in cell.initial_queries:
                query = " ".join(query.split())[:600]
                key = query.casefold()
                if key and key not in seen:
                    seen.add(key)
                    queries.append(query)
        merged = {
            "id": "cell1", "focus": "\n".join(c.focus for c in cells),
            "constraint_ids": list(dict.fromkeys(i for c in cells for i in c.constraint_ids)),
            "initial_queries": queries[:6],
        }
        obj = {**obj, "cells": [merged]}
        notes.append("merged_planner_cells_before_assignment_validation")

    # There is only one destination when the plan has exactly one cell. Preserve
    # all declared constraints and IDs, instead of discarding the whole plan for
    # an omitted assignment. Unknown IDs still fail ordinary Plan validation.
    if (isinstance(obj.get("cells"), list) and len(obj["cells"]) == 1
            and isinstance(obj.get("constraints"), list)
            and isinstance(obj["cells"][0], dict)):
        required = [c.get("id") for c in obj["constraints"]
                    if isinstance(c, dict) and c.get("required", True)]
        assigned = obj["cells"][0].get("constraint_ids")
        if isinstance(assigned, list) and all(isinstance(i, str) for i in required + assigned):
            missing = [i for i in required if i not in assigned]
            if missing:
                obj = {**obj, "cells": [{**obj["cells"][0],
                                       "constraint_ids": [*assigned, *missing]}]}
                notes.append("assigned_missing_constraints_to_only_cell:" + ",".join(missing))
    return Plan.model_validate(obj), notes


def parse_reader_output(text: str, finish_reason: str | None) -> tuple[ReaderReply, bool]:
    """Only recover fully decoded evidence entries from a length-truncated reply.

    Never add braces, guess string endings, repair a quote, salvage a nested object
    as the whole response, or recover a partial next-query/retraction. Recovered
    entries still go through the ordinary source validator. Raw history is untouched.
    """
    try:
        return ReaderReply.model_validate(parse_json_object(text)), False
    except ValueError as exc:
        if finish_reason != "length" or not str(exc).startswith("Malformed/truncated JSON:"):
            raise
    raw = re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.S).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].lstrip()
    if not raw.startswith("{"):
        raise ValueError("Truncated Reader output has no top-level object")
    decoder = json.JSONDecoder()
    pos = 1
    def skip(i):
        while i < len(raw) and raw[i].isspace():
            i += 1
        return i
    try:
        while True:
            pos = skip(pos)
            key, pos = decoder.raw_decode(raw, pos)
            if not isinstance(key, str):
                raise ValueError("Invalid top-level key")
            pos = skip(pos)
            if pos >= len(raw) or raw[pos] != ":":
                raise ValueError("Missing key separator")
            pos = skip(pos + 1)
            if key == "evidence":
                if pos >= len(raw) or raw[pos] != "[":
                    raise ValueError("Evidence is not an array")
                pos = skip(pos + 1)
                entries = []
                while pos < len(raw) and raw[pos] != "]":
                    try:
                        entry, end = decoder.raw_decode(raw, pos)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(entry, dict):
                        raise ValueError("Evidence entry is not an object")
                    entries.append(entry)
                    if len(entries) > 32:
                        raise ValueError("Too many evidence entries")
                    pos = skip(end)
                    if pos < len(raw) and raw[pos] == ",":
                        pos = skip(pos + 1)
                    else:
                        break
                if entries:
                    return ReaderReply(evidence=entries), True
                raise ValueError("No complete evidence entries in truncated output")
            _, pos = decoder.raw_decode(raw, pos)
            pos = skip(pos)
            if pos >= len(raw) or raw[pos] != ",":
                break
            pos += 1
    except json.JSONDecodeError as exc:
        raise ValueError("No recoverable top-level evidence array") from exc
    raise ValueError("No complete evidence entries in truncated output")
