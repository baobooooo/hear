"""Source-span validation and idempotent evidence handling, not an entailment oracle."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import unicodedata
from typing import Any
from .schemas import EvidenceProposal, ReaderReply

Record = dict[str, Any]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def candidate_key(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def normalize_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def source_record(docid: str, text: str, start: int, end: int, title: str = "") -> Record:
    if not 0 <= start < end <= len(text):
        raise ValueError("Invalid source span")
    version = hashlib.sha256(text.encode()).hexdigest()
    source_id = "s_" + stable_hash([str(docid), version, start, end])[:20]
    return {"source_id": source_id, "docid": str(docid), "document_sha256": version,
            "start": start, "end": end, "total_chars": len(text),
            "next_offset": end if end < len(text) else None,
            "title": title, "text": text[start:end]}


def merge_sources(left: Record, right: Record) -> Record:
    merged = deepcopy(left)
    for key, source in right.items():
        if key != source["source_id"]:
            raise ValueError("Source key mismatch")
        if key in merged and merged[key] != source:
            raise ValueError("Immutable source-id collision")
        merged[key] = deepcopy(source)
    return merged


def merge_cells(left: Record | None, right: Record | None) -> Record:
    """LangGraph reducer. Cells emit one versioned snapshot per lifecycle/reopen."""
    merged = deepcopy(left or {})
    for key, value in (right or {}).items():
        if key != value["cell_id"]:
            raise ValueError("Cell key mismatch")
        previous = merged.get(key)
        if previous is not None:
            if previous["revision"] > value["revision"]:
                continue
            if previous["revision"] == value["revision"]:
                if previous != value:
                    raise ValueError("Conflicting snapshots at the same cell revision")
                continue
        merged[key] = deepcopy(value)
    return merged



def source_aliases(sources: Record) -> dict[str, str]:
    return {key: f"S{i}" for i, key in enumerate(sources, 1)}


def evidence_aliases(evidence: Record) -> dict[str, str]:
    return {key: f"E{i}" for i, key in enumerate(evidence, 1)}


def exact_span(text: str, quote: str) -> tuple[int, int] | None:
    """Exact text, or whitespace-only equivalence with original-span mapping.

    Punctuation, capitalization, dates, names and missing words are NOT repaired.
    In particular, ellipses cannot join disconnected passages.
    """
    at = text.find(quote)
    if at >= 0:
        return at, at + len(quote)
    def canonical(value: str):
        chunks = list(re.finditer(r"\S+", value))
        chars, origins = [], []
        for i, match in enumerate(chunks):
            if i:
                chars.append(" ")
                origins.append((chunks[i - 1].end(), match.start()))
            for j in range(match.start(), match.end()):
                chars.append(value[j])
                origins.append((j, j + 1))
        return "".join(chars), origins
    normalized, locations = canonical(text)
    wanted, _ = canonical(quote)
    if not wanted:
        return None
    start = normalized.find(wanted)
    if start < 0:
        return None
    return locations[start][0], locations[start + len(wanted) - 1][1]


def accept_reply(reply: ReaderReply, sources: Record, constraints: Record,
                 previous: Record, cell_id: str, turn: int) -> tuple[Record, list[str], int]:
    ledger = deepcopy(previous)
    errors: list[str] = []
    changes = 0
    old_aliases = {alias: key for key, alias in evidence_aliases(previous).items()}
    visible_sources = {alias: key for key, alias in source_aliases(sources).items()}
    for ident in reply.retract_ids:
        ident = old_aliases.get(ident, ident)
        item = ledger.get(ident)
        if item is None or item["cell_id"] != cell_id:
            errors.append(f"Cannot retract unowned/unknown evidence: {ident}")
        elif item.get("active", True):
            item.update(active=False, revision=item["revision"] + 1, updated_turn=turn)
            changes += 1
    for index, raw in enumerate(reply.evidence):
        try:
            raw = dict(raw)
            source_key = raw.get("source_id")
            if isinstance(source_key, str):
                raw["source_id"] = visible_sources.get(source_key, source_key)
            constraint_key = raw.get("constraint_id")
            constraint = constraints.get(constraint_key) if isinstance(constraint_key, str) else None
            # The exact label is metadata, not something the LLM needs to copy.
            if constraint and ("time_scope" not in raw or raw["time_scope"] is None):
                raw["time_scope"] = constraint["time_scope"]
            p = EvidenceProposal.model_validate(raw)
            constraint = constraints.get(p.constraint_id)
            if constraint is None:
                raise ValueError("Unknown constraint_id")
            if p.time_scope != constraint["time_scope"]:
                raise ValueError("time_scope must match the relevant constraint's label")
            source = sources.get(p.source_id)
            if source is None:
                raise ValueError("Source was not actually shown to this cell")
            span = exact_span(source["text"], p.quote)
            if span is None:
                raise ValueError("quote is not a contiguous exact span in the named source")
            offset, quote_end = span
            original_quote = source["text"][offset:quote_end]
            if candidate_key(p.candidate) in {"unknown", "none", "n/a", "null", "tbd"}:
                raise ValueError("Placeholder candidate")
            if p.answer_value:
                if not constraint["answer_target"] or p.relation != "SUPPORTS":
                    raise ValueError("answer_value is allowed only on supporting answer-target evidence")
            ident = "e_" + stable_hash([cell_id, candidate_key(p.candidate), p.constraint_id,
                                        p.source_id, original_quote, p.relation, p.answer_value])[:20]
            if ident in ledger:
                # A repeated identical proposal cannot silently reactivate a retraction.
                continue
            row = p.model_dump()
            row.update(quote=original_quote, quote_match="exact" if p.quote == original_quote else "whitespace_only")
            if not row["claim"]:
                row["claim"] = original_quote[:1200]
            row.update(evidence_id=ident, candidate_key=candidate_key(p.candidate),
                       quote_start=offset, quote_end=quote_end,
                       docid=source["docid"], cell_id=cell_id, revision=1,
                       created_turn=turn, updated_turn=turn, active=True)
            ledger[ident] = row
            changes += 1
        except ValueError as exc:
            errors.append(f"evidence[{index}]: {exc}")
    return ledger, errors, changes


def candidate_status(evidence: Record, constraints: Record) -> Record:
    required = {k for k, c in constraints.items() if c.get("required", True)}
    result: Record = {}
    for e in evidence.values():
        if not e.get("active", True) or e["constraint_id"] not in constraints:
            continue
        c = constraints[e["constraint_id"]]
        if e["time_scope"] != c["time_scope"]:
            continue
        key = e["candidate_key"]
        row = result.setdefault(key, {"candidate": e["candidate"], "supported": set(),
                                     "contradicted": set(), "answers": set(),
                                     "evidence_ids": []})
        row["evidence_ids"].append(e["evidence_id"])
        relation_set = "supported" if e["relation"] == "SUPPORTS" else "contradicted"
        row[relation_set].add(e["constraint_id"])
        if e["relation"] == "SUPPORTS" and c.get("answer_target") and e.get("answer_value"):
            row["answers"].add(e["answer_value"])
    for row in result.values():
        row["missing"] = sorted(required - row["supported"])
        row["has_required_contradiction"] = bool(required & row["contradicted"])
        row["ready"] = bool(required and not row["missing"] and not row["has_required_contradiction"]
                            and len(row["answers"]) == 1)
        for k in ("supported", "contradicted", "answers"):
            row[k] = sorted(row[k])
    return result


def flatten_evidence(cells: Record) -> Record:
    ledger: Record = {}
    for cell_id in sorted(cells):
        for key, row in cells[cell_id].get("evidence", {}).items():
            if key in ledger and ledger[key] != row:
                raise ValueError("Evidence collision across cells")
            ledger[key] = deepcopy(row)
    return ledger


def validate_final(decision: Record, evidence: Record, constraints: Record,
                   require_full_coverage: bool = False) -> tuple[bool, str]:
    key = candidate_key(decision.get("candidate", ""))
    status = candidate_status(evidence, constraints).get(key)
    if not status:
        return False, "No validated evidence for the chosen candidate"
    ids = decision.get("evidence_ids", [])
    if not ids:
        return False, "No decision citations"
    for ident in ids:
        e = evidence.get(ident)
        if not e or not e.get("active", True) or e["candidate_key"] != key:
            return False, "Invalid, retracted, or other-candidate decision citation"
    if status["has_required_contradiction"]:
        return False, "Unresolved contradiction for chosen candidate"
    if len({candidate_key(v) for v in status["answers"]}) > 1:
        return False, "Conflicting supported target values require explicit resolution"
    if require_full_coverage and not status["ready"]:
        return False, "Required evidence coverage is incomplete"
    # answer_value is optional Reader metadata, not the only grounding route.
    # Keep the original explicit-target route. If metadata was omitted, a Main
    # answer may also be grounded in an actually cited, accepted SUPPORTS quote.
    # This verifies provenance/lexical presence, NOT semantic entailment. Main
    # must still resolve identity/attribute meaning, and the official judge
    # independently determines correctness. Never promote missing coverage.
    answer = candidate_key(decision.get("exact_answer", ""))
    target = [evidence[i] for i in ids if constraints[evidence[i]["constraint_id"]].get("answer_target")
              and evidence[i]["relation"] == "SUPPORTS" and evidence[i].get("answer_value")]
    explicit = bool(answer) and any(candidate_key(e["answer_value"]) == answer for e in target)
    # A known target value remains authoritative even if Main omits its citation.
    # Quotation fallback is only for absent metadata, not conflicting answers or
    # a way to avoid citing the evidence that established the requested value.
    if status["answers"]:
        if not explicit:
            return False, "Answer does not match a cited target value"
        return True, "complete" if status["ready"] else "answer_with_gaps"
    def contains_answer(quote: str) -> bool:
        if not answer:
            return False
        text = candidate_key(quote)
        start = text.find(answer)
        while start >= 0:
            end = start + len(answer)
            # Avoid matching e.g. Ann in Joanna or 48 in 1948.
            left = start == 0 or not (answer[0].isalnum() and text[start - 1].isalnum())
            right = end == len(text) or not (answer[-1].isalnum() and text[end].isalnum())
            if left and right:
                return True
            start = text.find(answer, start + 1)
        return False
    direct = any(evidence[i]["relation"] == "SUPPORTS" and contains_answer(evidence[i]["quote"])
                 for i in ids)
    if not (explicit or direct):
        return False, "Answer does not match a cited target value or cited supporting quote"
    return True, "complete" if status["ready"] else "answer_with_gaps"


def render_answer(decision: Record) -> str:
    def line(text: str) -> str:
        # Prevent source/model text from injecting a second evaluator field.
        return re.sub(r"\s+", " ", text).strip()
    answer = line(decision.get("exact_answer", ""))
    explanation = line(decision.get("explanation", ""))
    if not answer:
        answer = "Unable to determine"
    # Model-facing E identifiers are NOT BrowseComp corpus document identifiers.
    # Render only docids attached by the validated decision path, never an invented
    # E-number/hash suffix that the official evaluator could mistake for a docid.
    docs = list(dict.fromkeys(str(d) for d in decision.get("citation_docids", [])))
    allowed = set(docs)
    aliases = decision.get("citation_aliases", {})
    def citation(match):
        tokens = [t for t in re.split(r"[,;\s]+", match.group(1).strip()) if t]
        mapped = [str(aliases.get(t, t)) for t in tokens]
        if mapped and all(t in allowed for t in mapped):
            return " ".join(f"[{t}]" for t in dict.fromkeys(mapped))
        # Preserve prose but not a spurious citation recognized by the grader.
        return "(" + match.group(1) + ")"
    explanation = re.sub(r"\[([^\[\]]*)\]", citation, explanation)
    explanation = re.sub(r"cite(.*?)", citation, explanation)
    if docs:
        explanation = (explanation + " Sources: " + " ".join(f"[{d}]" for d in docs)).strip()
    # Confidence is intentionally absent, not replaced with a fabricated constant.
    return f"Explanation: {explanation or 'The available evidence is insufficient.'}\n\nExact Answer: {answer}"
