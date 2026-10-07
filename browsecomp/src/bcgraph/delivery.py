"""Separate answer-support citations from contextual references without relabelling evidence.

The legacy validate_final still decides whether the selected candidate and answer
are supported. Context cannot fill coverage, supply an answer, or remove a conflict.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Collection

from .evidence import candidate_key


def partition_decision_citations(
    decision: dict[str, Any], evidence: dict[str, dict[str, Any]],
    visible_ids: Collection[str],
) -> tuple[dict[str, Any], list[str]]:
    """Return a support-only decision and explicitly retained context IDs.

    Only citations actually emitted by Main and visible in its current input are
    eligible. Unknown, omitted, or retracted references fail; no new references
    are synthesized. Context must share an immutable source span with a cited
    supporting entry for the chosen candidate. Cross-document/unknown links and
    other-candidate contradiction entries are NOT silently filtered. The evidence
    ledger and original model response are untouched.
    """
    key = candidate_key(decision.get("candidate", ""))
    visible = set(visible_ids)
    support: list[str] = []
    context: list[str] = []
    seen: set[str] = set()
    for ident in decision.get("evidence_ids", []):
        if not isinstance(ident, str) or ident not in visible:
            raise ValueError("Main cited unknown evidence or evidence omitted from its prompt")
        row = evidence.get(ident)
        if row is None or not row.get("active", True):
            raise ValueError("Main cited unknown or retracted evidence")
        if ident in seen:
            continue
        seen.add(ident)
        (support if row["candidate_key"] == key else context).append(ident)
    if not support or not any(evidence[i]["relation"] == "SUPPORTS" for i in support):
        raise ValueError("No cited supporting evidence for the chosen candidate")
    supporting_sources = {evidence[i]["source_id"] for i in support
                          if evidence[i]["relation"] == "SUPPORTS"}
    if any(evidence[i]["relation"] != "SUPPORTS"
           or evidence[i]["source_id"] not in supporting_sources for i in context):
        raise ValueError("Other-candidate context is not in a cited supporting source span")
    normalized = deepcopy(decision)
    normalized["evidence_ids"] = support
    return normalized, context
