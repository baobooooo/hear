"""Read only query ID/question fields; ground-truth answers never enter the graph."""
from __future__ import annotations
import json
from pathlib import Path


def selection_ids(ids: str | None = None, ids_file: str | None = None) -> list[str] | None:
    if ids is not None and ids_file is not None:
        raise ValueError("Use either --ids or --ids-file")
    if ids is None and ids_file is None:
        return None
    values = ids.split(",") if ids is not None else Path(ids_file).read_text().splitlines()
    selected = [value.strip() for value in values if value.strip()]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Query selection must contain nonempty, unique IDs")
    return selected


def load_queries(path: str | Path, ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    path = Path(path)
    queries, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if path.suffix.lower() == ".jsonl":
            raw = json.loads(line)
            ident, question = raw.get("query_id", raw.get("id")), raw.get("query", raw.get("question"))
        else:
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(f"Expected exactly two TSV columns, ID<TAB>question, at line {number}; extra columns may contain answer labels")
            ident, question = fields
            if number == 1 and ident.strip().lower() in {"id", "query_id"} and question.strip().lower() in {"query", "question"}:
                continue
        if ident is None or not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing ID/question at line {number}")
        ident = str(ident).strip()
        if not ident or ident in seen:
            raise ValueError(f"Empty/duplicate query ID at line {number}: {ident!r}")
        seen.add(ident)
        queries.append({"query_id": ident, "question": question.strip()})
    if ids:
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate selected query IDs")
        missing = set(ids) - seen
        if missing:
            raise ValueError(f"Unknown query IDs: {sorted(missing)}")
        indexed = {q["query_id"]: q for q in queries}
        queries = [indexed[ident] for ident in ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        queries = queries[:limit]
    if not queries:
        raise ValueError("No selected queries")
    return queries
