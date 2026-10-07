"""Raw-passage provenance. Selection is NOT a semantic proof or a generated quote.

All model-facing P numbers point to exact contiguous substrings of text actually
read by this cell. Internal IDs are immutable; aliases never change on append.
"""
from __future__ import annotations
from copy import deepcopy
import json
import re
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .evidence import stable_hash, normalize_query
from .schemas import ReadMore, parse_json_object
from .retrieval import normalize_docid, RetrievalError

PROTOCOL = 'passages-v2'


class OutputModel(BaseModel):
    # Model commentary is not configuration. Unknown metadata is logged by the
    # parser but cannot invalidate a complete decision (e.g. missing_fact).
    model_config = ConfigDict(extra='ignore')


class Selection(OutputModel):
    selected_passage_ids: list[str] = Field(default_factory=list, max_length=32)
    drop_passage_ids: list[str] = Field(default_factory=list, max_length=16)
    candidate_answer: str = Field(default='', max_length=500)
    next_queries: list[str] = Field(default_factory=list, max_length=6)
    read_more: list[ReadMore] = Field(default_factory=list, max_length=8)
    missing_fact: str = Field(default='', max_length=1500)

    @model_validator(mode='before')
    @classmethod
    def optional(cls, data):
        if not isinstance(data, dict): return data
        d = dict(data)
        if 'selected_passage_ids' not in d:
            for key in ('selected_passages', 'passage_ids'):
                if key in d: d['selected_passage_ids'] = d[key]; break
        for key in ('selected_passage_ids','drop_passage_ids','next_queries','read_more'):
            if d.get(key) is None: d[key] = []
        for key in ('candidate_answer','missing_fact'):
            if d.get(key) is None: d[key] = ''
        return d


class Decision(OutputModel):
    action: Literal['answer','research','unresolved']
    exact_answer: str = Field(default='', max_length=1000)
    explanation: str = Field(default='', max_length=5000)
    citations: list[str] = Field(default_factory=list, max_length=40)
    reopen_cell_id: str | None = None
    next_queries: list[str] = Field(default_factory=list, max_length=4)
    read_more: list[ReadMore] = Field(default_factory=list, max_length=4)

    @model_validator(mode='before')
    @classmethod
    def optional(cls, data):
        if not isinstance(data, dict): return data
        d = dict(data)
        if 'citations' not in d:
            for key in ('passage_ids', 'evidence_ids'):
                if key in d: d['citations'] = d[key]; break
        if not d.get('explanation') and isinstance(d.get('missing_fact'), str):
            d['explanation'] = d['missing_fact']
        for key in ('citations','next_queries','read_more'):
            if d.get(key) is None: d[key] = []
        for key in ('exact_answer','explanation'):
            if d.get(key) is None: d[key] = ''
        return d

    @model_validator(mode='after')
    def nonempty_answer(self):
        if self.action == 'answer' and not self.exact_answer.strip():
            raise ValueError('answer requires a nonempty exact_answer')
        return self


def output_object(text: str) -> tuple[dict, list[str]]:
    """A single extra terminal brace is repairable; ambiguity is not."""
    try:
        return parse_json_object(text), []
    except ValueError:
        t = re.sub(r'^\s*<think>.*?</think>\s*', '', text, count=1, flags=re.S).strip()
        if t.startswith('```') and t.endswith('```'):
            t = '\n'.join(t.splitlines()[1:-1]).strip()
        if t.startswith('{'):
            try: obj, end = json.JSONDecoder().raw_decode(t)
            except json.JSONDecodeError: pass
            else:
                if isinstance(obj, dict) and t[end:].strip() == '}':
                    return obj, ['removed_one_surplus_terminal_brace']
        raise


def parse_selection(text: str, finish_reason: str | None) -> tuple[Selection, list[str], bool]:
    partial = False
    try:
        obj, notes = output_object(text)
    except ValueError:
        # Only recover COMPLETE string IDs from the FIRST top-level selection
        # array on length truncation. No synthetic text/answer/query is created.
        match = re.match(r'^\s*\{\s*"selected_passage_ids"\s*:\s*\[', text)
        if finish_reason != 'length' or not match: raise
        pos, ids = match.end(), []
        decoder = json.JSONDecoder()
        while pos < len(text) and len(ids) < 32:
            while pos < len(text) and text[pos].isspace(): pos += 1
            try: value, end = decoder.raw_decode(text, pos)
            except json.JSONDecodeError: break
            if not isinstance(value, str): break
            ids.append(value); pos = end
            while pos < len(text) and text[pos].isspace(): pos += 1
            if pos >= len(text) or text[pos] != ',': break
            pos += 1
        if not ids: raise ValueError('No complete passage ID in truncated selection')
        obj, notes, partial = {'selected_passage_ids': ids}, ['partial_selection_ids_only'], True
    allowed = set(Selection.model_fields) | {'selected_passages','passage_ids'}
    extra = sorted(set(obj) - allowed)
    if extra: notes.append('ignored_metadata:' + ','.join(extra))
    return Selection.model_validate(obj), notes, partial


class ResearchSelection(Selection):
    clear_candidate: bool = Field(default=False, strict=True)


def parse_research_output(text: str, finish_reason: str | None, *, bounded=False):
    """Decode only the leading control object; never search report prose for JSON."""
    body = text.lstrip()
    wrapper_notes = []
    if bounded and re.match(r'^```(?:json)?[ \t]*\r?\n', body):
        body = re.sub(r'^```(?:json)?[ \t]*\r?\n', '', body, count=1)
        wrapper_notes.append('markdown_control_fence')
    try:
        obj, end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError:
        # The existing truncation path recovers only complete selected IDs.
        if not body.startswith('{') or finish_reason != 'length':
            raise ValueError('research control must start with a complete JSON object')
        selection, notes, partial = parse_selection(body, finish_reason)
        return ResearchSelection.model_validate(selection.model_dump()), notes, partial, '', 'invalid'
    if not isinstance(obj, dict):
        raise ValueError('research control must be an object')
    notes = list(wrapper_notes)
    if bounded:
        supplied = set(obj)
        if 'selected_passage_ids' not in supplied and supplied & {'selected_passages', 'passage_ids'}:
            supplied.add('selected_passage_ids')
        obj = {key: value for key, value in Selection.optional(obj).items() if key in supplied}
        for key, limit in [('selected_passage_ids', 32), ('drop_passage_ids', 16)]:
            if key not in obj:
                continue
            values = obj[key]
            if isinstance(values, list):
                kept = list(dict.fromkeys(v for v in values
                            if isinstance(v, str) and re.fullmatch(r'P[1-9]\d*', v)))[:limit]
                if kept != values:
                    notes.append(f'bounded_control:{key}:{len(values)}->{len(kept)}')
                obj[key] = kept
        # Validate fields independently so malformed optional control cannot erase
        # valid positive selections or executable queries.
        filtered = {}
        rejected = []
        for key, value in obj.items():
            if key not in ResearchSelection.model_fields:
                continue
            if key == 'next_queries' and isinstance(value, list):
                value = list(dict.fromkeys(q for q in value if isinstance(q, str) and q.strip()))[:6]
            if key == 'read_more' and isinstance(value, list):
                reads = []
                for item in value:
                    try:
                        if not isinstance(item, dict) or type(item.get('offset')) is not int or item['offset'] < 0:
                            raise ValueError('invalid read offset')
                        reads.append(ReadMore.model_validate({**item, 'docid': normalize_docid(item.get('docid'))}).model_dump())
                    except (ValueError, RetrievalError):
                        notes.append('invalid_control_item:read_more')
                value = reads[:8]
            try:
                ResearchSelection.model_validate({key: value})
                filtered[key] = value
            except ValueError:
                rejected.append(key)
                notes.append('invalid_control_field:' + key)
        if rejected and not filtered:
            raise ValueError('no valid control fields')
        obj = filtered
    selection = ResearchSelection.model_validate(obj)
    extra = sorted(set(obj) - set(ResearchSelection.model_fields) - {'selected_passages', 'passage_ids'})
    if extra:
        notes.append('ignored_metadata:' + ','.join(extra))
    suffix = body[end:]
    if wrapper_notes:
        suffix = re.sub(r'^\s*```[ \t]*(?:\r?\n|$)', '\n', suffix, count=1)
        suffix = re.sub(r'\r?\n```\s*$', '', suffix, count=1)
    if not suffix.strip():
        return selection, notes, False, '', 'missing'
    if suffix.lstrip().startswith('{'):
        raise ValueError('ambiguous second control object')
    marker = re.match(r'^\s*RESEARCH_REPORT\r?\n', suffix)
    if marker is None:
        if not bounded:
            return selection, [*notes, 'invalid_report_boundary'], False, '', 'invalid'
        # This is already-generated unverified analysis, never control or evidence.
        report = suffix.lstrip()
        notes.append('report_marker_missing:preserved_unverified_text')
    else:
        report = suffix[marker.end():]
    status = 'truncated' if finish_reason == 'length' else 'complete' if report.strip() else 'missing'
    return selection, notes, False, report, status


REPORT_REFERENCES = re.compile(r'\[\s*(P\d+(?:\s*[-–]\s*P?\d+)?(?:\s*,\s*P\d+(?:\s*[-–]\s*P?\d+)?)*)\s*\]')


def report_aliases(text: str) -> list[str]:
    aliases = []
    for item in text.split(','):
        match = re.fullmatch(r'\s*P(\d+)(?:\s*[-–]\s*P?(\d+))?\s*', item)
        if match is None:
            raise ValueError('invalid report passage reference')
        first = int(match[1])
        last = int(match[2]) if match[2] else first
        if not 0 <= last - first < 32:
            # Preserve the invalid reference for warnings / NOT_SHOWN mapping.
            aliases.append(item.strip())
        else:
            aliases.extend(f'P{i}' for i in range(first, last + 1))
    return list(dict.fromkeys(aliases))


def question_clue_catalog(question: str) -> dict[str, str]:
    """Stable references to task text; these are not evidence or proof labels."""
    parts = [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+', question) if part.strip()]
    return {f'Q{i+1}': part for i, part in enumerate(parts)}


def grounded_direction(cell: dict, question: str) -> tuple[dict, list[str]]:
    cell = dict(cell)
    ids = cell.get('starting_clue_ids')
    if ids is not None:
        catalog = question_clue_catalog(question)
        if not isinstance(ids, list) or not ids or not all(isinstance(key, str) and key in catalog for key in ids):
            raise ValueError('starting_clue_ids must select supplied Q identifiers')
        cell['starting_clues'] = [catalog[key] for key in dict.fromkeys(ids)]
    unknown = re.compile(r'\b(person|author|researcher|candidate)\s*\d+\b', re.I)
    queries = cell['initial_queries']
    kept = [query for query in queries if not unknown.search(query)]
    notes = []
    if kept != queries:
        notes.append('removed_placeholder_queries')
        if not kept:
            # Use the selected question text, never guess an unknown person's name.
            clues = cell.get('starting_clues', [])
            if isinstance(clues, list):
                kept = [unknown.sub('', clue).strip(' :;,')[:600]
                        for clue in clues if isinstance(clue, str)][:2]
                kept = [query for query in kept if query]
            notes.append('queries_from_grounded_clues')
        cell['initial_queries'] = kept
    return cell, notes


def validate_direction_plan(cells: list[dict], question: str, max_cells: int) -> None:
    """Check executable plan contracts, not semantic truth or answer coverage."""
    if not 1 <= len(cells) <= max_cells:
        raise ValueError('direction count exceeds the configured limit')
    normalize = lambda text: ' '.join(re.findall(r'\w+', text.casefold()))
    original = normalize(question)
    seen = set()
    for cell in cells:
        clues = cell.get('starting_clues')
        if (not isinstance(clues, list) or not clues or
                not all(isinstance(c, str) and len(c.strip()) >= 4 and normalize(c) in original for c in clues)
                or not any(len(c.strip()) >= 8 for c in clues)):
            raise ValueError('each direction needs exact starting_clues from the question')
        if re.search(r'(first|second|other|another|previous)\s+(direction|cell|researcher)|第[一二三123]个方向',
                     cell['focus'], re.I):
            raise ValueError('direction depends on another direction; use independently available clues')
        queries = cell['initial_queries']
        if not queries:
            raise ValueError('direction has no executable search query')
        if any(re.search(r'\b(person|author|researcher|candidate)\s*\d+\b', q, re.I) for q in queries):
            raise ValueError('replace unknown person-number placeholders with descriptive search clues')
        normalized = {normalize(q) for q in queries}
        if normalized and normalized <= seen:
            raise ValueError('direction repeats all earlier search queries')
        seen.update(normalized)


def control_handoff(state: dict, selected: dict, candidate: str, gap: str) -> str:
    """A labeled delivery fallback from state, never synthetic research findings."""
    lookup = aliases(registry(state['sources'], state.get('passage_chars', 1200)))
    ids = [lookup[key] for key, row in selected.items() if row.get('active', True) and key in lookup][:8]
    return ('PROGRAM-GENERATED CONTROL SUMMARY; no usable narrative report was supplied.\n'
            'Assigned direction: ' + state['focus'] + '\n'
            'Unverified requested-value hypothesis: ' + (candidate or '(none)') + '\n'
            'Selected original passages for Main to inspect (not verified claims): ' +
            (' '.join('[' + ident + ']' for ident in ids) or '(none)') + '\n'
            'Reader-reported remaining gap: ' + (gap or '(not specified)'))


def report_references(report: str, sources: dict, passage_chars: int):
    rows = registry(sources, passage_chars)
    lookup = {alias: rows[key] for key, alias in aliases(rows).items()}
    refs, warnings = {}, []
    aliases_used = (alias for match in REPORT_REFERENCES.finditer(report)
                    for alias in report_aliases(match.group(1)))
    for alias in dict.fromkeys(aliases_used):
        if alias in lookup:
            refs[alias] = {k: lookup[alias][k] for k in
                           ('passage_id', 'docid', 'document_sha256', 'start', 'end')}
        else:
            warnings.append('unknown_report_reference:' + alias)
    return refs, warnings


def main_reports(cells: dict, rows: list[dict], counter, token_cap: int, *, directions=False) -> list[dict]:
    """Map local aliases via physical provenance, never by P-number equality."""
    if directions:
        eligible = [(key, cell) for key, cell in sorted(cells.items())
                    if cell.get('reader_mode') == 'researcher'
                    and cell.get('latest_research_report')
                    and cell.get('report_status') in {'complete', 'truncated', 'fallback'}
                    and cell.get('report_turn') == cell.get('turn')]
        # Allocate after mapping: NOT_SHOWN labels can be longer than local IDs.
        needs = {}
        for key, cell in eligible:
            full = main_reports({key: cell}, rows, counter, 10**9)
            if full:
                needs[key] = counter.text(full[0]['unverified_analysis'])
        shares = dict.fromkeys(needs, 0)
        remaining = max(0, token_cap)
        while remaining and needs:
            quota = max(1, remaining // len(needs))
            for key in list(needs):
                give = min(quota, needs[key], remaining)
                shares[key] += give
                remaining -= give
                needs[key] -= give
                if not needs[key]:
                    del needs[key]
        reports = []
        for key, cell in eligible:
            for report in main_reports({key: cell}, rows, counter, shares.get(key, 0)):
                reports.append({**report, 'objective': cell.get('focus', '')})
        return reports
    from .tokenization import prefix_chars
    physical = lambda r: (r['docid'], r['document_sha256'], r['start'], r['end'])
    shown = {physical(row): row['display_id'] for row in rows}
    reports = []
    remaining = token_cap
    for cell_id, cell in sorted(cells.items()):
        report = cell.get('latest_research_report', '')
        if (cell.get('reader_mode') != 'researcher' or not report or
                cell.get('report_status') not in {'complete', 'truncated', 'fallback'} or
                cell.get('report_turn') != cell.get('turn')):
            continue
        refs = cell.get('report_passage_refs', {})
        mapping = {}
        def replace(match):
            labels = []
            for alias in report_aliases(match.group(1)):
                ref = refs.get(alias)
                display = shown.get(physical(ref)) if ref else None
                mapping[alias] = display
                labels.append('[' + (display or f'NOT_SHOWN:{cell_id}/{alias}') + ']')
            return ', '.join(labels)
        mapped = REPORT_REFERENCES.sub(replace, report)
        if counter.text(mapped) > remaining:
            paragraphs = re.split(r'\n\s*\n', mapped)
            selected = ''
            for paragraph in paragraphs:
                trial = selected + ('\n\n' if selected else '') + paragraph
                if counter.text(trial) > remaining:
                    break
                selected = trial
            if not selected:
                selected = mapped[:prefix_chars(mapped, remaining, counter)]
                # Do not deliver half of a bracketed provenance label.
                if selected.rfind('[') > selected.rfind(']'):
                    selected = selected[:selected.rfind('[')]
            mapped = selected
        if not mapped.strip():
            continue
        remaining -= counter.text(mapped)
        reports.append({'cell_id': cell_id, 'turn': cell['report_turn'],
                        'status': cell['report_status'], 'unverified_analysis': mapped,
                        'reference_mapping': mapping})
    return reports


def parse_decision(text: str) -> tuple[Decision, list[str]]:
    obj, notes = output_object(text)
    allowed = set(Decision.model_fields) | {'missing_fact','evidence_ids','passage_ids'}
    extra = sorted(set(obj) - allowed)
    if extra: notes.append('ignored_metadata:' + ','.join(extra))
    return Decision.model_validate(obj), notes


def _spans(text: str, cap: int):
    start = 0
    while start < len(text):
        end = min(len(text), start + cap)
        if end < len(text):
            # Prefer a paragraph/sentence boundary, but never omit whitespace or
            # concatenate fragments. Spans exactly tile the input source.
            floor = start + cap // 2
            boundaries = [m.end() for m in re.finditer(r'\n\s*\n|[.!?。！？](?:\s+|$)', text[start:end])
                          if start + m.end() >= floor]
            if boundaries: end = start + boundaries[-1]
        yield start, end
        start = end


def registry(sources: dict, passage_chars: int = 1200) -> dict[str, dict]:
    out = {}
    for sid, source in sources.items():
        for lo, hi in _spans(source['text'], passage_chars):
            if not source['text'][lo:hi].strip(): continue
            ident = 'p_' + stable_hash([sid, lo, hi])[:24]
            out[ident] = {'passage_id': ident, 'source_id': sid,
                'docid': source['docid'], 'document_sha256': source['document_sha256'],
                'start': source['start'] + lo, 'end': source['start'] + hi,
                'title': source.get('title',''), 'text': source['text'][lo:hi]}
    return out


def aliases(passages: dict) -> dict[str, str]:
    return {key: f'P{i}' for i, key in enumerate(passages, 1)}


def apply_selection(reply: Selection, sources: dict, previous: dict, turn: int,
                    passage_chars: int = 1200) -> tuple[dict, list[str], int]:
    all_rows = registry(sources, passage_chars)
    lookup = {a: key for key, a in aliases(all_rows).items()}
    selected, errors, changes = deepcopy(previous), [], 0
    for a in reply.drop_passage_ids:
        key = lookup.get(a)
        if key is None or key not in selected:
            errors.append('unknown_or_unselected_drop:' + a)
        elif selected[key].get('active', True):
            selected[key]['active'] = False; changes += 1
    for a in reply.selected_passage_ids:
        key = lookup.get(a)
        if key is None:
            errors.append('unknown_passage_id:' + a); continue
        if key in selected: continue  # retracted IDs do not reactivate implicitly
        selected[key] = {**all_rows[key], 'active': True, 'selected_turn': turn}
        changes += 1
    return selected, errors, changes


def relevance(text: str, question: str) -> float:
    terms = set(re.findall(r'\w{4,}', question.casefold())) - {
        'which','what','person','that','this','with','from','were','their','following','between'}
    low = text.casefold()
    return sum(min(low.count(t), 3) for t in terms) / max(1, len(terms))


def final_pool(cells: dict, question: str, passage_chars: int = 1200, *, directions=False) -> list[dict]:
    """Selected passages + a document-diverse fallback, all original source text.

    Missing/invalid Reader selections never erase the raw data Main can inspect.
    Withdrawn selections are omitted, but other passages of that document remain.
    """
    if directions:
        pools = [final_pool({key: cell}, question, passage_chars)
                 for key, cell in sorted(cells.items())]
        ordered, physical_rows = [], {}
        for rank in range(max((len(pool) for pool in pools), default=0)):
            for pool in pools:
                if rank >= len(pool):
                    continue
                row = pool[rank]
                physical = (row['docid'], row['document_sha256'], row['start'], row['end'])
                if physical in physical_rows:
                    existing = physical_rows[physical]
                    if row['cell_id'] not in existing['cell_ids']:
                        existing['cell_ids'].append(row['cell_id'])
                    if row['selection'] == 'reader':
                        existing['selection'] = 'reader'
                else:
                    item = {**row, 'cell_ids': [row['cell_id']]}
                    ordered.append(item)
                    physical_rows[physical] = item
        return ordered
    preferred, fallback, seen = [], [], set()
    for cell_id, cell in sorted(cells.items()):
        selected = cell.get('selected_passages', {})
        by_doc = {}
        for key, row in registry(cell.get('sources', {}), passage_chars).items():
            physical = (row['docid'], row['document_sha256'], row['start'], row['end'])
            if physical in seen: continue
            if key in selected and not selected[key].get('active', True): continue
            seen.add(physical)
            item = {**row, 'cell_id': cell_id, 'pool_id': cell_id + ':' + key,
                    'selection': 'reader' if key in selected else 'source_fallback'}
            if key in selected: preferred.append(item)
            else: by_doc.setdefault(row['docid'], []).append(item)
        for group in by_doc.values():
            group.sort(key=lambda r: (-relevance(r['text'], question), r['start']))
        # Round robin across documents, rather than filling the pack with one bio.
        for rank in range(max((len(v) for v in by_doc.values()), default=0)):
            for group in by_doc.values():
                if rank < len(group): fallback.append(group[rank])
    # Interleave real-source fallback even when the Reader selected many items.
    ordered = []
    while preferred or fallback:
        ordered.extend(preferred[:2]); del preferred[:2]
        if fallback: ordered.append(fallback.pop(0))
    return ordered


def visible_pack(rows: list[dict]) -> list[dict]:
    return [{'passage_id': r.get('display_id', f'P{i}'), **{k: r[k] for k in ('docid','title','start','end','text')},
             **({'cell_ids': r['cell_ids']} if 'cell_ids' in r else {})}
            for i, r in enumerate(rows, 1)]


def commit_decision(decision: Decision, rows: list[dict]) -> dict:
    """Verify provenance only. Main + the external Judge check answer semantics.

    No lexical-substring test or Reader candidate/target label is a proof gate.
    Unknown, absent, or omitted citations still fail and must be repaired.
    """
    if decision.action != 'answer': raise ValueError('not_an_answer')
    if normalize_query(decision.exact_answer) in {'unknown','none','null','n/a','unable to determine','unresolved'}:
        raise ValueError('placeholder_answer')
    lookup = {row.get('display_id', f'P{i}'): row for i, row in enumerate(rows, 1)}
    if not decision.citations: raise ValueError('missing_passage_citations')
    unknown = [key for key in decision.citations if key not in lookup]
    if unknown: raise ValueError('unknown_or_omitted_passage_citations:' + ','.join(unknown))
    ids = list(dict.fromkeys(decision.citations))
    docs = list(dict.fromkeys(lookup[key]['docid'] for key in ids))
    return {**decision.model_dump(), 'evidence_ids': [], 'citation_docids': docs,
        'citation_aliases': {key: lookup[key]['docid'] for key in ids},
        'cited_passages': [{k: lookup[key][k] for k in
                          ('pool_id','docid','document_sha256','start','end')} for key in ids],
        'provenance_checked': True, 'semantic_correctness_verified': False}
