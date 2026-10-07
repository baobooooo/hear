"""Archived outputs test parsing, NOT new prompts or new benchmark accuracy."""
import json
from pathlib import Path
import pytest
from bcgraph.passages import parse_decision,output_object
from bcgraph.passage_demo import make_demo_runtime,QUESTION

DATA=json.loads((Path(__file__).parent/'fixtures/glm10_contracts.json').read_text())

@pytest.mark.parametrize('row',DATA['decisions'],ids=lambda r:r['arm']+'-'+str(r['query_id']))
def test_every_archived_final_json_parses_without_optional_metadata_failure(row):
    d,notes=parse_decision(row['content'])
    assert d.action in {'answer','research','unresolved'}
    # Citation correctness is NOT inferred here; scoped validation/repair has
    # separate tests, and no old result is relabeled completed by this replay.

@pytest.mark.parametrize('row',DATA['plans'],ids=lambda r:r['arm']+'-'+str(r['query_id']))
async def test_archived_plan_payloads_do_not_require_redundant_constraint_metadata(store,row):
    rt,b,http=make_demo_runtime(store)
    b.mutate=lambda obj,m,r,body: row['content'] if not r else None
    try:
        state=await rt.plan({'query_id':'frozen','question':QUESTION,'scope':'frozen','attempt':1})
        assert state['jobs'] and all(j['initial_queries'] for j in state['jobs'])
        assert not any(e.startswith('planner_format_fallback') for e in state['errors'])
    finally: await rt.close(); await http.aclose()
