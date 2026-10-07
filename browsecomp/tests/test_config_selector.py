"""Development/formal separation, feasibility, freeze integrity and CLI integration."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from bcgraph.config_selector import digest, freeze_selection, load_selection, rank_profiles

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def profiles(tmp_path):
    shutil.copytree(ROOT / 'examples/selector', tmp_path / 'profiles')
    return tmp_path / 'profiles/profiles.synthetic.json'


def change(profiles, arm, *, measurement=None, config=None):
    p = json.loads(profiles.read_text())
    c = next(c for c in p['candidates'] if c['name'] == arm)
    if measurement:
        file = profiles.parent / c['evidence']
        data = json.loads(file.read_text())
        measurement(data)
        file.write_text(json.dumps(data))
        c['evidence_sha256'] = digest(file.read_bytes())
    if config:
        file = profiles.parent / c['config']
        data = yaml.safe_load(file.read_text())
        config(data)
        file.write_text(yaml.safe_dump(data))
        c['config_sha256'] = digest(file.read_bytes())
    profiles.write_text(json.dumps(p))


def verify(candidate='C', reason='recommendation_confirmed'):
    return {'checked': True, 'accepted_candidate': candidate, 'reason': reason, 'evidence': 'test audit'}


def test_minimum_feasible_makespan_not_a_hardcoded_mapping(profiles):
    rank = rank_profiles(profiles, {'held-out'})
    assert rank['recommended'] == 'C'
    assert rank['rejected']['F'] == ['quality']
    change(profiles, 'B', measurement=lambda x: x.update(makespan_seconds=1))
    assert rank_profiles(profiles, {'held-out'})['recommended'] == 'B'


@pytest.mark.parametrize('field,value,reason', [
    ('valid', False, 'incomplete_or_invalid'), ('finished_queries', 1, 'incomplete_or_invalid'),
    ('quality_pass', False, 'quality'), ('stability_pass', False, 'stability'),
    ('compatible', False, 'unsupported_mode'), ('max_query_concurrency', 1, 'capacity_or_concurrency'),
    ('supported_reader', ['vllm'], 'unsupported_mode')])
def test_feasibility_filters_faster_candidate(profiles, field, value, reason):
    change(profiles, 'C', measurement=lambda x: x.update({field: value}))
    rank = rank_profiles(profiles, {'held-out'})
    assert rank['recommended'] == 'D'
    assert reason in rank['rejected']['C']


def test_model_compatibility(profiles):
    change(profiles, 'C', config=lambda x: x['reader'].update(model='another-model'))
    assert rank_profiles(profiles, {'held-out'})['rejected']['C'] == ['model_mismatch']


@pytest.mark.parametrize('mutation', [
    lambda x: x.update(split='formal'),
    lambda x: x.update(query_ids=['held-out', 'synthetic-dev-2']),
    lambda x: x.update(query_ids=['synthetic-dev-1', 'synthetic-dev-1']),
    lambda x: x.update(query_ids=['another-cohort', 'synthetic-dev-2']),
    lambda x: x.update(makespan_seconds=float('nan')),
    lambda x: x['workload'].update(shape='unmeasured-shape'),
    lambda x: x.update(engine_revisions={})])
def test_rejects_leakage_or_invalid_measurement(profiles, mutation):
    change(profiles, 'C', measurement=mutation)
    with pytest.raises(ValueError):
        rank_profiles(profiles, {'held-out'})


def test_changed_evidence_and_missing_holdout_fail_closed(profiles):
    with pytest.raises(ValueError, match='held-out'):
        rank_profiles(profiles, set())
    (profiles.parent / 'A-evidence.json').write_text('{}')
    with pytest.raises(ValueError, match='changed since profiling'):
        rank_profiles(profiles, {'held-out'})


def test_frozen_config_is_independent_of_later_profiles(profiles, tmp_path):
    ranking = rank_profiles(profiles, {'held-out'})
    frozen = tmp_path / 'frozen.json'
    freeze_selection(ranking, verify(), frozen)
    change(profiles, 'B', measurement=lambda x: x.update(makespan_seconds=1))
    config = load_selection(frozen, allow_synthetic=True)
    assert config.main.method == 'vanilla' and config.reader.method == 'h2o'
    assert config.engine_manifest['configuration_selection']['actual_engine_modes'] is None
    with pytest.raises(FileExistsError):
        freeze_selection(ranking, verify(), frozen)
    with pytest.raises(ValueError, match='Synthetic'):
        load_selection(frozen)
    data = json.loads(frozen.read_text())
    data['selected']['config']['workflow']['query_concurrency'] = 1
    frozen.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash/schema'):
        load_selection(frozen, allow_synthetic=True)


def test_verifier_can_only_choose_valid_candidates_with_allowed_reason(profiles, tmp_path):
    rank = rank_profiles(profiles, {'held-out'})
    for v in [verify('F', 'reliability'), verify('B', 'prefer_formal_results'), {'checked': False}]:
        with pytest.raises(ValueError):
            freeze_selection(rank, v, tmp_path/'bad.json')
    freeze_selection(rank, verify('D', 'reliability'), tmp_path/'valid.json')
    assert load_selection(tmp_path/'valid.json', allow_synthetic=True).main.method == 'omnikv'


def test_cli_freeze_then_existing_run_dry_run(profiles, tmp_path):
    formal = tmp_path/'formal.ids'; formal.write_text('held-out\n')
    query = tmp_path/'query.jsonl'; query.write_text('{"query_id":"held-out","query":"A fixture question"}\n')
    frozen = tmp_path/'frozen.json'
    base = [sys.executable, '-m', 'bcgraph.cli']
    subprocess.run([*base, 'select-config', '--profiles', str(profiles), '--formal-ids', str(formal),
                    '--verification', str(profiles.parent/'verification.synthetic.json'), '--output', str(frozen)],
                   check=True, capture_output=True, text=True)
    cmd = [*base, 'run', '--selection', str(frozen), '--queries', str(query), '--output', str(tmp_path/'unused')]
    output = subprocess.run([*cmd, '--dry-run'], check=True, capture_output=True, text=True)
    assert json.loads(output.stdout)['config']['reader']['method'] == 'h2o'
    assert not (tmp_path/'unused').exists()
    rejected = subprocess.run(cmd, capture_output=True, text=True)
    assert rejected.returncode and 'Synthetic' in rejected.stderr

