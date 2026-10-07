"""Offline development ranking and a frozen, once-per-workflow role mapping.

This is the coarse configuration policy in HEAR Appendix A.1, not the
per-cell BenefitRouter or the document selector. No model is called here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
import yaml

from .config import AppConfig, StrictModel, _expand


class Workload(StrictModel):
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    objective: Literal['makespan'] = 'makespan'
    query_concurrency: int = Field(ge=1)
    # Profile matching is deliberately exact: no extrapolation to another shape.
    shape: str = Field(min_length=1)


class Candidate(StrictModel):
    name: str = Field(min_length=1)
    config: str
    config_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    evidence: str
    evidence_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ProfileSet(StrictModel):
    schema_version: Literal[1]
    split: Literal['development']
    synthetic: bool = False
    workload: Workload
    candidates: list[Candidate] = Field(min_length=1)


class Measurement(StrictModel):
    split: Literal['development']
    workload: Workload
    query_ids: list[str] = Field(min_length=1)
    expected_queries: int = Field(ge=1)
    finished_queries: int = Field(ge=0)
    valid: bool
    compatible: bool
    quality_pass: bool
    stability_pass: bool
    makespan_seconds: float = Field(gt=0, allow_inf_nan=False)
    supported_main: list[Literal['vllm', 'omnikv']]
    supported_reader: list[Literal['vllm', 'h2o', 'snapkv']]
    max_query_concurrency: int = Field(ge=1)
    engine_revisions: dict[str, str]
    engine_parameters: dict[str, dict[str, Any]]
    # A content hash links quality/stability decisions to an external audit.
    audit_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_digest(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode())


def modes(config: AppConfig) -> dict[str, str]:
    return {role: ('vllm' if endpoint.engine == 'vllm' else endpoint.method)
            for role, endpoint in [('main', config.main), ('reader', config.reader)]}


def checked_file(base: Path, name: str, expected: str) -> bytes:
    data = (base / name).read_bytes()
    if digest(data) != expected:
        raise ValueError(f'Content changed since profiling: {name}')
    return data


def rank_profiles(path: str | Path, formal_ids: set[str]) -> dict:
    """Fail closed on damaged provenance; filter valid but unsuitable candidates."""
    if not formal_ids:
        raise ValueError('A nonempty held-out formal cohort is required')
    path = Path(path)
    source = path.read_bytes()
    profiles = ProfileSet.model_validate_json(source)
    names = [c.name for c in profiles.candidates]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate candidate names')
    accepted, rejected = [], {}
    cohort = None
    for c in profiles.candidates:
        cfg_bytes = checked_file(path.parent, c.config, c.config_sha256)
        ev_bytes = checked_file(path.parent, c.evidence, c.evidence_sha256)
        evidence = Measurement.model_validate_json(ev_bytes)
        raw = yaml.safe_load(cfg_bytes)
        config = AppConfig.model_validate(_expand(raw))
        ids = set(evidence.query_ids)
        if len(ids) != len(evidence.query_ids) or ids & formal_ids:
            raise ValueError(f'{c.name}: duplicate IDs or development/formal cohort overlap')
        if cohort is None:
            cohort = ids
        elif ids != cohort:
            raise ValueError('Development candidates must use the same question cohort')
        if evidence.workload != profiles.workload:
            raise ValueError(f'{c.name}: workload/model/shape differs from the profile set')
        if any(not evidence.engine_revisions.get(r) for r in ['main', 'reader']):
            raise ValueError(f'{c.name}: both engine revisions are required')
        if any(r not in evidence.engine_parameters for r in ['main', 'reader']):
            raise ValueError(f'{c.name}: both engine parameter sets are required')
        mapping = modes(config)
        reasons = []
        if not (evidence.valid and evidence.finished_queries == evidence.expected_queries == len(ids)):
            reasons.append('incomplete_or_invalid')
        if not evidence.quality_pass:
            reasons.append('quality')
        if not evidence.stability_pass:
            reasons.append('stability')
        if not evidence.compatible or mapping['main'] not in evidence.supported_main or mapping['reader'] not in evidence.supported_reader:
            reasons.append('unsupported_mode')
        if any(x.model != profiles.workload.model for x in [config.main, config.reader]):
            reasons.append('model_mismatch')
        if config.workflow.query_concurrency != profiles.workload.query_concurrency or evidence.max_query_concurrency < profiles.workload.query_concurrency:
            reasons.append('capacity_or_concurrency')
        if config.workflow.reader_routing != 'off' or config.workflow.cold_dense_below_tokens or config.dense_reader or config.reader_replicas:
            reasons.append('requires_fixed_two_role_mapping')
        if reasons:
            rejected[c.name] = reasons
        else:
            accepted.append({'name': c.name, 'makespan_seconds': evidence.makespan_seconds,
                             'mapping': mapping, 'config': raw,
                             'engine_revisions': evidence.engine_revisions,
                             'engine_parameters': evidence.engine_parameters,
                             'evidence_sha256': c.evidence_sha256,
                             'audit_sha256': evidence.audit_sha256})
    accepted.sort(key=lambda row: (row['makespan_seconds'], row['name']))
    if not accepted:
        raise ValueError(f'No feasible development candidate: {rejected}')
    return {'schema_version': 1, 'synthetic': profiles.synthetic,
            'workload': profiles.workload.model_dump(), 'profile_sha256': digest(source),
            'formal_cohort_sha256': object_digest(sorted(formal_ids)),
            'recommended': accepted[0]['name'], 'ranked': accepted, 'rejected': rejected}


def freeze_selection(ranking: dict, verification: dict, output: str | Path) -> dict:
    """A verifier confirms the recommendation or gives a restricted exception."""
    if verification.get('checked') is not True or not verification.get('evidence'):
        raise ValueError('Verification must confirm checks and cite log/audit evidence')
    accepted = verification.get('accepted_candidate')
    selected = next((x for x in ranking['ranked'] if x['name'] == accepted), None)
    if selected is None:
        raise ValueError('Verifier may only select a feasible development candidate')
    allowed = {'invalid_measurement', 'unsupported_execution_path', 'feasibility', 'reliability'}
    if accepted != ranking['recommended'] and verification.get('reason') not in allowed:
        raise ValueError('Override requires a measurement, execution, feasibility or reliability reason')
    frozen = {'schema_version': 1, 'policy': 'development-makespan-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'synthetic': ranking['synthetic'], 'workload': ranking['workload'],
              'profile_sha256': ranking['profile_sha256'],
              'formal_cohort_sha256': ranking['formal_cohort_sha256'],
              'recommended': ranking['recommended'], 'selected': selected,
              'verification': verification}
    frozen['sha256'] = object_digest(frozen)
    # A freeze is append-only. Reranking cannot overwrite a prior accepted mapping.
    with Path(output).open('x', encoding='utf-8') as f:
        json.dump(frozen, f, indent=2, allow_nan=False)
        f.write('\n')
    return frozen


def load_selection(path: str | Path, *, allow_synthetic: bool = False) -> AppConfig:
    """Resolve endpoints once; normal ChatClient dispatch and retries remain intact."""
    frozen = json.loads(Path(path).read_text())
    expected = frozen.pop('sha256', None)
    if expected != object_digest(frozen) or frozen.get('schema_version') != 1:
        raise ValueError('Frozen selection hash/schema mismatch')
    if frozen.get('policy') != 'development-makespan-v1':
        raise ValueError('Unsupported selector policy')
    if frozen['synthetic'] and not allow_synthetic:
        raise ValueError('Synthetic demo selections cannot start a real run')
    selection = frozen['selected']
    config = AppConfig.model_validate(_expand(selection['config']))
    if modes(config) != selection['mapping']:
        raise ValueError('Frozen role mapping does not match service configuration')
    workload = Workload.model_validate(frozen['workload'])
    if (any(ep.model != workload.model for ep in [config.main, config.reader])
            or config.workflow.query_concurrency != workload.query_concurrency):
        raise ValueError('Runtime model/concurrency differs from the profiled workload')
    config.engine_manifest['configuration_selection'] = {
        'sha256': expected, 'candidate': selection['name'],
        'requested_modes': selection['mapping'], 'resolved_service_modes': modes(config),
        'engine_revisions': selection['engine_revisions'],
        'engine_parameters': selection['engine_parameters'],
        'actual_engine_modes': None,
        'note': 'Service binding is configuration, not remote proof of execution mode.'}
    return config
