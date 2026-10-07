"""Empirical request costs. Unknown regions never become H2O speed claims."""
from __future__ import annotations

import asyncio
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .evidence import stable_hash


def quantile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError('Empty cost sample')
    x = (len(values) - 1) * q
    lo = int(x)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (x - lo)


class CostTable:
    def __init__(self, data):
        if data.get('version') != 1 or not data.get('samples'):
            raise ValueError('Invalid or empty Reader cost table')
        if set(data['train_query_ids']) & set(data['validation_query_ids']):
            raise ValueError('Cost table leaks validation queries')
        self.data = data
        self.groups = defaultdict(list)
        self.margin = data['gain_margin']
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError('Invalid calibrated gain margin')
        for row in data['samples']:
            if row['query_id'] not in data['train_query_ids']:
                raise ValueError('Cost sample outside training cohort')
            self.groups[(row['engine'], row['reader_mode'], row['cache_state'])].append(row)

    @classmethod
    def load(cls, path, digest):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Frozen Reader cost table hash mismatch')
        return cls(json.loads(raw))

    def output_estimate(self, mode, stage, cap, previous=None):
        values = self.data['output_estimates'].get(mode + ':' + stage)
        if not values:
            return None
        weight = values.get('previous_weight', 0) if previous is not None else 0
        return min(cap, (1 - weight) * values['p75'] + weight * (previous or 0))

    def predict(self, feature, engine, occupancy, *, exclude_query=None):
        candidates = []
        for row in self.groups[(engine, feature['reader_mode'], feature['cache_state'])]:
            if engine == 'h2o' and not row.get('server_chain_mode', True):
                continue
            if (row['engine'] != engine or row['reader_mode'] != feature['reader_mode']
                    or row['cache_state'] != feature['cache_state'] or row['query_id'] == exclude_query):
                continue
            distances = [abs(math.log2((feature[k] + offset) / (row[k] + offset)))
                         for k, offset in [('prompt_tokens', 1024), ('new_tokens', 1024),
                                           ('expected_output', 128)]]
            load_distance = abs(math.log2((occupancy + 1) / (row['occupancy'] + 1)))
            # Fixed support bounds prevent extrapolating c16 findings to c1 or short inputs.
            if max(distances) > .65 or load_distance > .65:
                continue
            candidates.append((sum(distances) + 2 * load_distance, row))
        chosen = [r for _, r in sorted(candidates, key=lambda pair: pair[0])[:24]]
        queries = {r['query_id'] for r in chosen}
        runs = {r['run'] for r in chosen}
        repeats = {r.get('service_repetition', r['run']) for r in chosen}
        if len(chosen) < 6 or len(queries) < 3 or len(runs) < 2 or len(repeats) < 2:
            return None
        if any(r.get('failed') for r in chosen):
            return {'supported': False, 'reason': 'observed_failure', 'n': len(chosen)}
        # Group by query so repeated requests from one chain do not create fake confidence.
        blocks = defaultdict(list)
        for row in chosen:
            blocks[row['query_id']].append(row['http_seconds'])
        means = [statistics.mean(v) for v in blocks.values()]
        return {'supported': True, 'mean': statistics.mean(means),
                'lower': quantile(means, .2), 'upper': quantile(means, .8),
                'protocol_error_rate': statistics.mean(r['protocol_error'] for r in chosen),
                'n': len(chosen), 'queries': len(queries), 'runs': len(runs),
                'service_repetitions': len(repeats),
                'sample_ids': [r['sample_id'] for r in chosen]}

    def select_batch(self, requests, loads, free_slots):
        n = len(requests)
        choices = []
        memo = {}
        # Same ready batch, no timer and no invented future requests to fill H2O.
        for k in range(n + 1):
            estimates = []
            for req in requests:
                dload = loads['dense_reader']['active'] + loads['dense_reader']['pending'] + n - k + (k > 0)
                hload = loads['reader']['active'] + loads['reader']['pending'] + max(1, k)
                pair = {}
                for name, engine, load in [('dense_reader', 'dense', dload), ('reader', 'h2o', hload)]:
                    cap = loads[name]['capacity']
                    occupancy = min(cap, max(1, load))
                    key = (engine, tuple(sorted(req[name].items())), occupancy)
                    if key not in memo:
                        memo[key] = self.predict(req[name], engine, occupancy)
                    prediction = memo[key]
                    if prediction and prediction.get('supported'):
                        prediction = dict(prediction)
                        # Approximate outstanding work in service-time waves; logged as an estimate.
                        factor = max(1, load / cap)
                        for field in ('mean', 'lower', 'upper'):
                            prediction[field] *= factor
                    pair[name] = prediction
                d, h = pair['dense_reader'], pair['reader']
                supported = bool(d and h and d.get('supported') and h.get('supported'))
                quality = supported and h['protocol_error_rate'] <= d['protocol_error_rate'] + .05
                gain = d['lower'] - h['upper'] * (1 + self.margin) if quality else -math.inf
                # On an existing chain, switching to Dense must itself clear the margin.
                retain = bool(req['h2o_warm'] and (not supported or
                              d['upper'] * (1 + self.margin) >= h['lower']))
                estimates.append({'pair': pair, 'gain': gain, 'retain': retain})
            mandatory = [i for i, e in enumerate(estimates) if e['retain']]
            eligible = sorted((i for i, e in enumerate(estimates) if not e['retain'] and e['gain'] > 0),
                              key=lambda i: estimates[i]['gain'], reverse=True)
            if len(mandatory) > k or len(mandatory) + len(eligible) < k:
                continue
            selected = set(mandatory + eligible[:k - len(mandatory)])
            required = sum(requests[i]['h2o_reserve_tokens'] for i in selected)
            if selected and (free_slots is None or required > free_slots):
                continue
            predicted = []
            for i, e in enumerate(estimates):
                p = e['pair']['reader' if i in selected else 'dense_reader']
                predicted.append(p['upper'] if p and p.get('supported') else 0)
            choices.append((max(predicted, default=0), k, selected, estimates, required))
        if not choices:
            return [('reader' if free_slots is None and req['h2o_warm'] else 'dense_reader',
                     {'reason': 'telemetry_unavailable_keep_committed_chain'
                      if free_slots is None and req['h2o_warm'] else 'capacity_or_no_supported_assignment',
                      'features': req}) for req in requests]
        _, _, selected, estimates, required = min(choices, key=lambda x: (x[0], x[1]))
        result = []
        for i, e in enumerate(estimates):
            backend = 'reader' if i in selected else 'dense_reader'
            reason = ('measured_gain' if e['gain'] > 0 else 'migration_not_beneficial') if i in selected else 'no_verified_h2o_gain'
            if i not in selected:
                d, h = e['pair']['dense_reader'], e['pair']['reader']
                if not d or not h:
                    reason = 'insufficient_cost_evidence'
                elif not d.get('supported') or not h.get('supported'):
                    reason = 'calibration_failure_veto'
                elif h['protocol_error_rate'] > d['protocol_error_rate'] + .05:
                    reason = 'protocol_quality_veto'
                elif e['gain'] > 0:
                    reason = 'capacity_or_batch_assignment'
            result.append((backend, {'reason': reason, 'predictions': e['pair'],
                'margin': self.margin, 'ready_batch': n, 'assigned_h2o': len(selected),
                'h2o_reserve_total': required, 'h2o_free_slots': free_slots,
                'features': requests[i]}))
        return result


class BenefitRouter:
    def __init__(self, config, clients, counters):
        self.config, self.clients, self.counters = config, clients, counters
        self.table = CostTable.load(config.workflow.routing_cost_table,
                                    config.workflow.routing_cost_table_sha256)
        if self.table.data['model'] != config.reader.model:
            raise ValueError('Reader cost table model mismatch')
        for key, value in self.table.data.get('endpoint_signature', {}).items():
            if getattr(config.reader, key) != value:
                raise ValueError('Reader cost table endpoint mismatch: ' + key)
        signature = self.table.data.get('h2o_engine_signature')
        if signature:
            launches = config.engine_manifest.get('routing_services', [])
            launch = next((x for x in launches if x.get('kind') == 'h2o'), None)
            if not launch or any(launch.get('engine_kwargs', {}).get(k) != v for k, v in signature.items()):
                raise ValueError('Reader cost table H2O engine settings mismatch')
        self.pending = []
        self.task = None
        self.reserved = {}

    @staticmethod
    def operation(state):
        return (state['scope'], state['cell_id'], state['turn'])

    def release(self, state):
        self.reserved.pop(self.operation(state), None)

    async def choose(self, state):
        future = asyncio.get_running_loop().create_future()
        self.pending.append((state, future))
        if self.task is None:
            self.task = asyncio.create_task(self._flush())
        try:
            return await future
        except BaseException:
            self.release(state)
            raise

    async def _flush(self):
        try:
            await asyncio.sleep(0)
            while self.pending:
                batch, self.pending = self.pending, []
                batch = [(s, f) for s, f in batch if not f.cancelled()]
                if not batch:
                    continue
                client = self.clients['reader']
                base = client.config.base_url.rstrip('/')
                if base.endswith('/v1'):
                    base = base[:-3]
                started = time.monotonic()
                try:
                    response = await client.http.get(base + '/v1/worker/load', timeout=2)
                    response.raise_for_status()
                    load = response.json()
                    cache = load.get('cache') or {}
                    free = cache.get('free_slots')
                    if free is None:
                        values = [v for key, v in cache.items() if key.startswith('free_slots') and isinstance(v, int)]
                        free = min(values) if values else None
                except Exception as exc:
                    # Missing telemetry fails closed for a new H2O assignment.
                    load, free = {'error': str(exc)}, None
                requests = []
                for state, _ in batch:
                    messages = [*state['raw_history'], state['packed']['message']]
                    handle = state.get('handle')
                    warm = bool(handle and handle.get('chain_id') and handle.get('endpoint') == client.url
                        and handle.get('engine_epoch') == client.config.engine_epoch
                        and handle.get('client_epoch') == client.client_epoch
                        and handle.get('history_hash') == stable_hash(messages[:-1]))
                    if warm:
                        try:
                            r = await client.http.post(base + '/v1/chain_cache/routing_match',
                                                       json={'chain_id': handle['chain_id']}, timeout=2)
                            r.raise_for_status()
                            status = r.json()
                            # Unknown schema cannot certify that GPU history is resident.
                            warm = bool(status.get('enabled') and status.get('present')
                                        and str(status.get('state', '')).upper() == 'IDLE')
                        except Exception:
                            warm = False
                    mode = self.config.workflow.reader_mode
                    stage = 'first' if state['turn'] == 0 else 'followup'
                    cap = state['requested_output_tokens']
                    previous = ((state.get('metrics') or [{}])[-1].get('usage') or {}).get('completion_tokens')
                    expected = self.table.output_estimate(mode, stage, cap, previous)
                    if expected is None:
                        expected = cap
                    item = {'h2o_warm': warm}
                    for name in ('reader', 'dense_reader'):
                        counter = self.counters[name]
                        prompt = counter.messages(messages)
                        cached = warm if name == 'reader' else state['turn'] > 0 and state.get('backend') == name
                        new = counter.text(state['packed']['message'].get('content') or '') if cached else prompt
                        item[name] = {'reader_mode': mode, 'cache_state': 'warm' if cached else 'cold',
                                      'prompt_tokens': prompt, 'new_tokens': new, 'expected_output': expected}
                    item['h2o_reserve_tokens'] = item['reader']['new_tokens'] + cap
                    requests.append(item)
                loads = {name: self.clients[name].gate.snapshot() for name in ('reader', 'dense_reader')}
                outstanding = sum(self.reserved.values())
                available = max(0, free - outstanding) if free is not None else None
                choices = self.table.select_batch(requests, loads, available)
                for (state, future), req, (backend, detail) in zip(batch, requests, choices):
                    detail.update(telemetry=load, outstanding_reserve_upper_bound=outstanding,
                                  decision_batch_ms=(time.monotonic() - started) * 1000)
                    if not future.cancelled():
                        if backend == 'reader':
                            self.reserved[self.operation(state)] = req['h2o_reserve_tokens']
                        future.set_result((backend, detail))
        except BaseException as exc:
            for _, future in [*locals().get('batch', []), *self.pending]:
                if not future.done():
                    future.set_exception(exc)
            self.pending = []
        finally:
            self.task = None
