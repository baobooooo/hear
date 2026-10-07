"""Reaggregate archived results, without GPUs, network calls or third-party packages."""
import ast
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'cache_coordination/analysis'


def main():
    # Reuse the original aggregation functions without importing matplotlib.
    tree = ast.parse((ANALYSIS / 'analyze_trace.py').read_text(encoding='utf-8'))
    funcs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name in ('q', 'ttft', 'summarize')], type_ignores=[])
    ns = {'defaultdict': defaultdict}
    exec(compile(funcs, 'archived_analyze_trace.py', 'exec'), ns)
    summary = {'scbench': [], 'mooncake': [], 'deepresearchbench': [], 'mismatches': []}
    expected_sc = {
        'baseline': [63.1, 74.2, 79.0, 481, 19.2],
        'kvaware': [4.6, 75.8, 93.9, 224, 87.6],
        'kvp-W40': [28.3, 51.1, 65.0, 299, 66.1],
        'kvp-W60': [4.5, 64.0, 80.4, 230, 84.7],
    }
    for tag, expected in expected_sc.items():
        run = json.loads((ANALYSIS / f'timeline/n60win60/{tag}.json').read_text())
        records = run['records']
        tt = [ns['ttft'](r) for r in records]
        later = [r for r in records if r['r'] > 0]
        values = [ns['q'](tt, .5), ns['q'](tt, .95), max(tt),
                  max(r['decode_end'] for r in records),
                  100 * sum(r['cached'] for r in later) / sum(r['prompt_len'] for r in later)]
        rounded = [round(v, 0 if i == 3 else 1) for i, v in enumerate(values)]
        summary['scbench'].append(dict(tag=tag, sessions=len({r['inst'] for r in records}),
                                        turns=len(records), values=values, paper=expected))
        if rounded != expected:
            summary['mismatches'].append(dict(benchmark='scbench', tag=tag, actual=rounded, paper=expected))
    expected_mc = {
        '0.5': [[.7,2.8,6.2,13.7,86.1,19.5],[.6,2.4,5.9,13.3,86.0,20.2],[.6,2.8,7.4,13.6,86.2,19.6],[.6,2.5,6.3,13.0,86.6,31.9],[.5,2.3,6.1,12.7,86.8,32.1]],
        '0.75': [[10.9,51.9,56.7,50.2,116.4,8.0],[8.7,42.9,49.6,42.6,119.2,10.1],[9.1,42.3,47.9,42.5,119.3,8.1],[10.5,39.0,45.1,43.2,119.5,16.1],[8.2,39.8,45.4,39.8,121.0,16.9]],
        '1.0': [[163.9,277.0,281.5,256.9,122.7,5.2],[86.8,209.9,220.9,156.1,128.3,10.0],[97.1,209.2,217.6,178.7,122.8,8.0],[93.0,199.2,207.8,171.8,124.6,12.7],[161.7,267.4,277.7,243.2,108.9,12.5]],
    }
    tags = ['baseline','kvaware-nh','W40','retainU','W40-retain']
    for frac, rows in expected_mc.items():
        for tag, expected in zip(tags, rows):
            run = json.loads((ANALYSIS / f'timeline/h100mc/mc-f{frac}-{tag}.json').read_text())
            s = ns['summarize'](run, 180.0)
            values = [s[k] for k in ['ttft_p50','ttft_p95','ttft_max','sess_svc_mean','thr_turns_min','hit_pct']]
            rounded = [round(v, 1) for v in values]
            summary['mooncake'].append(dict(load=frac, tag=tag, values=values, paper=expected))
            if rounded != expected:
                summary['mismatches'].append(dict(benchmark='mooncake', load=frac, tag=tag, actual=rounded, paper=expected))
    evidence = ROOT / 'deepresearchbench/evidence/metrics'
    def read_csv(name):
        with (evidence / name).open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))
    instances, qualities, arms = [read_csv(n) for n in ['per_instance.csv','quality.csv','per_arm.csv']]
    cohort = []
    expected_race = {'dense-dense': 40.65, 'omnikv-dense': 41.14, 'dense-h2o': 40.46,
                     'omnikv-h2o': 41.05, 'dense-snapkv': 40.01, 'omnikv-snapkv': 40.50}
    for arm in arms:
        tag = arm['arm']
        ok = [r for r in instances if r['arm'] == tag and r['status'] == 'COMPLETE']
        make = 0
        for lo in range(1, 101, 10):
            block = [r for r in ok if lo <= int(r['instance']) < lo + 10]
            make += max(float(r['finished_at']) for r in block) - min(float(r['started_at']) for r in block)
        qrows = [r for r in qualities if r['arm'] == tag]
        cohort.append({int(r['instance']) for r in qrows})
        result = dict(arm=tag, completed=len(ok), makespan_h=make/3600,
                      judged=len(qrows), race_percent=100*statistics.mean(float(r['overall_score']) for r in qrows))
        summary['deepresearchbench'].append(result)
        if round(result['race_percent'], 2) != expected_race[tag]:
            summary['mismatches'].append(dict(benchmark='deepresearchbench_quality', arm=tag,
                                               actual=result['race_percent'], paper=expected_race[tag]))
        if round(make/3600, 2) != float(arm['makespan_h']):
            summary['mismatches'].append(dict(benchmark='deepresearchbench', arm=tag, actual=round(make/3600, 2), paper=arm['makespan_h']))
    summary['quality_common_cohort'] = len(set.intersection(*cohort))
    summary['quality_rows'] = len(qualities)
    summary['instance_rows'] = len(instances)
    summary['notes'] = ['DRB makespan is the sum of retained successful block spans, not total historical retry wall time.',
                        'Cache table values are recomputed from per-turn records; DRB uses archived per-instance CSVs.',
                        'This verifies archived numbers, not GPU rerun reproducibility.']
    dest = ROOT / 'verification.json'
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['mismatches']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
