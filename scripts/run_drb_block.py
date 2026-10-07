"""Run one fresh ten-instance block against user-provided, already-running services."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deepresearchbench/harness/src'))


async def run(a):
    # Import only on execution, so --help works in an offline inspection environment.
    from tokenizers import Tokenizer
    from mtbench.config import load_config
    from mtbench.runner import run_experiment
    Tokenizer.from_file(str(Path(a.tokenizer).resolve()))  # Fail before the first request if the exact tokenizer is missing.
    os.environ['MTBENCH_TOKENIZER']=str(Path(a.tokenizer).resolve())
    os.environ['MTBENCH_EXTRACTOR_WORKERS']='16'
    os.environ['MTBENCH_FETCH_BATCH']='16'
    os.environ['MTBENCH_SEARCH_PAGE_BATCH']='4'
    out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    if a.arm.endswith('-snapkv'):
        os.environ.update(MTBENCH_PREFILL_GATE_FILE=str(out/'prefill-gate.json'),MTBENCH_PREFILL_GATE_TOKENS='456837',
                          MTBENCH_PREFILL_GATE_ROLE='researcher',MTBENCH_PREFILL_GATE_CHAIN_TOKENS='6000')
    else:
        for key in list(os.environ):
            if key.startswith('MTBENCH_PREFILL_GATE_'):os.environ.pop(key)
    config=load_config(a.config)
    expected='full_transcript' if a.arm.endswith('-dense') else 'chain_delta'
    if config.researcher_model.continuation_mode!=expected:
        raise ValueError(f'{a.arm} requires continuation_mode={expected}')
    # Separate processes match the original concurrency and file-lock gate behavior.
    procs=[]
    for i in range(a.start,a.start+10):
        argv=[sys.executable,str(Path(__file__).resolve()),'--config',a.config,'--arm',a.arm,
              '--output',str(out),'--tokenizer',a.tokenizer,'--instance',str(i)]
        procs.append(await asyncio.create_subprocess_exec(*argv))
    codes=await asyncio.gather(*(p.wait() for p in procs))
    return 0 if all(c==0 for c in codes) else 1


async def instance(a):
    from mtbench.config import load_config
    from mtbench.runner import run_experiment
    cfg=load_config(a.config)
    cfg.run.instance_id=a.instance
    run_dir, _ = await run_experiment(ROOT/'deepresearchbench/harness',cfg,run_id=str(Path(a.output).resolve()/f'i{a.instance:03d}'))
    status=json.loads((run_dir/'status.json').read_text(encoding='utf8'))
    if status.get('status')!='COMPLETE':
        raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--arm',required=True,choices=['dense-dense','omnikv-dense','dense-h2o','omnikv-h2o','dense-snapkv','omnikv-snapkv'])
    p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--start',type=int,default=1,choices=list(range(1,92,10)))
    p.add_argument('--instance',type=int,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.instance is not None:asyncio.run(instance(a));return
    raise SystemExit(asyncio.run(run(a)))


if __name__=='__main__':main()
