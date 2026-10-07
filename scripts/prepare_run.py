"""Generate explicit Linux commands and configurations; never start or stop a process."""
import argparse
import json
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['dense-dense', 'omnikv-dense', 'dense-h2o', 'omnikv-h2o', 'dense-snapkv', 'omnikv-snapkv']


def command(argv, env, log):
    args = ['nohup', 'env'] + [f'{k}={v}' for k, v in env.items()] + argv
    return shlex.join([str(x) for x in args]) + ' > ' + shlex.quote(str(log)) + ' 2>&1 < /dev/null &'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('benchmark', choices=['scbench', 'mooncake', 'deepresearchbench'])
    p.add_argument('--model', required=True, help='Local model weight directory')
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--researcher-gpu', type=int)
    p.add_argument('--python', default='python', help='Python from the appropriate activated Conda environment')
    p.add_argument('--sparse-python', default='python', help='SparseEngine environment Python for DRB')
    p.add_argument('--client-python', default='python')
    p.add_argument('--policy', choices=['fcfs', 'cache', 'guard40', 'guard60', 'session', 'combined'], default='fcfs')
    p.add_argument('--arm', choices=ARMS, default='omnikv-h2o')
    p.add_argument('--load', choices=['0.5', '0.75', '1.0'], default='0.75')
    p.add_argument('--port', type=int, default=19081)
    p.add_argument('--events-port', type=int, default=5557)
    p.add_argument('--researcher-port', type=int, default=23908)
    p.add_argument('--keys-file', help='Serper key file, not copied into this repository')
    p.add_argument('--output', required=True, help='New output directory; existing directories are rejected')
    a = p.parse_args()
    if a.benchmark == 'scbench' and a.policy in ['session','combined']:
        p.error('The SCBench paper table has only fcfs, cache, guard40, guard60.')
    if a.benchmark == 'mooncake' and a.policy == 'guard60':
        p.error('The Mooncake paper table uses Guard-40.')
    if a.benchmark == 'deepresearchbench':
        if a.researcher_gpu is None or a.researcher_gpu == a.gpu or not a.keys_file:
            p.error('DRB requires distinct role GPUs and --keys-file.')
        if a.port == a.researcher_port:
            p.error('Role ports must differ.')
    out = Path(a.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = str(Path(a.model).expanduser().resolve())
    commands = []
    if a.benchmark != 'deepresearchbench':
        server = ROOT / 'cache_coordination' / ('server_pro6000' if a.benchmark == 'scbench' else 'server_h100')
        guard = a.policy in ['guard40', 'guard60', 'combined']
        retain = a.policy in ['session', 'combined']
        env = dict(CUDA_VISIBLE_DEVICES=str(a.gpu), VLLM_SERVER_DEV_MODE='1',
                   PYTHONPATH=str(server/'engine_patch'), KVP_PROTECT=str(int(guard)),
                   KVP_KEEPALIVE=str(int(guard or retain)), KVP_FEEDBACK=str(int(guard or retain)),
                   KVP_RETAIN=str(int(retain)), KVP_RETAIN_TAU='400', KVP_RETAIN_MAX_WAIT='1000000000')
        events = dict(enable_kv_cache_events=True, publisher='zmq', endpoint=f'tcp://*:{a.events_port}',
                      topic='', buffer_steps=20000, hwm=200000)
        transfer = dict(kv_connector='OffloadingConnector', kv_role='kv_both', engine_id=out.name,
                        kv_connector_extra_config=dict(spec_name='CPUOffloadingSpec', cpu_bytes_to_use=103079215104,
                        eviction_policy='lru', offload_prompt_only=True, self_describing_kv_events=True, blocks_per_chunk=16))
        argv = [a.python,'-m','vllm.entrypoints.cli.main','serve',model,'--served-model-name','Qwen3-8B',
                '--host','127.0.0.1','--port',str(a.port),'--tensor-parallel-size','1','--dtype','bfloat16',
                '--max-model-len','40960','--block-size','16','--gpu-memory-utilization',
                '0.92' if a.benchmark=='scbench' else '0.90','--max-num-seqs','40',
                '--enable-prefix-caching','--enable-prompt-tokens-details','--enable-per-request-metrics',
                '--generation-config','vllm','--scheduling-policy','priority','--kv-events-config',json.dumps(events),
                '--kv-transfer-config',json.dumps(transfer)]
        commands.append(dict(step='start isolated engine on an idle GPU', command=command(argv,env,out/'engine.log')))
        inp = ROOT/'cache_coordination/inputs_pro6000'
        group = 'kvprotect' if guard else ('kvaware' if a.policy=='cache' else 'baseline')
        client = [a.client_python,str(server/'harness/run_timeline.py'),'--group',group,
                  '--instances',str(inp/'data/instances_320.json'),'--traj',str(inp/'trajectories/trajectories_320.json'),
                  '--seed','2026','--gpu',str(a.gpu),'--base',f'http://127.0.0.1:{a.port}/v1',
                  '--events',f'tcp://127.0.0.1:{a.events_port}','--out',str(out/'result.json')]
        if a.benchmark=='scbench':
            client += ['--n','60','--arrival-dist','poisson','--arrival-window','60']
        else:
            client += ['--n','320','--workload',str(ROOT/f'cache_coordination/inputs_h100/workloads/mc-f{a.load}-s600-d900.json')]
            if a.policy=='cache':client += ['--max-hold','0']
        if guard:client += ['--protect-s','60' if a.policy=='guard60' else '40']
        if retain:client += ['--p-return-table','{"1":0.31,"2":0.42,"3":0.55,"4":0.56,"5":0.81}']
        commands.append(dict(step='after /health succeeds, start replay against this dedicated fresh engine',
                             command=command(client,{},out/'client.log')))
    else:
        for role, mode, gpu, port in [('main',a.arm.split('-')[0],a.gpu,a.port),
                                      ('researcher',a.arm.split('-')[1],a.researcher_gpu,a.researcher_port)]:
            env = dict(CUDA_VISIBLE_DEVICES=str(gpu))
            if mode=='dense':
                argv=[a.python,'-m','vllm.entrypoints.openai.api_server','--model',model,'--served-model-name','GLM-4.7-Flash',
                      '--host','127.0.0.1','--port',str(port),'--dtype','bfloat16','--max-model-len','200000',
                      '--gpu-memory-utilization','0.90','--max-num-seqs','10','--max-num-batched-tokens','32768',
                      '--enable-prefix-caching','--reasoning-parser','glm47','--enable-auto-tool-choice','--tool-call-parser','glm47']
            else:
                env.update(PYTHONPATH=str(ROOT/'deepresearchbench/sparseengine/src'),SPARSEENGINE_MASTER_PORT=str(port+8000))
                argv=[a.sparse_python,'-m','sparseengine.entrypoints.openai.api_server','--model',model,
                      '--served-model-name','GLM-4.7-Flash','--host','127.0.0.1','--port',str(port),
                      '--engine-kwargs',str(ROOT/f'configs/engines/{a.arm}/{role}.json'),'--response-parser','glm47']
            commands.append(dict(step=f'start fresh {role} service on its idle GPU',command=command(argv,env,out/f'{role}.log')))
        template='long-context.toml' if a.arm.endswith('-dense') else 'long-context-chain.toml'
        text=(ROOT/'configs/deepresearchbench'/template).read_text(encoding='utf8')
        text=text.replace('http://127.0.0.1:23907/v1',f'http://127.0.0.1:{a.port}/v1')
        text=text.replace('http://127.0.0.1:23908/v1',f'http://127.0.0.1:{a.researcher_port}/v1')
        text=text.replace('"secrets/serper.keys"',json.dumps(str(Path(a.keys_file).expanduser().resolve())))
        (out/'workflow.toml').write_text(text,encoding='utf8')
        argv=[a.client_python,str(ROOT/'scripts/run_drb_block.py'),'--config',str(out/'workflow.toml'),
              '--arm',a.arm,'--start','1','--output',str(out/'block-001'), '--tokenizer',str(Path(model)/'tokenizer.json')]
        commands.append(dict(step='after both services are healthy and sparse kernels verified, run one 10-instance block',
                             command=command(argv,{},out/'block-001.log')))
    payload={'benchmark':a.benchmark,'commands':commands,'note':'Commands are generated, not executed. Use fresh services for each arm/block. No automatic stop/kill or GPU reservation.'}
    (out/'commands.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    main()
