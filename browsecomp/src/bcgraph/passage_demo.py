"""Synthetic real-node/real-HTTP-client fixture for the active passage protocol.
No GPU timings or benchmark accuracy can be inferred from this fixture.
"""
from copy import deepcopy
import json
import httpx
from .config import AppConfig, EndpointConfig, WorkflowConfig
from .passage_runtime import PassageRuntime
from .retrieval import FixtureRetriever, RecordedRetriever
from .transport import ChatClient
from .tokenization import Utf8Counter
from .demo import CORPUS, QUESTION

class PassageHTTP:
    def __init__(self, chain=True, evict_once=False, two_cells=False):
        self.chain,self.evict_once,self.two_cells=chain,evict_once,two_cells
        self.evicted=False; self.chains={}; self.requests=[]; self.sequence=0
        self.mutate=None

    def generate(self, logical, is_reader):
        content=logical[-1]['content']
        if content.startswith('TASK: SELECT_DOCUMENTS'):
            if not content.split('\n', 1)[1].startswith('{'):
                content = next(m['content'] for m in reversed(logical[:-1])
                               if m['role']=='user' and m['content'].startswith('TASK: SELECT_DOCUMENTS\n{'))
            data=json.loads(content.split('\n',1)[1])
            return {'documents':[{'docid':r['docid']} for r in data['candidate_documents_NOT_EVIDENCE'][:data['max_documents']]], 'next_queries':[]}
        if content.startswith('TASK: COORDINATE_RESEARCH'):
            data=json.loads(content.split('\n',1)[1])
            return {'action':'continue','tasks':[{'cell_id':c['cell_id'],
                'instruction':'Check the instrument using the shared observation.',
                'next_queries':['Eira Stone instrument '+str(data['sync_round'])],
                'evidence_ids':[r['passage_id'] for r in data['original_passages'][:1]]}
                for c in data['eligible_directions']]}
        if content.startswith('TASK: PLAN_RESEARCH'):
            cells=[{'focus':'Find the director and the instrument',
                    'initial_queries':['Iris Observatory director 2007']}]
            if self.two_cells: cells.append(deepcopy(cells[0]))
            return {'cells':cells}
        if is_reader:
            if content.startswith('TASK: REPAIR_RESEARCH_DELIVERY'):
                content = next(m['content'] for m in reversed(logical)
                               if m['role']=='user' and m['content'].startswith('{'))
            data=json.loads(content)
            rows=[p for d in data['source_documents'] for p in d['passages']]
            has_answer=any('Aurora spectrograph' in p['text'] for p in rows)
            return {'selected_passage_ids':[p['passage_id'] for p in rows],
                    'candidate_answer':'Aurora spectrograph' if has_answer else '',
                    'next_queries':[] if has_answer else ['Eira Stone designed instrument'],
                    'missing_fact':'' if has_answer else 'instrument name'}
        # A correction references the SAME previously displayed original pack.
        original=next(m['content'] for m in reversed(logical) if m.get('role')=='user'
                      and m['content'].startswith('TASK: DECIDE_FROM_ORIGINAL_PASSAGES'))
        data=json.loads(original[original.index('{"question"'):])
        rows=data['original_passages']
        answer=any('Aurora spectrograph' in p['text'] for p in rows)
        return {'action':'answer' if answer else 'unresolved',
                'exact_answer':'Aurora spectrograph' if answer else '',
                'citations':[p['passage_id'] for p in rows],
                'explanation':'The annual report identifies Eira Stone; the design archive names the instrument.' if answer else 'The instrument is not in these passages.'}

    def __call__(self, request):
        if request.method=='GET': return httpx.Response(200,json={'data':[{'id':'demo-model'}]})
        body=json.loads(request.content); self.requests.append(deepcopy(body))
        messages=body['messages']; is_reader=request.url.host=='mock-reader'
        ident=body.get('chain_id'); reused=0
        if ident:
            if self.evict_once and not self.evicted:
                self.evicted=True; self.chains.pop(ident,None)
                return httpx.Response(410,json={'detail':{'code':'chain_gone','message':'synthetic eviction'}})
            old=self.chains.get(ident)
            if old is None:
                return httpx.Response(404,json={'detail':{'code':'chain_not_found'}})
            if body.get('chain_append_start')==1:
                if messages[0]!=old[-1]:
                    return httpx.Response(409,json={'detail':{'code':'chain_prefix_mismatch'}})
                logical=[*old,*messages[1:]]
            else:
                if messages[:-1]!=old:
                    return httpx.Response(409,json={'detail':{'code':'chain_prefix_mismatch'}})
                logical=messages
            reused=len(json.dumps(old))//4
        else: logical=messages
        obj=self.generate(logical,is_reader)
        finish='stop'
        if self.mutate:
            changed=self.mutate(obj,logical,is_reader,body)
            if isinstance(changed,tuple): obj,finish=changed
            elif changed is not None: obj=changed
        content=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=1)
        assistant={'role':'assistant','content':content}
        prompt=len(json.dumps(logical))//4
        response={'choices':[{'message':assistant,'finish_reason':finish}],
                  'usage':{'prompt_tokens':prompt,'completion_tokens':len(content)//4,
                           'prompt_tokens_details':{'cached_tokens':reused}}}
        if is_reader and self.chain:
            status='resumed' if ident else 'created'
            if not ident:
                self.sequence+=1; ident=f'synthetic-chain-{self.sequence}'
            self.chains[ident]=[*deepcopy(logical),deepcopy(assistant)]
            response.update(chain_id=ident,chain_status=status)
            response['usage']['reused_tokens']=reused
        return httpx.Response(200,json=response)


def make_demo_runtime(store, *, chain=True, evict_once=False, two_cells=False):
    main=EndpointConfig(base_url='http://mock-main/v1',model='demo-model')
    reader=EndpointConfig(base_url='http://mock-reader/v1',model='demo-model',
         engine='sparse-vllm' if chain else 'vllm', method='h2o' if chain else 'vanilla',
         cache='chain' if chain else 'prefix')
    config=AppConfig(main=main,reader=reader,workflow=WorkflowConfig(
        research_protocol='passages-v2',answer_policy='best_effort',
        allow_approximate_tokenizer=True,max_cells_per_query=2 if two_cells else 1))
    backend=PassageHTTP(chain,evict_once,two_cells)
    http=httpx.AsyncClient(transport=httpx.MockTransport(backend))
    clients={n:ChatClient(spec,store,http=http) for n,spec in [('main',main),('reader',reader)]}
    retrieval=RecordedRetriever(FixtureRetriever(CORPUS),store)
    return PassageRuntime(config,store,retrieval,clients,
                          {'main':Utf8Counter(),'reader':Utf8Counter()}),backend,http
