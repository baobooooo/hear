import asyncio
import json
from copy import deepcopy
import httpx
import pytest
from bcgraph.config import EndpointConfig
from bcgraph.demo import DemoHTTP
from bcgraph.evidence import stable_hash
from bcgraph.storage import JournalConflict
from bcgraph.transport import (ChatClient, observed_usage, ModelHTTPError,
                               AmbiguousCommitError, CacheProtocolError)

MESSAGES=[{'role':'system','content':'s'},{'role':'user','content':'u'}]
ASSISTANT={'role':'assistant','content':'  {"ok": true}\n','reasoning_content':'raw reasoning'}

def response(chain=True):
    value={'choices':[{'message':deepcopy(ASSISTANT),'finish_reason':'stop'}],
           'usage':{'prompt_tokens':200,'completion_tokens':10,'prompt_tokens_details':{'cached_tokens':120}}}
    if chain: value.update(chain_id='chain1',chain_status='created')
    return value

def config(chain=True, **kw):
    return EndpointConfig(base_url='http://mock/v1',engine='sparse-vllm' if chain else 'vllm',
                          method='h2o' if chain else 'vanilla',cache='chain' if chain else 'prefix',**kw)

def test_missing_counters_remain_unknown():
    value=observed_usage({'prompt_tokens':100,'completion_tokens':12})
    assert value['reused_tokens'] is None and value['prefilled_tokens'] is None

@pytest.mark.parametrize('raw,expected', [({'prompt_tokens':100,'reused_tokens':20},80),
    ({'prompt_tokens':100,'prompt_tokens_details':{'cached_tokens':30}},70),
    ({'prompt_tokens':100,'prefilled_tokens':17},17)])
def test_cache_usage_sources(raw,expected): assert observed_usage(raw)['prefilled_tokens']==expected

def test_inconsistent_counter_never_becomes_negative_prefill():
    usage=observed_usage({'prompt_tokens':10,'reused_tokens':50})
    assert usage['prefilled_tokens'] is None and usage['warnings']

@pytest.mark.parametrize('v',[-1,True,'40'])
def test_invalid_usage_counter(v): assert observed_usage({'reused_tokens':v})['reused_tokens'] is None

@pytest.mark.asyncio
async def test_raw_assistant_delta_and_exact_journal_replay(store):
    calls=[]
    def backend(request): calls.append(json.loads(request.content)); return httpx.Response(200,json=response())
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(),store,http=http)
        first=await client.complete(MESSAGES,32,operation_id='a',writer_key='cell')
        assert first['assistant']==ASSISTANT
        messages=[*MESSAGES,first['assistant'],{'role':'user','content':'next'}]
        second=await client.complete(messages,32,operation_id='b',writer_key='cell',handle=first['handle'])
        assert calls[1]['messages']==messages[-2:] and calls[1]['chain_append_start']==1
        replay=await client.complete(messages,32,operation_id='b',writer_key='cell',handle=first['handle'])
        assert replay['journal_replay'] and len(calls)==2
        with pytest.raises(JournalConflict):
            await client.complete(messages,33,operation_id='b',writer_key='cell')

@pytest.mark.asyncio
async def test_tampered_raw_history_is_rejected_locally(store):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=response()))) as http:
        client=ChatClient(config(),store,http=http)
        first=await client.complete(MESSAGES,32,operation_id='a',writer_key='cell')
        messages=[*MESSAGES,{**first['assistant'],'content':'rewritten'}, {'role':'user','content':'next'}]
        with pytest.raises(CacheProtocolError): client.build_payload(messages,32,first['handle'])

@pytest.mark.asyncio
async def test_new_process_handle_is_cold_not_assumed_resident(store):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=response()))) as http:
        old=ChatClient(config(),store,http=http); new=ChatClient(config(),store,http=http)
        first=await old.complete(MESSAGES,32,operation_id='a',writer_key='cell')
        body,mode=new.build_payload([*MESSAGES,first['assistant'],{'role':'user','content':'next'}],32,first['handle'])
        assert 'chain_id' not in body and mode=='cold_stale_handle'

@pytest.mark.asyncio
@pytest.mark.parametrize('code,status,recover', [('chain_gone',410,True),('chain_not_found',404,True),
                                               ('not_found',404,False),('chain_busy',409,False),('chain_prefix_mismatch',409,False)])
async def test_only_specific_chain_loss_is_cold_recoverable(store,code,status,recover):
    calls=[]
    def backend(request):
        body=json.loads(request.content); calls.append(body)
        if len(calls)==2: return httpx.Response(status,json={'error':{'code':code,'message':'test'}})
        return httpx.Response(200,json=response())
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(),store,http=http)
        first=await client.complete(MESSAGES,32,operation_id='a',writer_key='cell')
        messages=[*MESSAGES,first['assistant'],{'role':'user','content':'next'}]
        if recover:
            out=await client.complete(messages,32,operation_id='b',writer_key='cell',handle=first['handle'])
            assert out['recoveries']==1 and len(calls)==3 and 'chain_id' not in calls[-1]
            assert calls[-1]['messages']==messages
        else:
            with pytest.raises(ModelHTTPError):
                await client.complete(messages,32,operation_id='b',writer_key='cell',handle=first['handle'])
            assert len(calls)==2

@pytest.mark.asyncio
async def test_timeout_is_not_blindly_retried(store):
    calls=[]
    def backend(request): calls.append(request); raise httpx.ReadTimeout('response lost',request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(),store,http=http)
        for _ in range(2):
            with pytest.raises(AmbiguousCommitError):
                await client.complete(MESSAGES,32,operation_id='a',writer_key='cell')
        assert len(calls)==1 and store.get('a')['status']=='ambiguous'
        assert store.get('request-input:a')['value']['messages'] == MESSAGES

@pytest.mark.asyncio
async def test_stateless_prefix_read_error_is_safely_retried(store):
    calls=[]
    def backend(request):
        calls.append(request)
        if len(calls)==1:
            raise httpx.ReadError('response lost',request=request)
        return httpx.Response(200,json=response(False))
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(False,max_safe_retries=1),store,http=http)
        result=await client.complete(MESSAGES,32,operation_id='prefix',writer_key='main')
        assert len(calls)==2 and result['attempts']==2
        assert store.get('prefix')['status']=='success'
        attempts=[json.loads(line) for line in (store.meta/'events.jsonl').read_text().splitlines()
                  if json.loads(line).get('kind')=='model_attempt']
        assert attempts[0]['safe_replay'] is True and attempts[0]['usage_unknown'] is True

@pytest.mark.asyncio
async def test_chain_read_error_remains_ambiguous_without_retry(store):
    calls=[]
    def backend(request):
        calls.append(request)
        raise httpx.ReadError('response lost',request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(True,max_safe_retries=2),store,http=http)
        with pytest.raises(AmbiguousCommitError):
            await client.complete(MESSAGES,32,operation_id='chain',writer_key='cell')
        assert len(calls)==1 and store.get('chain')['status']=='ambiguous'

@pytest.mark.asyncio
@pytest.mark.parametrize('chain',[True,False])
async def test_model_requests_disable_http_keepalive(store,chain):
    seen=[]
    def backend(request):
        seen.append(request)
        return httpx.Response(200,json=response(True))
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(chain),store,http=http)
        await client.complete(MESSAGES,32,operation_id='chain-close',writer_key='cell')
    assert seen[0].headers['connection'] == 'close'


@pytest.mark.asyncio
async def test_release_chain_uses_control_route_and_is_idempotent_for_absent_chain(store):
    seen=[]
    def backend(request):
        seen.append(request)
        if json.loads(request.content)['chain_id'] == 'gone':
            return httpx.Response(410, json={'error': {'code': 'chain_gone', 'message': 'gone'}})
        return httpx.Response(200, json={'status': 'invalidated'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(),store,http=http)
        handle={'chain_id':'chain1','endpoint':client.url}
        assert (await client.release_chain(handle))['released'] is True
        handle['chain_id']='gone'
        assert (await client.release_chain(handle))['already_absent'] is True
    assert all(req.url.path == '/v1/chain_cache/invalidate' for req in seen)
    assert all(req.headers['connection'] == 'close' for req in seen)


@pytest.mark.asyncio
async def test_release_chain_does_not_mask_unknown_route_as_absent(store):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(404, json={'detail': 'Not Found'}))) as http:
        client=ChatClient(config(),store,http=http)
        with pytest.raises(ModelHTTPError, match='Not Found'):
            await client.release_chain({'chain_id':'chain1','endpoint':client.url})

@pytest.mark.asyncio
async def test_same_cell_has_one_writer(store):
    active=0; peak=0
    async def backend(request):
        nonlocal active,peak
        active+=1; peak=max(peak,active); await asyncio.sleep(.01); active-=1
        return httpx.Response(200,json=response(False))
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as http:
        client=ChatClient(config(False),store,http=http)
        await asyncio.gather(*(client.complete(MESSAGES,32,operation_id=str(i),writer_key='same') for i in range(3)))
    assert peak==1

def test_sparse_payload_has_no_response_format_or_seed(store):
    client=ChatClient(config(),store,http=httpx.AsyncClient())
    body,_=client.build_payload(MESSAGES,32,None)
    assert body['enable_thinking'] is False
    assert 'response_format' not in body and 'seed' not in body


def test_deepseek_non_thinking_uses_native_api_parameter(store):
    cfg = EndpointConfig(engine='deepseek', model='deepseek-v4-pro', enable_thinking=False)
    client = ChatClient(cfg, store, http=httpx.AsyncClient())
    body, mode = client.build_payload(MESSAGES, 1536, None)
    assert body['thinking'] == {'type': 'disabled'}
    assert body['messages'] == MESSAGES and body['max_tokens'] == 1536
    assert 'enable_thinking' not in body and 'chat_template_kwargs' not in body
    assert mode == 'full_prefix' and 'chain_id' not in body


def test_deepseek_cache_usage_remains_observed_not_assumed():
    usage = observed_usage({'prompt_tokens': 100, 'completion_tokens': 7,
                            'prompt_cache_hit_tokens': 64, 'prompt_cache_miss_tokens': 36})
    assert usage['reused_tokens'] == 64 and usage['prefilled_tokens'] == 36
    assert usage['reuse_counter_source'] == 'deepseek.prompt_cache_hit_tokens'
    assert observed_usage({'prompt_tokens': 100})['reused_tokens'] is None
