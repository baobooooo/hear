import json
from copy import deepcopy
from bcgraph.config import WorkflowConfig
from bcgraph.packer import pack_documents
from bcgraph.passage_prompts import READER_SYSTEM
from bcgraph.tokenization import Utf8Counter
from bcgraph.passages import registry

def test_four_round_schedule_leaves_room_without_context_truncation():
    cfg=WorkflowConfig(answer_policy='best_effort',max_reader_turns=3,max_reopens=1,
                       context_reserve_tokens=768,passage_chars=1200)
    counter=Utf8Counter()
    s={'protocol':'passages-v2','passage_chars':1200,'question':'Which instrument?',
       'focus':'Find the instrument','turn':0,'turn_limit':3,'revision':1,
       'raw_history':[{'role':'system','content':READER_SYSTEM}], 'sources':{},
       'search_limit':8,'searches_used':0,'doc_catalog':{},'selected_passages':{}}
    snapshots=[]
    for turn in range(4):
        if turn==3: s.update(turn_limit=4,revision=2)
        documents=[{'docid':f'doc{turn}','text':('Instrument research details. '*10000)}]
        output=cfg.reader_first_output_tokens if turn==0 else cfg.reader_followup_output_tokens
        result=pack_documents(s,documents,counter,cfg,40960,output)
        assert result['message'] is not None
        assert result['logical_prompt_tokens']+output+cfg.context_reserve_tokens<=40960
        snapshots.append(result)
        original=deepcopy(s['raw_history'])
        s['raw_history'] += [result['message'],{'role':'assistant','content':'{"selected_passage_ids":[],"next_queries":["next clue"]}'}]
        assert s['raw_history'][:len(original)]==original
        s['sources'].update(result['sources']); s['turn']+=1
    assert snapshots[0]['reserved_future_tokens']==3*(5000+1024+1024)
    assert snapshots[2]['reserved_future_tokens']>0 and snapshots[3]['reserved_future_tokens']==0
    assert snapshots[0]['logical_prompt_tokens']<20000
    assert len(s['sources'])==4


def test_prefix_selection_does_not_tokenize_whole_huge_document():
    from bcgraph.tokenization import prefix_chars
    class Counter:
        def __init__(self): self.largest=0
        def text(self,text):
            self.largest=max(self.largest,len(text))
            return len(text)
    counter=Counter()
    n=prefix_chars('x'*1000000,5000,counter)
    assert n==5000 and counter.largest<=10000
