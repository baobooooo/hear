"""Consolidated production path: research -> raw passages -> one final decision.

Reuses the tested transports, retrieval journal, admission and linear chain
lifecycle. The old proof ledger is NOT a gate on this runtime's answer path.
"""
from __future__ import annotations
import asyncio
from copy import deepcopy
from .runtime import Runtime, distribute, unique_queries
from .schemas import message_text, FinalDecision
from .transport import ModelRequestError
from .retrieval import normalize_docid, RetrievalError
from .passages import (PROTOCOL, Selection, Decision, parse_selection, parse_decision,
                       output_object, apply_selection, final_pool, commit_decision,
                       parse_research_output, report_references, main_reports, validate_direction_plan, control_handoff,
                       grounded_direction, registry, visible_pack)
from .passage_prompts import (MAIN_SYSTEM, READER_SYSTEM, RESEARCHER_SYSTEM, DIRECTION_RESEARCHER_INSTRUCTIONS, DIRECTION_OUTPUT_CONTRACT, direction_output_contract, planner_message,
                              final_message, reader_message, js)


class PassageRuntime(Runtime):
    async def plan(self, state: dict) -> dict:
        cfg = self.config.workflow
        history = [{'role':'system','content':MAIN_SYSTEM},
                   planner_message(state['question'], cfg.max_cells_per_query,
                                   directions=cfg.research_directions, guards=cfg.research_direction_guards)]
        errors, usage, cells = [], [], []
        for attempt in range(2 if cfg.research_direction_guards else 1):
            count = self.counters['main'].messages(history)
            if count + cfg.planner_output_tokens + cfg.context_reserve_tokens > self.config.main.max_context_tokens:
                raise ModelRequestError('Question/plan prompt exceeds Main context budget')
            reply = await self.clients['main'].complete(history, cfg.planner_output_tokens,
                operation_id=state['scope']+(':main:plan' if not attempt else ':main:plan:repair'),
                writer_key=state['scope']+':main', local_prompt_tokens=count)
            history.append(reply['assistant'])
            usage.append({'stage': 'plan' if not attempt else 'plan_repair', 'usage': reply['usage']})
            try:
                if reply['assistant'].get('tool_calls'): raise ValueError('unexpected planner tool calls')
                obj, notes = output_object(message_text(reply['assistant']))
                errors.extend('planner_normalized: '+note for note in notes)
                raw_cells = obj.get('cells')
                if raw_cells is None:
                    raw_cells=[{'focus':obj.get('focus',state['question']),
                                'initial_queries':obj.get('initial_queries',obj.get('queries',[]))}]
                if not isinstance(raw_cells,list) or not raw_cells: raise ValueError('no cells/queries')
                cells=[]
                for item in raw_cells:
                    if not isinstance(item,dict): raise ValueError('non-object cell')
                    queries=item.get('initial_queries',item.get('queries',[]))
                    if not isinstance(queries,list) or not all(isinstance(q,str) for q in queries):
                        raise ValueError('queries must be strings')
                    queries=unique_queries(queries,[],6)
                    if not queries: continue
                    focus=item.get('focus')
                    cell={'focus':focus[:2000] if isinstance(focus,str) and focus.strip() else state['question'],
                          'initial_queries':queries}
                    if cfg.research_direction_guards:
                        cell['starting_clues']=item.get('starting_clues')
                        if 'starting_clue_ids' in item: cell['starting_clue_ids']=item['starting_clue_ids']
                        cell, normalized = grounded_direction(cell, state['question'])
                        errors.extend('planner_normalized: '+note for note in normalized)
                    cells.append(cell)
                if not cells: raise ValueError('no usable initial query')
                if cfg.research_direction_guards:
                    validate_direction_plan(cells, state['question'], cfg.max_cells_per_query)
                elif cfg.research_directions and len(cells)>cfg.max_cells_per_query:
                    cells=cells[:cfg.max_cells_per_query]
                    errors.append('planner_normalized: direction_limit_truncated')
                elif len(cells)>cfg.max_cells_per_query:
                    cells=[{'focus':state['question'],'initial_queries':unique_queries(
                        [q for c in cells for q in c['initial_queries']],[],6)}]
                    errors.append('planner_normalized: merged_to_configured_cell_limit')
                break
            except ValueError as exc:
                if cfg.research_direction_guards and not attempt:
                    self.store.event('planner_repair', scope=state['scope'], reason=str(exc)[:600])
                    history.append({'role':'user','content':'TASK: PLAN_RESEARCH_REPAIR\nCorrect the complete plan JSON once. '+str(exc)[:600]+
                                    '. Select valid starting_clue_ids from the supplied catalog and independently executable directions. Do not answer the question.'})
                    continue
                errors.append('planner_format_fallback: '+str(exc)[:600])
                cells=[{'focus':state['question'],'initial_queries':[state['question']]}]
        n=len(cells)
        reserved_search=min(cfg.reopen_searches,max(0,cfg.max_searches_per_query-n)) if cfg.max_reopens else 0
        reserved_docs=min(cfg.followup_documents*cfg.reopen_turns,max(0,cfg.max_document_fetches_per_query-n)) if cfg.max_reopens else 0
        reserved_output=min((cfg.reader_followup_output_tokens+cfg.reader_delivery_repair_tokens)*cfg.reopen_turns,
                            max(0,cfg.max_total_reader_output_tokens-n*128)) if cfg.max_reopens else 0
        searches=distribute(cfg.max_searches_per_query-reserved_search,n)
        docs=distribute(cfg.max_document_fetches_per_query-reserved_docs,n)
        outputs=distribute(cfg.max_total_reader_output_tokens-reserved_output,n)
        # Original question is authoritative. These IDs are compatibility state,
        # not model-generated semantic assertions and never determine completion.
        constraints={'target':{'id':'target','description':state['question'],'time_scope':'unspecified',
                               'required':True,'answer_target':True}}
        jobs=[{**c,'id':f'cell{i+1}','constraint_ids':['target'], 'search_limit':searches[i],
               'document_limit':docs[i],'output_limit':outputs[i],'turn_limit':cfg.max_reader_turns}
              for i,c in enumerate(cells)]
        self.store.event('plan',scope=state['scope'],query_id=state['query_id'],cells=n,errors=errors,protocol=PROTOCOL,
                         directions=cfg.research_directions, jobs=jobs)
        return {'protocol':PROTOCOL,'constraints':constraints,'jobs':jobs,'cells':{},
                'main_history':history, 'decision_round':0,'reopens':0,
                'errors':errors,'main_usage':usage,
                'final_repair_count':0,'final_review_count':0,'delivery_issues':[]}

    def init_cell(self, worker: dict) -> dict:
        state=super().init_cell(worker)
        # Structured coordination metadata comes from validated original passages,
        # never from parsing Main's free-form instructions. Fetch before citing.
        for document in worker['job'].get('shared_documents', []):
            state['doc_catalog'].setdefault(document['docid'], deepcopy(document))
        state['protocol']=PROTOCOL
        state['passage_chars']=self.config.workflow.passage_chars
        state['expanded_research_reports'] = self.config.workflow.expanded_research_reports
        if self.config.workflow.research_sync_interval:
            state['stage_until'] = min(state['turn_limit'], state['turn'] + self.config.workflow.research_sync_interval)
        mode = self.config.workflow.reader_mode
        if worker.get('previous') and state.get('reader_mode', 'selector') != mode:
            raise ValueError('reader_mode cannot change during a cell lifetime')
        if worker.get('previous') and state.get('research_directions', False) != self.config.workflow.research_directions:
            raise ValueError('research_directions cannot change during a cell lifetime')
        if worker.get('previous') and state.get('research_direction_guards', False) != self.config.workflow.research_direction_guards:
            raise ValueError('research_direction_guards cannot change during a cell lifetime')
        if not worker.get('previous'):
            if self.config.workflow.research_direction_guards:
                state['research_direction_guards'] = True
            if self.config.workflow.research_directions:
                state['research_directions'] = True
            state.update(raw_history=[{'role':'system','content':READER_SYSTEM}],
                         selected_passages={},candidate_answer='',missing_fact='',selection_turns=[])
            if mode == 'researcher':
                state.update(reader_mode=mode, latest_research_report='', report_status='missing',
                             report_turn=0, report_passage_refs={},
                             raw_history=[{'role': 'system', 'content': RESEARCHER_SYSTEM + (
                                 DIRECTION_RESEARCHER_INSTRUCTIONS if self.config.workflow.research_directions else '') + (
                                 '\n' + direction_output_contract(self.config.workflow.expanded_research_reports)
                                 if self.config.workflow.research_direction_guards else '')}])
                if self.config.workflow.researcher_select_documents:
                    state['raw_history'][0]['content'] = (
                        'You are a persistent researcher with TWO distinct task phases. '
                        'When the latest user task is SELECT_DOCUMENTS, return only a JSON object '
                        'with documents (a list of displayed docid objects) and next_queries. '
                        'Candidate snippets are leads, not read original passages. Do not produce '
                        'selected_passage_ids or RESEARCH_REPORT in that phase. '
                        'The following reading/report instructions apply ONLY to SELECT_AND_RESEARCH '
                        'and REPAIR_RESEARCH_DELIVERY, after actual original passages are supplied:\n'
                        + state['raw_history'][0]['content'])
        return state

    def _fallback_queries(self, state: dict) -> list[str]:
        # Search hypotheses, never manufacture evidence or a final answer.
        candidate=state.get('candidate_answer','').strip()
        gap=state.get('missing_fact','').strip()
        if candidate and gap: return [candidate+' '+gap]
        if gap: return [gap]
        return [state['focus']]

    @staticmethod
    def _unread_document_requests(cell: dict) -> list[dict]:
        shown={s['docid'] for s in cell.get('sources',{}).values()}
        unseen=[{'docid':d,'offset':0} for d in cell.get('doc_catalog',{}) if d not in shown]
        if unseen: return unseen[:4]
        # Union intervals within a document version. A stale next_offset from
        # an earlier excerpt must not advertise a tail that was read later.
        versions={}
        for source in cell.get('sources',{}).values():
            key=(source['docid'],source.get('document_sha256',''))
            versions.setdefault(key,[]).append(source)
        requests=[]
        for (docid,_),sources in versions.items():
            cursor=0
            for source in sorted(sources,key=lambda x:x['start']):
                if source['start']>cursor: break
                cursor=max(cursor,source['end'])
            total=max(source.get('total_chars',source['end']) for source in sources)
            if cursor<total and not any(r['docid']==docid for r in requests):
                requests.append({'docid':docid,'offset':cursor})
        return requests[:4]

    async def cell_fetch(self, state: dict) -> dict:
        # Do not spend search/document budget after the logical context is full.
        if (state.get('turn') or state.get('reader_mode') == 'researcher') and not self._has_reader_context(state):
            return {'stop_reason':'context_budget_exhausted','packed':{'message':None}}
        out=await super().cell_fetch(state)
        packed=out.get('packed',{})
        self.store.event('context_schedule',scope=state['scope'],cell_id=state['cell_id'],
                         turn=state['turn']+1,hard_prompt_cap=packed.get('hard_prompt_cap'),
                         scheduled_prompt_cap=packed.get('scheduled_prompt_cap'),
                         reserved_future_tokens=packed.get('reserved_future_tokens'),
                         stop_reason=out.get('stop_reason'))
        return out

    async def select_documents(self, state: dict, requests: list[dict], limit: int, *, repair_reason: str | None = None):
        """Select only discovered documents on the existing Reader history/chain."""
        cfg = self.config.workflow
        backend = state.get('backend') or 'reader'
        counter = self.counters[backend]
        output = cfg.document_selection_repair_tokens if repair_reason else cfg.document_selection_output_tokens
        if state['output_limit'] - state['output_used'] < output + cfg.reader_followup_output_tokens:
            return [], {'stop_reason': 'output_budget_exhausted'}
        candidates = []
        for req in requests:
            # Large catalogs must not starve the shared MCP transport loop.
            await asyncio.sleep(0)
            hit = state['doc_catalog'].get(req['docid'], {})
            item = {**req, 'title': hit.get('title', ''), 'snippet': hit.get('snippet', ''),
                    'previously_read': any(s['docid'] == req['docid'] for s in state['sources'].values())}
            if counter.text(js([*candidates, item])) > cfg.document_catalog_tokens:
                continue
            candidates.append(item)
        if not candidates:
            if state.get('reopen_instruction') and state.get('last_consumed_feedback_revision', 0) < state['revision']:
                return [], {}
            return [], {'stop_reason': 'no_new_lead'}
        selection_payload = {
            'question': state['question'], 'objective': state['focus'],
            'current_queries': state.get('retrieval_context', ''),
            'missing_fact': state.get('missing_fact', ''),
            'global_feedback': state.get('reopen_instruction', ''),
            'candidate_documents_NOT_EVIDENCE': candidates, 'max_documents': limit,
            'instructions': 'Choose documents likely to resolve the current gap, not simply the first ranks. '
                'Return ONE JSON object: {"documents":[{"docid":"shown id"}],"next_queries":[]}. '
                'Use requested_offset only for a known text position. Omit it to locate a relevant unread passage. '
                'If candidates are irrelevant, return no documents and concrete revised next_queries. '
                'Separate distinctive discovery clues from name-based verification; never invent docids.',
            # Keep the exact current-round namespace at the end of the prompt. Long-chain sparse
            # attention can otherwise retain the selection instruction but lose IDs in the catalog.
            'current_selection_contract': {
                'round': state['turn'] + 1,
                'allowed_document_objects_exact': [{'docid': item['docid']} for item in candidates],
                'documents_schema': [{'docid': 'copy one exact object from allowed_document_objects_exact'}],
                'rules': ['documents must be an array of objects, never bare strings',
                          'use only IDs from this current-round allowlist, never prior-round IDs'],
            },
        }
        message = {'role': 'user', 'content': 'TASK: SELECT_DOCUMENTS\n' + js(selection_payload)}
        if repair_reason:
            import json
            payload = json.loads(message['content'].split('\n', 1)[1])
            payload['correction_reason'] = repair_reason
            payload['correction_instruction'] = 'Correct control ONCE. Return only control JSON, without a report.'
            payload['current_selection_contract'] = payload.pop('current_selection_contract')
            message['content'] = 'TASK: SELECT_DOCUMENTS\n' + js(payload)
        history = [*state['raw_history'], message]
        count = counter.messages(history)
        reserve = cfg.reader_followup_output_tokens + cfg.reader_delivery_repair_tokens + cfg.context_reserve_tokens + 2048
        if count + output + reserve > self.clients[backend].config.max_context_tokens:
            return [], {'stop_reason': 'context_budget_exhausted'}
        request_seq = state.get('request_seq', 0) + 1
        operation_id = (f"{state['scope']}:reader:{state['cell_id']}:turn:{state['turn']+1}"
                        f":request:{request_seq}:select-documents" + (':repair' if repair_reason else ''))
        try:
            reply = await self.clients[backend].complete(history, output,
                operation_id=operation_id,
                writer_key=f"{state['scope']}:reader:{state['cell_id']}", handle=state.get('handle'),
                continuation=bool(state.get('handle') or state['turn']), local_prompt_tokens=count)
        except ModelRequestError as exc:
            return [], {'stop_reason': 'reader_service_error', 'last_error': str(exc),
                        'output_used': state['output_used'] + output}
        charge = reply['usage']['completion_tokens']
        charge = output if charge is None else charge
        metric = {k: reply[k] for k in ('request_mode', 'finish_reason', 'duration_ms', 'usage', 'recoveries', 'journal_replay')}
        metric.update(turn=state['turn']+1, document_selection=True, backend=backend,
                      document_selection_repair=bool(repair_reason), output_budget_charged=charge, local_logical_prompt_tokens=count)
        updates = {'request_seq': request_seq, 'selection_operation_id': operation_id, 'raw_history': [*history, deepcopy(reply['assistant'])], 'handle': reply['handle'],
                   'backend': backend, 'output_used': state['output_used']+charge,
                   'metrics': [*state['metrics'], metric]}
        try:
            if reply['assistant'].get('tool_calls'):
                raise ValueError('unexpected tool calls')
            obj, _ = output_object(message_text(reply['assistant']))
            shown = {normalize_docid(c['docid']) for c in candidates}
            known = set(state['doc_catalog']) | {s['docid'] for s in state['sources'].values()}
            errors, result = [], []
            selected = obj.get('documents')
            if not isinstance(selected, list):
                errors.append({'reason': 'bad_item_shape', 'field': 'documents'})
                selected = []
            for item in selected:
                raw = item.get('docid') if isinstance(item, dict) else item
                diagnostic = {'raw_id': repr(raw)[:200], 'raw_type': type(raw).__name__,
                              'normalized_id': None, 'shown': False, 'known': False}
                reason = ''
                if not isinstance(item, dict):
                    reason = 'bad_item_shape'
                else:
                    try:
                        docid = normalize_docid(raw)
                        diagnostic.update(normalized_id=docid, shown=docid in shown, known=docid in known)
                        if docid not in shown:
                            reason = 'known_not_shown' if docid in known else 'unknown_docid'
                        offset = item.get('requested_offset')
                        if not reason and offset is not None and (type(offset) is not int or offset < 0):
                            reason = 'invalid_offset'
                        totals = [s.get('total_chars', s['end']) for s in state['sources'].values()
                                  if s['docid'] == docid]
                        if not reason and offset is not None and totals and offset >= max(totals):
                            reason = 'invalid_offset'
                        if not reason:
                            req = {'docid': docid}
                            if offset is not None:
                                req['requested_offset'] = offset
                            if req not in result:
                                result.append(req)
                    except RetrievalError:
                        reason = 'bad_docid_type'
                if reason:
                    errors.append({**diagnostic, 'reason': reason})
            queries = obj.get('next_queries', [])
            if not isinstance(queries, list):
                errors.append({'reason': 'bad_item_shape', 'field': 'next_queries'})
                queries = []
            valid_queries = [q for q in queries if isinstance(q, str) and q.strip()]
            if len(valid_queries) != len(queries):
                errors.append({'reason': 'bad_item_shape', 'field': 'next_queries'})
            updates['selection_next_queries'] = unique_queries(valid_queries, state['seen_queries'], 3)
            valid_count = len(result)
            result = result[:limit]
            self.store.event('document_selection', scope=state['scope'], cell_id=state['cell_id'],
                operation_id=operation_id, request_seq=request_seq, phase='document_selection',
                turn=state['turn']+1, displayed=len(candidates), shown_ids=sorted(shown), selected=result,
                rejected=errors, proposed=len(selected), valid=valid_count, executed=len(result),
                overflow=max(0, valid_count-limit),
                catalog_tokens=counter.text(js(candidates)), next_queries=updates['selection_next_queries'])
            if errors and not result and not updates['selection_next_queries']:
                raise ValueError(js(errors))
            return result, updates
        except ValueError as exc:
            if not repair_reason:
                self.store.event('document_selection_repair', scope=state['scope'], cell_id=state['cell_id'],
                                 turn=state['turn']+1, operation_id=operation_id, request_seq=request_seq,
                                 reason=str(exc), status='attempted')
                result, corrected = await self.select_documents({**state, **updates}, requests, limit, repair_reason=str(exc))
                self.store.event('document_selection_repair', scope=state['scope'], cell_id=state['cell_id'],
                                 turn=state['turn']+1, operation_id=corrected.get('selection_operation_id', operation_id),
                                 request_seq=corrected.get('request_seq', request_seq), status='failed' if corrected.get('stop_reason') else 'nonempty_recovered' if result else 'empty')
                return result, {**updates, **corrected}
            self.store.event('document_selection_error', scope=state['scope'], cell_id=state['cell_id'], reason=str(exc))
            return [], {**updates, 'stop_reason': 'document_selection_error', 'last_error': str(exc)}

    async def cell_read(self, state: dict) -> dict:
        if self.config.workflow.store_raw_requests and state.get('packed', {}).get('message'):
            from .evidence import stable_hash
            context = {'question': state['question'], 'query_id': state['query_id'],
                       'cell_id': state['cell_id'], 'focus': state['focus'],
                       'reader_mode': state.get('reader_mode', 'selector'),
                       'turn': state['turn'] + 1, 'revision': state['revision'],
                       'sources': state['packed']['sources'],
                       'doc_catalog': state['doc_catalog'],
                       'searches_used': state['searches_used'],
                       'documents_used': state['documents_used'],
                       **{key: state[key] for key in
                          ('search_limit', 'document_limit', 'output_limit', 'turn_limit')}}
            key = f"reader-context:{state['scope']}:reader:{state['cell_id']}:turn:{state['turn'] + 1}"
            self.store.put(key, stable_hash(context), 'success', context)
        out = await super().cell_read(state)
        out['delivery_original_reply'] = {}
        if state.get('reader_mode') == 'researcher' and self.config.workflow.reader_delivery_repair_tokens and out.get('reply'):
            out = await self._repair_delivery({**state, **out}, out)
        if state.get('reader_mode') == 'researcher':
            # Even a failed new request must not present an older report as its
            # result. The immutable old assistant remains in raw_history.
            out.update(latest_research_report='', report_status='invalid',
                       report_turn=out.get('turn', state['turn']), report_passage_refs={})
        return out

    def _delivery_problem(self, state: dict) -> str:
        reply = state['reply']
        try:
            if reply['assistant'].get('tool_calls'):
                return 'unexpected tool calls'
            parsed, notes, partial, report, status = parse_research_output(
                message_text(reply['assistant']), reply.get('finish_reason'), bounded=True)
            # A valid complete control is independent of report delivery quality.
            if not partial:
                return ''
            _, issues, changes = apply_selection(parsed, state['sources'], state.get('selected_passages', {}),
                                                 state['turn'], self.config.workflow.passage_chars)
            if changes or (parsed.selected_passage_ids and not issues):
                return ''
            return 'truncated control without a safe positive selection'
        except ValueError as exc:
            return str(exc)[:400]
        return ''

    async def _repair_delivery(self, state: dict, out: dict) -> dict:
        """One additional delivery attempt in the same research turn and chain."""
        reason = self._delivery_problem(state)
        if not reason:
            return out
        cfg = self.config.workflow
        backend = state['backend']
        message = {'role':'user', 'content':
            'TASK: REPAIR_RESEARCH_DELIVERY\nCorrect only the control JSON once: ' + reason +
            '\nReturn one complete control JSON object. Do not write or rewrite RESEARCH_REPORT. '
            'Use only supplied P identifiers. Do not infer clear/drop/candidate changes from truncated text.\n'
            + js({'supplied_sources': [{**row, 'text': row['text'][:160]}
                for row in visible_pack(list(registry(state['sources'], cfg.passage_chars).values()))]})}
        messages = [*state['raw_history'], message]
        count = self.counters[backend].messages(messages)
        available = min(state['output_limit'] - state['output_used'],
                        self.clients[backend].config.max_context_tokens - count - cfg.context_reserve_tokens)
        event = {'scope':state['scope'], 'cell_id':state['cell_id'], 'turn':state['turn'], 'reason':reason}
        if available < cfg.reader_delivery_repair_tokens:
            self.store.event('reader_delivery_repair', **event, status='skipped_budget', available_tokens=available)
            return out
        budget = cfg.reader_delivery_repair_tokens
        out['delivery_original_reply'] = deepcopy(state['reply'])
        out['reader_delivery_repairs'] = state.get('reader_delivery_repairs', 0) + 1
        self.store.event('reader_delivery_repair', **event, status='requested', max_tokens=budget, full_report_format_rewrite_count=0)
        try:
            reply = await self.clients[backend].complete(
                messages, budget,
                operation_id=f"{state['scope']}:reader:{state['cell_id']}:turn:{state['turn']}:delivery-repair",
                writer_key=f"{state['scope']}:reader:{state['cell_id']}", handle=state['handle'],
                continuation=True, local_prompt_tokens=count)
        except ModelRequestError as exc:
            out['output_used'] += budget
            out.update(stop_reason='reader_service_error', last_error=str(exc))
            self.store.event('reader_delivery_repair', **event, status='service_error', error=str(exc))
            return out
        charge = reply['usage']['completion_tokens']
        charge = budget if charge is None else charge
        metric = {k:reply[k] for k in ('request_mode','finish_reason','duration_ms','usage','recoveries','journal_replay')}
        metric.update(turn=state['turn'], delivery_repair=True, backend=backend,
                      output_budget_charged=charge, local_logical_prompt_tokens=count)
        out.update(raw_history=[*messages, deepcopy(reply['assistant'])], handle=reply['handle'],
                   reply={k:reply[k] for k in ('assistant','usage','finish_reason')},
                   output_used=state['output_used'] + charge, metrics=[*state['metrics'], metric])
        remaining_problem = self._delivery_problem({**state, **out})
        self.store.event('reader_delivery_repair', **event,
                         status='unrepaired' if remaining_problem else 'repaired', remaining_problem=remaining_problem,
                         completion_tokens=charge)
        return out

    def cell_validate(self, state: dict) -> dict:
        cfg=self.config.workflow
        selected=state.get('selected_passages',{})
        issues=[]; partial=False; failed=False; changes=0
        candidate=state.get('candidate_answer',''); gap=state.get('missing_fact','')
        parsed=None
        research = state.get('reader_mode') == 'researcher'
        report, report_status, refs = '', 'invalid', {}
        try:
            if state['reply']['assistant'].get('tool_calls'): raise ValueError('unexpected reader tool calls')
            if research:
                parsed,notes,partial,report,report_status = parse_research_output(
                    message_text(state['reply']['assistant']), state['reply'].get('finish_reason'),
                    bounded=cfg.research_direction_guards)
            else:
                parsed,notes,partial=parse_selection(message_text(state['reply']['assistant']),state['reply'].get('finish_reason'))
            selected,issues,changes=apply_selection(parsed,state['sources'],selected,state['turn'],cfg.passage_chars)
            issues.extend(notes)
            if research and parsed.clear_candidate:
                candidate = ''
                if parsed.candidate_answer.strip():
                    issues.append('candidate_cleared_then_replaced')
            if parsed.candidate_answer.strip():
                candidate = parsed.candidate_answer
            gap=parsed.missing_fact
            queries=unique_queries([*parsed.next_queries,*state.get('pending_queries',[])],state['seen_queries'],8)
            reads=[r.model_dump() for r in parsed.read_more]
        except ValueError as exc:
            failed=True
            issues=['reader_format_error: '+str(exc)[:800]]
            queries=unique_queries([*state.get('pending_queries',[]),*self._fallback_queries(state)],state['seen_queries'],3)
            reads=[]
        if research:
            original = state.get('delivery_original_reply')
            if original:
                # Keep the actual first report separately; never splice assistant history.
                text = message_text(original['assistant'])
                try:
                    *_, report, report_status = parse_research_output(
                        text, original.get('finish_reason'), bounded=True)
                except ValueError:
                    # Only the explicit report boundary is recoverable from bad control.
                    import re
                    marker = re.search(r'(?:^|\r?\n)RESEARCH_REPORT\r?\n', text)
                    report = text[marker.end():] if marker else ''
                    report_status = ('truncated' if original.get('finish_reason') == 'length'
                                     else 'complete') if report.strip() else 'missing'
            refs, warnings = report_references(report, state['sources'], cfg.passage_chars)
            issues.extend(warnings)
            if warnings:
                report_status = 'invalid'
            if original and report and report_status != 'invalid':
                if changes or candidate != state.get('candidate_answer', ''):
                    report_status = 'stale'
        known=set(state.get('doc_catalog',{}))|{s['docid'] for s in state['sources'].values()}
        valid_reads=[]; read_keys=set()
        for r in reads:
            totals=[s.get('total_chars',s['end']) for s in state['sources'].values() if s['docid']==r['docid']]
            if r['docid'] not in known or (totals and r['offset']>=max(totals)):
                issues.append('invalid_read_request:'+r['docid']+':'+str(r['offset']))
            elif (r['docid'],r['offset']) not in read_keys:
                valid_reads.append(r);read_keys.add((r['docid'],r['offset']))
        reads=valid_reads
        no_progress=0 if changes else state.get('no_progress_turns',0)+1
        active=any(v.get('active',True) for v in selected.values())
        explicit_new=bool(parsed and (unique_queries(parsed.next_queries,state['seen_queries'],2) or reads))
        can_search=state['searches_used']<state['search_limit'] and bool(queries)
        can_read=bool(reads) or bool(self._unread_document_requests(state))
        stop=''
        if state['turn']>=state['turn_limit']: stop='turn_budget_exhausted'
        elif state['output_limit']-state['output_used']<128: stop='output_budget_exhausted'
        elif state['documents_used']>=state['document_limit']: stop='document_budget_exhausted'
        elif parsed is not None and not partial and active and not explicit_new:
            stop='ready_for_main'  # decision opportunity, NOT a proof of correctness
        elif no_progress>=cfg.max_no_progress_turns and not (
                cfg.answer_policy=='best_effort' and no_progress==cfg.max_no_progress_turns and (can_search or can_read)):
            stop='no_selection_progress'
        elif not can_search and not can_read: stop='no_new_lead'
        if not stop and state.get('stage_until') and state['turn'] >= state['stage_until']:
            stop = 'sync_due'
        self.store.event('cell_validated',scope=state['scope'],cell_id=state['cell_id'],turn=state['turn'],
            protocol=PROTOCOL,accepted_changes=changes,selected_passage_count=sum(v.get('active',True) for v in selected.values()),
            validation_errors=issues,stop_reason=stop,partial_recovery=partial,format_failed=failed,
            finish_reason=state['reply'].get('finish_reason'))
        report_update = {}
        if research:
            report_update = {'latest_research_report': report, 'report_status': report_status,
                             'report_turn': state['turn'], 'report_passage_refs': refs}
            self.store.event('research_report', scope=state['scope'], cell_id=state['cell_id'],
                turn=state['turn'], revision=state['revision'], reader_mode='researcher',
                **{k: v for k, v in report_update.items() if k != 'report_turn'},
                finish_reason=state['reply'].get('finish_reason'), partial=partial,
                candidate_before=state.get('candidate_answer', ''), candidate_after=candidate,
                clear_candidate=bool(parsed and getattr(parsed, 'clear_candidate', False)),
                next_queries=queries, stop_reason=stop,
                full_report_format_rewrite_count=0,
                control_repair_count=state.get('reader_delivery_repairs', 0),
                report_tokens=self.counters[state.get('backend') or 'reader'].text(report),
                token_count_method='standalone_report_not_additive_to_completion')
        return {**report_update, 'selected_passages':selected,'candidate_answer':candidate,'missing_fact':gap,
                'pending_queries':queries,'pending_read_more':reads,'validation_errors':issues,
                'protocol_issues':[*state.get('protocol_issues',[]),*issues],
                'reader_format_failures':state.get('reader_format_failures',0)+int(failed),
                'reader_partial_recoveries':state.get('reader_partial_recoveries',0)+int(partial),
                'no_progress_turns':no_progress,'stop_reason':stop,
                'selection_turns':[*state.get('selection_turns',[]),{'turn':state['turn'],'changes':changes,'issues':issues}]}

    def collect(self,state:dict)->dict:
        # Keep legacy-compatible fields empty. A selection is not a validated
        # claim and must not inflate accepted_evidence_count in exported metrics.
        return {'evidence':{},'candidates':{}}

    def _research_options(self, state: dict) -> list[tuple[FinalDecision,dict]]:
        if state['reopens']>=self.config.workflow.max_reopens: return []
        options=[]
        for ident,cell in sorted(state['cells'].items()):
            if cell.get('last_error'): continue
            queries=unique_queries(cell.get('pending_queries',[]),cell['seen_queries'],self.config.workflow.reopen_searches)
            reads=cell.get('pending_read_more',[])
            if not queries and not reads: reads=self._unread_document_requests(cell)
            if not queries and not reads:
                queries=unique_queries(self._fallback_queries(cell),cell['seen_queries'],2)
            decision=FinalDecision(action='research',reopen_cell_id=ident,next_queries=queries,read_more=reads,
                explanation=cell.get('missing_fact') or 'Verify the strongest candidate against the original question.')
            job=self._reopen_job(state,decision)
            if job: options.append((decision,job))
        return options

    def _can_reopen(self,state:dict)->bool:
        cfg=self.config.workflow; b=self._research_budget(state)
        if not b['reopens'] or b['document_fetches']<1 or b['reader_output_tokens']<128: return False
        return any(not c.get('last_error') and self._has_reader_context(c) and
                   (b['searches']>0 or self._unread_document_requests(c)) for c in state['cells'].values())

    def _reopen(self,state,updates,decision=None):
        proposed=None
        if decision and decision.action=='research':
            old=FinalDecision(action='research',reopen_cell_id=decision.reopen_cell_id,
                              next_queries=decision.next_queries,read_more=decision.read_more,
                              explanation=decision.explanation)
            job=self._reopen_job(state,old)
            if job: proposed=(old,job)
        if proposed is None and self.config.workflow.research_directions:
            # Do not silently redirect a model-requested check to the first cell.
            # Deterministic recovery is safe only when there is one eligible target.
            if decision and decision.action == 'research':
                return None
            opts = self._research_options(state)
            if len(opts) == 1:
                proposed = opts[0]
            else:
                return None
        if proposed is None:
            opts=self._research_options(state)
            if opts: proposed=opts[0]
        if proposed is None: return None
        old,job=proposed
        self.store.event('policy_reopen',scope=state['scope'],protocol=PROTOCOL,
                         cell_id=job['id'],remaining_budget=self._research_budget(state),
                         next_queries=job['initial_queries'],read_more=job['read_more'])
        return {**updates,'decision':old.model_dump(),'next_job':job,'status':'researching',
                'reopens':state['reopens']+1}

    async def coordinate(self, state: dict, eligible: dict) -> dict:
        """A bounded Main checkpoint; resumed workers keep their original histories."""
        cfg = self.config.workflow
        counter = self.counters['main']
        epoch = state.get('sync_round', 0) + 1
        pool = final_pool(state['cells'], state['question'], cfg.passage_chars, directions=True)
        rows = []
        base = state['main_history']
        def prompt(items):
            return {'role': 'user', 'content': 'TASK: COORDINATE_RESEARCH\n' + js({
                'question': state['question'], 'sync_round': epoch,
                'reports_NOT_EVIDENCE': main_reports(state['cells'], items, counter,
                    cfg.main_research_report_token_cap, directions=True),
                'original_passages': [{**r, 'passage_id': r['display_id']} for r in items],
                'eligible_directions': [{'cell_id': k, 'objective': c['focus'],
                    'remaining_turns': c['turn_limit']-c['turn']} for k,c in eligible.items()],
                'instructions': 'Synthesize findings and resolve overlap. Answer now if defensible from original '
                    'passages, using {"action":"answer","exact_answer":"value","citations":["S1P1"],"explanation":"reason"}. '
                    'Otherwise return {"action":"continue","tasks":[{"cell_id":"eligible id",'
                    '"instruction":"specific next investigation", "next_queries":["concrete search"],'
                    '"evidence_ids":["S1P1"]}]}. Continue multiple useful directions for at most two more rounds; '
                    'Use only cell_id values in the CURRENT eligible_directions list, each at most once. '
                    'Earlier reports and previous tasks may mention directions that are no longer eligible. '
                    'share newly found entities with exact source passages, distinguish hypotheses and contradictions. '
                    'Only cite IDs displayed here. Do not restart histories or repeat failed broad queries. '
                    'If no useful investigation remains return {"action":"unresolved","explanation":"concrete gap"}.'})}
        cap = self.config.main.max_context_tokens - cfg.final_output_tokens - cfg.context_reserve_tokens
        for row in pool:
            await asyncio.sleep(0)
            trial = [*rows, {**row, 'display_id': f'S{epoch}P{len(rows)+1}'}]
            if counter.text(js([r['text'] for r in trial])) > cfg.final_evidence_tokens:
                continue
            if counter.messages([*base, prompt(trial)]) <= cap:
                rows = trial
        history = [*base, prompt(rows)]
        if counter.messages(history) > cap:
            return {**self._unresolved(state, 'Main synchronization context exhausted'), 'next_jobs': []}
        try:
            reply = await self.clients['main'].complete(history, cfg.final_output_tokens,
                operation_id=f"{state['scope']}:main:sync:{epoch}", writer_key=state['scope']+':main',
                local_prompt_tokens=counter.messages(history))
        except ModelRequestError as exc:
            return {**self._unresolved(state, 'Main synchronization service error: '+str(exc)), 'next_jobs': []}
        updates = {'sync_round': epoch, 'next_jobs': [], 'next_job': None,
                   'main_history': [*history, deepcopy(reply['assistant'])],
                   'main_usage': [*state['main_usage'], {'stage':'coordinate', 'usage':reply['usage']}]}
        try:
            if reply['assistant'].get('tool_calls') or reply['finish_reason'] == 'length':
                raise ValueError('incomplete coordination response')
            obj, _ = output_object(message_text(reply['assistant']))
            if obj.get('action') == 'answer':
                decision = commit_decision(parse_decision(message_text(reply['assistant']))[0], rows)
                return {**updates, 'decision': decision, 'status':'completed',
                        'answer_support':'cited_original_passages_semantics_unverified', 'decision_exit_cause':'answered'}
            if obj.get('action') == 'unresolved':
                # Reuse the existing final source review before committing abstention.
                return await self._final_after_sync(state, updates)
            tasks = obj.get('tasks')
            if obj.get('action') != 'continue' or not isinstance(tasks, list) or not tasks:
                raise ValueError('coordination requires nonempty tasks')
            jobs, seen = [], set()
            originals = {r['display_id']: r for r in rows}
            for task in tasks:
                if not isinstance(task, dict) or task.get('cell_id') not in eligible or task['cell_id'] in seen:
                    raise ValueError('invalid or duplicate continuation direction')
                ident = task['cell_id']; seen.add(ident); cell = eligible[ident]
                queries = task.get('next_queries', [])
                refs = task.get('evidence_ids', [])
                if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                    raise ValueError('invalid synchronization queries')
                if not isinstance(refs, list) or not all(isinstance(k, str) and k in originals for k in refs):
                    raise ValueError('shared evidence was not displayed to Main')
                instruction = task.get('instruction')
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError('continuation needs a concrete instruction')
                shared_candidates = [{'shared_id': k, 'docid': originals[k]['docid'],
                           'start': originals[k]['start'], 'end': originals[k]['end'],
                           'text': originals[k]['text']} for k in dict.fromkeys(refs)]
                shared, omitted = [], []
                for source in shared_candidates:
                    if counter.text(js([*shared, source])) + counter.text(instruction) > 11000:
                        omitted.append(source['shared_id'])
                    else:
                        shared.append(source)
                feedback = js({'Main_instruction_NOT_EVIDENCE': instruction, 'shared_originals': shared,
                    'omitted_shared_ids_due_to_budget': omitted,
                    'citation_rule':'Shared S identifiers are not your local P identifiers. Retrieve the original document to cite it locally.'})
                if counter.text(feedback) > 12000:
                    raise ValueError('shared feedback exceeds 12000 token limit')
                jobs.append({'id':ident, 'focus':cell['focus'], 'instruction':feedback,
                    'shared_documents': [{'docid': source['docid'], 'snippet': source['text'],
                        'shared_id': source['shared_id'], 'shared_start': source['start'],
                        'shared_end': source['end'], 'shared_sync_round': epoch} for source in shared],
                    'initial_queries':unique_queries(queries or cell['pending_queries'], cell['seen_queries'], 6),
                    'read_more':cell.get('pending_read_more', []),
                    **{k:cell[k] for k in ('search_limit','document_limit','output_limit','turn_limit')}})
            self.store.event('research_sync', scope=state['scope'], sync_round=epoch,
                             continued=[j['id'] for j in jobs], jobs=jobs)
            return {**updates, 'next_jobs':jobs}
        except ValueError as exc:
            self.store.event('research_sync_error', scope=state['scope'], sync_round=epoch, reason=str(exc))
            return await self._final_after_sync(state, {**updates,
                'errors': [*state['errors'], 'coordination_error: '+str(exc)]})

    async def _final_after_sync(self, state, updates):
        result = await self._decide_final({**state, **updates})
        return {**updates, **result, 'next_jobs': []}

    async def decide(self, state: dict) -> dict:
        cfg = self.config.workflow
        max_syncs = (cfg.max_reader_turns - 1) // cfg.research_sync_interval if cfg.research_sync_interval else 0
        if cfg.research_sync_interval and state.get('sync_round', 0) < max_syncs:
            eligible = {k:c for k,c in state['cells'].items()
                        if c['turn'] < c['turn_limit'] and c['stop_reason'] in {'sync_due','ready_for_main','no_selection_progress','no_new_lead'}
                        and c['searches_used'] < c['search_limit'] and c['documents_used'] < c['document_limit']}
            if eligible:
                return await self.coordinate(state, eligible)
        result = await self._decide_final(state)
        return {**result, 'next_jobs': []}

    async def _decide_final(self,state:dict)->dict:
        cfg=self.config.workflow; counter=self.counters['main']
        pool=final_pool(state['cells'],state['question'],cfg.passage_chars, directions=cfg.research_directions)
        can_reopen=self._can_reopen(state)
        options=self._research_options(state) if can_reopen else []
        prompt_options=[{'cell_id':j['id'],'next_queries':j['initial_queries'],'read_more':j['read_more'],
                         **({'objective': j['focus']} if cfg.research_directions else {})} for _,j in options]
        cap=self.config.main.max_context_tokens-cfg.final_output_tokens-cfg.context_reserve_tokens
        rows=[]
        display_report_cap = cfg.main_research_report_token_cap
        def prompt(items):
            report_cap = display_report_cap
            reports = (main_reports(state['cells'], items, counter, report_cap, directions=cfg.research_directions)
                       if cfg.main_use_research_report else [])
            while True:
                message = final_message(state['question'],state['cells'],items,can_reopen,
                                        self._research_budget(state),prompt_options,cfg.answer_policy, reports=reports,
                                        directions=cfg.research_directions)
                if cfg.research_direction_guards or not reports or counter.messages([*base, message]) <= pack_cap:
                    return message
                if cfg.research_directions:
                    report_cap //= 2
                    reports = main_reports(state['cells'], items, counter, report_cap, directions=True)
                else:
                    reports.pop()  # optional analysis must not displace usable original text
        base=list(state['main_history'])
        # Reserve one short corrective/terminal-choice message BEFORE requesting
        # the initial decision. Never reset/truncate Main's cached history later.
        correction_reserve=min(1024,max(256,cfg.final_output_tokens))
        pack_cap=cap-cfg.final_output_tokens-correction_reserve
        if cfg.research_direction_guards and cfg.main_use_research_report:
            # Freeze a balanced report allowance before filling original text.
            # Reserve space for evidence as well; reports alone cannot prove facts.
            minimum = min(display_report_cap, 128 * len(state['cells']))
            evidence_reserve = 2048 * len(state['cells'])
            while (display_report_cap > minimum and
                   counter.messages([*base, prompt([])]) > pack_cap - evidence_reserve):
                await asyncio.sleep(0)
                display_report_cap = max(minimum, display_report_cap // 2)
        if counter.messages([*base,prompt([])])>cap:
            return self._unresolved(state,'Main context exhausted; history was not silently rewritten')
        for row in pool:
            await asyncio.sleep(0)
            # New decision epochs use distinct aliases; a citation from an old
            # Main pack must never silently point to a different new passage.
            row = {**row, 'display_id': f"D{state['decision_round']+1}P{len(rows)+1}"}
            trial=[*rows,row]
            if counter.text(js([r['text'] for r in trial]))>cfg.final_evidence_tokens: continue
            if counter.messages([*base,prompt(trial)])>pack_cap: continue
            rows=trial
        history=[*base,prompt(rows)]
        updates={'next_job':None,'main_usage':list(state['main_usage']),
                 'decision_round':state['decision_round'],'main_history':base,
                 'errors':list(state.get('errors',[])),
                 'delivery_issues':list(state.get('delivery_issues',[])),
                 'final_repair_count':state.get('final_repair_count',0),
                 'final_review_count':state.get('final_review_count',0)}
        self.store.event('final_passage_pack',scope=state['scope'],protocol=PROTOCOL,
            selected=len(rows),available=len(pool),source_chars=sum(len(r['text']) for r in rows),
            reader_selected=sum(r['selection']=='reader' for r in rows),
            raw_source_fallback=sum(r['selection']=='source_fallback' for r in rows),
            can_reopen=can_reopen,budget=self._research_budget(state))
        last_reason='no_decision'
        # ONE shared recovery opportunity per decision stage, not separate
        # recursive loops for schema, citations, confidence and abstention.
        for attempt in range(2):
            if counter.messages(history)>cap:
                last_reason='Main context exhausted before bounded correction'; break
            try:
                # A malformed first response may be a repetition loop that filled
                # the normal decision budget. Keep recovery deliberately compact
                # so the correction cannot repeat until it is truncated again.
                output_tokens=(cfg.final_output_tokens if not attempt else
                               min(cfg.final_output_tokens,1024))
                reply=await self.clients['main'].complete(history,output_tokens,
                    operation_id=f"{state['scope']}:main:decision:{updates['decision_round']+1}",
                    writer_key=state['scope']+':main',local_prompt_tokens=counter.messages(history))
            except ModelRequestError as exc:
                return {**updates,**self._unresolved({**state,**updates},'Main service error: '+str(exc))}
            history=[*history,deepcopy(reply['assistant'])]
            updates.update(main_history=history,decision_round=updates['decision_round']+1)
            updates['main_usage'].append({'stage':'decide' if not attempt else 'decision_recovery','usage':reply['usage']})
            parsed=None; problem=''
            try:
                if reply['assistant'].get('tool_calls'): raise ValueError('unexpected Main tool calls')
                parsed,notes=parse_decision(message_text(reply['assistant']))
                if notes:
                    updates['delivery_issues'].extend(notes)
                    self.store.event('decision_normalized',scope=state['scope'],notes=notes)
                if parsed.action=='answer':
                    committed=commit_decision(parsed,rows)
                    self.store.event('final_decision',scope=state['scope'],protocol=PROTOCOL,
                        action='answer',status='completed',repair_attempt=attempt,
                        semantics_verified=False,citations=committed['cited_passages'])
                    return {**updates,'decision':committed,'status':'completed',
                            'answer_support':'cited_original_passages_semantics_unverified',
                            'decision_exit_cause':'answered'}
            except ValueError as exc:
                problem=('final_validation_failed: ' if parsed is not None else 'final_format_error: ')+str(exc)[:700]
            if parsed is not None and not problem:
                # All valid nonanswers reach the SAME research/terminal handler.
                if cfg.answer_policy=='best_effort' or parsed.action=='research':
                    reopened=self._reopen(state,updates,parsed)
                    if reopened: return reopened
                if not attempt and rows and cfg.answer_policy=='best_effort':
                    updates['final_review_count']+=1
                    if cfg.research_directions and can_reopen:
                        feedback=('No valid targeted research was selected. Research may still be available. '
                                  'Choose an eligible cell_id suited to the concrete gap and executable queries '
                                  'or known-document read_more; do not default to the first cell. ')
                    else:
                        feedback='No executable research remains. Review the displayed original passages once. '
                    feedback+='Choose the best defensible requested answer with its P citations; incomplete auxiliary confirmation alone is not grounds to abstain. '
                    feedback+='If no defensible requested answer exists, use unresolved and state the concrete gap. Do not invent a candidate.'
                else:
                    explanation=parsed.explanation or 'No defensible requested value was identified in the available sources.'
                    return {**updates,'status':'unresolved','decision':{'action':'unresolved','exact_answer':'',
                        'explanation':explanation,'citations':[]},'answer_support':'none',
                        'decision_exit_cause':'research_exhausted_or_no_defensible_answer'}
            else:
                last_reason=problem
                self.store.event('decision_contract_issue',scope=state['scope'],issue=problem,attempt=attempt)
                if attempt: break
                updates['final_repair_count']+=1
                updates['delivery_issues'].append(problem)
                feedback='Correct the preceding decision using the SAME displayed original passages. '+problem+'. '
                feedback+='Allowed citation IDs: '+','.join(r['display_id'] for r in rows)+'. '
                feedback+='Return JSON with action, exact_answer, citations and explanation. No confidence or extra proof fields. '
                feedback+='Use at most two short sentences in explanation and do not repeat any sentence or phrase. '
                feedback+='Do not invent citations; revise the answer or request a specific available investigation instead.'
            history=[*history,{'role':'user','content':feedback}]
        # One failed correction cannot trigger a recursive LLM loop. A meaningful
        # pending research continuation is still permitted within the global cap.
        if cfg.answer_policy=='best_effort':
            reopened=self._reopen(state,updates)
            if reopened:
                reopened['delivery_issues'].append(last_reason)
                return reopened
        return {**updates,**self._unresolved({**state,**updates},last_reason),
                'decision_exit_cause':'delivery_failed_after_bounded_recovery'}
