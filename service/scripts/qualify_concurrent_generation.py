#!/usr/bin/env python3
"""Bounded actual-batching client. Run it only in a guarded trial.

Records raw HTTP/SSE/health and per-request choices; no service mutation. A
passing semantic/lifecycle scope is separate from whole-choice equivalence.
"""
import argparse,concurrent.futures,copy,hashlib,json,pathlib,re,threading,time,urllib.request

BAD_PIECES=('<|assistant|>','<|user|>','<|observation|>','<|im_start|>','<think>')

def visible(response):
    return response['choices'][0].get('message',{}).get('content','') or ''

def valid_count_prefix(content,reason):
    if not content or any(piece in content for piece in BAD_PIECES):return False
    lines=content.strip().splitlines()
    for index,line in enumerate(lines):
        if line==str(index+1):continue
        if index==len(lines)-1 and reason=='length' and line.isdigit() and str(index+1).startswith(line):return True
        return False
    return bool(lines)


def validate(case,response):
    choices=response.get('choices',[])
    if len(choices)!=1 or choices[0].get('finish_reason') not in ('stop','length'):return False
    message=choices[0].get('message',{})
    if any(piece in (message.get('content','') or '')+(message.get('reasoning_content','') or '') for piece in BAD_PIECES):return False
    content=visible(response).strip();oracle=case['oracle']
    if oracle['kind']=='integer':return re.fullmatch(r'\s*'+str(oracle['value'])+r'\s*[.!]?\s*',content) is not None
    if oracle['kind']=='ordered_markers':
        values=oracle['values'];found=re.findall(r'[A-Z]+-\d{4}',content)
        return found==values
    raise ValueError('Unsupported bounded oracle')

class Client:
    def __init__(self,base,out,timeout):self.base=base.rstrip('/');self.out=out;self.timeout=timeout;self.lock=threading.Lock();self.serial=0
    def save(self,name,value):
        (self.out/name).write_text(json.dumps(value,indent=2)+'\n')
    def health(self,tag):
        with urllib.request.urlopen(self.base+'/health',timeout=min(15,self.timeout)) as r:body=json.load(r)
        self.save(tag+'_health.json',body);return body
    def post(self,body,tag):
        self.save(tag+'_request.json',body);started=time.monotonic()
        req=urllib.request.Request(self.base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=self.timeout) as r:raw=r.read();status=r.status
        (self.out/(tag+'_raw.json')).write_bytes(raw);answer=json.loads(raw)
        self.save(tag+'_receipt.json',dict(status=status,wall_ms=(time.monotonic()-started)*1000,request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()))
        return answer
    def stream(self,body,tag,first,cancel=None):
        self.save(tag+'_request.json',body);start=time.monotonic();events=[];text=[];thought=[];first_ms=None;usage=None;reason=None
        req=urllib.request.Request(self.base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        raw=[]
        with urllib.request.urlopen(req,timeout=self.timeout) as response:
            for line in response:
                raw.append(line)
                if not line.startswith(b'data: '):continue
                data=line[6:].strip()
                if data==b'[DONE]':break
                event=json.loads(data);events.append(event)
                if event.get('usage'):usage=event['usage']
                for choice in event.get('choices',[]):
                    delta=choice.get('delta',{})
                    if choice.get('finish_reason'):reason=choice['finish_reason']
                    content=delta.get('content','') or '';thinking=delta.get('reasoning_content','') or ''
                    if content or thinking:
                        if first_ms is None:first_ms=(time.monotonic()-start)*1000;first.set()
                    text.append(content);thought.append(thinking)
                if cancel is not None and cancel.is_set():break
        (self.out/(tag+'_raw.sse')).write_bytes(b''.join(raw))
        result=dict(events=events,content=''.join(text),reasoning_content=''.join(thought),first_text_ms=first_ms,wall_ms=(time.monotonic()-start)*1000,
                    usage=usage,finish_reason=reason,client_disconnect=bool(cancel is not None and cancel.is_set()))
        self.save(tag+'_receipt.json',result);return result
    def idle(self,tag):
        deadline=time.monotonic()+min(self.timeout,120)
        while time.monotonic()<deadline:
            h=self.health(tag)
            if not h.get('busy') and h.get('active_sequences',1)==0 and h.get('pending_sequences',1)==0:return h
            time.sleep(.1)
        raise RuntimeError('Service did not settle to owned idle')


def campaign(client,fixture,repeats=1,cache_tokens=393216,context_length=392960):
    checks=[];comparisons=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    initial=client.idle('initial')
    check('explicit_concurrency',initial.get('experimental_concurrent_generation') is True)
    check('resident_TP_and_capacity',initial.get('target_tp') is True and initial.get('max_active_sequences')==2)
    check('no_paired_retention',initial.get('prompt_cache') is None and initial.get('target_cpu_cache_budget_bytes')==0)
    check('full_cache_geometry',initial.get('cache_tokens')==cache_tokens and initial.get('context_length')==context_length)
    check('fixed_K7',initial.get('draft_num_tokens')==7 and initial.get('dynamic_draft_tokens') is False)
    if not all(c['passed'] for c in checks):
        return dict(schema=1,qualified=False,collection_completed=False,checks=checks,comparisons=[],whole_choice_equivalence=False,initial_health=initial,scope='Startup profile refused before inference')
    serial={}
    for case in fixture['parallel']:
        response=client.post(case['request'],'serial_'+case['id']);serial[case['id']]=response
        check('serial_oracle_'+case['id'],validate(case,response))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for round in range(repeats):
            barrier=threading.Barrier(2)
            def run(case):barrier.wait(timeout=15);return client.post(case['request'],f'parallel_{round}_{case["id"]}')
            pending=[pool.submit(run,case) for case in fixture['parallel']]
            for case,future in zip(fixture['parallel'],pending):
                response=future.result(timeout=client.timeout+15)
                check(f'parallel_{round}_oracle_{case["id"]}',validate(case,response))
                comparisons.append(dict(id=case['id'],round=round,whole_choices_equal=response['choices']==serial[case['id']]['choices'],
                                        token_counts_equal=all(response.get('usage',{}).get(k)==serial[case['id']].get('usage',{}).get(k) for k in ('prompt_tokens','completion_tokens','total_tokens'))))
        # Arrival during a real holder stream, followed by a complete long prompt.
        first=threading.Event();holder=pool.submit(client.stream,fixture['holder'],'mixed_holder',first)
        check('mixed_holder_text_before_arrival',first.wait(timeout=45))
        response=client.post(fixture['mixed_prefill']['request'],'mixed_prefill')
        check('mixed_prefill_oracle',validate(fixture['mixed_prefill'],response))
        hresult=holder.result(timeout=client.timeout+15)
        check('mixed_holder_complete',hresult['finish_reason'] in ('stop','length') and hresult['usage'] is not None)
        check('mixed_holder_count_prefix',valid_count_prefix(hresult['content'],hresult['finish_reason']))
        check('mixed_holder_no_control_pieces',not any(piece in hresult['content']+hresult['reasoning_content'] for piece in BAD_PIECES))
        client.idle('before_cancel')
        # Cancel one unfinished SSE while the second still has its own producer.
        a_first,b_first,cancel=threading.Event(),threading.Event(),threading.Event()
        a=pool.submit(client.stream,fixture['holder'],'cancel_A',a_first,cancel)
        b=pool.submit(client.stream,fixture['holder'],'survivor_B',b_first)
        check('both_streams_produced_before_cancel',a_first.wait(45) and b_first.wait(45))
        cancel.set();ares=a.result(timeout=client.timeout+15)
        check('cancel_was_unfinished',ares['client_disconnect'] and ares['finish_reason'] is None)
        response=client.post(fixture['slot_reuse']['request'],'reuse_C')
        check('reused_slot_arithmetic',validate(fixture['slot_reuse'],response))
        bres=b.result(timeout=client.timeout+15)
        check('unrelated_survivor_completed',bres['finish_reason'] in ('stop','length') and bres['usage'] is not None)
        check('survivor_count_prefix',valid_count_prefix(bres['content'],bres['finish_reason']))
        check('survivor_no_control_pieces',not any(piece in bres['content']+bres['reasoning_content'] for piece in BAD_PIECES))
    final=client.idle('final');ring=final.get('concurrent_ring',{})
    check('real_batched_draft_observed',ring.get('peak_draft_batch')==2 and ring.get('batched_draft_rounds',0)>0)
    check('native_batched_verification_observed',ring.get('peak_target_batch')==2 and ring.get('batched_verifier_rounds',0)>0)
    check('generation_and_prefill_overlap_observed',ring.get('mixed_prefill_steps',0)>0)
    check('all_ring_leases_released',ring.get('active_ring_leases')==0)
    check('ring_slot_reuse_observed',ring.get('reused_slots',0)>initial.get('concurrent_ring',{}).get('reused_slots',0))
    return dict(schema=1,qualified=all(c['passed'] for c in checks),checks=checks,comparisons=comparisons,
                whole_choice_equivalence=all(c['whole_choices_equal'] for c in comparisons),initial_health=initial,final_health=final,
                collection_completed=True,scope='Bounded concurrency semantics/lifecycle; whole-choice equality reported separately; no universal losslessness or speed claim')

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:8013');p.add_argument('--fixture',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--request-timeout',type=float,default=180);p.add_argument('--repeats',type=int,default=1)
    # Expected /health geometry: a k_idx ring pool (-cs 655360 --context-cap 392960) reports a larger cache_tokens
    p.add_argument('--expect-cache-tokens',type=int,default=393216);p.add_argument('--expect-context-length',type=int,default=392960)
    args=p.parse_args()
    if not 1<=args.repeats<=3 or not 15<=args.request_timeout<=300:p.error('Bounded repeats1..3/timeouts15..300 required')
    args.output.mkdir(parents=True,exist_ok=False);fixture=json.loads(args.fixture.read_text());client=Client(args.base_url,args.output,args.request_timeout)
    try:
        report=campaign(client,fixture,args.repeats,args.expect_cache_tokens,args.expect_context_length)
    except Exception as error:
        report=dict(schema=1,qualified=False,collection_completed=False,error_type=type(error).__name__,error=str(error))
    report['fixture_sha256']=hashlib.sha256(args.fixture.read_bytes()).hexdigest();client.save('report.json',report)
    print(json.dumps({'qualified':report['qualified'],'whole_choice_equivalence':report.get('whole_choice_equivalence')}))
    return 0 if report['qualified'] else 2
if __name__=='__main__':raise SystemExit(main())
