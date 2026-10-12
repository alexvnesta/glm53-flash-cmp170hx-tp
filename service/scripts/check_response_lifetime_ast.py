"""Execute frozen actual complete() AST against real response/admission classes.

Only the model/tokenizer/job are CPU stand-ins. The complete() body, admission
helper, response wrapper and Starlette/AnyIO ASGI call are actual supplied source.
Never imports the engine, Torch, a model, or accesses a service/network endpoint.
"""
import argparse
import ast
import asyncio
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
import time
import traceback
import uuid

import anyio
from starlette.exceptions import HTTPException
from starlette.responses import StreamingResponse

EXPECTED_HELPER='61379a3b677dd518688445956e54d6e4b4e61a8da5c35d7dadf8f3b05217f62e'
SCOPE={'type':'http','asgi':{'version':'3.0','spec_version':'2.4'},'http_version':'1.1',
       'method':'POST','path':'/v1/completions','headers':[]}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


class Lock(asyncio.Lock):
    def __init__(self):super().__init__();self.releases=0
    def release(self):self.releases+=1;super().release()


async def run_case(api,label,expect_owned,header_failure,helper_dir):
    source=api.read_text();tree=ast.parse(source)
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
              and n.name in ('complete','stream_text')]
    admission=load(helper_dir/'glm_admission.py',label+'_admission')
    cleanup_path=helper_dir/'glm_response_cleanup.py'
    helper=None
    if expect_owned:
        if digest(cleanup_path)!=EXPECTED_HELPER:raise AssertionError('Frozen response helper hash changed')
        helper=load(cleanup_path,label+'_cleanup')
    lock=Lock();jobs=[];cancel_observations=[]
    class AsyncJob:
        def __init__(self,*args,**kwargs):
            assert lock.locked();self.live=True;self.iterations=0;self.cancel_calls=0;jobs.append(self)
        def __aiter__(self):return self
        async def __anext__(self):
            self.iterations+=1
            if self.iterations>1:raise StopAsyncIteration
            return {'stage':'streaming','text':'4','eos':True,'new_tokens':1,
                    'eos_reason':'stop','cached_tokens':0}
        async def cancel(self):
            self.cancel_calls+=1
            cancel_observations.append({'lock_held':lock.locked(),'body_iterations':self.iterations})
            self.live=False
    engine=ModuleType('exllamav3');engine.AsyncJob=AsyncJob
    engine.ComboSampler=lambda **kwargs:object();engine.GreedySampler=lambda:object()
    runtime={'lock':lock,'admission':admission.RequestAdmission(admission.AdmissionPolicy()),
             'args':NS(model_dir='/synthetic/model'),'context_length':393216,
             'generator':NS(error=None),'tokenizer':NS(encode=lambda *a,**k:NS(shape=(1,10))),
             'config':NS(eos_token_id_list=[2]),'draft_options':NS(record_stats=False)}
    scope=dict(runtime=runtime,Path=Path,MODEL_ID='glm',HTTPException=HTTPException,
               NamedToolChoice=type('NamedToolChoice',(),{}),AdmissionError=admission.AdmissionError,
               anyio=anyio,json=json,time=time,uuid=uuid,traceback=traceback,
               StreamingResponse=StreamingResponse)
    if helper:scope.update(OwnedJobCleanup=helper.OwnedJobCleanup,OwnedStreamingResponse=helper.OwnedStreamingResponse)
    for node in selected:node.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(api),'exec'),scope)
    body=NS(store=False,model='glm',tool_choice=None,tools=None,prompt='Synthetic arithmetic fixture',
            messages=None,max_completion_tokens=None,max_tokens=8,min_tokens=0,temperature=0,
            frequency_penalty=0,presence_penalty=0,ignore_eos=False,stop=None,seed=42,
            stream=True,stream_options={'include_usage':True},parallel_tool_calls=True)
    async def disconnected():return False
    request=NS(is_disconnected=disconnected)
    never=asyncio.Event();messages=[]
    async def receive():await never.wait();return {'type':'http.disconnect'}
    async def send(message):
        messages.append(message)
        if header_failure and message['type']=='http.response.start':raise RuntimeError('injected header send failure')
    with patch.dict(sys.modules,{'exllamav3':engine}):
        response=await scope['complete'](body,request,False)
    assert lock.locked() and len(jobs)==1 and jobs[0].live
    error=None
    try:await response(SCOPE,receive,send)
    except BaseException as failure:error=repr(failure)
    result=dict(api=api.name,api_sha256=digest(api),helper_sha256=digest(cleanup_path) if helper else None,
                admission_sha256=digest(helper_dir/'glm_admission.py'),label=label,
                header_failure=header_failure,response_class=type(response).__name__,
                lock_held_after=lock.locked(),release_calls=lock.releases,
                job_live_after=jobs[0].live,job_cancel_calls=jobs[0].cancel_calls,
                body_iterations=jobs[0].iterations,cancel_observations=cancel_observations,
                admission=runtime['admission'].statistics(),response_error=error)
    if header_failure:
        assert error is not None
        assert jobs[0].iterations==0
        if expect_owned:
            assert not lock.locked() and not jobs[0].live and jobs[0].cancel_calls==1 and lock.releases==1
        else:
            assert lock.locked() and jobs[0].live and jobs[0].cancel_calls==0 and lock.releases==0
            # Dispose only this CPU stand-in after saving the leaked ownership observations.
            await jobs[0].cancel();lock.release()
    else:
        assert error is None and not lock.locked() and not jobs[0].live
        assert jobs[0].cancel_calls==1 and lock.releases==1
        payloads=[]
        for message in messages:
            raw=message.get('body',b'').decode()
            if raw.startswith('data: {'):payloads.append(json.loads(raw[6:].strip()))
        result['semantic_payloads']=[{k:v for k,v in payload.items() if k in ('choices','usage')}
                                     for payload in payloads]
        assert any(payload.get('choices',[{}])[0].get('text')=='4'
                   for payload in payloads if payload.get('choices'))
    return result


def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,default=repo/'tests/fixtures/response_before_cleanup.py.txt')
    p.add_argument('--candidate',action='append',type=Path,help='Actual API file; repeat to compare LS and TP copies')
    p.add_argument('--helper-dir',type=Path,default=repo/'glm')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    args.candidate=args.candidate or [repo/'glm/glm_api.py']
    metadata=json.loads((repo/'tests/fixtures/response_before_cleanup.json').read_text())
    if args.baseline.resolve()==(repo/'tests/fixtures/response_before_cleanup.py.txt').resolve():
        if digest(args.baseline)!=metadata['fixture_sha256']:
            raise AssertionError('Historical API function fixture drifted')
    async def run():
        results=[]
        for label,path,owned in [('baseline',args.baseline,False),
                                 *[(f'candidate{i}',path,True) for i,path in enumerate(args.candidate)]]:
            for failure in (True,False):results.append(await run_case(path.resolve(),label,owned,failure,args.helper_dir.resolve()))
        normal=[r['semantic_payloads'] for r in results if not r['header_failure']]
        assert all(payloads==normal[0] for payloads in normal)
        return {'passed':True,'tests':len(results),'qualification':'actual frozen complete AST with real Starlette/admission/cleanup, CPU job stand-ins',
                'python':sys.version.split()[0],'starlette':version('starlette'),'anyio':version('anyio'),
                'torch_imported':'torch' in sys.modules,'normal_stream_semantic_payloads_equal':True,'cases':results}
    result=asyncio.run(run())
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':main()
