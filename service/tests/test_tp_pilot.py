"""Offline HTTP workflow tests. No sockets or model imports."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import qualify_tp_session_cache as runner

from qualify_tp_session_cache import Client,content,qualify,signature


class ClientDouble:
    def __init__(self,early_stop=False,cache_tokens=65536,content_tokens=40000):
        self.cache_tokens=cache_tokens;self.content_tokens=content_tokens
        self.tag='initial';self.early_stop=early_stop;self.calls=[]
    def health(self,tag='health'):
        step=tag.replace('_health','')
        before=step in ('initial','cold_A')
        return {'status':'ok','busy':False,'target_tp':True,'experimental_tp_dflash2':True,'cache_tokens':self.cache_tokens,'context_length':self.cache_tokens-256,
                'prompt_cache':{'mode':'tp_multi_session','hits':0 if step=='initial' else 2,
                  'completed_checkpoints':0 if step=='initial' else 2,
                  'paired_host_bytes':400000000,'paired_budget_bytes':1073741824,
                  'target_cpu_tier':{'reservation_policy':'lazy_exact_registered_rank_slabs_on_gpu_eviction',
                     'pinned_bytes':0 if before else 1000000,'reserved_budget_bytes':1073741824,
                     'metrics':{'pushes':0 if before else 8,'restores':0 if step=='cold_B' else 3}}}}
    def request(self,method,path,payload,tag):
        self.calls.append(tag)
        warm=tag in ('resume_A','resume_B','recover_B')
        label=payload['label']
        return {'choices':[{'message':{'role':'assistant','content':'stable_'+label},
                            'finish_reason':'stop' if self.early_stop else 'length'}],
                'usage':{'prompt_tokens':self.content_tokens if label in ('A','B') else 4096,
                         'completion_tokens':3 if self.early_stop else 128,
                         'total_tokens':(self.content_tokens if label in ('A','B') else 4096)+(3 if self.early_stop else 128),
                         'prompt_tokens_details':{'cached_tokens':self.content_tokens//256*256 if warm else 0}}}
    def cancel_stream(self,payload):
        self.calls.append('cancel_D');return self.health('cancel_recovery_health')


class HTTPWorkflow(unittest.TestCase):
    def fixture(self): return {'requests':{x:{'label':x} for x in 'ABCD'}}
    def test_complete_workflow_and_early_stop(self):
        for early in (False,True):
            c=ClientDouble(early)
            self.assertTrue(qualify(c,self.fixture())['qualified'])
            self.assertEqual(c.calls,['cold_A','cold_B','resume_A','cold_C','resume_B','cancel_D','recover_B'])
    def test_no_actual_pressure_refuses(self):
        c=ClientDouble();original=c.health
        def health(tag):
            r=original(tag);r['prompt_cache']['target_cpu_tier']['metrics']['pushes']=0
            return r
        c.health=health
        with self.assertRaises(AssertionError):qualify(c,self.fixture())
    def test_output_mismatch_refuses(self):
        c=ClientDouble();original=c.request
        def request(*args,**kw):
            r=original(*args,**kw)
            if kw['tag']=='resume_A':r['choices'][0]['message']['content']='changed'
            return r
        c.request=request
        with self.assertRaisesRegex(AssertionError,'output mismatch'):qualify(c,self.fixture())
    def test_production_port_refused(self):
        with self.assertRaisesRegex(ValueError,'8012'):Client('http://127.0.0.1:8012',None)
    def test_distinct_session_tags_before_shared_ledger(self):
        a,b=content('A',2),content('B',2)
        self.assertNotEqual(a.splitlines()[0],b.splitlines()[0])
        self.assertEqual(a.splitlines()[1:-1],b.splitlines()[1:-1])

    def test_full384_workflow_uses_explicit_geometry(self):
        client=ClientDouble(early_stop=True,cache_tokens=393216,content_tokens=220000)
        report=qualify(client,self.fixture(),393216,220000)
        self.assertTrue(report['qualified'])
        self.assertEqual(report['cache_tokens'],393216)
        self.assertEqual(report['session_content_tokens'],220000)

    def test_wrong_live_geometry_refuses(self):
        with self.assertRaisesRegex(AssertionError,'geometry'):
            qualify(ClientDouble(),self.fixture(),393216,220000)

    def test_campaign_geometry_defaults_and_full384(self):
        self.assertEqual(runner.campaign_geometry(),65280)
        self.assertEqual(runner.campaign_geometry(393216,220000,300),392960)

    def test_insufficient_pressure_or_context_headroom_refuses(self):
        for tokens in (1,32640,65000,220000):
            with self.subTest(tokens=tokens),self.assertRaises(ValueError):
                runner.campaign_geometry(65536,tokens)

    def test_invalid_page_geometry_and_timeout_refuse(self):
        for cache in (0,8191,65537,True,65536.0):
            with self.subTest(cache=cache),self.assertRaises(ValueError):
                runner.campaign_geometry(cache,40000)
        for timeout in (0,-1,601,True,1.5):
            with self.subTest(timeout=timeout),self.assertRaises(ValueError):
                runner.campaign_geometry(request_timeout=timeout)

    def test_large_fixture_calibration_is_bounded_and_saved(self):
        class TokenizerDouble:
            def __init__(self,out):self.out=out;self.calls=[]
            def request(self,method,path,payload,tag):
                rows=len(payload['content'].splitlines())-2
                self.calls.append((tag,rows))
                return {'token_length':rows*30}
        with tempfile.TemporaryDirectory() as tmp:
            client=TokenizerDouble(Path(tmp))
            fixture=runner.build_fixture(client,220000)
            self.assertTrue(any(rows>2000 for tag,rows in client.calls if tag.startswith('tokenize_A')))
            self.assertLessEqual(len(client.calls),56)
            for label in ('A','B'):
                self.assertLessEqual(abs(fixture['tokenized_content_lengths'][label]-220000),40)
            self.assertEqual(json.loads((Path(tmp)/'fixture.json').read_text()),fixture)
            self.assertTrue((Path(tmp)/'fixture.sha256').is_file())
            self.assertEqual(fixture['requests']['A']['max_tokens'],128)
            self.assertEqual(fixture['requests']['A']['seed'],12345)
            self.assertEqual(fixture['requests']['A']['reasoning_effort'],'low')

    def test_configured_http_timeout_and_fixed_health_timeout(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner.http.client,'HTTPConnection') as connection:
            response=connection.return_value.getresponse.return_value
            response.status=200;response.read.return_value=b'{}';response.getheaders.return_value=[]
            client=Client('http://127.0.0.1:8013',Path(tmp),300)
            client.request('GET','/check')
            self.assertEqual(connection.call_args.kwargs['timeout'],300)
            client.health()
            self.assertEqual(connection.call_args.kwargs['timeout'],10)

    def test_configured_cancel_timeout_and_saved_headers(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner.http.client,'HTTPConnection') as connection:
            response=connection.return_value.getresponse.return_value
            response.status=200;response.getheaders.return_value=[]
            response.readline.return_value=b'data: {"choices":[{"delta":{"content":"x"}}]}\n'
            client=Client('http://127.0.0.1:8013',Path(tmp),300)
            client.health=lambda tag:{'status':'ok','busy':False,'target_tp':True,'experimental_tp_dflash2':True}
            client.cancel_stream({'messages':[]})
            self.assertEqual(connection.call_args.kwargs['timeout'],300)
            self.assertEqual(json.loads((Path(tmp)/'cancel_http.json').read_text())['status'],200)


if __name__=='__main__':unittest.main()
