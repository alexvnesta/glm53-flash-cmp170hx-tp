"""Only benchmark clients are interrupted. Public router traffic always wins."""
import datetime, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

LAB=Path('/data2/llm-models/glm53-stream-usage-20261017')
PY=LAB/'bench-venv/bin/python'
SCRIPT=LAB/'llm_decode_bench-v0.8.0.py'
OBSERVER=LAB/'observe.py'
OUT=LAB/'runs';OUT.mkdir(exist_ok=True)
EXPECTED=json.loads((LAB/'cancel-fix/after-health.json').read_text())['profile_digest']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def get(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=5) as r:return json.load(r)
def psi():return float(re.search(r'avg10=([\d.]+)',Path('/proc/pressure/cpu').read_text()).group(1))
def check():
    h=get(8014);router=get(8012).get('switcher',{})
    if router.get('active_model')!='glm' or router.get('inflight') or router.get('waiting'):
        raise RuntimeError('Public client arrived; yield benchmark requests')
    if h.get('profile_digest')!=EXPECTED or h.get('status')!='ok':raise RuntimeError('Serving identity changed')
    if psi()>=2:raise RuntimeError('CPU pressure above qualification threshold')
    return h
def idle(timeout=1200):
    began=time.monotonic();stable=0
    while time.monotonic()-began<timeout:
        try:
            h=check()
            clean=not any(h.get(k) for k in ('busy','active_sequences','pending_sequences','admitted_requests'))
            stable=stable+1 if clean else 0
            if stable>=3:return h
        except Exception:stable=0
        time.sleep(2)
    raise TimeoutError('No quiet window; clients left running')
def cancel(p):
    p.send_signal(signal.SIGINT)
    try:p.wait(timeout=8)
    except subprocess.TimeoutExpired:
        p.terminate()
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:p.kill();p.wait()

def run(label,effort,context,finite=False,smoke=False):
    for attempt in range(1,7):
        folder=OUT/(label+f'-attempt{attempt}')
        if folder.exists():
            receipt_path=folder/'receipt.json'
            if receipt_path.is_file():
                old=json.loads(receipt_path.read_text())
                if old.get('accepted') and old.get('profile_digest')==EXPECTED:
                    return
            continue
        idle()
        folder.mkdir()
        began=now();start=time.monotonic();samples=[]
        command=[str(PY),str(OBSERVER),str(SCRIPT),'--','--host','http://127.0.0.1:8014',
                 '--model','GLM-5.3-Flash-'+effort,'--concurrency','1' if smoke else '1,2,4',
                 '--contexts',str(context),'--prefill-contexts',str(context),'--skip-prefill',
                 '--kv-budget','655360','--max-tokens','512' if finite or smoke else '2048',
                 '--temperature','0','--duration','3' if smoke else '30','--token-targeting','exact',
                 '--decode-warmup-seconds','0' if smoke else '3',
                 '--display-mode','plain','--no-resume','--output','benchmark_results.json']
        if finite:command+=['--request-count','4','--warmup-request-count','1']
        environment=os.environ.copy();environment['LLM_BENCH_NO_UPDATE_CHECK']='1'
        environment['LLM_BENCH_EVENT_FILE']=str(folder/'events.jsonl')
        with (folder/'run.log').open('w') as output:
            p=subprocess.Popen(command,cwd=folder,stdout=output,stderr=subprocess.STDOUT,env=environment)
            error=None
            try:
                while p.poll() is None:
                    h=check();samples.append({'at':now(),'psi_cpu_avg10':psi(),
                        'active_sequences':h['active_sequences'],'pending_sequences':h['pending_sequences'],
                        'efforts':sorted(set(r['reasoning_effort'] for r in h['request_metrics']['active']))})
                    if time.monotonic()-start>1800:raise TimeoutError('Benchmark group exceeded 30 minutes')
                    time.sleep(.5)
                if p.returncode:raise RuntimeError('Benchmark process failed; inspect retained run.log')
                scan=subprocess.run(['journalctl','-u','glm-qwen-switcher.service','--since',began,
                                     '--no-pager','-o','cat','--grep','"POST '],capture_output=True,text=True)
                if scan.returncode not in (0,1):raise RuntimeError('Cannot audit public traffic')
                if scan.stdout.strip():raise RuntimeError('Public POST during group; exclude and yield')
                report=json.loads((folder/'benchmark_results.json').read_text())
                cells=report['results'];assert len(cells)==(1 if smoke else 3)
                for cell in cells:
                    assert cell['aggregate_source']==('openai_completed_usage' if finite else 'openai_continuous_usage'),cell
                    assert cell['aggregate_tps']>0 and cell['num_errors']==0 and not cell.get('failure_reason'),cell
                    assert not cell['warmup_timed_out'] and not cell.get('capacity_limited'),cell
                aliases={v for s in samples for v in s['efforts']};assert aliases=={effort},aliases
                idle(timeout=120)
            except BaseException as exc:
                error=str(exc)
                if p.poll() is None:cancel(p)
            receipt={'label':label,'attempt':attempt,'effort':effort,'context':context,'finite':finite,
                     'started':began,'finished':now(),'wall_seconds':time.monotonic()-start,
                     'profile_digest':EXPECTED,'command':command,'exit_code':p.returncode,
                     'accepted':error is None,'error':error,'health_samples':samples}
            (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k] for k in ('label','attempt','accepted','error','wall_seconds')}),flush=True)
        if error is None:return
        if not any(x in error for x in ('Public','CPU pressure')):raise RuntimeError(error)
    raise RuntimeError('Repeated external activity; retained excluded attempts')

if sys.argv[1]=='smoke':run('smoke','low',0,smoke=True)
else:
    for context in (0,32768,131072):
        for effort in ('low','high','max'):
            run(f'{effort}-ctx{context}',effort,context)
    for effort in ('low','high','max'):run(f'{effort}-finite-ctx0',effort,0,finite=True)
    print('ALL COMMUNITY BENCHMARK GROUPS COMPLETE',flush=True)
