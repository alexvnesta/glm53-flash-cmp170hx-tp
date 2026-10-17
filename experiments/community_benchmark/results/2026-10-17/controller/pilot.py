import datetime,json,subprocess,time
from pathlib import Path
LAB=Path('/data2/llm-models/glm53-stream-usage-20261017')
PY=LAB/'bench-venv/bin/python'
start=time.monotonic()
while not (LAB/'cancel-fix/after-health.json').exists():
    if time.monotonic()-start>900:raise TimeoutError('Qualified service activation did not finish')
    time.sleep(2)
with (LAB/'benchmark-controller.log').open('a',buffering=1) as output:
    for script,arguments in [('qualify.py',[]),('run_guarded.py',['smoke']),('run_guarded.py',['full'])]:
        qualification=LAB/'live-stream-qualification-v2.json'
        if script=='qualify.py' and qualification.exists():
            old=json.loads(qualification.read_text())
            expected=json.loads((LAB/'cancel-fix/after-health.json').read_text())['profile_digest']
            if old.get('passed') and old.get('profile_digest')==expected:continue
        output.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'begin':script,'arguments':arguments})+'\n')
        subprocess.run([str(PY),str(LAB/script),*arguments],stdout=output,stderr=subprocess.STDOUT,check=True)
    output.write('ALL QUALIFICATION AND BENCHMARK WORK COMPLETE\n')
