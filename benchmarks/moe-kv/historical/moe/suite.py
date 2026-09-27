"""Resume completed configurations; keep all failed-probe logs."""
import json,time
from pathlib import Path
import experiment as e

def fitted(context,budget,start):
    p=e.OUT/f'fit_c{context+1024}_k{budget or 0}.json'
    if p.exists():
        x=json.loads(p.read_text())
        if x.get('best_ncpu') is not None:return x['best_ncpu']
    print('PHASE fit '+str((context,budget,start)),flush=True)
    return e.fit(context+1024,budget,start)

def case(name,context,ncpu,budget):
    p=e.OUT/name/'summary.json'
    if p.exists():return json.loads(p.read_text())
    print('PHASE benchmark '+str((name,context,ncpu,budget)),flush=True)
    return e.run_case(name,context,ncpu,budget)

try:
    while not (e.ROOT/'download.done').exists():
        time.sleep(5)
    while not (e.OUT/'sanity.json').exists():
        pid=int((e.ROOT/'sanity.pid').read_text())
        if not Path(f'/proc/{pid}').exists():raise RuntimeError('Sanity failed; inspect logs/sanity.log')
        time.sleep(5)
    sanity=json.loads((e.OUT/'sanity.json').read_text())
    if not (e.ROOT/'validation.done').exists():raise RuntimeError('Numerical validation has not been reviewed')
    time.sleep(5)
    base64=fitted(65536,None,24)
    k16=fitted(65536,16384,16)
    k32=fitted(65536,32768,18)
    results=[]
    results.append(case('full_64k',65536,base64,None))
    results.append(case('k16_64k_control',65536,base64,16384))
    results.append(case('k16_64k',65536,k16,16384))
    results.append(case('k32_64k',65536,k32,32768))
    eligible=[r for r in results if r['budget'] and all(r['recall'].values())]
    if not eligible:raise RuntimeError('No KVMem configuration passed recall; investigate')
    best=max(eligible,key=lambda r:r['decode_tps_median'])
    base128=fitted(131072,None,34)
    results.append(case('full_128k',131072,base128,None))
    results.append(case('best_128k',131072,best['ncpu'],best['budget']))
    e.save(e.OUT/'suite-summary.json',results)
    (e.ROOT/'suite.done').write_text('completed\n')
finally:
    e.stop_server()
