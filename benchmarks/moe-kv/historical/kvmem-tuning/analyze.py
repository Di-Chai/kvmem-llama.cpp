"""Audit completed full-context requests and summarize without treating capacity probes as speed results."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPECTED='4515ef3274b39b3cb6a3b04920793eb1b59f7dea36e01d521327d3027ed68df9'
rows=[];checks=[]
for p in sorted((ROOT/'results').glob('*/summary.json')):
 s=json.loads(p.read_text());assert s['prompt_sha256']==EXPECTED
 assert s['outputs']==[512]*3 and s['prompt_tokens']==260992
 for i in range(3):
  r=json.loads((p.parent/f'rep{i}.json').read_text())
  req=json.loads((p.parent/f'rep{i}.request.json').read_text())
  assert hashlib.sha256(json.dumps(req['messages']).encode()).hexdigest()==EXPECTED
  assert r['error'] is None and r['finish']=='length'
  assert r['usage']['prompt_tokens']==260992 and r['usage']['completion_tokens']==512
  assert r['timings']['prompt_n']==260992 if i==0 else r['timings']['cache_n']>=260096
  assert r['timings']['cache_n']+r['timings']['prompt_n']==260992
  assert r['memory_checks']['process_swap_peak']==0
  assert r['peaks']['gpu_mib']<=16000
  assert r['timings']['predicted_per_second']==s['decode_tps'][i]
  events=json.loads((p.parent/f'rep{i}.events.json').read_text())
  times=[e['elapsed_s'] for e in events if any(c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning_content') for c in e['event'].get('choices',[]))]
  assert len(times)==512,'Unexpected stream coalescing; review token-rate crosscheck'
  client=511/(times[-1]-times[0]);delta=100*(client/r['timings']['predicted_per_second']-1)
  assert abs(delta)<3
  checks.append(dict(name=s['name'],rep=i,client_tps=client,server_tps=r['timings']['predicted_per_second'],difference_percent=delta))
 assert statistics.median(s['decode_tps'])==s['decode_median']
 rows.append(s)
rows.sort(key=lambda x:x['decode_median'],reverse=True)
failures=[dict(name=p.parent.name,**json.loads(p.read_text())) for p in (ROOT/'results').glob('*/failure.json')]
report=dict(status='measurement_integrity_passed',completed_cases=len(rows),audited_requests=len(checks),checks=checks,failures=failures)
(ROOT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'ranking.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['name','budget','kv','cpu_expert_layers','gpu_expert_layers','threads','ubatch','decode_median','decode_rounds','cold_prefill','gpu_peak_GiB','rss_peak_GiB','recall_pass'])
 for s in rows:
  c=s['config'];w.writerow([s['name'],c['budget'],c['kv'],c['ncpu'],48-c['ncpu'],c['threads'],c['ub'],s['decode_median'],json.dumps(s['decode_tps']),s['cold_prefill_tps'],s['peaks']['gpu_mib']/1024,s['peaks']['rss']/2**30,all(s['quality'].values())])
print(json.dumps({k:report[k] for k in ['status','completed_cases','audited_requests','failures']},indent=2))
