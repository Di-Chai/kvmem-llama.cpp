"""Check one GPU-expert neighbor for each independently validated objective."""
import fcntl,json,os,time
import tune as t
r=t.ROOT
(r/'edge-runner.pid').write_text(str(os.getpid()))
deadline=time.monotonic()+8*3600
while not (r/'complete.json').exists():
 if time.monotonic()>deadline:raise RuntimeError('Main search did not finish; no GPU action performed')
 time.sleep(10)
lock=(r/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
try:
 initial=json.loads((r/'complete.json').read_text());t.save(r/'main-complete.json',initial)
 decisions={}
 for field in ['speed_winner','winner']:
  best=initial.get(field,initial['winner']);cfg=dict(best['config'])
  decision=dict(original=best['name'],require_recall=field=='winner',policy='One additional GPU expert layer. Probe <=15950 MiB, full-run <=16000 MiB; require >2% gain in independent 3x512 confirmation to prefer smaller headroom.')
  valid=t.good if field=='winner' else t.resource_valid
  if cfg['ncpu']>0:
   cfg['ncpu']-=1;t.status('checking_expert_neighbor',objective=field,config=cfg)
   probe=t.probe(cfg);decision['probe']=probe
   if probe.get('gpu_mib',float('inf'))<=15950:
    candidate=t.run(cfg,'edge')
    if candidate:initial['candidates'].append(candidate)
    if valid(candidate) and candidate['decode_median']>best['decode_median']*1.02:
     confirm=t.run(cfg,'final_edge')
     if confirm:initial['candidates'].append(confirm)
     if valid(confirm) and confirm['decode_median']>best['decode_median']*1.02:
      best=confirm;decision['selected_edge']=True
  initial[field]=best;decisions[field]=decision
 initial['edge_decisions']=decisions;t.save(r/'complete.json',initial);t.save(r/'edge-complete.json',decisions)
 best=initial['winner'];fast=initial['speed_winner']
 t.status('experiment_complete_extended',winner=best['name'],decode_median=best['decode_median'],speed_winner=fast['name'],speed_median=fast['decode_median'])
finally:
 t.b.stop_server();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
