"""Bounded KVMem search; every scored case runs real cold 260,992-token prefill."""
import fcntl, hashlib, importlib.util, json, math, os, resource, statistics, subprocess, time
from pathlib import Path
ROOT=Path.home()/'kvmem-tuning-256k'
OUT=ROOT/'results'
ROOT.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('bench',Path.home()/'kvmem-moe-256k/bench256.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.ROOT=ROOT;b.OUT=OUT;b.URL='http://127.0.0.1:1942'
MSG=json.loads((Path.home()/'kvmem-moe-256k/prompt.json').read_text())
HASH=hashlib.sha256(json.dumps(MSG).encode()).hexdigest()
KV={'q8q8':('q8_0','q8_0'),'q4q8':('q4_0','q8_0'),'q4q4':('q4_0','q4_0')}
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
WARM=[{'role':'user','content':'Capacity planning review. '*200+b.previous.WARM[0]['content']}]

def save(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):
 d=dict(time=time.time(),stage=stage,**kw);save(ROOT/'status.json',d);print('STATUS '+json.dumps(d),flush=True)
def sample(pid):
 ps=b.previous.psutil;mem=ps.virtual_memory();proc=ps.Process(pid);sw=ps.swap_memory()
 vms=next(int(s.split()[1])*1024 for s in Path(f'/proc/{pid}/status').read_text().splitlines() if s.startswith('VmSwap:'))
 vals=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory','--format=csv,noheader,nounits'],text=True).strip().split(',')
 return dict(t=time.time(),rss=proc.memory_info().rss,system_used=mem.total-mem.available,available=mem.available,swap=sw.used,process_swap=vms,system_swap_in=sw.sin,system_swap_out=sw.sout,gpu_mib=float(vals[0]),gpu_util=float(vals[1]),gpu_temp=float(vals[2]),gpu_watts=float(vals[3]),gpu_graphics_mhz=float(vals[4]),gpu_memory_mhz=float(vals[5]))
b.previous.sample=sample

def config(budget,kv='q8q8',ncpu=20,t=6,ub=512):
 return dict(budget=budget,kv=kv,ncpu=ncpu,threads=t,tb=6,ub=ub,reserve=4096)
def tag(c):
 return f"k{c['budget']//1024}_{c['kv']}_n{c['ncpu']}_t{c['threads']}_u{c['ub']}"
def launch(name,c):
 b.stop_server();d=OUT/name;d.mkdir(exist_ok=True);b.LOG=d/'server.log'
 k,v=KV[c['kv']]
 cmd=[str(b.KVMEM),'-m',str(b.MODEL),'--alias','Qwen3-MoE','--host','127.0.0.1','--port','1942','-c','262144','-ngl','99','--n-cpu-moe',str(c['ncpu']),'-t',str(c['threads']),'-tb',str(c['tb']),'-b',str(max(512,c['ub'])),'-ub',str(c['ub']),'-fa','on','-ctk',k,'-ctv',v,'--load-mode','none','--threads-http','4','--timeout','7200','--spec-type','none','--no-ui','--kvmem-budget',str(c['budget']),'--kvmem-gen-reserve',str(c['reserve']),'--no-kvmem-trace']
 env=b.ENV.copy()
 for key in list(env):
  if key.startswith(('GGML_','LLAMA_','KVMEM_')) or key.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(key,None)
 env['GGML_OP_OFFLOAD_MIN_BATCH']='32'
 b.ACTIVE=dict(name=name,config=c,command=cmd,env_overrides={'GGML_OP_OFFLOAD_MIN_BATCH':'32'})
 save(d/'config.json',b.ACTIVE)
 with b.LOG.open('w') as f:b.PROCESS=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 (ROOT/'server.pid').write_text(str(b.PROCESS.pid));status('starting',name=name,config=c,server_pid=b.PROCESS.pid)
 for _ in range(240):
  if b.PROCESS.poll() is not None:raise RuntimeError(f'server exited {b.PROCESS.returncode}')
  try:
   if b.get('/health').get('status')=='ok':
    save(d/'props.json',b.get('/props'));return
  except (OSError,ValueError):pass
  time.sleep(1)
 raise RuntimeError('server startup timeout')

def request(name,msgs=MSG,tokens=512,reset=False,full=True):
 status('measuring',name=name,config=b.ACTIVE['config'],server_pid=b.PROCESS.pid,full_context=full,reset=reset)
 r=b.request(name,msgs,tokens,reset)
 samples=[s for s in json.loads((OUT/(name+'.samples.json')).read_text()) if 'error' not in s]
 assert samples,'No valid resource samples'
 r['memory_checks']=dict(process_swap_peak=max(s['process_swap'] for s in samples),system_swap_in_delta=samples[-1]['system_swap_in']-samples[0]['system_swap_in'],system_swap_out_delta=samples[-1]['system_swap_out']-samples[0]['system_swap_out'])
 times=[e['elapsed_s'] for e in json.loads((OUT/(name+'.events.json')).read_text()) if any(c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning_content') for c in e['event'].get('choices',[]))]
 r['content_event_count']=len(times)
 if len(times)==tokens:r['client_decode_tps']=(tokens-1)/(times[-1]-times[0])
 save(OUT/(name+'.json'),r)
 if full:
  assert r['usage']['prompt_tokens']==260992,r['usage']
  assert r['usage']['completion_tokens']==tokens and r['finish']=='length',r['usage']
  assert r['prompt_sha256']==HASH
  assert r['timings']['prompt_n']==260992 if reset else r['timings']['cache_n']>=260096,r['timings']
  assert r['peaks']['gpu_mib']<=16000,'Insufficient GPU headroom'
  assert r['memory_checks']['process_swap_peak']==0,'Model process swapped'
  if 'client_decode_tps' in r:assert abs(r['client_decode_tps']/r['timings']['predicted_per_second']-1)<.03,'Client/server throughput mismatch'
 return r

def failed(name,e):
 d=OUT/name;d.mkdir(exist_ok=True);save(d/'failure.json',dict(error=str(e),time=time.time()));status('candidate_failed',name=name,error=str(e));b.stop_server()

def probe(c):
 name='probe_'+tag(c);p=OUT/name/'probe.json'
 if p.exists():return json.loads(p.read_text())
 try:
  launch(name,c);r=request(name+'/warmup',WARM,96,True,False)
  result=dict(config=c,ok=r['peaks']['gpu_mib']<=15550,gpu_mib=r['peaks']['gpu_mib'])
 except Exception as e:
  failed(name,e);result=dict(config=c,ok=False,error=str(e))
 finally:b.stop_server()
 save(p,result);return result

def fit(budget,kv,ub=512):
 p=ROOT/f'fit_k{budget//1024}_{kv}_u{ub}.json'
 if p.exists():return json.loads(p.read_text())['config']
 pool=(budget+4096)*{'q8q8':51,'q4q8':39,'q4q4':27}[kv]/1024**2
 n=max(0,min(48,20+math.ceil((pool-3.38671875)/.345)))
 tested=[];best=None
 # Move toward the feasible frontier, stopping at the first failure below it.
 while 0<=n<=48:
  c=config(budget,kv,n,ub=ub);r=probe(c);tested.append(r)
  if r['ok']:
   best=c
   if n==0:break
   n-=1
  elif best is not None:break
  else:n+=1
 if best is None:raise RuntimeError('No fitting placement')
 save(p,dict(config=best,probes=tested,probe_limit_mib=15550));return best

def run(c,prefix='case'):
 name=prefix+'_'+tag(c);p=OUT/name/'summary.json'
 if p.exists():return json.loads(p.read_text())
 # Retry a transient CUDA startup failure once, retaining the entire failed attempt.
 for attempt in range(2):
  attempt_name=name if attempt==0 else name+'_retry1'
  if (OUT/attempt_name/'summary.json').exists():return json.loads((OUT/attempt_name/'summary.json').read_text())
  try:
   launch(attempt_name,c);request(attempt_name+'/warmup',WARM,64,True,False)
   rows=[request(attempt_name+f'/rep{i}',reset=i==0) for i in range(3)]
   recall_msg=MSG+[{'role':'assistant','content':rows[-1]['content']},{'role':'user','content':'Look up the exact recovery phrase for project ORCHID and the vault identifier for project COBALT in the earlier archive. Return only those two values, preserving all letters and digits.'}]
   recall=request(attempt_name+'/recall',recall_msg,96,False,False)
   quality=dict(orchid='COPPER-HERON-731' in recall['content'],cobalt='SILVER-MAPLE-284' in recall['content'])
   out=dict(name=attempt_name,config=c,prompt_tokens=260992,prompt_sha256=HASH,outputs=[r['usage']['completion_tokens'] for r in rows],decode_tps=[r['timings']['predicted_per_second'] for r in rows],decode_median=statistics.median(r['timings']['predicted_per_second'] for r in rows),cold_prefill_tps=rows[0]['timings']['prompt_per_second'],cold_ttft_s=rows[0]['ttft_s'],cached_ttft_s=[r['ttft_s'] for r in rows[1:]],quality=quality,peaks={k:max(r['peaks'][k] for r in rows+[recall]) for k in rows[0]['peaks']},process_swap_peak=max(r['memory_checks']['process_swap_peak'] for r in rows+[recall]),source='fresh process; full cold prefill and 2 cached repeats; 512 tokens each')
   save(OUT/attempt_name/'summary.json',out);print('SUMMARY '+json.dumps(out),flush=True);return out
  except Exception as e:
   failed(attempt_name,e)
   log=(OUT/attempt_name/'server.log').read_text(errors='replace') if (OUT/attempt_name/'server.log').exists() else ''
   if 'illegal instruction' not in log or attempt: return None
  finally:b.stop_server()

def good(r):return r and all(r['quality'].values()) and r['process_swap_peak']==0 and r['peaks']['gpu_mib']<=16000

def resource_valid(r):
 return r and r['process_swap_peak']==0 and r['peaks']['gpu_mib']<=16000 and r['outputs']==[512]*3

def main():
 lock=(ROOT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (ROOT/'runner.pid').write_text(str(os.getpid()))
 rows=[]
 def add(r):
  if r:rows.append(r)
  save(ROOT/'partial-summary.json',rows)
 try:
  for budget in [65536,32768,16384,131072]:
   for kv in ['q8q8','q4q8']:
    add(run(fit(budget,kv)))
  assert any(resource_valid(r) for r in rows),'No resource-valid candidate'
  # Preserve both goals when smaller windows fail recall: speed ceiling and usable recall.
  fastest=max([r for r in rows if resource_valid(r)],key=lambda r:r['decode_median'])
  add(run(fit(fastest['config']['budget'],'q4q4')))
  fastest=max([r for r in rows if resource_valid(r)],key=lambda r:r['decode_median'])
  for threads in [4,8,10]:
   add(run({**fastest['config'],'threads':threads},'threads'))
  fastest=max([r for r in rows if resource_valid(r)],key=lambda r:r['decode_median'])
  c=fit(fastest['config']['budget'],fastest['config']['kv'],256);c['threads']=fastest['config']['threads']
  add(run(c,'batch'))
  fastest=max([r for r in rows if resource_valid(r)],key=lambda r:r['decode_median'])
  speed_final=run(fastest['config'],'final_speed');add(speed_final)
  assert resource_valid(speed_final),'Speed final resource check failed'
  if good(speed_final):
   recommended=speed_final
  else:
   qualified=[r for r in rows if good(r)];assert qualified,'No recall-passing candidate'
   best=max(qualified,key=lambda r:r['decode_median'])
   # Only two extra CPU thread choices for the separate recall-passing layout.
   for threads in [4,10]:
    c={**best['config'],'threads':threads}
    if not any(r['config']==c for r in rows):add(run(c,'quality_threads'))
   qualified=sorted([r for r in rows if good(r)],key=lambda r:r['decode_median'],reverse=True)
   seen=set();recommended=None
   for candidate in qualified:
    key=json.dumps(candidate['config'],sort_keys=True)
    if key in seen:continue
    seen.add(key)
    final=run(candidate['config'],'final_quality');add(final)
    if good(final):recommended=final;break
    if len(seen)>=3:break
   assert recommended,'No final configuration passed recall'
  save(ROOT/'complete.json',dict(winner=recommended,speed_winner=speed_final,candidates=rows))
  status('experiment_complete',winner=recommended['name'],decode_median=recommended['decode_median'],speed_winner=speed_final['name'],speed_median=speed_final['decode_median'])
 finally:b.stop_server();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
