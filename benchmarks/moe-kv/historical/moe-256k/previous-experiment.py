"""Single-slot Qwen3 MoE: full KV vs retrieval, with fixed precision and hardware."""
import argparse, hashlib, json, os, re, statistics, subprocess, threading, time, urllib.request
from pathlib import Path
import psutil
ROOT=Path.home()/'kvmem-moe-test'
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
MODEL=ROOT/'models/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
URL='http://127.0.0.1:1922'
ENV=os.environ.copy()
for k in list(ENV):
    if k.lower() in ('http_proxy','https_proxy','all_proxy') or k.startswith('LLAMA_ARG_') or k in ('LLAMA_API_KEY','KVMEM_TRACE','KVMEM_PERF'):
        ENV.pop(k,None)
ENV['LD_LIBRARY_PATH']=str(Path.home()/'kvmem-llama-test/cuda-packages/nvidia/cu13/lib')+':'+str(Path.home()/'freetoken-test/.venv/lib/python3.12/site-packages/nvidia/cu13/lib')+':/usr/lib/wsl/lib'
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))
BINARY=ROOT/'build/bin/llama-kvmem-server'
PROCESS=None; LOG=None

def get(route):
    return json.load(OPENER.open(URL+route,timeout=5))

def save(path,x):
    Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2))

def sample(pid):
    m=psutil.virtual_memory(); p=psutil.Process(pid)
    g=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().split(',')
    return dict(t=time.time(),rss=p.memory_info().rss,system_used=m.total-m.available,available=m.available,swap=psutil.swap_memory().used,gpu_mib=float(g[0]),gpu_util=float(g[1]),gpu_temp=float(g[2]),gpu_watts=float(g[3]))

def request(name,messages,max_tokens=256,reset=True):
    assert PROCESS and PROCESS.poll() is None
    body=dict(model='Qwen3-MoE',messages=messages,max_tokens=max_tokens,stream=True,stream_options={'include_usage':True},temperature=0,seed=42,cache_reset=reset)
    dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True)
    save(str(dest)+'.request.json',body)
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:samples.append(sample(PROCESS.pid))
            except Exception as e:samples.append(dict(t=time.time(),error=str(e)))
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    offset=LOG.stat().st_size
    events=[];content='';reasoning='';usage={};timings={};first=None;finish=None
    t0=time.perf_counter();error=None
    try:
        req=urllib.request.Request(URL+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with OPENER.open(req,timeout=3600) as response:
            for line in response:
                if not line.startswith(b'data:'):continue
                raw=line[5:].strip()
                if raw==b'[DONE]':break
                event=json.loads(raw);events.append(event)
                if event.get('error'):raise RuntimeError(event['error'])
                if event.get('usage'):usage=event['usage']
                if event.get('timings'):timings=event['timings']
                for choice in event.get('choices',[]):
                    d=choice.get('delta',{});c=d.get('content') or '';r=d.get('reasoning_content') or ''
                    if first is None and (c or r):first=time.perf_counter()-t0
                    content+=c;reasoning+=r;finish=choice.get('finish_reason') or finish
    except Exception as e:
        error=str(e)
    finally:
        elapsed=time.perf_counter()-t0;stop.set();thread.join(timeout=10)
        save(str(dest)+'.samples.json',samples);save(str(dest)+'.events.json',events)
        with LOG.open('rb') as f:f.seek(offset);Path(str(dest)+'.server.log').write_bytes(f.read())
    good=[s for s in samples if 'error' not in s]
    peak={k:max(s[k] for s in good) for k in ['rss','system_used','swap','gpu_mib','gpu_temp','gpu_watts']} if good else {}
    result=dict(name=name,prompt_sha256=hashlib.sha256(json.dumps(messages).encode()).hexdigest(),ttft_s=first,elapsed_s=elapsed,timings=timings,usage=usage,peaks=peak,finish=finish,content=content,reasoning=reasoning,error=error)
    save(str(dest)+'.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('content','reasoning')},ensure_ascii=False),flush=True)
    if error:raise RuntimeError(error)
    if not usage or not content:raise RuntimeError('Incomplete generation: '+name)
    return result

def stop_server():
    global PROCESS
    if PROCESS and PROCESS.poll() is None:
        PROCESS.terminate()
        try:PROCESS.wait(timeout=20)
        except subprocess.TimeoutExpired:PROCESS.kill();PROCESS.wait()
    PROCESS=None
    time.sleep(2)

def launch(name,context,ncpu,budget=None,trace=True,kv_dtype="q8_0",reserve=4096):
    global PROCESS,LOG
    stop_server()
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    LOG=dest/'server.log'
    cmd=[str(BINARY),'-m',str(MODEL),'--alias','Qwen3-MoE','--host','127.0.0.1','--port','1922','-c',str(context),'-ngl','99','--n-cpu-moe',str(ncpu),'-t','6','-tb','6','-b','512','-ub','512','-fa','on','--kv-dtype',kv_dtype,'--spec-type','none','--load-mode','none','--no-ui','--threads-http','4']
    if budget is None:cmd+=['--no-kvmem']
    else:cmd+=['--kvmem-budget',str(budget),'--kvmem-gen-reserve',str(reserve)]
    cmd+=['--kvmem-trace' if trace else '--no-kvmem-trace']
    save(dest/'config.json',dict(context=context,ncpu=ncpu,budget=budget,command=cmd,trace=trace))
    with LOG.open('w') as f:PROCESS=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=ENV)
    (ROOT/'server.pid').write_text(str(PROCESS.pid))
    for _ in range(180):
        if PROCESS.poll() is not None:raise RuntimeError('Server exited: '+str(LOG))
        try:
            if get('/health').get('status')=='ok':return
        except Exception:pass
        time.sleep(1)
    raise RuntimeError('Server startup timeout: '+str(LOG))

SMOKE=[{'role':'user','content':'请先计算 17×19，再用一句话解释 GPU 显存的作用。'}]
WARM=[{'role':'user','content':'Explain why capacity planning and monitoring are important for a reliable distributed application. Provide a detailed answer of at least 1000 words.'}]

def count(text):
    p=subprocess.run([str(ROOT/'build/bin/llama-tokenize'),'-m',str(MODEL),'--stdin','--ids'],input=text,text=True,capture_output=True,env=ENV,check=True)
    return len(json.loads(p.stdout))

def messages_for(target):
    file=ROOT/f'prompt_{target}.json'
    if file.exists():return json.loads(file.read_text())
    topics=['latency','capacity','retry policy','monitoring','backup retention','request queue','worker scheduling','audit records','storage layout','failure recovery','network timeout']
    rows=[f'Record {i}: The {topics[i%len(topics)]} review for service {i%137} requires validation in staging. Team {i%23} will inspect metrics and document the outcome before rollout.\n' for i in range(16000)]
    raw=''.join(rows);cut=int((target-256)*3.5)
    for _ in range(4):
        n=count(raw[:cut])
        if abs(n-(target-256))<40:break
        cut=int(cut*(target-256)/n)
    doc=raw[:cut]
    facts=[(.18,'Project ORCHID recovery phrase is COPPER-HERON-731.'),(.43,'Project COBALT vault identifier is SILVER-MAPLE-284.')]
    for fraction,fact in reversed(facts):
        pos=doc.find('\n',int(len(doc)*fraction));doc=doc[:pos+1]+fact+'\n'+doc[pos+1:]
    msgs=[{'role':'system','content':'You are a precise technical assistant. Follow user instructions and use supplied records when relevant.'},{'role':'user','content':'Reference archive:\n'+doc},{'role':'assistant','content':'I have received the reference archive.'},{'role':'user','content':'Write a detailed engineering guide of at least 1200 words about monitoring, rollback, capacity planning, retry policy and recovery procedures. Explain tradeoffs in normal prose; do not repeat the archive verbatim.'}]
    save(file,msgs);return msgs

def fit(context,budget,start):
    tag=f'fit_c{context}_k{budget or 0}'
    tested=[];best=None
    n=start
    while 0<=n<=48:
        name=f'{tag}_n{n}'
        try:
            launch(name,context,n,budget)
            r=request(name+'/warmup',WARM,128)
            peak=r['peaks']['gpu_mib']
            ok=peak<=15550
            tested.append(dict(ncpu=n,ok=ok,gpu_mib=peak,decode=r['timings']['predicted_per_second']))
            if ok:
                best=n
                if n==0:break
                n=max(0,n-2)
                continue
        except Exception as e:
            tested.append(dict(ncpu=n,ok=False,error=str(e)))
        if best is not None:break
        n+=2
    stop_server()
    save(OUT/(tag+'.json'),dict(context=context,budget=budget,best_ncpu=best,probes=tested))
    if best is None:raise RuntimeError('No fitting placement')
    return best

def run_case(name,target,ncpu,budget,repeats=3):
    msgs=messages_for(target)
    launch(name,target+1024,ncpu,budget)
    request(name+'/warmup',WARM,64)
    results=[]
    for rep in range(repeats):
        results.append(request(name+f'/rep{rep}',msgs,256,reset=(rep==0)))
    recall_msgs=msgs+[{'role':'assistant','content':results[-1]['content']},{'role':'user','content':'Look up the exact recovery phrase for project ORCHID and the vault identifier for project COBALT in the earlier archive. Return only those two values, preserving all letters and digits.'}]
    recall=request(name+'/recall',recall_msgs,96,False)
    check=dict(orchid='COPPER-HERON-731' in recall['content'],cobalt='SILVER-MAPLE-284' in recall['content'])
    summary=dict(name=name,target=target,ncpu=ncpu,budget=budget,prompt_tokens=results[0]['usage']['prompt_tokens'],cold_ttft_s=results[0]['ttft_s'],cold_prefill_tps=results[0]['timings']['prompt_per_second'],decode_tps_median=statistics.median(x['timings']['predicted_per_second'] for x in results),decode_tps=[x['timings']['predicted_per_second'] for x in results],output_tokens=[x['usage']['completion_tokens'] for x in results],recall=check,peaks={k:max(x['peaks'][k] for x in results+[recall]) for k in results[0]['peaks']})
    save(OUT/name/'summary.json',summary);print('SUMMARY '+json.dumps(summary),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['sanity','fit','case']);parser.add_argument('--context',type=int,default=65536);parser.add_argument('--budget',type=int,default=0);parser.add_argument('--ncpu',type=int,default=24);parser.add_argument('--name',default='case');args=parser.parse_args()
    try:
        if args.mode=='sanity':
            answers=[]
            for enabled in (False,True):
                name='sanity_'+str(enabled).lower();launch(name,4096,24,3072 if enabled else None)
                r=request(name+'/smoke',SMOKE,128);answers.append(r['content'])
                if '323' not in r['content']:raise RuntimeError('Arithmetic failed')
            save(OUT/'sanity.json',dict(identical=answers[0]==answers[1],answers=answers))
        elif args.mode=='fit':print('FIT_RESULT '+str(fit(args.context+1024,args.budget or None,args.ncpu)),flush=True)
        else:run_case(args.name,args.context,args.ncpu,args.budget or None)
    finally:stop_server()
if __name__=='__main__':main()
