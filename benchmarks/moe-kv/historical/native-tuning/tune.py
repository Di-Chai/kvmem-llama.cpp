"""Bounded, resumable native llama.cpp parameter search at real 260,992-token context.

Uses the existing measurement client; never changes the model or server source.
KV snapshots are only reused within one K/V type pair and model/binary identity.
Final validation must prefill from scratch using the winning configuration.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path.home() / 'llama-native-tuning-256k'
ROOT.mkdir(exist_ok=True)
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
CACHE = ROOT / 'cache'
CACHE.mkdir(exist_ok=True)
BASE = Path.home() / 'kvmem-moe-256k'
spec = importlib.util.spec_from_file_location('bench', BASE / 'bench256.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.ROOT = ROOT
b.OUT = OUT
b.URL = 'http://127.0.0.1:1941'
MSG = json.loads((BASE / 'prompt.json').read_text())
PROMPT_TOKENS = 260992
PORT = 1941
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
KV = {
    'q4q4': ('q4_0', 'q4_0'),
    'q8q4': ('q8_0', 'q4_0'),
    'q4q8': ('q4_0', 'q8_0'),
    'q5q5': ('q5_0', 'q5_0'),
    'q8q8': ('q8_0', 'q8_0'),
}
WARM = [{'role': 'user', 'content': 'Capacity planning review. ' * 200 + b.previous.WARM[0]['content']}]


def resource_sample(pid):
    # One nvidia-smi call per sample, as in the original client, with clock data
    # added to help distinguish small tuning differences from laptop power limits.
    memory = b.previous.psutil.virtual_memory()
    process = b.previous.psutil.Process(pid)
    swap = b.previous.psutil.swap_memory()
    process_swap = next(int(line.split()[1])*1024 for line in
                        Path(f'/proc/{pid}/status').read_text().splitlines()
                        if line.startswith('VmSwap:'))
    fields = 'memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,clocks_event_reasons.active'
    values = subprocess.check_output(
        ['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu='+fields,
         '--format=csv,noheader,nounits'], text=True).strip().split(',')
    return dict(t=time.time(), rss=process.memory_info().rss,
                system_used=memory.total-memory.available, available=memory.available,
                swap=swap.used, process_swap=process_swap,
                system_swap_in=swap.sin, system_swap_out=swap.sout,
                gpu_mib=float(values[0]), gpu_util=float(values[1]),
                gpu_temp=float(values[2]), gpu_watts=float(values[3]),
                gpu_graphics_mhz=float(values[4]), gpu_memory_mhz=float(values[5]),
                gpu_clock_event_reasons=values[6].strip())


b.previous.sample = resource_sample


def save(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def status(stage, **kwargs):
    record = dict(time=time.time(), stage=stage, **kwargs)
    save(ROOT / 'status.json', record)
    print('STATUS ' + json.dumps(record), flush=True)


def config(kv, ncpu, threads=6, ub=512, tb=6, op_min=32, op_offload=True):
    return dict(kv=kv, ncpu=ncpu, threads=threads, ub=ub, tb=tb,
                op_min=op_min, op_offload=op_offload)


def tag(cfg):
    base = f"{cfg['kv']}_n{cfg['ncpu']}_t{cfg['threads']}_u{cfg['ub']}_tb{cfg['tb']}_op{cfg['op_min']}_{int(cfg['op_offload'])}"
    if 'poll' in cfg:
        base += '_poll'+str(cfg['poll'])
    if cfg.get('spec','none') != 'none':
        base += '_spec'+cfg['spec']
        base += f"_n{cfg.get('ngram_n',8)}m{cfg.get('ngram_m',8)}d{cfg.get('draft_max',8)}"
    return base


def launch(name, cfg):
    b.stop_server()
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    b.LOG = dest / 'server.log'
    k, v = KV[cfg['kv']]
    cmd = [str(b.NATIVE), '-m', str(b.MODEL), '--alias', 'Qwen3-MoE',
           '--host', '127.0.0.1', '--port', str(PORT), '-c', '262144', '-np', '1',
           '-ngl', '99', '--n-cpu-moe', str(cfg['ncpu']),
           '-t', str(cfg['threads']), '-tb', str(cfg['tb']),
           '-b', str(max(512, cfg['ub'])), '-ub', str(cfg['ub']),
           '-fa', 'on', '-ctk', k, '-ctv', v, '--load-mode', 'none',
           '--fit', 'off', '--no-context-shift', '--cache-ram', '0',
           '--slot-save-path', str(CACHE) + '/', '--slots',
           '--threads-http', '4', '--timeout', '7200',
           '--chat-template-kwargs', '{"enable_thinking":false}', '--no-warmup']
    if not cfg['op_offload']:
        cmd.append('--no-op-offload')
    if 'poll' in cfg:
        cmd += ['--poll', str(cfg['poll'])]
    if cfg.get('spec','none') != 'none':
        cmd += ['--spec-type',cfg['spec'],'--spec-draft-n-max',str(cfg.get('draft_max',8)),
                '--spec-ngram-simple-size-n',str(cfg.get('ngram_n',8)),
                '--spec-ngram-simple-size-m',str(cfg.get('ngram_m',8))]
    env = b.ENV.copy()
    for key in list(env):
        if key.startswith('GGML_') or key.startswith('LLAMA_'):
            env.pop(key, None)
    env['GGML_OP_OFFLOAD_MIN_BATCH'] = str(cfg['op_min'])
    b.ACTIVE = dict(name=name, config=cfg, command=cmd,
                    env_overrides={'GGML_OP_OFFLOAD_MIN_BATCH': str(cfg['op_min'])})
    save(dest / 'config.json', b.ACTIVE)
    with b.LOG.open('w') as log:
        b.PROCESS = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
    (ROOT / 'server.pid').write_text(str(b.PROCESS.pid))
    status('starting', name=name, config=cfg, server_pid=b.PROCESS.pid)
    for _ in range(240):
        if b.PROCESS.poll() is not None:
            raise RuntimeError(f'server exited ({b.PROCESS.returncode}): {b.LOG}')
        try:
            if b.get('/health').get('status') == 'ok':
                slots = b.get('/slots')
                save(dest / 'slots-start.json', slots)
                save(dest / 'props.json', b.get('/props'))
                assert len(slots) == 1 and slots[0]['n_ctx'] == 262144, slots
                return
        except (ConnectionError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError('startup timed out')


def slot(action, kv):
    body = json.dumps({'filename': kv + '.bin'}).encode()
    req = b.urllib.request.Request(b.URL + '/slots/0?action=' + action,
                                   data=body, headers={'Content-Type': 'application/json'})
    with b.OPENER.open(req, timeout=300) as response:
        result = json.load(response)
    if action == 'restore':
        assert result['n_restored'] >= PROMPT_TOKENS, result
    if action == 'save':
        assert result['n_saved'] >= PROMPT_TOKENS, result
    return result


def measure(name, *, tokens=256, reset=False, cached=False):
    status('measuring', name=name, server_pid=b.PROCESS.pid,
           config=b.ACTIVE['config'], expected_prompt_tokens=PROMPT_TOKENS, cached=cached)
    row = b.request(name, MSG, tokens, reset=reset)
    samples = [s for s in json.loads((OUT/(name+'.samples.json')).read_text()) if 'error' not in s]
    row['memory_checks'] = dict(
        process_swap_peak=max(s['process_swap'] for s in samples),
        system_swap_start=samples[0]['swap'], system_swap_end=samples[-1]['swap'],
        system_swap_in_delta=samples[-1]['system_swap_in']-samples[0]['system_swap_in'],
        system_swap_out_delta=samples[-1]['system_swap_out']-samples[0]['system_swap_out'])
    save(OUT/(name+'.json'),row)
    assert row['usage']['prompt_tokens'] == PROMPT_TOKENS, row['usage']
    assert row['usage']['completion_tokens'] == tokens, row['usage']
    if cached:
        assert row['timings']['cache_n'] >= PROMPT_TOKENS - 4, row['timings']
    else:
        assert row['timings']['prompt_n'] >= PROMPT_TOKENS - 4, row['timings']
    return row


def recall(name, generated):
    messages = MSG + [{'role':'assistant', 'content':generated},
                     {'role':'user', 'content':'Look up the exact recovery phrase for project ORCHID and the vault identifier for project COBALT in the earlier archive. Return only those two values, preserving all letters and digits.'}]
    row = b.request(name, messages, 96, reset=False)
    checks = {'orchid':'COPPER-HERON-731' in row['content'],
              'cobalt':'SILVER-MAPLE-284' in row['content']}
    return dict(checks=checks, result=row)


def summary(name, cfg, rows, source, quality=None):
    values = [r['timings']['predicted_per_second'] for r in rows]
    out = dict(name=name, config=cfg, source=source, prompt_tokens=PROMPT_TOKENS,
               decode_tps=values, decode_median=statistics.median(values),
               peaks={k:max(r['peaks'][k] for r in rows) for k in rows[0]['peaks']},
               outputs=[r['usage']['completion_tokens'] for r in rows], quality=quality)
    if all('memory_checks' in r for r in rows):
        out['memory_checks'] = dict(
            process_swap_peak=max(r['memory_checks']['process_swap_peak'] for r in rows),
            system_swap_in_delta=sum(r['memory_checks']['system_swap_in_delta'] for r in rows),
            system_swap_out_delta=sum(r['memory_checks']['system_swap_out_delta'] for r in rows))
    save(OUT / name / 'summary.json', out)
    print('SUMMARY ' + json.dumps(out), flush=True)
    return out


def probe(cfg):
    name = 'probe_' + tag(cfg)
    path = OUT / name / 'probe.json'
    if path.exists():
        return json.loads(path.read_text())
    try:
        launch(name, cfg)
        row = b.request(name + '/warmup', WARM, 96, reset=True)
        result = dict(config=cfg, ok=row['peaks']['gpu_mib'] <= 15650,
                      gpu_mib=row['peaks']['gpu_mib'], decode_short=row['timings']['predicted_per_second'])
    except Exception as exc:
        result = dict(config=cfg, ok=False, error=str(exc))
    finally:
        b.stop_server()
    save(path, result)
    print('PROBE ' + json.dumps(result), flush=True)
    return result


def fit(kv, start, *, op_min=32, ub=512):
    n = start
    best = None
    while 0 <= n <= 48:
        row = probe(config(kv, n, ub=ub, op_min=op_min))
        if row['ok']:
            best = n
            if n == 0:
                break
            n -= 2
            continue
        if best is None:
            n += 2
            continue
        if best - n == 2 and probe(config(kv, n+1, ub=ub, op_min=op_min))['ok']:
            best = n+1
        break
    if best is None:
        raise RuntimeError(f'No fitting placement: {kv}, op_min={op_min}, ub={ub}')
    return config(kv, best, ub=ub, op_min=op_min)


def seed(kv):
    name = 'seed_' + kv
    done = OUT / name / 'summary.json'
    if done.exists() and (CACHE / (kv+'.bin')).exists():
        return json.loads(done.read_text())
    # Leave expert-weight headroom to amortize prefill with a bigger ubatch.
    for ub in ([512] if kv == 'q8q8' else [2048,1024,512]):
        cfg = config(kv, 48, ub=ub)
        try:
            launch(name + f'_u{ub}', cfg)
            b.request(name + f'_u{ub}/warmup', WARM, 64, reset=True)
            row = measure(name + f'_u{ub}/cold', tokens=512, reset=True)
            saved = slot('save', kv)
            quality = recall(name + f'_u{ub}/recall', row['content'])
            (OUT/name).mkdir(exist_ok=True)
            save(OUT/name/'slot-save.json', saved)
            out = summary(name, cfg, [row], 'real full cold prefill; snapshot for same-KV screening', quality['checks'])
            return out
        except Exception as exc:
            save(OUT / (name+f'_u{ub}') / 'failure.json', {'error':str(exc)})
            print('SEED_FAILURE',kv,ub,str(exc),flush=True)
        finally:
            b.stop_server()
    raise RuntimeError('Could not seed real long context for '+kv)


def screen(cfg, *, tokens=256, repeats=1, prefix='screen'):
    name = prefix + '_' + tag(cfg)
    path = OUT / name / 'summary.json'
    if path.exists():
        return json.loads(path.read_text())
    try:
        launch(name, cfg)
        save(OUT/name/'slot-restore.json', slot('restore', cfg['kv']))
        rows = [measure(name+f'/rep{i}', tokens=tokens, cached=True) for i in range(repeats)]
        return summary(name,cfg,rows,'restored same-model same-KV real 260,992-token snapshot')
    finally:
        b.stop_server()


def guarded_screen(cfg, **kwargs):
    try:
        return screen(cfg, **kwargs)
    except Exception as exc:
        name = kwargs.get('prefix','screen') + '_' + tag(cfg)
        save(OUT/name/'failure.json', {'error':str(exc),'config':cfg})
        print('SCREEN_FAILURE',name,str(exc),flush=True)
        return None


def phase1():
    rows = []
    fits = {}
    for kv,start in [('q4q4',34),('q8q4',42),('q5q5',38),('q8q8',48)]:
        status('phase1', kv=kv)
        chosen = fit(kv,start) if kv != 'q8q8' else config(kv,48)
        fits[kv] = chosen
        save(ROOT/'fits.json',fits)
        seed(kv)
        for n in sorted(set([chosen['ncpu'], min(48,chosen['ncpu']+2)])):
            row = guarded_screen({**chosen, 'ncpu':n})
            if row:
                rows.append(row)
        save(ROOT/'phase1.json',rows)
    status('phase1_complete', configurations=len(rows))


def final(cfg, name):
    # Always use a new directory/name for each independent final validation.
    launch(name,cfg)
    try:
        b.request(name+'/warmup',WARM,64,reset=True)
        rows = [measure(name+'/rep0',tokens=512,reset=True)]
        rows += [measure(name+f'/rep{i}',tokens=512,cached=True) for i in (1,2)]
        quality = recall(name+'/recall',rows[-1]['content'])
        save(OUT/name/'slots-end.json',b.get('/slots'))
        return summary(name,cfg,rows,'fresh cold prefill with this exact configuration, followed by 2 cached repeats',quality['checks'])
    finally:
        b.stop_server()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['phase1','seed','probe','screen','final'])
    parser.add_argument('--kv',choices=KV,default='q4q4')
    parser.add_argument('--ncpu',type=int,default=32)
    parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--ub',type=int,default=512)
    parser.add_argument('--tb',type=int,default=6)
    parser.add_argument('--op-min',type=int,default=32)
    parser.add_argument('--no-op-offload',action='store_true')
    parser.add_argument('--name',default='final_best')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--tokens',type=int,default=256)
    parser.add_argument('--spec',default='none',choices=['none','ngram-simple'])
    parser.add_argument('--poll',type=int,choices=range(101),default=None)
    parser.add_argument('--ngram-n',type=int,default=8)
    parser.add_argument('--ngram-m',type=int,default=8)
    parser.add_argument('--draft-max',type=int,default=8)
    args=parser.parse_args()
    lock=(ROOT/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'runner.pid').write_text(str(os.getpid()))
    cfg=config(args.kv,args.ncpu,args.threads,args.ub,args.tb,args.op_min,not args.no_op_offload)
    if args.spec != 'none':
        cfg.update(spec=args.spec,ngram_n=args.ngram_n,ngram_m=args.ngram_m,draft_max=args.draft_max)
    if args.poll is not None: cfg['poll']=args.poll
    try:
        if args.mode=='phase1': phase1()
        elif args.mode=='seed': seed(args.kv)
        elif args.mode=='probe': probe(cfg)
        elif args.mode=='screen': screen(cfg,tokens=args.tokens,repeats=args.repeats)
        elif args.mode=='final': final(cfg,args.name)
        status('complete', mode=args.mode)
    finally:
        b.stop_server()


if __name__=='__main__':
    main()
