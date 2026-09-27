"""Native full KV/all CPU experts vs KVMem 64K/mixed expert placement."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

ROOT = Path.home() / 'kvmem-moe-256k'
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
OLD = Path.home() / 'kvmem-moe-test'
MODEL = OLD / 'models/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
NATIVE = ROOT / 'native-build/bin/llama-server'
KVMEM = OLD / 'build/bin/llama-kvmem-server'
CONTEXT = 262144
BUDGET = 65536
RESERVE = 4096
OUTPUT = 512
URL = 'http://127.0.0.1:1923'
ENV = os.environ.copy()
for key in list(ENV):
    if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy') or key.startswith('LLAMA_ARG_') or key in ('LLAMA_API_KEY', 'KVMEM_TRACE', 'KVMEM_PERF', 'GGML_CUDA_DISABLE_FUSION'):
        ENV.pop(key, None)
ENV['LD_LIBRARY_PATH'] = str(Path.home()/'kvmem-llama-test/cuda-packages/nvidia/cu13/lib') + ':' + str(Path.home()/'freetoken-test/.venv/lib/python3.12/site-packages/nvidia/cu13/lib') + ':/usr/lib/wsl/lib'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
spec = importlib.util.spec_from_file_location('previous', OLD / 'experiment.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
PROCESS = None
LOG = None
ACTIVE = None


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def get(route):
    return json.load(OPENER.open(URL + route, timeout=10))


def stop_server():
    global PROCESS
    if PROCESS is not None and PROCESS.poll() is None:
        PROCESS.terminate()
        try:
            PROCESS.wait(timeout=20)
        except subprocess.TimeoutExpired:
            PROCESS.kill()
            PROCESS.wait()
    PROCESS = None
    time.sleep(2)


def launch(name, native, ncpu=48, context=CONTEXT):
    global PROCESS, LOG, ACTIVE, URL
    stop_server()
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    LOG = dest / 'server.log'
    port = 1923 if name == 'smoke_native' else 1924 if name == 'smoke_kvmem' else 1933 if native else 20000+ncpu if name.startswith('fit_') else 1934
    URL = f'http://127.0.0.1:{port}'
    cmd = [str(NATIVE if native else KVMEM), '-m', str(MODEL),
           '--alias', 'Qwen3-MoE', '--host', '127.0.0.1', '--port', str(port),
           '-c', str(context), '-ngl', '99', '-t', '6', '-tb', '6',
           '-b', '512', '-ub', '512', '-fa', 'on', '--load-mode', 'none',
           '--threads-http', '4', '--timeout', '7200']
    if native:
        cmd += ['--cpu-moe', '-ctk', 'q8_0', '-ctv', 'q8_0', '-np', '1',
                '--fit', 'off', '--no-context-shift', '--cache-ram', '0',
                '--chat-template-kwargs', '{"enable_thinking":false}', '--no-warmup']
    else:
        cmd += ['--n-cpu-moe', str(ncpu), '--kv-dtype', 'q8_0',
                '--spec-type', 'none', '--no-ui', '--kvmem-budget', str(BUDGET),
                '--kvmem-gen-reserve', str(RESERVE), '--no-kvmem-trace']
    ACTIVE = dict(name=name, native=native, context=context, cpu_expert_layers=48 if native else ncpu,
                  gpu_expert_layers=0 if native else 48-ncpu, kv_gpu_history_tokens=context if native else BUDGET,
                  kv_gpu_reserve_tokens=0 if native else RESERVE, command=cmd)
    save(dest / 'config.json', ACTIVE)
    with LOG.open('w') as f:
        PROCESS = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=ENV)
    (ROOT / 'server.pid').write_text(str(PROCESS.pid))
    for _ in range(240):
        if PROCESS.poll() is not None:
            raise RuntimeError('Server exited: ' + str(LOG))
        try:
            if get('/health').get('status') == 'ok':
                if native:
                    save(dest / 'props.json', get('/props'))
                    save(dest / 'slots-start.json', get('/slots'))
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError('Server startup timeout: ' + str(LOG))


def request(name, messages, max_tokens=OUTPUT, reset=False):
    assert PROCESS is not None and PROCESS.poll() is None
    body = dict(model='Qwen3-MoE', messages=messages, max_tokens=max_tokens,
                stream=True, stream_options={'include_usage': True}, temperature=0, seed=42,
                top_p=1, top_k=0, min_p=0, presence_penalty=0, frequency_penalty=0,
                repeat_penalty=1, chat_template_kwargs={'enable_thinking': False},
                cache_prompt=not reset, cache_reset=reset)
    dest = OUT / name
    save(str(dest) + '.request.json', body)
    stop = threading.Event()
    samples = []
    def monitor():
        while not stop.is_set():
            try:
                samples.append(previous.sample(PROCESS.pid))
            except Exception as exc:
                samples.append(dict(t=time.time(), error=str(exc)))
            stop.wait(.5)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    offset = LOG.stat().st_size
    events = []
    text = ''
    reasoning = ''
    usage = {}
    timings = {}
    first = None
    finish = None
    error = None
    started = time.perf_counter()
    try:
        req = urllib.request.Request(URL + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with OPENER.open(req, timeout=7200) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                raw = line[5:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                events.append(dict(elapsed_s=time.perf_counter()-started, event=event))
                if event.get('error'):
                    raise RuntimeError(event['error'])
                usage = event.get('usage') or usage
                timings = event.get('timings') or timings
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    piece = delta.get('content') or ''
                    thought = delta.get('reasoning_content') or ''
                    if first is None and (piece or thought):
                        first = time.perf_counter()-started
                    text += piece
                    reasoning += thought
                    finish = choice.get('finish_reason') or finish
    except Exception as exc:
        error = str(exc)
    finally:
        elapsed = time.perf_counter()-started
        stop.set()
        worker.join(timeout=10)
        save(str(dest)+'.events.json', events)
        save(str(dest)+'.samples.json', samples)
        with LOG.open('rb') as f:
            f.seek(offset)
            Path(str(dest)+'.server.log').write_bytes(f.read())
    good = [s for s in samples if 'error' not in s]
    peaks = {k: max(s[k] for s in good) for k in ('rss', 'system_used', 'swap', 'gpu_mib', 'gpu_temp', 'gpu_watts')} if good else {}
    result = dict(name=name, prompt_sha256=hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                  ttft_s=first, elapsed_s=elapsed, usage=usage, timings=timings, peaks=peaks,
                  content=text, reasoning=reasoning, finish=finish, error=error)
    save(str(dest)+'.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('content','reasoning')}, ensure_ascii=False), flush=True)
    if error or not usage or not timings or not text:
        raise RuntimeError(error or 'Incomplete result: ' + name)
    return result


def messages():
    path = ROOT / 'prompt.json'
    if path.exists():
        return json.loads(path.read_text())
    result = previous.messages_for(CONTEXT - 1024)
    result[-1]['content'] = result[-1]['content'].replace('1200 words', '2000 words')
    save(path, result)
    return result


def fit():
    path = OUT / 'fit.json'
    if path.exists():
        return json.loads(path.read_text())['best_cpu_layers']
    probes = []
    best = None
    n = 24
    while 0 <= n <= 48:
        name = f'fit_k64_n{n}'
        try:
            launch(name, False, n)
            row = request(name + '/warmup', previous.WARM, 128, True)
            ok = row['peaks']['gpu_mib'] <= 15550
            probes.append(dict(ncpu=n, ok=ok, gpu_mib=row['peaks']['gpu_mib']))
            if ok:
                best = n
                if n == 0:
                    break
                n -= 1
                continue
        except Exception as exc:
            probes.append(dict(ncpu=n, ok=False, error=str(exc)))
        if best is not None:
            break
        n += 2
    stop_server()
    if best is None:
        raise RuntimeError('No fitting KVMem placement')
    save(path, dict(best_cpu_layers=best, probes=probes, gpu_peak_limit_mib=15550))
    return best


def smoke():
    rows = []
    for native in (True, False):
        name = 'smoke_native' if native else 'smoke_kvmem'
        launch(name, native, 48, context=4096)
        rows.append(request(name+'/arithmetic', previous.SMOKE, 128, True))
    stop_server()
    if not all('323' in row['content'] for row in rows):
        raise RuntimeError('Arithmetic smoke failed')
    if rows[0]['usage']['prompt_tokens'] != rows[1]['usage']['prompt_tokens']:
        raise RuntimeError('Native and KVMem chat templates have different token counts')
    save(OUT/'smoke.json', dict(prompt_tokens=rows[0]['usage']['prompt_tokens'], answers=[r['content'] for r in rows]))


def run_case(name, native, ncpu=48):
    path = OUT/name/'summary.json'
    if path.exists():
        return json.loads(path.read_text())
    msgs = messages()
    launch(name, native, ncpu)
    request(name+'/warmup', previous.WARM, 64, True)
    results = []
    for i in range(3):
        row = request(name+f'/rep{i}', msgs, OUTPUT, reset=(i==0))
        if row['usage']['completion_tokens'] != OUTPUT:
            raise RuntimeError('Output count differs from 512')
        if i == 0 and row['timings']['prompt_n'] < CONTEXT-2048:
            raise RuntimeError('The initial prompt was not fully evaluated')
        results.append(row)
    recall_msgs = msgs + [{'role':'assistant','content':results[-1]['content']},
                         {'role':'user','content':'Look up the exact recovery phrase for project ORCHID and the vault identifier for project COBALT in the earlier archive. Return only those two values, preserving all letters and digits.'}]
    recall = request(name+'/recall', recall_msgs, 96)
    summary = dict(ACTIVE, prompt_tokens=results[0]['usage']['prompt_tokens'], output_tokens=OUTPUT,
                   prompt_sha256=results[0]['prompt_sha256'], decode_tps=[r['timings']['predicted_per_second'] for r in results],
                   decode_tps_median=statistics.median(r['timings']['predicted_per_second'] for r in results),
                   cold_prefill_tps=results[0]['timings']['prompt_per_second'], cold_ttft_s=results[0]['ttft_s'],
                   cached_ttft_s=statistics.median(r['ttft_s'] for r in results[1:]), cold_elapsed_s=results[0]['elapsed_s'],
                   cached_elapsed_s=statistics.median(r['elapsed_s'] for r in results[1:]),
                   recall=dict(orchid='COPPER-HERON-731' in recall['content'], cobalt='SILVER-MAPLE-284' in recall['content']),
                   peaks={k:max(r['peaks'][k] for r in results+[recall]) for k in results[0]['peaks']})
    if native:
        save(OUT/name/'slots-end.json', get('/slots'))
    save(path, summary)
    print('SUMMARY '+json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'smoke', 'fit', 'native', 'kvmem', 'suite'])
    parser.add_argument('--ncpu', type=int)
    args = parser.parse_args()
    try:
        if args.mode == 'prepare':
            messages()
            return
        if args.mode in ('smoke','suite'):
            smoke()
        if args.mode == 'fit':
            print('FIT',fit(),flush=True)
        if args.mode == 'native':
            run_case('native_all_experts_cpu',True)
        if args.mode == 'kvmem':
            run_case('kvmem_64k',False,args.ncpu if args.ncpu is not None else fit())
        if args.mode == 'suite':
            ncpu=fit()
            rows=[run_case('native_all_experts_cpu',True),run_case('kvmem_64k',False,ncpu)]
            assert rows[0]['prompt_tokens']==rows[1]['prompt_tokens']
            assert rows[0]['prompt_sha256']==rows[1]['prompt_sha256']
            save(OUT/'summary.json',rows)
            (ROOT/'suite.done').write_text('completed\n')
    finally:
        stop_server()


if __name__=='__main__':
    main()
