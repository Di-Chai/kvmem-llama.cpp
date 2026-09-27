"""Second-stage search using verified full-context snapshots from phase 1."""
import fcntl
import json
import os
from pathlib import Path
import time
import tune as t


def screened():
    rows=[]
    for p in t.OUT.glob('*/summary.json'):
        r=json.loads(p.read_text())
        if r['name'].startswith(('screen_','threads_','batch_','gpuops_','ops_','poll_','ngram_','repeat_')):
            memory=r.get('memory_checks')
            stable_memory=(memory['process_swap_peak']==0 if memory is not None
                           else r['peaks']['swap']<=16*1024**2)
            if r['peaks']['gpu_mib']<=16000 and stable_memory:
                rows.append(r)
    return rows


def best_by_kv():
    rows=[r for r in screened() if r['config'].get('spec','none') == 'none']
    return {kv:max((r for r in rows if r['config']['kv']==kv),key=lambda r:r['decode_median'])
            for kv in t.KV if any(r['config']['kv']==kv for r in rows)}


def main():
    lock=(t.ROOT/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (t.ROOT/'runner.pid').write_text(str(os.getpid()))
    try:
        # Probe the main alternative execution path early, using already seeded
        # full contexts, before the last mixed-KV cold prefill.
        ready=best_by_kv()
        for kv in ['q8q8','q4q4']:
            if kv in ready:
                t.guarded_screen({**ready[kv]['config'],'spec':'ngram-simple',
                                  'ngram_n':1,'ngram_m':1,'draft_max':1},prefix='ngram',tokens=256)
        # The two mixed-precision directions have the same KV footprint but
        # different CUDA vector work. Check the reverse direction as well.
        placement=t.fit('q4q8',42)
        t.seed('q4q8')
        for n in sorted(set([placement['ncpu'],min(48,placement['ncpu']+2)])):
            t.guarded_screen({**placement,'ncpu':n})
        # Random-looking, fixed order to avoid always testing thread counts in increasing order.
        initial=best_by_kv()
        floor=0.75*max(r['decode_median'] for r in initial.values())
        pruned={kv:dict(decode=r['decode_median'],threshold=floor,
                        reason='More than 25% behind leading ordinary case; still test ngram separately.')
                for kv,r in initial.items() if r['decode_median']<floor}
        t.save(t.ROOT/'thread-search-pruned.json',pruned)
        for kv,row in initial.items():
            if kv in pruned: continue
            cfg=row['config']
            for threads in [8,4,2]:
                t.guarded_screen({**cfg,'threads':threads},prefix='threads',tokens=256)
            t.save(t.ROOT/'phase2-progress.json',screened())
        for row in sorted(best_by_kv().values(),key=lambda r:r['decode_median'],reverse=True)[:2]:
            for threads in [12,10]:
                t.guarded_screen({**row['config'],'threads':threads},prefix='threads',tokens=256)
        leaders=sorted(best_by_kv().values(),key=lambda r:r['decode_median'],reverse=True)[:2]
        for leader in leaders:
            cfg=leader['config']
            for ub in [256,1024]:
                try:
                    placement=t.fit(cfg['kv'],min(48,cfg['ncpu']+2),ub=ub)
                    placement.update(threads=cfg['threads'])
                    t.guarded_screen(placement,prefix='batch',tokens=256)
                except Exception as exc:
                    print('BATCH_SEARCH_FAILURE',str(exc),flush=True)
            # Explore GPU execution of CPU-resident expert tensors, not only CPU computation.
            try:
                placement=t.fit(cfg['kv'],min(48,cfg['ncpu']+6),op_min=1,ub=cfg['ub'])
                placement.update(threads=cfg['threads'])
                t.guarded_screen(placement,prefix='gpuops',tokens=256)
            except Exception as exc:
                print('GPU_OP_SEARCH_FAILURE',str(exc),flush=True)
        for row in sorted(best_by_kv().values(),key=lambda r:r['decode_median'],reverse=True)[:2]:
            t.guarded_screen({**row['config'],'poll':0},prefix='poll',tokens=256)
        leader=max(best_by_kv().values(),key=lambda r:r['decode_median'])
        t.guarded_screen({**leader['config'],'op_offload':False},prefix='ops',tokens=256)
        # Batched verification uses a different CUDA attention path. Test every KV
        # type: the best ordinary-decode type need not be the best speculative type.
        # No draft model is downloaded or substituted.
        for row in best_by_kv().values():
            for n,m in [(1,1),(2,4)]:
                t.guarded_screen({**row['config'],'spec':'ngram-simple',
                                  'ngram_n':n,'ngram_m':m,'draft_max':m},prefix='ngram',tokens=256)
        speculative=[r for r in screened() if r['config'].get('spec','none') != 'none']
        if speculative:
            leader=max(speculative,key=lambda r:r['decode_median'])
            for n,m in [(1,4),(4,8),(8,8)]:
                t.guarded_screen({**leader['config'],'ngram_n':n,'ngram_m':m,'draft_max':m},
                                 prefix='ngram',tokens=256)
            # Verification with >1 token uses n_threads_batch, not n_threads.
            candidate=max((r for r in screened() if r['config'].get('spec','none')!='none'),
                          key=lambda r:r['decode_median'])
            ordinary_best=max(best_by_kv().values(),key=lambda r:r['decode_median'])
            if candidate['decode_median'] >= 0.90*ordinary_best['decode_median']:
                for tb in [4,8,2]:
                    t.guarded_screen({**candidate['config'],'tb':tb},prefix='ngram',tokens=256)
        t.save(t.ROOT/'phase2.json',screened())
        ranked=sorted(screened(),key=lambda r:r['decode_median'],reverse=True)
        unique=[]
        seen=set()
        # Always confirm ordinary decode independently of speculation.
        ordinary=[r for r in ranked if r['config'].get('spec','none') == 'none']
        speculative=[r for r in ranked if r['config'].get('spec','none') != 'none']
        for row in ordinary:
            key=t.tag(row['config'])
            if key not in seen:
                unique.append(row); seen.add(key)
            if len(unique)==2: break
        if speculative: unique.append(speculative[0])
        # Repeat the strongest ordinary and speculative candidates in reverse order.
        for row in reversed(unique):
            t.guarded_screen(row['config'],prefix='repeat',tokens=512,repeats=3)
        t.save(t.ROOT/'phase2.json',screened())
        t.status('phase2_complete',tested=len(screened()))
    finally:
        t.b.stop_server()


if __name__=='__main__': main()
