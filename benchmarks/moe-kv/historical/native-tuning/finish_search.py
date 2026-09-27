"""Finish this bounded experiment after phase 2 releases the GPU runner lock.

Only repeat-confirmed, full-context configurations can enter final validation.
Preserve all results; never overwrite an existing final measurement.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import tune as t


def valid(row):
    return (row['peaks']['gpu_mib'] <= 16000
            and row.get('memory_checks', {}).get('process_swap_peak') == 0
            and row['outputs'] == [512, 512, 512])


def final_name(row):
    return 'final_' + t.tag(row['config'])


def main():
    (t.ROOT / 'finish-runner.pid').write_text(str(os.getpid()))
    deadline = time.monotonic() + 6 * 3600
    while True:
        state = json.loads((t.ROOT / 'status.json').read_text())
        if state['stage'] == 'phase2_complete':
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Phase 2 did not finish within the controller wait limit.')
        time.sleep(5)
    lock = (t.ROOT / 'runner.lock').open('a')
    # A short non-blocking retry lets the completed phase finish stopping its server.
    for _ in range(60):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(1)
    else:
        raise RuntimeError('Phase 2 completed, but the runner lock is still held.')
    (t.ROOT / 'runner.pid').write_text(str(os.getpid()))
    # The short leaders differ by less than the observed warm/steady gap.
    # Confirm close four-thread candidates from both leading KV types so
    # the final selection is not just two effectively identical t=10 cases.
    initial = [json.loads(p.read_text()) for p in t.OUT.glob('*/summary.json')]
    ordinary_screens = [r for r in initial if r['outputs'] == [256]
                        and r['config'].get('spec', 'none') == 'none'
                        and r['peaks']['gpu_mib'] <= 16000
                        and r.get('memory_checks', {}).get('process_swap_peak', 0) == 0]
    screen_leader = max(r['decode_median'] for r in ordinary_screens)
    extra = []
    for kv in ['q4q8', 'q8q8']:
        choices = [r for r in ordinary_screens if r['config']['kv'] == kv
                   and r['config']['threads'] == 4
                   and r['config']['op_offload'] and r['config']['op_min'] == 32
                   and r['decode_median'] >= 0.97 * screen_leader]
        if choices:
            extra.append(max(choices, key=lambda r:r['decode_median']))
    t.save(t.ROOT / 'additional-confirmation-plan.json', dict(
        reason='Short-run ranking differences are smaller than the observed warm/steady gap. '
               'Confirm close four-thread candidates from the two leading KV types.',
        candidates=[dict(name=r['name'], config=r['config']) for r in extra]))
    for candidate in extra:
        t.guarded_screen(candidate['config'], prefix='repeat', tokens=512, repeats=3)
    repeats = [json.loads(p.read_text()) for p in t.OUT.glob('repeat_*/summary.json')]
    ranked = sorted((r for r in repeats if valid(r)),
                    key=lambda r:r['decode_median'], reverse=True)
    assert ranked, 'No valid three-round confirmation is available.'
    tie_break = None
    # For ordinary batch-1 decode at op_min=32, host expert matmuls do not
    # cross the CUDA offload threshold. Disabling op_offload changes the
    # prefill path; a near-tie does not justify selecting that slower path.
    first = ranked[0]
    if first['config'].get('spec', 'none') == 'none' and not first['config']['op_offload']:
        twin_config = {**first['config'], 'op_offload': True}
        twins = [r for r in ranked if r['config'] == twin_config]
        if twins and first['decode_median'] <= twins[0]['decode_median'] * 1.02:
            twin = twins[0]
            ranked.remove(twin)
            ranked.insert(0, twin)
            tie_break = dict(preferred=twin['name'], other=first['name'],
                             reason='Within 2% for otherwise identical ordinary-decode settings; '
                                    'retain automatic GPU op offload for prefill.')
    t.save(t.ROOT / 'final-selection.json', dict(
        policy='Validate the highest repeat median from a fresh process and cold prefill. '
               'For otherwise identical ordinary settings within 2%, retain op_offload for prefill. '
               'If it fails quality/stability, try the next repeat-confirmed candidate. '
               'If its final median is over 5% below another repeat-confirmed candidate, '
               'cold-validate that candidate too.',
        tie_break=tie_break,
        repeat_ranking=[dict(name=r['name'], config=r['config'],
                            decode_median=r['decode_median']) for r in ranked]))
    finals = []
    failures = []
    try:
        for candidate in ranked:
            if finals and candidate['decode_median'] <= max(r['decode_median'] for r in finals) * 1.05:
                continue
            name = final_name(candidate)
            existing = t.OUT / name / 'summary.json'
            try:
                if existing.exists():
                    result = json.loads(existing.read_text())
                else:
                    result = t.final(candidate['config'], name)
                if not valid(result) or not all(result['quality'].values()):
                    raise RuntimeError('Final result failed resource or two-fact recall checks.')
                finals.append(result)
            except Exception as error:
                failures.append(dict(name=name, error=str(error)))
                t.save(t.ROOT / 'final-validation-failures.json', failures)
                print('FINAL_FAILURE', name, str(error), flush=True)
            finally:
                t.b.stop_server()
        assert finals, 'All final candidates failed; no final recommendation is supported.'
        winner = max(finals, key=lambda r:r['decode_median'])
        t.save(t.ROOT / 'final-complete.json', dict(
            winner=winner, validated=[r['name'] for r in finals], failures=failures))
        t.status('final_validation_complete', winner=winner['name'],
                 decode_median=winner['decode_median'])
    finally:
        t.b.stop_server()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    ordinary = [r for r in ranked if r['config'].get('spec', 'none') == 'none']
    speculative = [r for r in ranked if r['config'].get('spec', 'none') != 'none']
    if (ordinary and speculative
            and speculative[0]['decode_median'] > ordinary[0]['decode_median'] * 1.05):
        subprocess.run([str(Path.home() / 'freetoken-test/.venv/bin/python'),
                        '-u', str(t.ROOT / 'workload_check.py')], check=True)
    t.status('experiment_complete', winner=winner['name'],
             decode_median=winner['decode_median'])


if __name__ == '__main__':
    main()
