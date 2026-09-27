"""Audit saved measurements and produce a compact, reproducible ranking.

Run after copying the small result artifacts, excluding cache/*.bin.
This does not infer a global optimum from the bounded parameter search.
"""
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
EXPECTED_HASH = '4515ef3274b39b3cb6a3b04920793eb1b59f7dea36e01d521327d3027ed68df9'


def main():
    rows = []
    audits = []
    failures = []
    for path in sorted(OUT.glob('*/summary.json')):
        row = json.loads(path.read_text())
        cfg = row['config']
        assert row['prompt_tokens'] == 260992
        assert row.get('memory_checks',{}).get('process_swap_peak',0) == 0, path
        records = sorted(path.parent.glob('rep[0-9].json'))
        if not records and row['name'].startswith('seed_'):
            records = [OUT / (row['name'] + f"_u{cfg['ub']}") / 'cold.json']
        valid_records = []
        for record_path in records:
            record = json.loads(record_path.read_text())
            if record.get('error'):
                continue
            assert record['prompt_sha256'] == EXPECTED_HASH, record_path
            assert record['usage']['prompt_tokens'] == 260992, record_path
            assert record['usage']['completion_tokens'] in (256, 512), record_path
            assert record['finish'] == 'length', record_path
            timing = record['timings']
            assert max(timing.get('cache_n', 0), timing.get('prompt_n', 0)) >= 260988, record_path
            events = json.loads(record_path.with_suffix('.events.json').read_text())
            times = [e['elapsed_s'] for e in events if any(
                c.get('delta', {}).get('content') or c.get('delta', {}).get('reasoning_content')
                for c in e['event'].get('choices', []))]
            assert len(times) > 1, record_path
            client_tps = (record['usage']['completion_tokens'] - 1) / (times[-1] - times[0])
            engine_tps = timing['predicted_per_second']
            samples=[s for s in json.loads(record_path.with_suffix('.samples.json').read_text()) if 'error' not in s]
            audits.append(dict(case=row['name'], record=record_path.stem,
                               cache_n=timing.get('cache_n', 0), prompt_n=timing.get('prompt_n', 0),
                               output_tokens=record['usage']['completion_tokens'],
                               engine_tps=engine_tps, client_tps=client_tps,
                               difference_percent=100 * (client_tps / engine_tps - 1),
                               stream_content_events=len(times),
                               first256_client_tps=255/(times[255]-times[0]) if len(times)==512 else None,
                               last256_client_tps=255/(times[511]-times[256]) if len(times)==512 else None,
                               gpu_temp_median=statistics.median(s['gpu_temp'] for s in samples),
                               gpu_watts_median=statistics.median(s['gpu_watts'] for s in samples),
                               gpu_graphics_mhz_median=statistics.median(s['gpu_graphics_mhz'] for s in samples)
                                   if all('gpu_graphics_mhz' in s for s in samples) else None,
                               gpu_memory_mhz_median=statistics.median(s['gpu_memory_mhz'] for s in samples)
                                   if all('gpu_memory_mhz' in s for s in samples) else None,
                               system_swap_start=samples[0]['swap'],
                               system_swap_end=samples[-1]['swap'],
                               system_swap_range=max(s['swap'] for s in samples)-min(s['swap'] for s in samples),
                               process_swap_peak=max((s['process_swap'] for s in samples if 'process_swap' in s),default=None)))
            valid_records.append(record)
        assert valid_records, path
        assert len(valid_records) == len(row['decode_tps']), path
        assert abs(statistics.median(r['timings']['predicted_per_second'] for r in valid_records)
                   - row['decode_median']) < 1e-8, path
        cold = [r for r in valid_records if r['timings'].get('prompt_n', 0) >= 260988]
        draft_tokens = sum(r['timings'].get('draft_n', 0) for r in valid_records)
        accepted_tokens = sum(r['timings'].get('draft_n_accepted', 0) for r in valid_records)
        rows.append(dict(name=row['name'], kv=cfg['kv'], cpu_expert_layers=cfg['ncpu'],
                         gpu_expert_layers=48-cfg['ncpu'], threads=cfg['threads'],
                         ubatch=cfg['ub'], batch_threads=cfg['tb'], op_min=cfg['op_min'],
                         poll=cfg.get('poll', 'default'),
                         op_offload=cfg['op_offload'], speculation=cfg.get('spec', 'none'),
                         ngram_n=cfg.get('ngram_n',''), ngram_m=cfg.get('ngram_m',''),
                         draft_max=cfg.get('draft_max',''),
                         draft_tokens=draft_tokens, draft_tokens_accepted=accepted_tokens,
                         draft_acceptance_rate=accepted_tokens/draft_tokens if draft_tokens else '',
                         median_tps=row['decode_median'], min_tps=min(row['decode_tps']),
                         max_tps=max(row['decode_tps']), repetitions=len(row['decode_tps']),
                         output_tokens=','.join(map(str, row['outputs'])),
                         gpu_peak_gib=row['peaks']['gpu_mib']/1024,
                         rss_peak_gib=row['peaks']['rss']/2**30,
                         system_used_peak_gib=row['peaks']['system_used']/2**30,
                         system_swap_peak_mib=row['peaks']['swap']/2**20,
                         process_swap_peak_bytes=row.get('memory_checks',{}).get('process_swap_peak'),
                         temp_peak_c=row['peaks']['gpu_temp'], watts_peak=row['peaks']['gpu_watts'],
                         cold_prefill_tps=cold[0]['timings']['prompt_per_second'] if cold else '',
                         cold_ttft_s=cold[0]['ttft_s'] if cold else '',
                         quality=row.get('quality'), source=row['source']))
    for path in sorted(OUT.glob('*/failure.json')):
        failure=dict(name=path.parent.name, **json.loads(path.read_text()))
        log=path.parent/'server.log'
        if log.exists():
            errors=[line for line in log.read_text(errors='replace').splitlines()
                    if 'CUDA error:' in line]
            if errors: failure['server_error_lines']=errors[:3]
        failures.append(failure)
    rows.sort(key=lambda r: r['median_tps'], reverse=True)
    report = dict(status='measurement_integrity_passed' if rows else 'no_completed_measurements',
                  full_context_cases=len(rows), audited_requests=len(audits),
                  rows=rows, audits=audits, failures=failures,
                  limits=['Bounded search; no claim of global optimality.',
                          'Short capacity probes excluded from throughput ranking.',
                          'Snapshot screening and fresh cold validation labeled separately.',
                          'Ngram speculation is workload-dependent and reported separately.',
                          'Small static system swap occupancy is distinguished from model process swap and swap IO; phase-1 records only global occupancy.',
                          'Two-fact recall is not a comprehensive quality evaluation.'])
    (ROOT/'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if rows:
        with (ROOT/'ranking.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','audits')}, ensure_ascii=False, indent=2))
    for row in rows[:12]:
        print(f"{row['median_tps']:.3f} tok/s | {row['gpu_peak_gib']:.2f} GiB | {row['name']}")


if __name__ == '__main__':
    main()
