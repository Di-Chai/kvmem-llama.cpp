"""Cross-check server throughput against client-observed streamed decode."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / 'results/summary.json').read_text())
checks = []
for row in rows:
    streamed = []
    for index in range(3):
        prefix = ROOT / 'results' / row['name'] / f'rep{index}'
        result = json.loads(prefix.with_suffix('.json').read_text())
        events = json.loads(prefix.with_suffix('.events.json').read_text())
        assert result['error'] is None
        assert result['finish'] == 'length'
        assert result['usage']['completion_tokens'] == 512
        assert result['usage']['prompt_tokens'] == row['prompt_tokens']
        assert result['prompt_sha256'] == row['prompt_sha256']
        if index == 0:
            assert result['timings']['prompt_n'] >= 260096
        else:
            assert result['timings']['cache_n'] > 260096
        content_times = [event['elapsed_s'] for event in events
                         if any(choice.get('delta', {}).get('content')
                                or choice.get('delta', {}).get('reasoning_content')
                                for choice in event['event'].get('choices', []))]
        assert len(content_times) > 400, 'Unexpectedly coalesced stream'
        client_tps = 511 / (content_times[-1] - content_times[0])
        server_tps = result['timings']['predicted_per_second']
        streamed.append(client_tps)
        checks.append(dict(name=row['name'], repetition=index,
                           content_events=len(content_times), client_tps=client_tps,
                           server_tps=server_tps,
                           difference_percent=100 * (client_tps / server_tps - 1)))
    row['client_decode_tps_median'] = statistics.median(streamed)
assert rows[0]['prompt_sha256'] == rows[1]['prompt_sha256']
assert rows[0]['prompt_tokens'] == rows[1]['prompt_tokens']
validation = dict(
    status='passed', checks=checks,
    client_speedup=rows[1]['client_decode_tps_median'] / rows[0]['client_decode_tps_median'],
    server_speedup=rows[1]['decode_tps_median'] / rows[0]['decode_tps_median'],
    note='Client rate uses 511 token intervals between first and last content events; UTF-8 buffering may coalesce events.',
)
(ROOT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
print(json.dumps(validation, indent=2))
