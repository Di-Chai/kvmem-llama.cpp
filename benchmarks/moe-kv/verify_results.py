#!/usr/bin/env python3
"""Audit the exported historical measurements; does not run inference."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def median_matches(row, key):
    rounds = row['decode_tps']
    require(len(rounds) == 3, f"Expected three rounds: {row['name']}")
    require(all(math.isfinite(x) and x > 0 for x in rounds), 'Invalid throughput')
    require(math.isclose(statistics.median(rounds), row[key], rel_tol=1e-12),
            f"Median mismatch: {row['name']}")


def prompt_hash(relative):
    # Match the original runners' JSON serialization, not the file byte hash.
    return hashlib.sha256(json.dumps(read(relative)).encode()).hexdigest()


def main():
    manifest = read('manifest.json')
    for entry in manifest['files']:
        path = ROOT / entry['exported']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == entry['exported_sha256'], f'Checksum mismatch: {path}')

    hashes = {
        65536: prompt_hash('inputs/prompt-64k.json'),
        131072: prompt_hash('inputs/prompt-128k.json'),
        262144: prompt_hash('inputs/prompt-256k.json'),
    }
    ablation = read('data/ablation.json')
    require(len(ablation) == 6, 'Expected six ablation configurations')
    for row in ablation:
        median_matches(row, 'decode_tps_median')
        require(row['prompt_sha256'] == hashes[row['target']], 'Ablation input mismatch')
        require(row['output_tokens'] == [256] * 3, 'Ablation output mismatch')

    full = read('data/full256-comparison.json')
    require(len(full) == 2, 'Expected two original 256K arms')
    for row in full:
        median_matches(row, 'decode_tps_median')
        require(row['prompt_sha256'] == hashes[262144], '256K input mismatch')
        require(row['prompt_tokens'] == 260992, '256K input length mismatch')
        require(row['output_tokens'] == 512, '256K output length mismatch')

    native = read('data/native-final.json')
    median_matches(native, 'decode_median')
    require(native['outputs'] == [512] * 3, 'Native output length mismatch')
    require(native['prompt_tokens'] == 260992, 'Native input length mismatch')

    cases = {}
    for path in sorted((ROOT / 'data/kvmem-cases').glob('*.json')):
        row = json.loads(path.read_text())
        median_matches(row, 'decode_median')
        require(row['prompt_sha256'] == hashes[262144], 'Tuning input mismatch')
        require(row['prompt_tokens'] == 260992 and row['outputs'] == [512] * 3,
                f"Tuning token counts: {row['name']}")
        require(row['process_swap_peak'] == 0, f"Process swapped: {row['name']}")
        cases[row['name']] = row
    require(len(cases) == 20, 'Expected 20 completed tuning configurations')

    with (ROOT / 'data/kvmem-ranking.csv').open(newline='') as stream:
        ranking = list(csv.DictReader(stream))
    require(len(ranking) == 20, 'Ranking count mismatch')
    require({row['name'] for row in ranking} == set(cases), 'Ranking case mismatch')
    for entry in ranking:
        row = cases[entry['name']]
        require(math.isclose(float(entry['decode_median']), row['decode_median'], rel_tol=1e-12),
                'Ranking median mismatch')
        require((entry['recall_pass'] == 'True') == all(row['quality'].values()),
                'Ranking quality flag mismatch')

    recommended = cases['final_quality_k64_q8q8_n21_t4_u512']
    speed = cases['final_edge_k16_q4q8_n11_t4_u256']
    require(all(recommended['quality'].values()), 'Recommended case failed smoke recall')
    require(sum(speed['quality'].values()) == 1, 'Speed-only case must retain its failed recall')
    arms = {row['name']: row for row in ablation}
    placement_gain = arms['k16_64k']['decode_tps_median'] / arms['k16_64k_control']['decode_tps_median']
    print(f"Checksums: {len(manifest['files'])} files OK")
    print('Inputs, medians, output lengths, ranking and recall labels: OK')
    print('KVMem tuning: 20 configurations, 60 generation measurements')
    print(f'Extra expert placement at fixed 16K working set: {placement_gain:.3f}x')
    print(f"Tuned candidate/native: {recommended['decode_median'] / native['decode_median']:.3f}x")
    print(f"Speed-only candidate: {speed['decode_median']:.3f} tok/s, recall 1/2")
    print('This audits archived evidence; it is not a new GPU benchmark or a full quality evaluation.')


if __name__ == '__main__':
    main()
