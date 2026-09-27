"""Queue the existing phase-2 search after the phase-1 GPU runner exits."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

root = Path.home() / 'llama-native-tuning-256k'
with (root/'runner.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    status = json.loads((root/'status.json').read_text())
    assert status['stage'] == 'complete' and status.get('mode') == 'phase1', status
    assert all((root/'cache'/f'{kv}.bin').exists() for kv in ('q4q4','q8q4','q5q5','q8q8'))
with (root/'phase2.log').open('w') as log:
    subprocess.run([sys.executable, '-u', str(root/'search_more.py')],
                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
