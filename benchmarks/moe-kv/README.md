# MoE KV/expert experiments

The [design document (中文)](../../docs/moe-kv-design.md) describes the implementation, measurement protocol, results, and limitations. These are historical Alienware results, not new measurements performed during publication.

## Contents

- `inputs/`: exact synthetic chat messages for the 64K, 128K and 256K experiments. These are not RULER datasets. The final questions do not disclose the planted answers.
- `data/`: numerical summaries, rankings, timing/checkpoint checks, and all 20 completed KVMem tuning cases. Memory quantities retain their original byte/MiB units.
- `historical/`: the original experiment runners and selected analysis scripts. They retain the original WSL directory assumptions and are not a portable one-command suite.
- `manifest.json`: source and exported SHA256 values. Only machine-specific command paths in `full256-comparison.json` are normalized to `$HOME/`; those JSON strings are records, not shell-expanded commands.
- `verify_results.py`: verify exported checksums, input identity, medians, and quality flags without a GPU.
- [Publication validation](publication-validation.md): checks performed for this export, separated from historical GPU measurements.

Model weights, build outputs, KV snapshots, full logs, SSE events, memory samples and generated responses are not included. Some historical analysis scripts need these omitted raw files; use them after generating a new run, not as if this were a complete raw-data archive.

```bash
python3 benchmarks/moe-kv/verify_results.py
```

## Build and model

Build this branch from source; upstream prebuilt binaries do not contain the MoE adaptation. The commands below use SM86 for the measured RTX 3080 Ti Laptop. Choose the architecture for your hardware and use a validated CUDA toolchain; the measured compiler was 13.2.86.

```bash
git clone --branch moe-kv-offload --recurse-submodules \
  https://github.com/Di-Chai/kvmem-llama.cpp.git "$HOME/kvmem-moe-test"
cd "$HOME/kvmem-moe-test"
bash scripts/apply-patches.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DKVMEM_ENABLE_NVME=OFF
cmake --build build -j 4
```

Use `models/Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`. Verify SHA256 `382b4f5a164d200f93790ee0e339fae12852896d23485cfb203ce868fea33a95`. The source and exact revision are in the design document. If downloading, use `HF_ENDPOINT=https://hf-mirror.com` and unset HTTP/HTTPS/ALL proxy variables.

Build the **unpatched** baseline from the submodule's committed tree, not its patched working directory:

```bash
mkdir -p "$HOME/kvmem-moe-256k/native-source"
git -C llama.cpp archive b81c99b479d4c24e5eeca10de99032ebd343ef8f |
  tar -x -C "$HOME/kvmem-moe-256k/native-source"
cmake -S "$HOME/kvmem-moe-256k/native-source" \
  -B "$HOME/kvmem-moe-256k/native-build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build "$HOME/kvmem-moe-256k/native-build" --target llama-server -j 4
```

## Re-run using the historical WSL layout

Use a fresh experiment directory layout. These runners cache completed cases by name; reusing old result directories may skip measurement. Run one experiment process at a time.

```bash
python3 -m venv .venv
.venv/bin/pip install psutil
source .venv/bin/activate
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
mkdir -p "$HOME/llama-native-tuning-256k" "$HOME/kvmem-tuning-256k"
cp benchmarks/moe-kv/historical/moe/experiment.py ./experiment.py
cp benchmarks/moe-kv/historical/moe/suite.py ./suite.py
cp benchmarks/moe-kv/inputs/prompt-64k.json ./prompt_65536.json
cp benchmarks/moe-kv/inputs/prompt-128k.json ./prompt_131072.json
cp benchmarks/moe-kv/historical/moe-256k/bench256.py "$HOME/kvmem-moe-256k/"
cp benchmarks/moe-kv/inputs/prompt-256k.json "$HOME/kvmem-moe-256k/prompt.json"
cp benchmarks/moe-kv/historical/native-tuning/tune.py "$HOME/llama-native-tuning-256k/"
cp benchmarks/moe-kv/historical/kvmem-tuning/tune.py "$HOME/kvmem-tuning-256k/"
```

Before launching, adapt `LD_LIBRARY_PATH` in `experiment.py` and `bench256.py` to your CUDA runtime. The original paths refer to the measured WSL installation. The samplers use `/usr/lib/wsl/lib/nvidia-smi`, a single GPU, and Linux `/proc`; on native Linux replace that path with your `nvidia-smi` location. The scripts clear inherited proxy variables for the server and use a no-proxy HTTP client.

Initial 64K/128K ablations, then the original 256K comparison:

```bash
cd "$HOME/kvmem-moe-test"
python3 experiment.py sanity
python3 - <<'PY'
import experiment as e
cases = [
    ('full_64k', 65536, 20, None),
    ('k16_64k_control', 65536, 20, 16384),
    ('k16_64k', 65536, 14, 16384),
    ('k32_64k', 65536, 16, 32768),
    ('full_128k', 131072, 30, None),
    ('best_128k', 131072, 14, 16384),
]
try:
    for name, context, cpu_layers, budget in cases:
        e.run_case(name, context, cpu_layers, budget)
finally:
    e.stop_server()
PY
python3 "$HOME/kvmem-moe-256k/bench256.py" smoke
python3 "$HOME/kvmem-moe-256k/bench256.py" native
python3 "$HOME/kvmem-moe-256k/bench256.py" kvmem --ncpu 20
```

The archived `suite.py` was an orchestration script waiting for `download.done`, `sanity.pid` and `validation.done` from the original session. The direct calls above avoid those historical orchestration dependencies. Validate the model and short-context outputs before starting long runs.

Re-run the optimized native candidate (the final mode performs three 512-token measurements):

```bash
python3 "$HOME/llama-native-tuning-256k/tune.py" final \
  --kv q4q8 --ncpu 40 --threads 10 --ub 512 --tb 6 --op-min 32 \
  --name reproduced_native_q4q8_n40
```

Re-run the KVMem candidate with more VRAM headroom:

```bash
cd "$HOME/kvmem-tuning-256k"
python3 - <<'PY'
import tune
try:
    result = tune.run(tune.config(65536, 'q8q8', 21, t=4, ub=512), 'reproduced')
    if not tune.good(result):
        raise SystemExit('The resource/recall checks did not pass')
finally:
    tune.b.stop_server()
PY
```

The full KVMem search is `python3 "$HOME/kvmem-tuning-256k/tune.py"`. It performs real cold prefill per candidate and can take many hours. The historical search also used follow-up scripts to probe the placement boundary; the exported ranking records the completed cases, not a guarantee that a fresh search returns the same optimum.

## Interpretation

The headline 3.26x result compares Q8 KV with all 48 expert layers on CPU against a 64K+4K retrieval window and 28 expert layers on GPU. The later 3.07x result compares separately tuned configurations with different KV precisions. Neither is a full-attention equivalence claim.

The 16K result around 34 tok/s fails one of two historical fact checks. Passing both checks at 64K is only a smoke test, not a quality guarantee. Report cold prefill, cached TTFT, decode, memory and recall separately. Keep failed candidates and CUDA errors in the new run's records.
