# Qwen3 MoE with CPU expert offload

This extension adds Qwen3 MoE retrieval captures and exposes llama.cpp CPU expert placement in `llama-kvmem-server`. The target model family is `qwen3moe`, including Qwen3-30B-A3B-Instruct-2507 and Qwen3-Coder-30B-A3B-Instruct. Runtime validation is recorded separately; an architecture hook does not establish model quality.

## Placement

- `--cpu-moe` / `-cmoe`: keep all routed expert tensors on CPU.
- `--n-cpu-moe N` / `-ncmoe N`: keep routed expert tensors of the first N layers on CPU. N=0 disables this override.
- Environment: `LLAMA_ARG_CPU_MOE`, `LLAMA_ARG_N_CPU_MOE`. CLI values override the corresponding environment values; `--cpu-moe` takes precedence over a layer count.
- These reuse upstream expert tensor patterns. Attention, router, embeddings and non-expert tensors retain normal llama.cpp placement. Keep attention on GPU with `-ngl 99` for the CUDA KVMem adapter.
- This is layer-based placement, not dynamic caching of individual experts. CPU-resident routed experts normally run on CPU during decode; backend scheduling can choose GPU computation for larger prefill operations.

Example (tune CPU layer count for available VRAM):

```bash
build/bin/llama-kvmem-server -m model.gguf \
  -c 131072 -ngl 99 --n-cpu-moe 20 \
  --kvmem-budget 16384 --kvmem-gen-reserve 4096 \
  --kv-dtype q8_0 --spec-type none -fa on \
  -t 6 -tb 6 -b 512 -ub 512 --load-mode none
```

Compare with `--no-kvmem`, the same model, KV types and workload, increasing CPU expert layers only as needed to fit the full GPU KV. Account for generation reserve when comparing GPU cache sizes. Sweep CPU expert layers jointly with the KV working set; the smallest cache is not necessarily the best configuration.

## Captures and patches

`qwen3moe.cpp` captures normalized Q and K before RoPE, at the same points as the existing Qwen3 dense path. The optional V capture follows the existing adapter convention. Attention keeps original positions; routing and expert kernels are unchanged.

The cumulative `patches/llama-kvmem-current.patch` includes this change. `scripts/apply-patches.sh` also accepts a previously patched baseline through `patches/qwen3-moe-upgrade.patch`. No changes to the llama.cpp submodule pin are required.

The server saves partial recurrent checkpoints only for recurrent or hybrid models. On pure attention models, llama.cpp's partial-state flag still serializes the whole attention cache. Restoring that state would compact KV cells without updating KVMem's slot map. Pure attention rollback instead removes uncommitted rows and keeps the existing KV placement.

## Boundaries

This does not add Qwen3.5 MoE recurrent-state/MTP validation, SSD offload, concurrent slots, or a general per-expert cache. Qwen3 models without an MTP head use `--spec-type none`. Existing CPU RAM KV storage and retrieval are reused. Retrieval changes the attention working set and needs quality tests against full KV on each workload.

Verify short-context identity before testing eviction, then check retrieval after the history exceeds the GPU pool. Report cold prefill, cached-turn TTFT, decode, actual CPU/GPU placement, host RAM, VRAM, and recall quality separately. Report any slowdown as well as speedup.
