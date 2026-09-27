# Publication validation (2026-09-27)

This records the checks performed while exporting the MoE branch. It does not add GPU performance measurements.

| Check | Result |
|---|---|
| Apply cumulative patch to clean llama.cpp `b81c99b` | Passed |
| Apply the same script again | Already-applied detection passed |
| Upgrade original KVMem-patched tree, then repeat | Passed; resulting 39 patched files match the clean replay |
| Compare replayed files with the tested local MoE snapshot | All 39 patched llama.cpp files match |
| Parent tracked source comparison after applying the archived MoE patch | Matches local MoE snapshot |
| CPU storage tests, NVMe disabled | 5/5 passed |
| Existing server-options test, including MoE flags and environment variables | Passed |
| Exported file SHA256 checks | 46/46 passed |
| Input identity, medians, token counts, ranking and recall labels | Passed; 20 KVMem cases / 60 generation measurements |
| Python syntax and design/benchmark Markdown local links | Passed |

Commands for the CPU checks:

```bash
cmake -S . -B build-cpu -DKVMEM_BUILD_LLAMA=OFF \
  -DKVMEM_ENABLE_NVME=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j 4
ctest --test-dir build-cpu --output-on-failure
g++ -std=c++17 -O2 -Itools -Illama.cpp/include -Illama.cpp/ggml/include \
  tests/server-options-test.cpp -o build-cpu/kvmem-server-options-test
./build-cpu/kvmem-server-options-test
python3 benchmarks/moe-kv/verify_results.py
```

The publication environment had no CUDA compiler on PATH. No CUDA server rebuild, GPU benchmark, or complete long-context quality evaluation was performed in this step. Historical runtime evidence and failures are documented separately in the design document.
