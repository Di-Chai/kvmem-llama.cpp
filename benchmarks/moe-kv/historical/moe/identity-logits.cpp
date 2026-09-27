#include "common.h"
#include "llama.h"
#include "llama-kvmem-hooks.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static std::vector<float> evaluate(llama_model * model, bool enabled, ggml_type dtype, const std::vector<llama_token> & prompt, const std::vector<llama_token> & forced, int method = 1) {
    llama_kvmem_params kp{};
    kp.enabled = enabled;
    kp.block_tokens = 128;
    kp.budget = 3072;
    kp.gen_reserve = 1024;
    kp.method = method;
    kp.query_begin = std::max(0, (int) prompt.size() - 64);
    kp.query_end = prompt.size();
    kp.force_pos = -1;
    llama_kvmem_set_params(&kp);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 6;
    cp.type_k = cp.type_v = dtype;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context init");
    for (size_t off = 0; off < prompt.size(); off += 512) {
        auto n = std::min(size_t(512), prompt.size() - off);
        auto batch = llama_batch_get_one(const_cast<llama_token *>(prompt.data() + off), n);
        if (llama_decode(ctx, batch)) throw std::runtime_error("prefill");
    }
    if (enabled) llama_kvmem_end_prefill_capture();
    const auto n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> result;
    for (auto token : forced) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        result.insert(result.end(), logits, logits + n_vocab);
        auto batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, batch)) throw std::runtime_error("decode");
    }
    llama_free(ctx);
    return result;
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    common_init();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    std::vector<llama_model_tensor_buft_override> overrides;
    llm_add_n_cpu_ffn_overrides(24, LLM_FFN_EXPS_REGEX, overrides);
    overrides.push_back({nullptr, nullptr});
    mp.tensor_buft_overrides = overrides.data();
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    std::string text;
    for (int i = 0; i < 40; ++i) text += "The engineering team reviews capacity planning, monitoring and backup recovery before releasing the distributed application.\n";
    auto * vocab = llama_model_get_vocab(model);
    const auto prompt = common_tokenize(vocab, text, true, true);
    const auto forced = common_tokenize(vocab, "Monitoring helps engineers detect unusual latency and find capacity bottlenecks before failures affect users. Recovery procedures should be tested regularly and documented clearly.", false, true);
    int nv = llama_vocab_n_tokens(vocab);
    for (auto dtype : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        auto a = evaluate(model, false, dtype, prompt, forced);
        for (int method : {0, 1}) {
        auto b = evaluate(model, true, dtype, prompt, forced, method);
        double squared = 0, maxabs = 0, kl = 0;
        size_t matched = 0;
        for (size_t row = 0; row < forced.size(); ++row) {
            const float * x = a.data() + row * nv;
            const float * y = b.data() + row * nv;
            double mx = *std::max_element(x, x + nv), my = *std::max_element(y, y + nv), sx = 0, sy = 0;
            matched += std::max_element(x, x + nv) - x == std::max_element(y, y + nv) - y;
            for (int j = 0; j < nv; ++j) {
                if (!std::isfinite(x[j]) || !std::isfinite(y[j])) return 4;
                double d = x[j] - y[j]; squared += d*d; maxabs = std::max(maxabs, std::abs(d));
                sx += std::exp(x[j] - mx); sy += std::exp(y[j] - my);
            }
            for (int j = 0; j < nv; ++j) {
                double lp = x[j] - mx - std::log(sx), lq = y[j] - my - std::log(sy);
                kl += std::exp(lp) * (lp-lq);
            }
        }
        std::printf("IDENTITY method=%d dtype=%s rows=%zu rmse=%.9f maxabs=%.9f mean_kl=%.9f top1=%zu/%zu\n", method, ggml_type_name(dtype), forced.size(), std::sqrt(squared/a.size()), maxabs, kl/forced.size(), matched, forced.size());
        std::fflush(stdout);
        }
    }
    llama_model_free(model);
    llama_backend_free();
}
