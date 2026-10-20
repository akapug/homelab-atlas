// moe-trace: run a fixed chat workload greedily and log every MoE layer's router top-k expert ids per token.
// Built inside a llama.cpp tree as tools/moe-trace. Env: MOE_TRACE_WORKLOAD (jsonl with id + messages),
// MOE_TRACE_OUT (binary trace), MOE_TRACE_GEN (jsonl of generated text + timings).
// Record: int32 magic 'MOE2', prompt_id, phase (0 prefill, 1 decode), layer, pos0, k, n_tok, then n_tok*k int32 ids.
// pos0 = position of the record's first token, from a per-layer token counter reset per prompt. On the LAST layer
// llama.cpp keeps only output rows in prefill (inp_out_ids), so that layer's prefill rows are not every prompt token.
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct tracer {
    FILE * f = nullptr;
    int32_t pid = -1, phase = 0;
    std::vector<uint8_t> buf;
    std::vector<int32_t> ids;
    std::vector<int32_t> seen; // tokens seen per layer in this prompt
    long recs = 0;
};

static bool trace_cb(struct ggml_tensor * t, bool ask, void * ud) {
    const bool want = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask || !want) {
        return want || !ask;
    }
    auto * tr = (tracer *) ud;
    const int32_t il = atoi(t->name + 13), k = (int32_t) t->ne[0], n = (int32_t) t->ne[1];
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    const size_t nb = ggml_nbytes(t);
    tr->buf.resize(nb);
    ggml_backend_tensor_get(t, tr->buf.data(), 0, nb);
    tr->ids.resize((size_t) k * n);
    for (int32_t j = 0; j < n; j++) {
        for (int32_t i = 0; i < k; i++) {
            tr->ids[(size_t) j * k + i] = *(const int32_t *) (tr->buf.data() + j * t->nb[1] + i * t->nb[0]);
        }
    }
    if ((int32_t) tr->seen.size() <= il) {
        tr->seen.resize(il + 1, 0);
    }
    const int32_t hdr[7] = { 0x4D4F4532, tr->pid, tr->phase, il, tr->seen[il], k, n };
    tr->seen[il] += n;
    fwrite(hdr, sizeof(hdr), 1, tr->f);
    fwrite(tr->ids.data(), sizeof(int32_t), tr->ids.size(), tr->f);
    tr->recs++;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * wl = getenv("MOE_TRACE_WORKLOAD"), * out = getenv("MOE_TRACE_OUT"), * gen = getenv("MOE_TRACE_GEN");
    if (!wl || !out || !gen) {
        LOG_ERR("set MOE_TRACE_WORKLOAD, MOE_TRACE_OUT, MOE_TRACE_GEN\n");
        return 1;
    }
    tracer tr;
    tr.f = fopen(out, "wb");
    std::ofstream gout(gen);
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &tr;
    params.warmup = false;
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) {
        LOG_ERR("init failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tmpls = common_chat_templates_init(model, params.chat_template);
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::ifstream in(wl);
    std::string line;
    while (std::getline(in, line)) {
        auto row = nlohmann::json::parse(line);
        common_chat_templates_inputs inputs;
        for (auto & m : row["messages"]) {
            common_chat_msg msg;
            msg.role = m["role"];
            msg.content = m["content"];
            inputs.messages.push_back(msg);
        }
        inputs.add_generation_prompt = true;
        const std::string prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt;
        auto toks = common_tokenize(ctx, prompt, false, true);
        tr.pid = row["id"];
        tr.seen.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);
        const int64_t t0 = ggml_time_us();
        tr.phase = 0;
        common_batch batch = common_batch_get_one(ctx, toks);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("prefill failed on prompt %d\n", tr.pid);
            return 1;
        }
        const int64_t t1 = ggml_time_us();
        tr.phase = 1;
        std::string text;
        int n_gen = 0;
        for (; n_gen < params.n_predict; n_gen++) {
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }
            text += common_token_to_piece(ctx, id, true);
            common_batch b1 = common_batch_get_one(ctx, &id, 1);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b1.get())) {
                LOG_ERR("decode failed on prompt %d\n", tr.pid);
                return 1;
            }
        }
        const int64_t t2 = ggml_time_us();
        fflush(tr.f);
        nlohmann::json o = { { "id", tr.pid }, { "n_prompt", toks.size() }, { "n_gen", n_gen },
                             { "prefill_s", (t1 - t0) / 1e6 }, { "decode_s", (t2 - t1) / 1e6 },
                             { "prompt_head", prompt.substr(0, 160) }, { "text", text } };
        gout << o.dump() << "\n";
        gout.flush();
        LOG_INF("prompt %d: %zu prompt tok in %.1f s, %d gen in %.1f s, records %ld\n", tr.pid, toks.size(),
                (t1 - t0) / 1e6, n_gen, (t2 - t1) / 1e6, tr.recs);
    }
    fclose(tr.f);
    llama_sampler_free(smpl);
    llama_backend_free();
    return 0;
}
