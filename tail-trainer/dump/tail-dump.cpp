// tail-dump: run text through a GGUF model with stock llama.cpp and write, per window of tokens, the residual stream
// that enters the chosen blocks and the engine's own logits at a few positions. A torch tail (tail_trainer.py) starts
// from that residual and is checked against those logits.
//
//   tail-dump -m MODEL.gguf --out DIR --layers 40,44 [--ctx 1024] [--logit-rows 32] [--max-seqs N] [-t THREADS]
//             [--no-mmap] [--lora A.gguf ...] TEXT_FILE [TEXT_FILE ...]
//
// Each text file is tokenized whole (no special-token parsing; BOS when the model asks for one, at every window
// start) and cut into windows of --ctx tokens; a last window shorter than 64 tokens is dropped. Each window is one
// llama_decode of one ubatch (n_ubatch = --ctx), from an empty cache.
// --layers L: the input of block L, the graph tensor "l_out-(L-1)" ([n_tokens, n_embd]); L >= 1.
// --logit-rows K: logits are kept at K positions spread evenly over the second half of the window, the last included.
// --no-mmap: read the model into process memory (placed by the NUMA policy, e.g. numactl --interleave=0,1) instead
// of mapping the page cache, which may sit on a slow memory node.
// --lora: apply these adapters at scale 1 (the engine side of an adapter check); repeatable.
// Out, per window: DIR/seq-NNNNN/{tokens.i32, l_out-<L-1>.f32 [n_tokens, n_embd], logits.f32 [K, n_vocab],
// logit_pos.i32 [K], meta.tsv}; DIR/index.tsv: seq, source file, first token of the window, tokens.
// Needs only llama.h and ggml-backend.h (the public API): build it against a llama.cpp checkout with build.sh.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

struct dump_state {
    std::set<std::string>                     names;
    std::map<std::string, int64_t>            width;
    std::map<std::string, std::vector<float>> rows;
    std::string                               error;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    if (st->names.count(t->name) == 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    const std::string name = t->name;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        st->error = name + ": not a contiguous F32 tensor";
        return true;
    }
    const int64_t n = ggml_nelements(t);
    st->width[name] = t->ne[0];
    auto & buf = st->rows[name];
    const size_t at = buf.size();
    buf.resize(at + n);
    ggml_backend_tensor_get(t, buf.data() + at, 0, n * sizeof(float));
    return true;
}

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void write_file(const std::string & path, const void * p, size_t bytes) {
    FILE * o = fopen(path.c_str(), "wb");
    if (!o || fwrite(p, 1, bytes, o) != bytes) {
        fprintf(stderr, "tail-dump: cannot write %s\n", path.c_str());
        exit(1);
    }
    fclose(o);
}

static std::vector<int> logit_positions(int n, int k) {
    std::vector<int> pos;
    if (k <= 0) return pos;
    const int lo = n / 2;
    for (int i = 0; i < k; ++i) {
        const int p = k == 1 ? n - 1 : lo + (int) ((long long) (n - 1 - lo) * i / (k - 1));
        if (pos.empty() || pos.back() != p) pos.push_back(p);
    }
    return pos;
}

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf --out DIR [--layers L,L] [--ctx 1024] [--logit-rows 32] [--max-seqs N] "
            "[-t THREADS] [--no-mmap] [--lora A.gguf ...] TEXT_FILE [TEXT_FILE ...]\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path, out_dir, layers;
    int n_ctx = 1024, n_logit = 32, max_seqs = 0, n_threads = 0;
    bool use_mmap = true;
    std::vector<std::string> texts, loras;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); exit(2); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--out") out_dir = next();
        else if (a == "--layers") layers = next();
        else if (a == "--ctx") n_ctx = atoi(next().c_str());
        else if (a == "--logit-rows") n_logit = atoi(next().c_str());
        else if (a == "--max-seqs") max_seqs = atoi(next().c_str());
        else if (a == "-t" || a == "--threads") n_threads = atoi(next().c_str());
        else if (a == "--lora") loras.push_back(next());
        else if (a == "--no-mmap") use_mmap = false;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (!a.empty() && a[0] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0]); return 2; }
        else texts.push_back(a);
    }
    if (model_path.empty() || out_dir.empty() || texts.empty() || n_ctx < 64 || n_logit < 0) {
        usage(argv[0]);
        return 2;
    }
    dump_state st;
    std::vector<int> blocks;
    {
        std::istringstream ss(layers);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            const int L = atoi(tok.c_str());
            if (L < 1) { fprintf(stderr, "--layers: a tail starts at block 1 or later\n"); return 2; }
            blocks.push_back(L);
            st.names.insert("l_out-" + std::to_string(L - 1));
        }
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    if (!use_mmap) mp.load_mode = LLAMA_LOAD_MODE_NONE;   // llama.h of 2026-11 (older trees: mp.use_mmap = false)
    const double t_load = now_s();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "tail-dump: model load failed\n"); return 1; }
    for (int L : blocks) {
        if (L >= llama_model_n_layer(model)) {
            fprintf(stderr, "tail-dump: block %d does not exist (the model has %d)\n", L, llama_model_n_layer(model));
            return 2;
        }
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.n_seq_max = 1;
    if (n_threads > 0) { cp.n_threads = n_threads; cp.n_threads_batch = n_threads; }
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = &st;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "tail-dump: context init failed\n"); return 1; }
    std::vector<llama_adapter_lora *> ads;
    std::vector<float> scales;
    for (const auto & p : loras) {
        llama_adapter_lora * ad = llama_adapter_lora_init(model, p.c_str());
        if (!ad) { fprintf(stderr, "tail-dump: cannot load adapter %s\n", p.c_str()); return 1; }
        ads.push_back(ad);
        scales.push_back(1.0f);
    }
    if (!ads.empty() && llama_set_adapters_lora(ctx, ads.data(), ads.size(), scales.data()) != 0) {
        fprintf(stderr, "tail-dump: cannot set the adapters\n");
        return 1;
    }
    fprintf(stderr, "tail-dump: load %.1f s, %d adapter(s)\n", now_s() - t_load, (int) ads.size());

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    mkdir(out_dir.c_str(), 0755);
    FILE * index = fopen((out_dir + "/index.tsv").c_str(), "w");
    if (!index) { fprintf(stderr, "tail-dump: cannot write %s/index.tsv\n", out_dir.c_str()); return 1; }
    fprintf(index, "seq\tsource\tstart\ttokens\n");
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    int seq = 0;
    const double t0 = now_s();
    for (const auto & path : texts) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "tail-dump: cannot read %s\n", path.c_str()); return 1; }
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string text = ss.str();
        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
        }
        toks.resize(std::max(n, 0));
        const int body = n_ctx - (add_bos ? 1 : 0);
        for (size_t start = 0; start < toks.size(); start += body) {
            if (max_seqs > 0 && seq >= max_seqs) break;
            std::vector<llama_token> w;
            if (add_bos) w.push_back(llama_vocab_bos(vocab));
            const size_t end = std::min(toks.size(), start + body);
            w.insert(w.end(), toks.begin() + start, toks.begin() + end);
            const int N = (int) w.size();
            if (N < 64) break;
            const std::vector<int> lpos = logit_positions(N, n_logit);
            std::set<int> want(lpos.begin(), lpos.end());
            for (auto & kv : st.rows) kv.second.clear();
            llama_memory_clear(llama_get_memory(ctx), true);
            batch.n_tokens = 0;
            for (int j = 0; j < N; ++j) {
                batch.token[j] = w[j];
                batch.pos[j] = j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = want.count(j) ? 1 : 0;
            }
            batch.n_tokens = N;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "tail-dump: llama_decode failed\n"); return 1; }
            llama_synchronize(ctx);
            if (!st.error.empty()) { fprintf(stderr, "tail-dump: %s\n", st.error.c_str()); return 1; }
            char name[32];
            snprintf(name, sizeof(name), "seq-%05d", seq);
            const std::string d = out_dir + "/" + name;
            mkdir(d.c_str(), 0755);
            std::vector<float> logits;
            logits.reserve(lpos.size() * (size_t) n_vocab);
            for (int p : lpos) {
                const float * lg = llama_get_logits_ith(ctx, p);
                logits.insert(logits.end(), lg, lg + n_vocab);
            }
            write_file(d + "/tokens.i32", w.data(), w.size() * sizeof(int32_t));
            write_file(d + "/logits.f32", logits.data(), logits.size() * sizeof(float));
            write_file(d + "/logit_pos.i32", lpos.data(), lpos.size() * sizeof(int32_t));
            FILE * meta = fopen((d + "/meta.tsv").c_str(), "w");
            fprintf(meta, "tokens\t%d\nlogit_rows\t%d\nn_vocab\t%d\nadapters\t%d\n", N, (int) lpos.size(), n_vocab,
                    (int) ads.size());
            for (const auto & nm : st.names) {
                const auto & buf = st.rows[nm];
                const int64_t W = st.width[nm];
                if (buf.empty() || W <= 0 || (int64_t) buf.size() != W * N) {
                    fprintf(stderr, "tail-dump: %s: %zu values, expected %d rows (never evaluated, or cropped)\n",
                            nm.c_str(), buf.size(), N);
                    return 1;
                }
                write_file(d + "/" + nm + ".f32", buf.data(), buf.size() * sizeof(float));
                fprintf(meta, "%s\t%d\t%lld\n", nm.c_str(), N, (long long) W);
            }
            fclose(meta);
            fprintf(index, "%s\t%s\t%zu\t%d\n", name, path.c_str(), start, N);
            fflush(index);
            ++seq;
            fprintf(stderr, "tail-dump: %s %d tokens, %.1f s\n", name, N, now_s() - t0);
        }
    }
    fclose(index);
    fprintf(stderr, "tail-dump: done, %d windows, %.1f s\n", seq, now_s() - t0);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return seq > 0 ? 0 : 1;
}
