# Glossary

The words and shorthand this repo uses, in plain terms. Alphabetical within each group.

## Speed and size

- **tok/s**: tokens (word pieces) per second. **Decode** is the speed of writing the answer, one token after
  another; **prefill** (or "prompt") is the speed of reading the prompt, which runs many tokens at once and is much
  faster. Unless a line says "summed", a decode figure is for one request.
- **Stream**: one request being answered. "351 summed over 8 streams" = eight requests at once, their speeds added.
- **Depth**: how many tokens of context the request already holds. Decode slows as depth grows.
- **Window** (context window, `-c`, `--max-model-len`): the most tokens one request may hold, prompt plus answer.
- **bpw**: bits per weight, averaged over the model. A 3.05 bpw store keeps about 3 bits for each parameter.
- **25 tok/s bar**: the decode speed at which we would hand a model an agent's coding work. Not a law of nature;
  our line.

## Models

- **MoE** (mixture of experts): a model whose feed-forward layers are split into many "experts", of which each token
  uses a few. DeepSeek V4.1 Flash is 748B in total but reads a small fraction of that per token, which is why it can
  run from system memory at all.
- **Routed experts / picks**: the experts the model's router chooses for each token on each layer (8 per layer for
  GLM-5.3-Flash). **Shared experts** run for every token. **Dense weights**: everything that is not a routed expert
  (attention, shared experts, the output head).
- **Hot / cold experts**: experts that are picked often or rarely on real prompts. We count picks on coding prompts
  and put the hot ones in the fastest memory.
- **Active parameters**: the parameters one token actually reads ("35B-A3B" = 35B total, about 3B active).
- **Dense model**: every parameter is used for every token (Qwen3.8-27B).

## Quantization and stores

- **Quantization**: storing weights in fewer bits. Names you will see: **int4**, **AutoRound** and **GPTQ** (4-bit
  checkpoints for vLLM), **GGUF** files with llama.cpp types such as **Q4_K_M**, **Q8_0**, **IQ3_S**, **UD-Q4_K_XL**
  (unsloth's mixed-precision recipe), **EXL3** (exllamav3's format), **MXFP4** (the 4-bit float format DeepSeek
  released V4.1's experts in).
- **Ternary**, **PTQ1_0**: weights that are only -1, 0 or +1, with a scale per group. PTQ1_0 is PrismML's GGUF
  type for them: 1.75 bits a weight, packed in base 3. PQ2_0 is their 2-bit sibling.
- **Release precision**: the weights as the model's makers published them, no further quantization.
- **Split store**: one model file in which hot experts keep a higher precision and cold experts a lower one.
- **Route trim**: computing only the top few of a token's expert picks that are not in VRAM, on some layers, and
  skipping the rest. Faster, and it changes the output a little, so it must pass the quality check.

## Speculative decoding

- **Speculative decoding**: a cheap guesser drafts a few next tokens and the model checks them all in one pass. When
  the guesses are right, several tokens come out for the price of one pass.
- **MTP** (multi-token prediction): a small guessing layer that ships inside the model (Qwen3.5-family, GLM,
  Ornith). "MTP 3" = three drafted tokens a step.
- **Draft model**: a separate small model used as the guesser. **DSpark** is the draft model we use for DeepSeek V4.1.
- **n4 / n0**: four drafted tokens a step / no drafting. **n_max**: the most tokens drafted a step.
- **Acceptance**: the share of drafted tokens the model keeps.
- **Reduced draft vocabulary** (`draft-vocab`): the guesser scores only the most common token ids instead of the
  whole vocabulary, which makes each guess cheaper. +25-29 % at one stream on our cards.
- **Block drafter**: a guesser that drafts many tokens at once and is verified as a block. It did not pay on our
  boxes.

## Memory and placement

- **VRAM**: the GPU's own memory. **DRAM**: the system's RAM.
- **Optane PMem** (persistent memory): Intel memory modules that sit in DIMM slots, larger and slower than DRAM,
  much faster than an SSD. We run them in **App Direct** mode, onlined as a NUMA node with no CPUs.
- **NUMA node**: a group of cores with the memory attached to them. A two-socket EPYC 7601 server has 8; reading
  another node's memory is slower than reading your own.
- **Row split**: cutting each expert's (or dense weight's) rows into one slice per NUMA node, so each node's threads
  compute on their own memory. Our biggest single lever on the two-socket boxes.
- **Expert bank**: a GPU holding a set of routed experts, computing them there instead of on the CPU.
- **Tier split**: our change that gives CPU threads reading PMem-resident experts less work than those reading DRAM,
  so no thread waits on the slowest one.
- **KV cache**: the memory that holds the conversation so far. **fp8 / int8 KV**: kept in 8 bits to fit more.
  **k8v4**: keys in 8 bits, values in 4.
- **Agent slot**: the context one agent may grow to on a batched server; a 131,072-token slot does not mean the
  prompt is that long.

## Quality checks

- **KL** (KL divergence) and **top-1**: how far a fast configuration's next-token probabilities drift from a
  reference run of the same model, over the same prompts. Lower KL is closer; top-1 is how often both pick the same
  next token. Our bar: **KL at most 0.03 and top-1 at least 0.97**.
- **Recall probe**: facts planted at known depths in a long prompt; the model is asked for them at the end.
- **Agent eval**: our 12 repository tasks with hidden checks, run by Claude Code in a sandbox
  ([b70/agent-eval](b70/agent-eval/)).
- **Code-review screen**: a change with planted defects and a clean control; does the model find the defects
  without inventing any?
- **Measured / inferred / unknown**: how sure a number is. Recipe Maker tags these `seen`, `inferred`, `unknown`.

## Software

- **vLLM**, **llama.cpp**, **ik_llama.cpp**: serving engines. vLLM is Python and serves many requests well;
  llama.cpp is C++ and runs GGUF files on almost anything; ik_llama.cpp is a llama.cpp fork strong at CPU and MoE
  work, and [our fork of it](https://github.com/akapug/ik_llama.cpp) carries the engine changes in
  [moe-offload](moe-offload/#engine-changes).
- **llm-scaler**: Intel's Docker image of vLLM for Arc GPUs. **XPU**: what PyTorch and vLLM call an Intel GPU.
  **SYCL**: the programming layer of llama.cpp's Intel backend. **Xe2**: the B70's GPU generation. **XMX** (or
  **DPAS**, its instruction): the matrix engines of an Intel GPU, used for batches of tokens.
- **Strata / XeStrata**: an engine that serves Qwen3.8 Flash-Next with its experts in RAM and a VRAM expert cache;
  XeStrata is its port to Intel GPUs.
- **RDMA** / **RoCE**: one machine writing into another's memory through the network card, without the CPU copying
  each packet; RoCE is RDMA over Ethernet. **IPoIB**: ordinary IP traffic carried over InfiniBand.
- **Tool call**: the model asking its client to run something. **Agent client**: a program such as Claude Code
  that works through tool calls. **Proxy**: translates Anthropic's API to OpenAI's so Claude Code can talk to a
  local server.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
