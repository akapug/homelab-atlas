#!/usr/bin/env python3
"""Write a copy of a Qwen3.8 chat template that accepts the OpenAI reasoning-effort words.

Qwen3.8-27B's own template takes reasoning_effort xhigh (default), medium or low and raises on
anything else. An OpenAI-compatible client sends low, medium or high, and Claude Code through
cliproxy sends its effort level (default high): every such request failed with HTTP 400 "Unexpected
reasoning effort high" once vLLM served the model (2026-09-24; llama.cpp never passed the field to
the template). The copy maps high and max to xhigh, minimal and none to low, and a null to the
default, before the model's own check. Nothing else changes.

    effort-template.py <model dir> <output .jinja>
Serve it with vLLM's --chat-template (b70-vllm.sh: CHAT_TEMPLATE=<path under the models dir>).
"""
import sys

src, out = sys.argv[1] + "/chat_template.jinja", sys.argv[2]
t = open(src).read()
old = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
new = ("{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh', true) %}\n"
       "    {%- set resolved_reasoning_effort = {'high': 'xhigh', 'max': 'xhigh', 'minimal': 'low', 'none': 'low'}"
       ".get(resolved_reasoning_effort, resolved_reasoning_effort) %}")
if t.count(old) != 1:
    sys.exit(f"{src}: expected exactly one effort default line, found {t.count(old)}; not written")
open(out, "w").write(t.replace(old, new))
print(f"wrote {out}")
