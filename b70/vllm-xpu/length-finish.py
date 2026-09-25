"""A tool call cut off by max_tokens is reported as finish_reason "length", not "tool_calls"
(vllm/entrypoints/openai/chat_completion/serving.py in llm-scaler 0.26.0-b2).

The image answers "tool_calls" whenever a tool call was streamed (or parsed, non-streaming), even when
the engine stopped on the token limit. A turn cut inside a tool call's arguments then reads as a
complete tool call with broken JSON, and the client cannot know to continue. Measured with Claude Code
on Qwen3.8-27B: a turn that used exactly its 16,000 output tokens stopped inside a tool call's JSON, and
the client got a finished call it could not parse instead of a max_tokens stop it could resume from.
OpenAI's API reports "length" in that case. Upstream vLLM main fixed the streaming path the same way
(tool_calls only when the engine said "stop"); its non-streaming path still masks the length stop.
Unconditional: it changes only what a length stop is called. A proxy that translates to Anthropic stop
reasons must also let "length" win over a tool call it has already seen, or it masks the stop again.
Writes a .orig backup once and always patches from it.
  python3 length-finish.py <rootfs>/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/serving.py"""
import os
import shutil
import sys

p = sys.argv[1]
if not os.path.exists(p + ".orig"):
    shutil.copy2(p, p + ".orig")
s = open(p + ".orig").read()
for old, new in (
    ("                        if tools_streamed[i] and not tool_choice_function_name:\n",
     "                        # length-finish patch (homelab-atlas b70/vllm-xpu/length-finish.py): a length stop stays \"length\"\n"
     "                        if (tools_streamed[i] and not tool_choice_function_name\n"
     "                                and output.finish_reason == \"stop\"):\n"),
    ("            is_finish_reason_tool_calls = auto_tools_called or (\n",
     "            # length-finish patch (homelab-atlas b70/vllm-xpu/length-finish.py): a length stop stays \"length\"\n"
     "            is_finish_reason_tool_calls = (auto_tools_called and output.finish_reason == \"stop\") or (\n"),
):
    assert s.count(old) == 1, f"not found exactly once: {old.strip()}"
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"patched {p}")
