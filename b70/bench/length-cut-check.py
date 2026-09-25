#!/usr/bin/env python3
"""Force a tool call to be cut by max_tokens on a vLLM server and print the streamed finish_reason
(patched image: length; unpatched: tool_calls).  length-cut-check.py PORT SERVED [MAX_TOKENS] [TOOL_CHOICE]"""
import json
import sys
import urllib.request

port, served = sys.argv[1], sys.argv[2]
body = {
    "model": served, "stream": True, "max_tokens": int(sys.argv[3]) if len(sys.argv) > 3 else 80, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False}, "tool_choice": sys.argv[4] if len(sys.argv) > 4 else "auto",
    "messages": [{"role": "user", "content": "Call the save_note tool once. Its text argument must be a 400-word essay about rivers."}],
    "tools": [{"type": "function", "function": {"name": "save_note", "description": "Save a note.",
               "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}],
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
finish, tool = None, False
with urllib.request.urlopen(req, timeout=300) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for c in json.loads(line[6:]).get("choices", []):
            tool = tool or bool(c.get("delta", {}).get("tool_calls"))
            finish = c.get("finish_reason") or finish
print(f"{served}: tool call streamed={tool} finish_reason={finish}")
