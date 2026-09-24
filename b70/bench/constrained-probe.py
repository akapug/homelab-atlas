"""Probe a chat server with request shapes that speculative decoding could break: a JSON-schema
response, JSON mode, a tool call, and a reasoning (thinking) request."""
import json
import sys
import urllib.request

# Usage: constrained-probe.py [endpoint] [model]   (default: port 8083 on this host, and the
# first model its /v1/models lists)
EP = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8083/v1/chat/completions"
MODEL = sys.argv[2] if len(sys.argv) > 2 else json.loads(urllib.request.urlopen(
    EP.replace("/chat/completions", "/models"), timeout=30).read())["data"][0]["id"]


def ask(body):
    body = {"model": MODEL, "temperature": 0, **body}
    r = urllib.request.Request(EP, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


schema = {"type": "object", "properties": {"failed": {"type": "array", "items": {"type": "string"}},
                                           "count": {"type": "integer"}}, "required": ["failed", "count"]}
msg = [{"role": "user", "content": "Tests test_a and test_b failed, test_c passed. Report the failed tests."}]
no_think = {"chat_template_kwargs": {"enable_thinking": False}}

r = ask({"messages": msg, "max_tokens": 200, **no_think,
         "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": schema}}})
out = json.loads(r["choices"][0]["message"]["content"])
print("json_schema:", out, "OK" if sorted(out["failed"]) == ["test_a", "test_b"] and out["count"] == 2 else "WRONG")

r = ask({"messages": msg + [{"role": "user", "content": "Answer as a JSON object."}], "max_tokens": 200, **no_think,
         "response_format": {"type": "json_object"}})
print("json_object:", "OK" if isinstance(json.loads(r["choices"][0]["message"]["content"]), dict) else "WRONG")

tools = [{"type": "function", "function": {"name": "rerun", "description": "Rerun one test",
                                           "parameters": {"type": "object", "properties": {"test": {"type": "string"}},
                                                          "required": ["test"]}}}]
r = ask({"messages": [{"role": "user", "content": "Rerun test_a using the tool."}], "tools": tools, "max_tokens": 300, **no_think})
calls = r["choices"][0]["message"].get("tool_calls") or []
print("tool call:", [(c["function"]["name"], c["function"]["arguments"]) for c in calls], "OK" if calls else "NONE")

r = ask({"messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}], "max_tokens": 1500})
m = r["choices"][0]["message"]
print("thinking:", repr((m.get("content") or "").strip()[-20:]), "reasoning chars", len(m.get("reasoning_content") or m.get("reasoning") or ""),
      "OK" if "391" in (m.get("content") or "") else "CHECK")
t = r.get("timings", {})
print("draft stats:", {k: t[k] for k in t if "draft" in k})
