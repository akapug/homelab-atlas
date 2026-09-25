"""Grammar-constrained tool calls for tool_choice "auto" (vllm/tool_parsers/structural_tag_registry.py
in llm-scaler 0.26.0-b2).

The image already builds an xgrammar structural tag for the qwen3_coder parser: once the model
writes "<tool_call>\\n<function=", the name can only be a declared tool and the parameters must fit
that tool's schema. For tool_choice "auto" it does so only when some tool says "strict": true, and
Claude Code's tools (through an Anthropic-to-OpenAI proxy) never do, so the model got no constraint
and once emitted a tool named "Bash`\\n". With VLLM_STRICT_TOOLS_AUTO=1 every auto request with tools
gets the grammar. Text and thinking stay free: the tag only fires on the trigger, and the grammar waits
for the end of reasoning (enable_in_reasoning is off).
Measured: Claude Code's ~120 tools (built-ins + MCP servers) compile in 0.8 s for the whole set; the
grammar rejects "<function=Bash`" and accepts a real Bash call. The XPU bitmask (xgrammar's
torch_compile backend) compiles once in ~2 s and then applies in under 1 ms. A tool whose schema
xgrammar cannot validate (regex lookahead, an unresolved $ref) keeps its name constrained with free
arguments, and a combined tag that still fails means no constraint, never a 500. Unset, nothing
changes. Writes a .orig backup once and always patches from it.
  python3 strict-tools.py <rootfs>/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/structural_tag_registry.py"""
import os
import shutil
import sys

p = sys.argv[1]
if not os.path.exists(p + ".orig"):
    shutil.copy2(p, p + ".orig")
s = open(p + ".orig").read()
old = """    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None
"""
new = """    # strict-tools patch (homelab-atlas b70/vllm-xpu/strict-tools.py): VLLM_STRICT_TOOLS_AUTO=1 constrains
    # auto tool calls to the declared tools even when no tool says strict
    if tool_choice == "auto" and not _any_tool_strict(tools):
        if os.environ.get("VLLM_STRICT_TOOLS_AUTO") != "1":
            return None
        return _strict_tools_tag(model, tools, reasoning)
"""
assert s.count(old) == 1, "the auto/strict check was not found as expected"
s = s.replace(old, new)
head = "\nfrom collections.abc import Callable, Sequence\n"
assert s.count(head) == 1
s = s.replace(head, "\nimport os\nfrom collections.abc import Callable, Sequence\n", 1)
s += '''

# strict-tools patch (homelab-atlas b70/vllm-xpu/strict-tools.py): a schema xgrammar cannot take must never turn a
# request into a 500. Measured: one tool in a Claude Code tool set failed xgrammar's validation, vLLM's "auto"
# backend then fell back to guidance, which cannot read this tag format (KeyError 'triggers'), and every turn failed.
# xgrammar rejects e.g. regex lookahead in a pattern and a $ref that does not resolve. So each tool's schema is
# checked once (cached by content); a tool whose schema fails keeps its NAME constrained with free arguments; and a
# combined tag that still fails means no constraint.
_STRICT_SCHEMA_OK: dict[str, bool] = {}
_STRICT_TAG_OK: dict[str, bool] = {}


def _strict_copy(t, params):
    if isinstance(t, ChatCompletionToolsParam):
        return t.model_copy(update={"function": t.function.model_copy(update={"strict": True, "parameters": params})})
    return t.model_copy(update={"strict": True, "parameters": params})


def _strict_tools_tag(model, tools, reasoning):
    import json
    import logging

    import xgrammar as xgr

    log = logging.getLogger(__name__)

    def valid(tag):
        try:
            xgr.Grammar.from_structural_tag(json.dumps(tag.model_dump()))
            return True
        except Exception as e:  # noqa: BLE001 - any refusal means "do not constrain with this"
            log.warning("strict-tools: xgrammar refused a tag (%s)", str(e)[:200])
            return False

    out = []
    for t in tools:
        if not isinstance(t, (ChatCompletionToolsParam, FunctionTool)):
            out.append(t)
            continue
        fn = t.function if isinstance(t, ChatCompletionToolsParam) else t
        params = getattr(fn, "parameters", None) or {"type": "object"}
        key = json.dumps(params, sort_keys=True)
        if key not in _STRICT_SCHEMA_OK:
            try:
                one = get_model_structural_tag(model=model, tools=[_strict_copy(t, params)], tool_choice="auto", reasoning=reasoning)
            except Exception as e:  # noqa: BLE001
                log.warning("strict-tools: building %s's tag failed (%s)", getattr(fn, "name", "?"), str(e)[:200])
                one = None
            _STRICT_SCHEMA_OK[key] = one is not None and valid(one)
            if not _STRICT_SCHEMA_OK[key]:
                log.warning("strict-tools: tool %s keeps its name constrained; its arguments stay free", getattr(fn, "name", "?"))
        out.append(_strict_copy(t, params if _STRICT_SCHEMA_OK[key] else {"type": "object", "additionalProperties": True}))
    tag = get_model_structural_tag(model=model, tools=out, tool_choice="auto", reasoning=reasoning)
    if tag is None:
        return None
    key = json.dumps(tag.model_dump(), sort_keys=True)
    if key not in _STRICT_TAG_OK:
        _STRICT_TAG_OK[key] = valid(tag)
    return tag if _STRICT_TAG_OK[key] else None
'''
open(p, "w").write(s)
print("patched", p)
