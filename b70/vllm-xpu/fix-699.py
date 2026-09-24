"""llm-scaler 0.26.0-b2 issue 699: SymInt4LinearMethod.apply() calls logger.info_once() on
the path torch.compile traces, and Dynamo refuses a logging.Logger method, so every compiled
mode (XPU graph or compile-only) fails at startup and only --enforce-eager serves.

Remove the two log calls from apply(). The only loss is the one-time "which int4 kernel runs"
line; the two calls at weight-load time (lines ~171, ~197) are not traced and stay.
Writes a .orig backup once and always patches from it."""
import os
import shutil
import sys

p = sys.argv[1]
if not os.path.exists(p + ".orig"):
    shutil.copy2(p, p + ".orig")
s = open(p + ".orig").read()
for msg in ("sym_int4 linear execution is using the guarded ESIMD path.",
            "sym_int4 linear execution is using the XPU W4A16 kernel."):
    call = f'''logger.info_once(
                "{msg}",
                scope="local",
            )
''' if "ESIMD" in msg else f'''logger.info_once(
            "{msg}",
            scope="local",
        )
'''
    assert s.count(call) == 1, msg
    # the call's own indentation stays in front of the comment, and the next line keeps its own
    s = s.replace(call, f"# issue 699: no logging inside the traced apply() path ({msg})\n")
open(p, "w").write(s)
print("patched", p)
