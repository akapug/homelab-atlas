"""Text summaries of parsed requests."""
from collections import Counter


def summarize(requests, malformed, top=5):
    """Totals, requests per status class, and the `top` busiest endpoints."""
    classes = Counter(f"{r.status // 100}xx" for r in requests)
    out = [
        f"requests:        {len(requests)}",
        f"malformed lines: {malformed}",
        "status:          " + "  ".join(f"{c} {classes[c]}" for c in sorted(classes)),
        "",
        f"{'endpoint':<24} {'hits':>5} {'avg ms':>7} {'max ms':>7}",
    ]
    times = {}
    for r in requests:
        times.setdefault(r.endpoint, []).append(r.ms)
    busiest = sorted(times.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
    for endpoint, ms in busiest:
        out.append(f"{endpoint:<24} {len(ms):>5} {sum(ms) / len(ms):>7.1f} {max(ms):>7}")
    return "\n".join(out)


def server_errors(requests):
    """One line per 5xx request, in log order."""
    return "\n".join(f"{r.ts} {r.method} {r.path} {r.status}" for r in requests if r.status >= 500)
