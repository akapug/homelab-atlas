#!/usr/bin/env python3
"""The benchmark client of our V4.1 runs (standard library only), against a running llama-server.

Warm-up: one 64-token request on the first prompt, then one long prefill (13,000 characters, 8 tokens out) so every
expert page is touched before timing. Then each phase sends every prompt once: greedy, N tokens, no prompt cache,
with its own per-request draft length (n_max 0 = no draft). A phase repeated last (n4b) shows the drift in one load.
  PORT=8790 N=300 PHASES=n4:4,n0:0,n4b:4 python3 bench.py [prompts.json]
Prints one line per request and writes bench.json: per phase the decode and prefill tok/s per prompt, their means,
draft tokens proposed/accepted and a short hash of each text (the same text means the same greedy path)."""
import hashlib, json, os, sys, time, urllib.request

BASE = 'http://127.0.0.1:%s' % os.environ.get('PORT', '8790')
N = int(os.environ.get('N', '300'))
PHASES = [(t, int(n)) for t, n in (p.split(':') for p in os.environ.get('PHASES', 'n4:4,n0:0,n4b:4').split(','))]
prompts = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'prompts.json')))


def post(prompt, n, n_max=0):
    body = {'prompt': prompt + '\n', 'n_predict': n, 'temperature': 0, 'ignore_eos': True, 'cache_prompt': False,
            'speculative.n_max': n_max, 'speculative.n_min': 0}
    req = urllib.request.Request(BASE + '/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as f:
        return json.loads(f.read())


post(prompts[0], 64)
post(('\n\n'.join(prompts) * 2)[:13000], 8)
res = []
for tag, n_max in PHASES:
    ph = {'phase': tag, 'n_max': n_max, 'decode_tps': [], 'prefill_tps': [], 'draft_n': [], 'draft_acc': [], 'texts_sha': []}
    for i, p in enumerate(prompts):
        t0 = time.time()
        r = post(p, N, n_max)
        t = r['timings']
        ph['decode_tps'].append(round(t['predicted_per_second'], 2))
        ph['prefill_tps'].append(round(t['prompt_per_second'], 1))
        ph['draft_n'].append(t.get('draft_n', 0))
        ph['draft_acc'].append(t.get('draft_n_accepted', 0))
        ph['texts_sha'].append(hashlib.sha1(r.get('content', '').encode()).hexdigest()[:10])
        print('%s prompt%d decode %.2f prefill %.1f draft %d/%d wall %.0fs' % (
            tag, i, ph['decode_tps'][-1], ph['prefill_tps'][-1], ph['draft_n'][-1], ph['draft_acc'][-1], time.time() - t0), flush=True)
    ph['decode_mean'] = round(sum(ph['decode_tps']) / len(ph['decode_tps']), 2)
    ph['prefill_mean'] = round(sum(ph['prefill_tps']) / len(ph['prefill_tps']), 1)
    print('%s decode mean %.2f tok/s' % (tag, ph['decode_mean']), flush=True)
    res.append(ph)
json.dump({'n': N, 'phases': res}, open('bench.json', 'w'), indent=1)
