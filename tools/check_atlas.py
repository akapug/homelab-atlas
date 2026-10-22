#!/usr/bin/env python3
"""Check atlas.json against the pages it cites, and every relative Markdown link in the repo.

For each result: its source page exists, its #anchor is a heading there, and every number in decode_tps,
decode_tps_no_draft and prefill_tps appears in that page's text; every card in atlas.json has at least one result
(by card, or in a result's cards list). Exit 1 on any miss. Python 3, standard library.
Run from anywhere: python3 tools/check_atlas.py
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slug(heading):
    return re.sub(r'[^\w\- ]', '', heading.strip().lower()).replace(' ', '-')


def page(path):
    target = os.path.join(ROOT, path)
    return os.path.join(target, 'README.md') if os.path.isdir(target) else target


def anchors(md):
    text = re.sub(r'```.*?```', '', open(md).read(), flags=re.S)
    return {slug(h) for h in re.findall(r'^#+\s+(.*)$', text, re.M)}


def numbers(value):
    return value if isinstance(value, list) else [value]


def shown(n):
    return f'{n:,}' if isinstance(n, int) else f'{n:,.2f}'.rstrip('0').rstrip('.')


misses = []
atlas = json.load(open(os.path.join(ROOT, 'atlas.json')))
covered = {c for r in atlas['results'] for c in r.get('cards', [r['card']])}
misses += [f"card {c['id']}: no result names it (card or cards)" for c in atlas['cards'] if c['id'] not in covered]
misses += [f"{r['model']}: card {r['card']} is not listed in its cards" for r in atlas['results'] if r['card'] not in r.get('cards', [r['card']])]
for r in atlas['results']:
    path, _, frag = r['source'].partition('#')
    md = page(path)
    if not os.path.exists(md):
        misses.append(f"{r['card']} {r['model']}: no page {path}")
        continue
    if frag and frag not in anchors(md):
        misses.append(f"{r['card']} {r['model']}: no heading #{frag} in {path}")
    text = open(md).read()
    for key in ('decode_tps', 'decode_tps_no_draft', 'prefill_tps'):
        for n in numbers(r.get(key)) if r.get(key) is not None else []:
            if shown(n) not in text:
                misses.append(f"{r['card']} {r['model']}: {key} {shown(n)} not on {path}")

for d, dirs, files in os.walk(ROOT):
    dirs[:] = [x for x in dirs if x not in ('.git', 'docs')]
    for f in (x for x in files if x.endswith('.md')):
        md = os.path.join(d, f)
        for link in re.findall(r'\]\(([^)\s]+)\)', open(md).read()):
            if link.startswith(('http', 'mailto')):
                continue
            path, _, frag = link.partition('#')
            target = os.path.normpath(os.path.join(d, path)) if path else md
            if not os.path.exists(target):
                misses.append(f'{os.path.relpath(md, ROOT)}: broken link {link}')
                continue
            dest = os.path.join(target, 'README.md') if os.path.isdir(target) else target
            if frag and dest.endswith('.md') and os.path.exists(dest) and frag not in anchors(dest):
                misses.append(f'{os.path.relpath(md, ROOT)}: no heading for {link}')

print('\n'.join(misses) or 'atlas.json and links: OK')
sys.exit(1 if misses else 0)
