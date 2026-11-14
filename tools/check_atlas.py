#!/usr/bin/env python3
"""Check atlas.json against the pages it cites, and every relative Markdown link in the repo.

For each result: its source page exists, its #anchor is a heading there, and every number in decode_tps,
decode_tps_no_draft, prefill_tps and agent_useful_tps appears as a whole number in the section the anchor names (the heading down to
the next heading of its level or higher; the whole page when the source has no anchor). It catches a number missing
from its section, not one that drifted to another number the same section also prints. Every card in atlas.json has at least one result
(by card, or in a result's cards list). With --accuracy, also the planner accuracy table in recipe-maker/README.md
against `node report.js --accuracy`. --provenance projects active qualified seats
to UNKNOWN/unpinned_provenance when immutable pins are absent. No file is rewritten.
Exit 1 on invalid schema/link/claim, 3 on unknown admission, 0 otherwise. Python 3, standard library.
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


def section(md, frag):
    """the text under the heading whose slug is frag, down to the next heading of the same or a higher level"""
    lines, out, level = open(md).read().splitlines(), [], 0
    fence = False
    for l in lines:
        fence ^= l.startswith('```')
        m = None if fence else re.match(r'^(#+)\s+(.*)$', l)
        if level and m and len(m[1]) <= level:
            break
        if level:
            out.append(l)
        elif m and slug(m[2]) == frag:
            level = len(m[1])
    return '\n'.join(out)


def numbers(value):
    return value if isinstance(value, list) else [value]


def shown(n):
    return f'{n:,}' if isinstance(n, int) else f'{n:,.2f}'.rstrip('0').rstrip('.')


PIN_RE = re.compile(r'[0-9a-fA-F]{64}')
OID_RE = re.compile(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})')


def pinned(value, pattern=PIN_RE):
    return isinstance(value, str) and bool(pattern.fullmatch(value)) and set(value) != {'0'}


def seats(document, path='$'):
    """Recipe/seat records may be nested by model, serving role or named slot."""
    if isinstance(document, dict):
        if 'status' in document or 'provenance_pins' in document:
            yield path, document
        for key, value in document.items():
            if key != 'provenance_pins':
                yield from seats(value, path + '.' + key)
    elif isinstance(document, list):
        for i, value in enumerate(document):
            yield from seats(value, path + '[' + str(i) + ']')


def pin_errors(pins, required=False):
    if not isinstance(pins, dict):
        return ['provenance_pins missing or not an object']
    errors = [key + ': unknown provenance pin' for key in pins.keys() - {'weights_sha256','engine_commit','patch_sha256','outcome_probe_receipt'}]
    for key, regex in [('weights_sha256', PIN_RE), ('engine_commit', OID_RE), ('patch_sha256', PIN_RE)]:
        value = pins.get(key)
        if (required and key != 'patch_sha256') or value is not None:
            if not pinned(value, regex):
                errors.append(key + ' must be a full immutable hex value')
    receipt = pins.get('outcome_probe_receipt')
    if receipt is not None and not (pinned(receipt) or isinstance(receipt, dict) and pinned(receipt.get('sha256'))):
        errors.append('outcome_probe_receipt must pin a receipt sha256')
    return errors


def validate_schema(document):
    errors = []
    if document.get('schema') != 'homelab-atlas/1':
        errors.append('unsupported atlas schema')
    for path, seat in seats(document):
        if 'provenance_pins' in seat:
            # Null values are explicit unknown slots, not qualification.
            errors.extend(path + ': ' + e for e in pin_errors(seat['provenance_pins']))
    return errors


def provenance_checks(document):
    """Read-only admission projection: never rewrite historical measurement status."""
    for path, seat in seats(document):
        if seat.get('status') not in ('qualified', 'verified') or seat.get('active') is False:
            continue
        errors = pin_errors(seat.get('provenance_pins'), required=True)
        yield dict(path=path, claimed_status=seat['status'],
                   admission_status='unpinned_provenance' if errors else seat['status'],
                   status='UNKNOWN' if errors else 'PASS', reasons=errors)


def main():
    misses = []
    unknown = []
    with open(os.path.join(ROOT, 'atlas.json')) as source:
        atlas = json.load(source)
    misses.extend(validate_schema(atlas))
    if '--provenance' in sys.argv:
        documents = [('atlas.json', atlas)]
        recipes = os.path.join(ROOT, 'recipe-maker', 'recipes')
        if os.path.isdir(recipes):
            for name in sorted(os.listdir(recipes)):
                if name.endswith('.json'):
                    with open(os.path.join(recipes, name)) as source:
                        documents.append(('recipe-maker/recipes/' + name, json.load(source)))
        for source, document in documents:
            for result in provenance_checks(document):
                result['source'] = source
                print(json.dumps(result, sort_keys=True))
                if result['status'] == 'UNKNOWN':
                    unknown.append(result)
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
        text = section(md, frag) if frag else open(md).read()
        for key in ('decode_tps', 'decode_tps_no_draft', 'prefill_tps', 'agent_useful_tps'):
            for n in numbers(r.get(key)) if r.get(key) is not None else []:
                # a whole number: 12.69 is not inside 112.69 or 12.695; 69.0 may print as 69 or 69.0
                if not re.search(r'(?<![\d.,])%s%s(?![\d]|\.\d|,\d)' % (re.escape(shown(n)), '' if '.' in shown(n) else r'(?:\.0+)?'), text):
                    misses.append(f"{r['card']} {r['model']}: {key} {shown(n)} not in {r['source']}")

    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x != '.git' and os.path.relpath(os.path.join(d, x), ROOT) != os.path.join('docs', 'page')]
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

    # --accuracy: the planner accuracy table in recipe-maker/README.md against what `node report.js --accuracy` prints
    # (always rebuilds data.json first). Each cell "P % (N runs, bias B %)" must be a line of the matching section
    # for the row's box: column 1 the launch-time decode lines, 2 the no-draft lines, 3 the replay with today's planner.
    if '--accuracy' in sys.argv:
        import subprocess
        rm = os.path.join(ROOT, 'recipe-maker')
        subprocess.run([sys.executable, 'build.py'], cwd=rm, check=True, capture_output=True)
        out = subprocess.run(['node', 'report.js', '--accuracy'], cwd=rm, check=True, capture_output=True, text=True).stdout
        head, _, replay = out.partition("-- the same with today's planner")
        replay = replay.partition('\n-- ')[0]
        sections = [[l for l in head.splitlines() if l.startswith('decode_tps ')],
                    [l for l in head.splitlines() if l.startswith('decode_tps_nodraft ')],
                    [l for l in replay.splitlines() if l.startswith('decode_tps ')]]
        cell = re.compile(r'([\d.]+) % \((\d+) runs?, bias ([+-][\d.]+) %\)')
        group = re.compile(r'([\d.]+)-([\d.]+) % \((\d+)-(\d+) runs each\)')
        line = re.compile(r'^\S+\s+(\w+)/\S+\s+n\s+(\d+)\s+median \|err\|\s+([\d.]+) %')
        seen, groups = set(), []
        for row in open(os.path.join(rm, 'README.md')).read().splitlines():
            cols = [c.strip() for c in row.strip('|').split('|')]
            if len(cols) != 4 or not (cell.search(row) or group.search(row)):
                continue
            box = cols[0].split(',')[0].strip()
            if group.fullmatch(cols[1]):   # a row that covers several small groups by range, e.g. "odie / lassie, ..."
                groups.append(({b.strip() for b in box.split('/')}, group.fullmatch(cols[1])))
            for col, lines in zip(cols[1:], sections):
                m = cell.fullmatch(col)
                if not m:
                    continue
                want = re.compile(r'^\S+\s+%s/\S+.*\bn\s+%s\s+median \|err\|\s+%s %%\s+bias %s %%' % (re.escape(box), m[2], re.escape(m[1]), re.escape(m[3])))
                hit = [l for l in lines if want.search(l)]
                if not hit:
                    misses.append(f'recipe-maker/README.md accuracy table: "{cols[0]}" cell "{col}" is not what report.js --accuracy prints')
                if lines is sections[0]:
                    seen.update(hit)
        # the other direction: every launch-time line report.js prints is in the table, as its own row or inside a range row
        for l in sections[0]:
            b, n, err = line.match(l).groups() if line.match(l) else (None, 0, 0)
            covered = l in seen or any(b in boxes and float(g[1]) <= float(err) <= float(g[2]) and int(g[3]) <= int(n) <= int(g[4])
                                       for boxes, g in groups)
            if not covered:
                misses.append(f'recipe-maker/README.md accuracy table: no row for report.js line "{l.strip()}"')

    print('\n'.join(misses) or ('atlas.json and links: OK; provenance UNKNOWN' if unknown else 'atlas.json and links: OK') + (' (and the accuracy table)' if '--accuracy' in sys.argv else ''))
    return 1 if misses else 3 if unknown else 0


if __name__ == '__main__':
    sys.exit(main())
