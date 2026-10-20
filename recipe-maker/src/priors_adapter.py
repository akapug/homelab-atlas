# Applies sibling quant-priors.json (unknown schema) to the type KLD table.
# Looks for objects keyed by our type names whose values are numbers or dicts with a KLD-like field.
import re as _re

_KLD_KEYS = ('kld', 'mean_kld', 'kl', 'kl_div', 'kld_mean', 'kld_prior', 'err')


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norm(q):
    return q.lower().replace('-', '_').replace(' ', '')


def apply_priors(data):
    sp = data['siblings']['quant_priors']
    data['priors_applied'] = []
    if not sp.get('present') or 'data' not in sp:
        return
    names = {_norm(q): q for q in data['types']}
    found = []

    def fam_of(path):
        p = ' '.join(path).lower()
        if 'glm' in p:
            return ['glm']
        if 'v4' in p or 'deepseek' in p or 'ds4' in p:
            return ['v41']
        return ['v41', 'glm']

    def walk(o, path):
        if isinstance(o, dict):
            for k, val in o.items():
                q = names.get(_norm(str(k)))
                if q:
                    x = None
                    if _num(val):
                        x = val
                    elif isinstance(val, dict):
                        for kk in val:
                            if _norm(kk) in _KLD_KEYS and _num(val[kk]):
                                x = val[kk]
                                break
                    if x is not None:
                        found.append((q, float(x), fam_of(path + [str(k)]), '/'.join(path + [str(k)])))
                walk(val, path + [str(k)])
        elif isinstance(o, list):
            for i, it in enumerate(o):
                if isinstance(it, dict):
                    tname = it.get('type') or it.get('quant') or it.get('name')
                    q = names.get(_norm(str(tname))) if tname else None
                    if q:
                        for kk in it:
                            if _norm(kk) in _KLD_KEYS and _num(it[kk]):
                                fams = fam_of(path + [str(it.get('model', ''))])
                                found.append((q, float(it[kk]), fams, '/'.join(path + [str(i)])))
                                break
                walk(it, path + [str(i)])

    walk(sp['data'], [])
    for q, x, fams, where in found:
        for fam in fams:
            data['types'][q]['kld'][fam] = {'v': x, 'tag': 'PRIOR', 'src': 'quant-priors.json ' + where + ' (PRIOR (community))'}
        data['priors_applied'].append({'type': q, 'kld': x, 'families': fams, 'path': where})
