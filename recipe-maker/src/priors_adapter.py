# Applies explicit per-family/per-type KLD priors from sibling quant-priors.json when provided.
# Reads only explicit schema entries (`kld_priors` keyed by `{family: {type: value_or_kld_dict}}`),
# never guessing unknown schemas recursively or matching generic `'err'` keys.

_KLD_KEYS = ('kld', 'mean_kld', 'kl', 'kl_div', 'kld_mean', 'kld_prior')


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _norm(q):
    return q.lower().replace('-', '_').replace(' ', '')


def apply_priors(data):
    sp = data['siblings']['quant_priors']
    data['priors_applied'] = []
    if not sp.get('present') or not isinstance(sp.get('data'), dict):
        return
    names = {_norm(q): q for q in data['types']}
    kld_priors = sp['data'].get('kld_priors')
    if not isinstance(kld_priors, dict):
        return
    for fam, by_quant in kld_priors.items():
        if not isinstance(by_quant, dict):
            continue
        for k, val in by_quant.items():
            q = names.get(_norm(str(k)))
            if not q or fam not in data['types'][q]['kld']:
                continue
            x = None
            if _num(val):
                x = val
            elif isinstance(val, dict):
                for kk in _KLD_KEYS:
                    if kk in val and _num(val[kk]):
                        x = val[kk]
                        break
            if x is None:
                continue
            where = 'kld_priors/%s/%s' % (fam, k)
            data['types'][q]['kld'][fam] = {
                'v': float(x),
                'tag': 'PRIOR',
                'src': 'quant-priors.json ' + where + ' (PRIOR (community))',
            }
            data['priors_applied'].append({'type': q, 'kld': float(x), 'families': [fam], 'path': where})
