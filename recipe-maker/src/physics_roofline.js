// Physical, independent-token roofline. SI units: bytes, seconds, FLOPs.
// A bound, not a fitted speed prediction. Correlated routing needs ordered traces.
(function (root) {
  'use strict';
  const positive = (x, name, zero = false) => {
    if (typeof x !== 'number' || !Number.isFinite(x) || (zero ? x < 0 : x <= 0)) throw new TypeError(name);
    return x;
  };
  function expectedUnion(probabilities, batch) {
    positive(batch, 'batch');
    return probabilities.reduce((sum, p) => {
      positive(p, 'activation probability', true);
      if (p > 1) throw new RangeError('activation probability > 1');
      return sum + (p === 1 ? 1 : -Math.expm1(batch * Math.log1p(-p)));
    }, 0);
  }
  function residency(trace, hotExperts, batch = 1) {
    if (!trace || trace.status !== 'measured' || trace.trace_provenance !== 'measured' || !trace.source)
      return {status: 'unmeasured', trace_provenance: 'unmeasured', hit_rate: null};
    const layers = trace.activation_probabilities;
    if (!Array.isArray(layers) || !layers.length) throw new TypeError('per-layer activation_probabilities');
    let picks = 0, hits = 0, unique = 0, cold = 0;
    const rows = layers.map((ps, l) => {
      if (!Array.isArray(ps) || !ps.length) throw new TypeError('empty layer');
      const hot = new Set((hotExperts || [])[l] || []);
      for (const e of hot) if (!Number.isInteger(e) || e < 0 || e >= ps.length) throw new RangeError('hot expert ID');
      const u = expectedUnion(ps, batch), c = expectedUnion(ps.filter((_, e) => !hot.has(e)), batch);
      picks += ps.reduce((a, p) => a + p, 0);
      hits += ps.reduce((a, p, e) => a + (hot.has(e) ? p : 0), 0);
      unique += u; cold += c;
      return {unique: u, cold_unique: c};
    });
    if (!picks) return {status: 'unmeasured', trace_provenance: 'unmeasured', hit_rate: null};
    return {status: 'inferred', trace_provenance: 'measured', source: trace.source,
      hit_rate: hits / picks, unique_experts: unique, cold_unique_experts: cold, layers: rows};
  }
  function hotBank(trace, capacity) {
    if (!trace || trace.status !== 'measured' || trace.trace_provenance !== 'measured' || !trace.source) return null;
    return trace.activation_probabilities.map((ps, l) => {
      const n = Array.isArray(capacity) ? capacity[l] : capacity;
      if (!Number.isInteger(n) || n < 0 || n > ps.length) throw new RangeError('hot bank capacity');
      expectedUnion(ps, 1); // validate marginals before ranking
      return ps.map((p, e) => ({p, e})).sort((a, b) => b.p - a.p || a.e - b.e).slice(0, n).map(x => x.e);
    });
  }
  function beladyLookahead(trace, capacity, window = 16, initial = []) {
    if (!Number.isInteger(window) || window < 1) throw new RangeError('lookahead window');
    if (!trace || trace.status !== 'measured' || trace.trace_provenance !== 'measured' || !trace.source || !trace.expert_tokens)
      return {status:'unmeasured', trace_provenance:'unmeasured', hit_rate:null, window_tokens:window};
    const tokens = trace.expert_tokens, layers = trace.activation_probabilities;
    if (!Array.isArray(tokens) || !tokens.length || !Array.isArray(layers)) throw new TypeError('ordered expert_tokens');
    for (const token of tokens) {
      if (!Array.isArray(token) || token.length !== layers.length) throw new RangeError('ordered trace layers');
      token.forEach((ids, l) => {
        if (!Array.isArray(ids) || new Set(ids).size !== ids.length || ids.some(e => !Number.isInteger(e) || e < 0 || e >= layers[l].length))
          throw new RangeError('ordered expert IDs');
      });
    }
    let hits = 0, accesses = 0;
    layers.forEach((ps, l) => {
      const n = Array.isArray(capacity) ? capacity[l] : capacity;
      if (!Number.isInteger(n) || n < 0 || n > ps.length) throw new RangeError('lookahead capacity');
      const cache = new Set(initial[l] || []);
      if (cache.size > n || [...cache].some(e => !Number.isInteger(e) || e < 0 || e >= ps.length)) throw new RangeError('initial bank');
      tokens.forEach((token, t) => {
        // One token's picks at a layer execute in the same step, so they are
        // co-resident: an expert this token still needs is never the victim.
        const now = new Set(token[l]);
        for (const e of token[l]) {
          accesses++;
          if (cache.has(e)) { hits++; cache.delete(e); cache.add(e); continue; }
          if (!n) continue;
          if (cache.size === n) {
            let victim, farthest = -1;
            for (const resident of cache) {
              if (now.has(resident)) continue;
              let next = Infinity;
              for (let future = t + 1; future < Math.min(tokens.length, t + 1 + window); future++) {
                if (tokens[future][l].includes(resident)) { next = future - t; break; }
              }
              if (next > farthest) { victim = resident; farthest = next; }
            }
            // Bank smaller than this token's picks: streamed, not admitted.
            if (victim === undefined) continue;
            cache.delete(victim);
          }
          cache.add(e);
        }
      });
    });
    return {status:'inferred', trace_provenance:'measured', source:trace.source,
      algorithm:'finite-horizon Belady replay (LRU tie break)', window_tokens:window,
      accesses, hits, hit_rate:accesses ? hits/accesses : null};
  }
  function physicsRoofline(input) {
    const B = positive(input.batch ?? 1, 'batch');
    const W = input.lookahead_window_tokens ?? 0;
    if (!Number.isInteger(W) || W < 0) throw new RangeError('lookahead_window_tokens');
    const banks = input.hot_experts || (input.hot_expert_capacity !== undefined ? hotBank(input.router_trace, input.hot_expert_capacity) : null);
    const r = residency(input.router_trace, banks, B);
    const lookahead = W && input.hot_expert_capacity !== undefined
      ? beladyLookahead(input.router_trace, input.hot_expert_capacity, W, banks || []) : null;
    const base = {status: 'unmeasured', trace_provenance: r.trace_provenance, residency: r,
      aggregate_tps: null, step_s: null, lookahead_window_tokens: W};
    // No invented routing identities or Zipf exponent. Uniform is an explicit scenario only.
    let unique, cold;
    if (r.hit_rate !== null) { unique = r.unique_experts; cold = r.cold_unique_experts; }
    else if (input.uniform_scenario) {
      const {layers, experts, top_k, resident_experts = 0} = input.uniform_scenario;
      [layers, experts, top_k].forEach(x => positive(x, 'uniform scenario'));
      if (top_k > experts || resident_experts < 0 || resident_experts > experts) throw new RangeError('uniform scenario');
      unique = layers * experts * -Math.expm1(B * Math.log1p(-top_k / experts));
      cold = unique * (1 - resident_experts / experts);
    } else return {...base, why: 'router activations unmeasured; supply an explicitly hypothetical uniform_scenario'};
    const bytes = positive(input.expert_bytes, 'expert_bytes');
    const n = key => positive(input[key] ?? 0, key, true);
    const time = (amount, rate, name) => amount === 0 ? 0 : amount / positive(rate, name);
    const misses = cold * bytes;
    // KV read per stream per step, in bytes. kv_bytes_per_token is bytes per
    // *context* token, so it needs the attended depth; without one the product
    // B * kv_bytes_per_token would be bytes of a single cached token, not a read.
    const kvPerToken = n('kv_bytes_per_token');
    if (input.kv_read_bytes_per_stream !== undefined && kvPerToken) throw new TypeError('kv_read_bytes_per_stream excludes kv_bytes_per_token');
    if (kvPerToken && input.context_tokens === undefined) throw new TypeError('kv_bytes_per_token needs context_tokens');
    const kvStream = input.kv_read_bytes_per_stream !== undefined ? n('kv_read_bytes_per_stream') : kvPerToken * n('context_tokens');
    // Both resident and uploaded weights are read by the GPU; uploaded bytes
    // also consume VRAM write bandwidth on their DMA path.
    const terms = {
      vram_bw: time(n('dense_bytes') + unique * bytes + misses + B * kvStream, input.vram_bytes_per_s, 'vram_bytes_per_s'),
      pcie_transfer: time(misses, input.pcie_bytes_per_s, 'pcie_bytes_per_s'),
      host_mem_bw: time(misses, input.host_bytes_per_s, 'host_bytes_per_s'),
      compute: time(B * n('flops_per_token'), input.flops_per_s, 'flops_per_s'),
    };
    const sync = n('sync_overhead_s'), floor = Math.max(...Object.values(terms)) + sync;
    // W steps of overlap can hide latency, never the sustained transfer bandwidth.
    // This is an optimistic speculative-prefetch bound, not Belady cache hits:
    // frequencies alone cannot establish future order or draft acceptance.
    // A fetch issued W steps ahead overlaps W whole steps, stalls included, so
    // the stalled step solves step = floor + latency - W * step. (Overlapping
    // only W * floor made step_s fall as the per-step work grew.) No cold
    // bytes means no fetch and no latency.
    const latency = misses > 0 ? n('pcie_prefetch_latency_s') : (n('pcie_prefetch_latency_s'), 0);
    const step = Math.max(floor, (floor + latency) / (1 + W));
    const hidden = latency && W ? Math.min(1, Math.max(0, 1 - (step - floor) / latency)) : 0;
    const cap = input.capacity_streams;
    if (cap !== undefined) positive(cap, 'capacity_streams', true);
    if (cap !== undefined && B > cap) return {...base, status: 'capacity_exceeded', capacity_streams: cap};
    if (!(step > 0)) throw new RangeError('zero step time');
    return {...base, status: 'inferred', scenario: r.hit_rate === null ? 'uniform hypothetical' : 'independent measured marginals',
      step_s: step, aggregate_tps: B / step, per_stream_tps: 1 / step, terms,
      unique_experts: unique, sync_overhead_s: sync, hidden_prefetch_fraction: hidden,
      prefetch_status: W ? 'optimistic_bound_unmeasured' : 'disabled',
      belady_hit_rate: lookahead ? lookahead.hit_rate : null, lookahead,
      binding: Object.keys(terms).reduce((a, b) => terms[a] >= terms[b] ? a : b)};
  }
  const api = {expectedUnion, residency, hotBank, beladyLookahead, physicsRoofline};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PhysicsRoofline = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
