'use strict';
const assert = require('assert');
const P = require('../src/physics_roofline.js');
const FM = require('../src/model.js');
const base = {expert_bytes:1e7, dense_bytes:1e9, kv_bytes_per_token:1e4, context_tokens:1000,
  vram_bytes_per_s:5e11, pcie_bytes_per_s:2e10, host_bytes_per_s:8e10,
  flops_per_token:2e10, flops_per_s:2e13, sync_overhead_s:.001,
  uniform_scenario:{layers:4,experts:128,top_k:8,resident_experts:32}};
assert(Math.abs(P.expectedUnion([.1,.7], 1) - .8) < 1e-12);
assert(Math.abs(P.expectedUnion([.1,.7],2) - 1.1) < 1e-12);
// Exact expected union under independent routing, against brute-force enumeration
// of two tokens; E * (1 - prod(1 - p_e)^B) is a different (wrong) quantity.
{
  const ps = [.9,.05,.05], B = 2;
  let brute = 0;
  for (let mask = 0; mask < 1 << (ps.length * B); mask++) {
    let weight = 1; const seen = new Set();
    for (let t = 0; t < B; t++) ps.forEach((p, e) => {
      const on = mask >> (t * ps.length + e) & 1;
      weight *= on ? p : 1 - p; if (on) seen.add(e);
    });
    brute += weight * seen.size;
  }
  assert(Math.abs(P.expectedUnion(ps, B) - brute) < 1e-12);
  const wrong = ps.length * (1 - Math.pow(ps.reduce((a, p) => a * (1 - p), 1), B));
  assert(Math.abs(P.expectedUnion(ps, B) - wrong) > .5, 'product form is not the union');
}
for (const W of [0,1,16]) for (const latency of [0,.002,.2,5]) {
  let last = 0, lastStep = 0;
  for (let B = 1; B <= 128; B++) {
    const p = FM.physicsRoofline({...base,batch:B,lookahead_window_tokens:W,pcie_prefetch_latency_s:latency});
    assert(p.aggregate_tps + 1e-9 >= last, `throughput fell at B=${B} W=${W} latency=${latency}`); last = p.aggregate_tps;
    // more work per step never makes the step shorter
    assert(p.step_s * (1 + 1e-12) >= lastStep,`step_s fell at B=${B} W=${W} latency=${latency}`); lastStep = p.step_s;
    assert.equal(p.trace_provenance,'unmeasured');
    const floor = Math.max(...Object.values(p.terms)) + p.sync_overhead_s;
    assert(p.step_s >= floor, 'lookahead cannot erase bandwidth or compute');
    assert(p.step_s <= floor + latency + 1e-15, 'a step stalls at most one latency');
    assert(p.hidden_prefetch_fraction >= 0 && p.hidden_prefetch_fraction <= 1);
    if (!W) assert.equal(p.hidden_prefetch_fraction, 0);
  }
}
// self-consistent overlap: a stalled step is (floor + latency) / (1 + W)
{
  const p = P.physicsRoofline({...base,lookahead_window_tokens:16,pcie_prefetch_latency_s:5});
  const floor = Math.max(...Object.values(p.terms)) + p.sync_overhead_s;
  assert(Math.abs(p.step_s - (floor + 5) / 17) < 1e-12);
}
// KV operand is bytes read per stream per step: depth is required, and scales the term
assert.throws(() => P.physicsRoofline({...base,context_tokens:undefined}), /context_tokens/);
assert.throws(() => P.physicsRoofline({...base,kv_read_bytes_per_stream:1e7}), /excludes/);
{
  const kvOnly = {...base,dense_bytes:0,expert_bytes:1,flops_per_token:0,sync_overhead_s:0,batch:4};
  const a = P.physicsRoofline(kvOnly), b = P.physicsRoofline({...kvOnly,context_tokens:2000});
  assert.equal(a.binding,'vram_bw');
  assert(Math.abs(b.terms.vram_bw / a.terms.vram_bw - 2) < 1e-4);
  const direct = P.physicsRoofline({...kvOnly,kv_bytes_per_token:undefined,context_tokens:undefined,kv_read_bytes_per_stream:1e7});
  assert.equal(direct.terms.vram_bw, a.terms.vram_bw);
}
assert.equal(P.physicsRoofline({...base,uniform_scenario:undefined}).aggregate_tps,null);
assert.equal(P.physicsRoofline({...base,batch:9,capacity_streams:8}).status,'capacity_exceeded');
const trace = {status:'measured',trace_provenance:'measured',source:'fixture only',activation_probabilities:[[.8,.2],[.7,.3]]};
const r = P.residency(trace,[[0],[0]],2);
assert.equal(r.hit_rate,.75);
assert(Math.abs(r.cold_unique_experts - .87) < 1e-12);
assert.equal(P.residency({...trace,trace_provenance:'unmeasured'},[[0],[0]]).hit_rate,null);
assert.throws(() => P.expectedUnion([1.1],2));
const d = FM.decodeRoof({}, {batch:2,physicsRoofline:base});
assert.equal(d.status,'OK'); assert.equal(d.tps_total,FM.physicsRoofline({...base,batch:2}).aggregate_tps);
console.log('physical roofline, monotonic batch, trace provenance and prefetch bounds passed');
assert.deepEqual(P.hotBank(trace,1),[[0],[0]]);
assert.equal(P.beladyLookahead(trace,1,16).status,'unmeasured');
const ordered = {...trace, activation_probabilities:[[.6,.3,.1]], expert_tokens:[[[0]],[[1]],[[2]],[[0]],[[1]],[[0]]]};
const look = P.beladyLookahead(ordered,2,16);
assert.equal(look.accesses,6); assert.equal(look.hits,2);
assert.equal(look.trace_provenance,'measured');
assert.throws(()=>P.beladyLookahead({...ordered,expert_tokens:[[[9]]]},2,16));
// One token's picks are co-resident: admitting 2 must not evict 0, which the same token needs.
const pair = {...trace, activation_probabilities:[[.7,.7,.6]], expert_tokens:[[[0,1]],[[2,0]]]};
const co = P.beladyLookahead(pair,2,16);
assert.equal(co.accesses,4); assert.equal(co.hits,1);
// A bank smaller than one token's picks streams the overflow instead of thrashing itself.
const tiny = P.beladyLookahead({...pair, expert_tokens:[[[0,1]],[[0,1]]]},1,16);
assert.equal(tiny.hits,1);
// No ordered trace, no Belady number: marginals alone never yield a hit rate.
assert.equal(P.physicsRoofline({...base,uniform_scenario:undefined,router_trace:trace,hot_expert_capacity:1,lookahead_window_tokens:16}).belady_hit_rate,null);

const capture = {models:{toy:{k:1, heat:{sets:{decode:{status:'measured',trace_provenance:'measured',map:{digest:'fixture'},layers:[[8,2]]}}}}}};
assert.deepEqual(FM.physicsTrace(capture,'toy','decode').activation_probabilities,[[.8,.2]]);
assert.equal(FM.physicsTrace(capture,'toy','absent').status,'unmeasured');
assert.deepEqual(FM.recipeRoofQuery({}, {physicsRoofline:base}).physicsRoofline,base);

// A synthetic Zipf fixture exercises measured-trace plumbing, not model evidence.
const zipf = Array.from({length:128},(_,e)=>1/(e+1)), zsum=zipf.reduce((a,b)=>a+b,0);
const ztrace={status:'measured',trace_provenance:'measured',source:'synthetic Zipf test fixture',activation_probabilities:[zipf.map(p=>p/zsum)]};
let previous=0;
for(let B=1;B<=64;B++) {
  const point=P.physicsRoofline({...base,uniform_scenario:undefined,router_trace:ztrace,hot_expert_capacity:8,batch:B});
  assert(point.aggregate_tps+1e-9>=previous);previous=point.aggregate_tps;
}
const allHot=P.physicsRoofline({...base,router_trace:ztrace,hot_expert_capacity:128});
assert.equal(allHot.terms.pcie_transfer,0);
assert.equal(allHot.terms.host_mem_bw,0);
// nothing cold, nothing fetched: prefetch latency cannot stall a fully resident bank
const allHotLatent=P.physicsRoofline({...base,router_trace:ztrace,hot_expert_capacity:128,pcie_prefetch_latency_s:5});
assert.equal(allHotLatent.step_s,allHot.step_s);
console.log('exact union, self-consistent prefetch overlap, KV depth and co-resident Belady passed');
