'use strict';
const assert = require('assert');
const P = require('../src/physics_roofline.js');
const FM = require('../src/model.js');
const base = {expert_bytes:1e7, dense_bytes:1e9, kv_bytes_per_token:1e7,
  vram_bytes_per_s:5e11, pcie_bytes_per_s:2e10, host_bytes_per_s:8e10,
  flops_per_token:2e10, flops_per_s:2e13, sync_overhead_s:.001,
  uniform_scenario:{layers:4,experts:128,top_k:8,resident_experts:32}};
assert(Math.abs(P.expectedUnion([.1,.7], 1) - .8) < 1e-12);
assert(Math.abs(P.expectedUnion([.1,.7],2) - 1.1) < 1e-12);
for (const W of [0,16]) {
  let last = 0;
  for (let B = 1; B <= 128; B++) {
    const p = FM.physicsRoofline({...base,batch:B,lookahead_window_tokens:W,pcie_prefetch_latency_s:.2});
    assert(p.aggregate_tps + 1e-9 >= last, `B=${B}`); last = p.aggregate_tps;
    assert.equal(p.trace_provenance,'unmeasured');
    assert(p.step_s >= p.terms.pcie_transfer, 'lookahead cannot erase bandwidth');
  }
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
