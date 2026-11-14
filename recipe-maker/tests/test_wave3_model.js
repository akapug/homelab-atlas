'use strict';
const assert = require('assert');
const FM = require('../src/model.js');
const ADV = require('../src/advisor.js');
const D = require('../data.json');
const data = {models:{fixture:{kld_family:'fixture'}}, checks:{earf:{families:{fixture:{perturbation_floor:.968}}}}, upgrade:{settings:{exclude_tags:[]}}};
const recipe = {box:'odie',model:'fixture',engine:'strata',role:'author_under_judge',store:{types:['q4']}};
const gate = (patch = {}, d = data, opts) => FM.earfCheck(d, {...recipe, ...patch}, opts);
assert.equal(gate().status, 'UNKNOWN');
for (const earf of [NaN, Infinity, -1, 1.1, '0.99']) assert.equal(gate({earf}).status, 'FAIL');
for (const earf of [.955,.966,.968]) assert.equal(gate({earf}).status, 'PASS');
assert.equal(gate({earf:.9529}).status, 'FAIL');
assert.equal(gate({earf:.953}).status, 'PASS');
assert.equal(gate({earf:.955}).min_earf, .968 - .015);
assert.equal(gate({earf:.93,perturbation_floor:.94}).status, 'PASS');
assert.equal(gate({earf:.894}, {}).status, 'UNKNOWN');
assert.equal(gate({earf:.894}, {}).severity, 'degraded');
assert.equal(gate({earf:.90}, {}).status, 'PASS');
assert.equal(gate({}, {}, {minEarf:null}).status, 'UNKNOWN');
for (const [passed,total] of [[22,24],[11,12]]) {
  const check = gate({earf:.894, authoring_probe:{executed:true,passed,total}});
  assert.equal(check.status, 'PASS');
  assert.equal(check.admission_basis, 'outcome_probe');
  assert.equal(check.role, 'author_under_judge');
  assert.equal(check.diff_reviewer, false);
  assert.equal(check.capabilities.diff_review, 'advisory_excluded');
}
assert.equal(gate({battery_pass_rate:.4,earf:.99}).status, 'FAIL');
assert.equal(gate({authoring_probe:{executed:false,passed:24,total:24}}).status, 'UNKNOWN');
const screen = {executed:true,planted:true,clean:true,false_alarm_rate:.10,recall:.50};
assert.equal(gate({review_screen:screen}).diff_reviewer, true);
for (const patch of [{clean:false},{planted:false},{executed:false},{false_alarm_rate:.101},{recall:.499}])
  assert.equal(gate({review_screen:{...screen,...patch}}).diff_reviewer, false);
const partial = {checks:{earf:{entries:{known:.96}}}};
assert.equal(gate({store:{types:['known','unknown']}}, partial).status, 'UNKNOWN');
assert.equal(gate({store:{types:['known','unknown']},earf:.96}, partial).status, 'PASS');
const unknown = {id:'unknown',box:'odie',model:'fixture',recipe, measured:{kl_verdict:'PASS',decode_tps:{single:100}}};
assert.equal(ADV.runQuality(data, unknown, [unknown]).status, 'UNKNOWN');
assert.equal(ADV.runQuality(data, {...unknown,measured:{kl_verdict:'FAIL'}}, []).status, 'FAIL');
const local = {model:'qwen3.8-27b-int4',engine:'vllm-xpu',role:'coding'};
const recall = (model,ctx,patch={}) => FM.seatSafetyChecks(D,{...local,model,ctx,...patch}).find(c=>c.id==='context_recall');
assert.equal(recall(local.model,12288).status,'PASS');
for (const ctx of [12289,18000,23000]) assert.equal(recall(local.model,ctx).severity,'warn');
assert.equal(recall(local.model,23001).severity,'degraded');
for (const model of ['Qwen3.8-35B','Qwen3.8-Flash-Next','q27','Bonsai-27B']) {
  assert.equal(recall(model,24000).status,'UNKNOWN');
  assert.equal(recall(model,24000).severity,null);
}
assert.equal(recall('other',10001,{measured_recall_budget_tokens:10000}).severity,'degraded');
const batch = {model:'fnx',agents:33,serving:{batch:32}};
assert.equal(FM.seatSafetyChecks(D,batch)[0].status,'FAIL');
assert.equal(FM.seatSafetyChecks(D,{...batch,admission:{policy:'door',conversations:32}})[0].status,'PASS');
// Isolate the ranking seam; no solver/fixture prerequisites hide unknown seats.
const predict = FM.predictRecipe;
try {
  FM.predictRecipe = () => ({status:'PREDICTED',decode_tps:100});
  assert.equal(ADV.bestPassing(data,[unknown]).best[0].quality.badge,'unknown');
  const probed = {...unknown,id:'probed',recipe:{...recipe,battery_pass_rate:11/12},measured:{kl_verdict:'PASS',decode_tps:{single:20}}};
  for (const runs of [[unknown,probed],[probed,unknown]])
    assert.equal(ADV.bestPassing(data,runs).best[0].run.id,'probed');
  assert.equal(ADV.bestPassing(data,[probed],{role:'diff_review'}).best.length,0);
  assert.equal(ADV.bestPassing(data,[{...probed,recipe:{...probed.recipe,review_screen:screen}}],{role:'diff_review'}).best.length,1);
  const noRecall = {...probed,recipe:{...probed.recipe,ctx:24000}};
  assert.equal(ADV.bestPassing(data,[noRecall]).best.length,1);
} finally { FM.predictRecipe = predict; }
// A run that declares no agent role is not a seat: no behavioural admission and no recall row, so a decode
// benchmark keeps its KL verdict. Evidence it does bring is still judged, and a caller can ask for the seat reading.
const bench = {box:'odie',model:'fixture',engine:'strata',store:{types:['q4']}};
assert.equal(FM.earfCheck(data, bench).status, 'PASS');
assert.equal(FM.earfCheck(data, bench).admission_basis, 'not_required');
assert.equal(FM.earfCheck(data, {...bench, earf:.9529}).status, 'FAIL');
assert.equal(FM.earfCheck(data, bench, {role:'coding'}).status, 'UNKNOWN');
const benchRun = {id:'bench',box:'odie',model:'fixture',recipe:bench,measured:{kl_verdict:'PASS',decode_tps:{single:100}}};
assert.equal(ADV.runQuality(data, benchRun, [benchRun]).status, 'PASS');
assert.equal(FM.seatSafetyChecks(D,{model:'other',ctx:131072}).length, 0);
// The 12k/23k probe is the int4 file's: another quant of the 27B, or a context that is not a count, stays UNKNOWN.
assert.equal(recall('Qwen3.8-27B-GSQ-IQ2_XS',8000).status,'UNKNOWN');
assert.equal(recall(local.model,'deep').status,'UNKNOWN');
// Agent sweeps and tagged counts are judged at their largest value, not coerced to NaN.
assert.equal(FM.seatSafetyChecks(D,{model:'fnx',conditions:{workload:{agents:[1,16,40]}},serving:{batch:{v:32}}})[0].id,'slot_admission');
assert.equal(FM.seatSafetyChecks(D,{model:'fnx',conditions:{workload:{agents:[1,16,32]}},serving:{batch:32}}).length,0);
const original = FM.cardCurveFit(D);
D.multibox.card_curves.anchors.reverse();
const reordered = FM.cardCurveFit(D);
assert.deepStrictEqual(original.consts, reordered.consts);
assert(original.converged);
assert(original.solved.filter(x => x.status === 'SOLVED').every(x => Math.abs(x.residual) < 1e-5));
console.log('Wave 3 agent fidelity and admission checks passed');
