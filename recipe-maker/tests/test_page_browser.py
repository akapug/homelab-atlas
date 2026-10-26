"""Optional model-free check of the generated page in a real sandboxed browser."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMIUM = Path.home() / '.cache/ms-playwright/chromium-1228/chrome-linux64/chrome'
CHROMIUM = Path(os.environ.get('FITVIZ_CHROMIUM') or os.environ.get('FAB_BROWSER_EXECUTABLE') or DEFAULT_CHROMIUM)
DEFAULT_TIMEOUT_S = 120
TIMEOUT_S = float(os.environ.get('FITVIZ_BROWSER_TIMEOUT_S', str(DEFAULT_TIMEOUT_S)))


@unittest.skipUnless(CHROMIUM.is_file(), 'sandboxed Chromium not installed; set FITVIZ_CHROMIUM')
class StaticBrowser(unittest.TestCase):
    def test_real_generated_page_renders_the_model(self):
        self.assertTrue((ROOT / 'index.html').is_file(), 'run build.py first')
        # Use the same CDP pipe transport as Fab's sandboxed browser probe: Chrome 149's
        # --dump-dom can navigate without printing or exiting. No server or TCP debug port.
        incoming, writer = os.pipe()
        reader, outgoing = os.pipe()
        with tempfile.TemporaryDirectory(prefix='fitviz-static-browser-') as profile, tempfile.TemporaryFile() as log:
            browser = subprocess.Popen([
                'bash', '-c', 'exec 3<&"$1" 4>&"$2"; shift 2; exec "$@"', 'fitviz-cdp',
                str(incoming), str(outgoing), str(CHROMIUM), '--headless', '--disable-gpu',
                '--remote-debugging-pipe', '--user-data-dir=' + profile, 'about:blank',
            ], pass_fds=(incoming, outgoing), stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            os.close(incoming)
            os.close(outgoing)
            messages, events = bytearray(), []
            deadline, serial = time.monotonic() + TIMEOUT_S, 0
            with selectors.DefaultSelector() as selector:
                selector.register(reader, selectors.EVENT_READ)

                def receive():
                    while b'\0' not in messages:
                        remaining = deadline - time.monotonic()
                        self.assertGreater(remaining, 0, 'browser CDP deadline expired')
                        self.assertTrue(selector.select(remaining), 'browser CDP response timed out')
                        block = os.read(reader, 65536)
                        self.assertTrue(block, 'browser CDP pipe closed unexpectedly')
                        messages.extend(block)
                    stop = messages.index(0)
                    result = json.loads(messages[:stop])
                    del messages[:stop + 1]
                    return result

                def call(method, params=None, session=None):
                    nonlocal serial
                    serial += 1
                    request = {'id': serial, 'method': method, 'params': params or {}}
                    if session:
                        request['sessionId'] = session
                    payload = memoryview(json.dumps(request).encode() + b'\0')
                    while payload:
                        payload = payload[os.write(writer, payload):]
                    while True:
                        response = receive()
                        if response.get('id') == serial:
                            self.assertNotIn('error', response, 'CDP ' + method + ' failed')
                            return response.get('result', {})
                        events.append(response)

                try:
                    target = call('Target.createTarget', {'url': 'about:blank'})['targetId']
                    session = call('Target.attachToTarget', {'targetId': target, 'flatten': True})['sessionId']
                    call('Page.enable', session=session)
                    call('Runtime.enable', session=session)
                    events.clear()
                    navigation = call('Page.navigate', {'url': (ROOT / 'index.html').as_uri()}, session)
                    self.assertNotIn('errorText', navigation)
                    while not any(e.get('method') == 'Page.loadEventFired' and e.get('sessionId') == session for e in events):
                        events.append(receive())
                    result = call('Runtime.evaluate', {
                        'expression': "({chart:!!document.getElementById('plot'),navigation:!!document.querySelector('nav.toc'),loading:document.getElementById('app').textContent.includes('Loading the planner...'),model:typeof window.FitModel.checkRecipe==='function',recipes:window.FIT_DATA.recipes.length})",
                        'returnByValue': True,
                    }, session)
                    self.assertNotIn('exceptionDetails', result)
                    actual = result['result']['value']
                    self.assertTrue(actual['chart'], 'real model/solver did not render the interactive chart')
                    self.assertTrue(actual['navigation'], 'actual planner navigation missing')
                    self.assertFalse(actual['loading'], 'static scripts failed to finish loading')
                    self.assertTrue(actual['model'])
                    generated = json.loads((ROOT / 'data.json').read_text())
                    self.assertEqual(actual['recipes'], len(generated['recipes']))
                    feature = call('Runtime.evaluate', {
                        'expression': """(() => {
const data=window.FIT_DATA, source=data.recipes.find(r=>r.recipe && !r.recipe.kind && r.recipe.engine_build);
if (!source) return {missing:true};
const r=JSON.parse(JSON.stringify(source)); r.id='synthetic-browser-fallback'; r.label='synthetic browser fallback causal control';
delete r.recipe_audit_src; delete r.recipe.fallback_plan_basis;
r.recipe.fallback_plan={src:'synthetic browser intent',classes:{dense_op:{tier:'matrix',src:'synthetic browser intent'}}};
r.observed={fallback_ledger:{src:'synthetic browser receipt',identity:Object.fromEntries(['box','model','engine','engine_build'].map(k=>[k,r.recipe[k]])),events:[{class:'dense_op',tier:'cpu',proof:'observed',src:'synthetic browser receipt'}]}};
const control=data.checks.fallback.controls.find(c=>c.id==='groups-disable-mtp');
const groupSource=data.recipes.find(x=>x.recipe && x.recipe.model===control.model && x.recipe.engine===control.engine);
if (!groupSource) return {missing:true};
const groups=['split','unknown','single'].map(state=>{
const g=JSON.parse(JSON.stringify(groupSource));g.id='synthetic-browser-groups-'+state;g.label='synthetic browser grouped draft '+state;
delete g.recipe_audit_src;delete g.recipe.fallback_plan_basis;delete g.recipe.gpu_split;
g.recipe.engine_build=control.builds[0];g.recipe.conditions={engine_flags:'--batch-groups 10'};
g.recipe.fallback_plan={src:'synthetic browser grouped draft intent',classes:{batch_mtp:{tier:'on',src:'synthetic browser intent'}}};
g.observed={};
if (state==='split') g.recipe.gpu_split={banks_gib:[1,1]};
if (state==='single') g.recipe.gpu_split=null;
return g;});
data.recipes.push(r,...groups);window.FIT_APP.render();
const table=document.querySelector('table[aria-label="Recorded run checks"]');
const item=table && table.querySelector('tr[data-run-id="synthetic-browser-fallback"] [data-check-id="fallback"]');
const audits=data.recipes.filter(x=>x.recipe_audit_src).map(x=>{
const tr=[...table.querySelectorAll('tr[data-run-id]')].find(t=>t.dataset.runId===x.id),cell=tr && tr.querySelector('[data-check-id="fallback"]');
const ram=tr && tr.querySelector('[data-check-id="expert_residency"]');
return {id:x.id,status:cell && cell.textContent,ram:ram && ram.textContent,basis:!!cell && cell.title.includes('post-hoc maintained intent')};});
const headers=table && [...table.querySelectorAll('th')].map(x=>x.textContent),body=table && [...table.querySelectorAll('tr[data-run-id]')];
const groupCells=groups.map(g=>{const cell=table.querySelector('tr[data-run-id="'+g.id+'"] [data-check-id="fallback"]');return {status:cell && cell.textContent,detail:cell && cell.title};});
return {headers,failure:item && item.textContent,detail:item && item.title,audits,groups:groupCells,aligned:body.every(tr=>tr.cells.length===headers.length),rowIds:body.map(tr=>tr.dataset.runId)};
})()""", 'returnByValue': True,
                    }, session)
                    self.assertNotIn('exceptionDetails', feature)
                    rendered = feature['result']['value']
                    self.assertFalse(rendered.get('missing'), 'real data has no eligible recipe to exercise the existing check table')
                    self.assertIn('expert RAM', rendered['headers'])
                    self.assertIn('fallbacks', rendered['headers'])
                    self.assertTrue(rendered['aligned'], 'new headers and every current body row must have matching cells')
                    synthetic = {'synthetic-browser-fallback'} | {'synthetic-browser-groups-' + state for state in ('split', 'unknown', 'single')}
                    self.assertEqual(set(rendered['rowIds']), {r['id'] for r in generated['recipes']} | synthetic)
                    self.assertEqual(rendered['failure'], 'FAIL')
                    self.assertIn('dense_op:run FAIL', rendered['detail'])
                    split, unknown, single = rendered['groups']
                    self.assertEqual(split['status'], 'FAIL')
                    self.assertIn('control:groups-disable-mtp FAIL', split['detail'])
                    self.assertEqual(unknown['status'], 'UNKNOWN')
                    self.assertIn('control:groups-disable-mtp UNKNOWN', unknown['detail'])
                    self.assertEqual(single['status'], 'UNKNOWN')
                    self.assertNotIn('control:groups-disable-mtp', single['detail'])
                    expected_audits = sum(bool(r.get('recipe_audit_src')) for r in generated['recipes'])
                    self.assertEqual(len(rendered['audits']), expected_audits)
                    if (ROOT / 'recipes' / 'snoopy-v41-gsq-iq2xs-strata.json').is_file():
                        self.assertGreater(expected_audits, 0, 'maintained intent never reached generated data or the actual UI')
                    self.assertTrue(all(x['basis'] and x['status'] == 'UNKNOWN' and x['ram'] == 'UNKNOWN' for x in rendered['audits']), 'unverified maintained intent must render UNKNOWN, never a fabricated pass')
                    missing_training = call('Runtime.evaluate', {
                        'expression': """(() => {
const data=window.FIT_DATA, configs=Object.values((data.window||{}).configs||{});
if (!configs.length || !data.network) return {missing:true};
const saved=configs.flatMap(w=>w.points.filter(p=>p.use==='fit').map(p=>[p,p.tps]));
saved.forEach(([p])=>{p.tps={v:null};});window.FIT_APP.render();
const result={status:window.FIT_NETWORK_REPORT.status,text:document.getElementById('network-rank-summary').textContent};
saved.forEach(([p,tps])=>{p.tps=tps;});window.FIT_APP.render();
result.restored=window.FIT_NETWORK_REPORT.plan.total>0;return result;
})()""", 'returnByValue': True,
                    }, session)
                    self.assertNotIn('exceptionDetails', missing_training)
                    unknown_rendered = missing_training['result']['value']
                    if generated.get('network') and generated.get('window', {}).get('configs'):
                        self.assertFalse(unknown_rendered.get('missing'))
                        self.assertEqual(unknown_rendered['status'], 'UNKNOWN')
                        self.assertIn('No throughput rank is available', unknown_rendered['text'])
                        self.assertTrue(unknown_rendered['restored'])
                    self.assertFalse(any(e.get('method') == 'Runtime.exceptionThrown' for e in events), 'actual browser script threw an exception')
                    call('Browser.close')
                    browser.wait(timeout=max(1, deadline - time.monotonic()))
                finally:
                    os.close(writer)
                    os.close(reader)
                    if browser.poll() is None:
                        browser.kill()
                        browser.wait()
        print('PASS actual generated page through sandboxed Chromium CDP: maintained intent UNKNOWN, observed fallback FAIL, and grouped-draft split FAIL versus missing/single UNKNOWN reached the existing check table')


if __name__ == '__main__':
    unittest.main()
