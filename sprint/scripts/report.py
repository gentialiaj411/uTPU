import csv,json,statistics,subprocess,datetime
from pathlib import Path
D=Path('sprint/data');lines=[]
def load(name,default=None):
 try:return json.loads((D/name).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def emit(x=''):lines.append(str(x))
def table(headers,rows):
 emit('| '+' | '.join(headers)+' |');emit('| '+' | '.join(['---']*len(headers))+' |')
 for row in rows:emit('| '+' | '.join(str(x).replace('|','/') for x in row)+' |')
 emit()
def status():
 es=load('engineering_status.json',{});cc=[]
 if (D/'crosscheck_comparison.csv').exists():cc=list(csv.DictReader((D/'crosscheck_comparison.csv').open()))
 return es,cc
es,cc=status();small=load('15m_small.json',{});big=load('15m_big.json',{});p4=es.get('P4',{});runtime=load('15m_timing.json',[])
bounds={}
if (D/'argmax_sweep.csv').exists():
 with (D/'argmax_sweep.csv').open() as f:
  for r in csv.DictReader(f):
   key=(r['variant'],int(r['block']),r['ordering'],float(r['rate']))
   b=bounds.setdefault(key,dict(skip=[],bytes=[],tps=[],seeks=[],identical=True,metadata=int(r['bound_metadata'])))
   b['skip'].append(float(r['skipped_fraction']));b['bytes'].append(int(r['bytes_per_token']));b['tps'].append(float(r['estimated_tps']));b['seeks'].append(int(r['seeks']));b['identical'] &= r['identical']=='True'
best=max(((statistics.mean(b['tps']),key,b) for key,b in bounds.items() if key[3]==.25),default=None,key=lambda x:x[0])
go='NO-GO for hardware from these bounds' if best and best[0]<=50e6/(11633661/.25) else 'CONDITIONAL: screening advantage only; hardware and full traffic accounting still required'
complete_p1=len(cc)==11 and all(r['identical']=='True' and r['icarus']==r['verilator']=='pass' for r in cc) and big.get('status')=='pass' and len(big.get('prompts',[]))==8
p4status=p4.get('status','TIMEBOXED')
if p4status=='IN-PROGRESS':
 required=['regen_'+x for x in ['gemv','gemv15','actq','rms','vec','attn','attn15','qspi','uart']]+['fault_top','fault_gemv','fault_qspi','fault_board']
 p4status='DONE' if all(es.get(k,{}).get('rc')==0 for k in required) else 'TIMEBOXED; metadata smokes passed but full fault/suite acceptance incomplete'
summary=[
 'P0 DONE: integration closed at e8d9282; all owner log tails EXIT 0, fresh strict gate PASS, stale-comment FAIL, restored gate PASS (MEASURED; sprint/data/p0_*).',
 f'P1 {"DONE" if complete_p1 else "TIMEBOXED / acceptance incomplete"}: {sum(r["identical"]=="True" and r["icarus"]==r["verilator"]=="pass" for r in cc)}/{len(cc)} completed backend comparisons identical and passing (MEASURED; crosscheck_comparison.csv).',
 f'P1 15M small: {small.get("status","NOT RUN")}, {small.get("steps","N/A")} steps; wall {next((round(r["wall_seconds"],2) for r in runtime if r["label"]=="15m_small"),"N/A")} s (MEASURED; 15m_timing.json).',
 f'P1 large: {big.get("status","NOT RUN")}, {big.get("steps","N/A")} steps, {len(big.get("prompts",[]))} actual prompts; errors logit/index/argmax = {big.get("logit_mismatches","N/A")}/{big.get("index_errors","N/A")}/{big.get("argmax_errors","N/A")} (MEASURED; 15m_big.json).',
 'P2 DONE code/tests: simulator executable/compiler identity, Python/NumPy identity and fault model/golden dependencies tracked; see test_gate_final.log (MEASURED).',
 f'P3 {go}; MEASURED max mean row skip '+(f'{max(statistics.mean(b["skip"]) for b in bounds.values()):.6f}' if bounds else 'N/A')+' over the final screening variants; '+(f'best screening estimate {best[0]:.4f} tok/s at 0.25 B/clk (ESTIMATED), versus baseline bandwidth ceiling 1.0745 (DERIVED).' if best else 'NOT RUN / no completed screening data.'),
 f'P4 {p4status}; see metadata and final-regeneration artifacts for measured byte/cycle changes. No unvalidated arithmetic claim.',
 'P5 TIMEBOXED design-only: existing autonomous prefetch already overlaps serial compute until FIFO full; no extra RTL feature retained.',
 'P6 TIMEBOXED until physical regeneration: strict gate kept strict; OWNER_RUN2.ps1 contains sequential logged Vivado jobs and final gate. No push or hardware programming.'
]
emit('# uTPU engineering sprint report');emit('Generated '+datetime.datetime.now().isoformat()+'. Worktree uTPU-perf, branch perf, based on integration e8d9282.\n')
emit('## Summary');emit('\n'.join(str(i+1)+'. '+s for i,s in enumerate(summary)));emit()
emit('## Bugs and verification findings first')
emit('- Unit TB timing was unspecified: GEMV used #0.1 without timescale. Verilator initially reported 12,754 mismatches and stale done behavior. Explicit 1ns/1ps is now supplied to every TB lacking a time declaration for BOTH backends. Golden arithmetic and scoreboard remain unchanged. The clean backends agree on 100.2654 cycles/tile versus historical 101.2654; this difference is testbench sampling, not a compute speedup. Evidence gemv_*.json and SPRINT_LOG.md.\n- The 15M e2e runner silently truncated --prompts 8 to its two-entry prompt bank. It now extends the bank from TinyStories validation and rejects context overflow rather than silently under-testing. Existing default prompts are preserved.\n- Provenance previously missed simulator executable identity, Python/NumPy versions and fault model/numeric inputs. New inputs are required by coverage, so historical artifacts fail until regenerated.\n- P3 self-review caught a software screening error: maximum per-row bounds cannot be reconstructed from only groupwise block-header maxima. The initial CSV was explicitly archived INVALID and excluded. Final coefficient results actually use the declared groupwise block header.\n')
emit('## Evidence rules and environment')
emit('MEASURED means executed here, with command/artifact. DERIVED is arithmetic from named measurements or explicitly named historical inputs. ESTIMATED is a model with assumptions; never a board result. Numerical dimensions and test counts are configurations. No bench/results JSON was hand-edited. Tools: Windows Python/NumPy identities are recorded in each newly generated provenance; Verilator 5.032 runs under WSL Ubuntu using tools extracted in ../uTPU-research/research/data/tools. Icarus uses C:/iverilog/bin. No new Vivado run was attempted.\n')
emit('## P0 integration closure')
emit('Commands: strict gate python tools/decode_gate.py; stale demonstration script ../uTPU-integ/build/sprint_logs/close.py appends a comment to GEMV inside try/finally, checks the stale failure, restores exact bytes and reruns the gate. Owner logs were audited by final-line EXIT 0, saved as sprint/data/p0_owner_log_audit.csv. Copies of before/stale/restored gate output are p0_gate_*.log. Integration docs and owner-generated artifacts committed e8d9282; untracked owner clockInfo.txt was preserved.\n')
emit('## P1 portable simulator backend')
emit('tools/decode/sim.py implements --sim icarus|verilator (default icarus), shared by runners and inherited through UTPU_SIM for fault subprocesses. Windows memory paths are mapped to WSL only in the generated Verilator TB copy. Compile logs go to build/decode/*_compile.log. Generated C++/objects build under isolated /tmp/utpu-perf directories, which are removed after every completed run/failure to avoid filling the 7.7GB tmpfs. The initial exhaustion failure and recovery logs are preserved. Build failures preserve CalledProcessError so existing parameter-rejection tests remain meaningful. CLI help added to each runner. Explicit TB time precision applies identically to both backends. No expected outputs or tolerances were changed.\n')
emit('Commands: python sprint/scripts/crosscheck.py; python sprint/scripts/compare.py; python sprint/scripts/regress15.py. Crosscheck compiles clean and mutation variants through the original runners. Comparison includes every summary, mismatch count, cycle counter and variant recursively; stdout hashes and wall times are intentionally not asserted equal. Original output/logs are sprint/data/<unit>_<backend>.{json,log}.\n')
table(['Unit','Icarus status','Verilator status','Identical relevant fields MEASURED','Differences'],[[r['unit'],r['icarus'],r['verilator'],r['identical'],r['differences']] for r in cc])
if (D/'crosscheck.csv').exists():
 times={r['label']:r for r in csv.DictReader((D/'crosscheck.csv').open())}
 rows=[]
 for name in ['gemv','gemv15','actq','rms','vec','attn','attn15','qspi','uart','e2e_ideal','e2e_qspi']:
  a=times.get(name+'_icarus');b=times.get(name+'_verilator')
  if a and b:rows.append([name,f"{float(a['seconds']):.3f}",f"{float(b['seconds']):.3f}",f"{float(a['seconds'])/float(b['seconds']):.3f}"])
 table(['Runner','Icarus wall s MEASURED','Verilator wall s MEASURED','Speedup DERIVED I/V'],rows)
emit('These runner times include compilation, golden preparation and provenance; they are not simulation-only speedups. The earlier 87x was a small precompiled workload, not a guaranteed full 15M speedup. The measured 15M small and big wall times replace that extrapolation. Concurrent screening/regressions affect wall time.\n')
table(['15M workload','Actual prompts MEASURED','Steps MEASURED','Status','Logit mismatches MEASURED','Index / argmax MEASURED'],[[n,len(o.get('prompts',[])),o.get('steps','N/A'),o.get('status','NOT RUN'),o.get('logit_mismatches','N/A'),str(o.get('index_errors','N/A'))+' / '+str(o.get('argmax_errors','N/A'))] for n,o in [('small',small),('8x32',big)]])
emit('## P2 provenance')
emit('Simulator provenance records name, version, executable path/SHA256 and compiler SHA256. Version strings are cached only by executable content digest; stale checks recompute binary digests. Python sys.version and NumPy __version__ hashes are mandatory virtual dependencies. Fault provenance includes M260, M15, NUMERICS and LUTS in addition to fault runner and RTL dependencies. Tests cover missing identities, changed recorded identities, actual mocked runtime changes and actual mocked simulator digest changes, plus each model/numeric source digest. Command python -m pytest tools/decode/test_gate.py -q; test_gate_final.log. New virtual dependencies also make physical artifacts stale; OWNER_RUN2 regenerates their provenance rather than accepting historical results.\n')
emit('## P3 exact bounded classifier argmax')
emit('Command python sprint/scripts/argmax_bound.py. Dataset: first 50 research prompts derived from TinyStories validation, eight prompt tokens plus 120 generated tokens, KV4, maximum position127. The reference integer model computes all logits as an independent checker; the selection simulation reads only admitted blocks and never uses skipped logits to set the running best. This is a screening measurement of reads and decisions, NOT a Python inference speedup.\n')
emit('### Bound proof')
emit('For a weight group g, write d_rg=sum_i x_i*w_ri; a_g is its positive activation scale and m_rg=RNE(M_rg/2^16)>=0; s_rg=shift_rg-16>=0. L1 gives |d|<=max|x|*sum|w|; Cauchy gives |d|<=ceil(sqrt(sum x^2 * sum w^2)). Integer square comparisons correct the sqrt proposal. Multiplying by positive a*m preserves the bound; signed RNE is odd and monotone, so |RNE(d*a*m/2^s)|<=RNE(D*a*m/2^s). Checked product bounds reject INT64 overflow rather than silently wrap.\n')
emit('div127 uses Q32 reciprocal R=33818641, followed by at most one unit correction. For |n|<=N, a safe envelope is floor((N*R+2^31)/2^32)+1; this also covers the wide approximate division path and does not incorrectly assume exact division there. Summing per-group envelopes bounds the final signed logit. Reciprocal products are evaluated in high/low limbs to prevent overflow. Runtime checks confirm every reference logit lies below its bound.\n')
emit('Coefficient variant: precompute c_rg=ceil(L1_rg*m_rg*2^32/2^s_rg). Store C_bg=max_rows_in_block c_rg separately for every group. For each activation, N_bg=ceil(C_bg*a_g*max|x_g|/2^32)+1 bounds the rounded scaled magnitude of every row in that block. Apply the reciprocal envelope and sum. Measured model coefficients fit u32; the script asserts this. Groupwise block maxima are used even though they are looser than max of activation-specific per-row sums.\n')
emit('Skip a block only if U<best, or U==best and every row index in the block is greater than the current best index. RTL and integer lexsort choose the lowest index on ties. Independent reference checks require identical argmax AND maximum logit on every sampled position. No probability or quality tolerance is used. Because no sampled block was skipped, empirical equality does not exercise a successful pruning branch; the skip safety argument is mathematical, not evidence of useful pruning.\n')
rows=[]
for key,b in sorted(bounds.items()):
 if key[3]!=.25:continue
 v=sorted(b['skip']);q=lambda p:v[min(len(v)-1,int(p*(len(v)-1)))]
 other=bounds.get((*key[:3],.5));rows.append([*key[:3],len(v),f"{statistics.mean(v):.5f}",f"{q(.1):.4f}/{q(.5):.4f}/{q(.9):.4f}",b['metadata'],f"{statistics.mean(b['bytes']):.1f}",f"{statistics.mean(b['tps']):.4f}",f"{statistics.mean(other['tps']):.4f}" if other else 'N/A',b['identical']])
table(['Bound','Block rows','Ordering','Positions MEASURED','Mean skipped MEASURED','p10/p50/p90 MEASURED','Metadata B DERIVED','Mean B/token DERIVED','tok/s .25 ESTIMATED','tok/s .5 ESTIMATED','All identical MEASURED'],rows)
emit('Traffic model uses pre-P4 format: 144 code bytes +45 scale bytes per row (189B). L1/L2 screening streams all per-row norm+scale metadata (9 groups*(2+5)*32000=2,016,000 B), then rereads admitted blocks including scales; this deliberately pays both passes. Coefficient headers cost ceil(32000/block)*9*4 bytes: block16 72,000B, block64 18,000B, block256 4,500B. Against eight BRAM36=36,864B nominal capacity, only block64/256 header caches fit before port/fragmentation/control costs. Upper-first ordering needs header storage and sort/control; hardware cost is NOT MEASURED. No headers are assumed free.\n')
emit('QSPI model counts one seek for each non-contiguous admitted block: 6Bh command8+address24+dummy8=40 SCLK cycles=80 core clocks at HALF=1. Stored CSV predictions are optimistic protocol-only ceilings: cycles=(11,633,661-6,048,000+classifier_bytes)/rate+80*seeks; tok/s=50,000,000/cycles. CS-high/capture/pad/control latency and compute are additional; therefore these are upper bounds on architectural benefit, not measured end-to-end tok/s. Full non-contiguous reads preserve aligned packed tiles; blocks16/64/256 are all multiples of16.\n')
emit('P3 decision: '+go+'. A conditional screening advantage does not authorize RTL. Required follow-up: measured controller/metadata bandwidth and seek latency, then an independently checked hardware prototype. A failure of these norm bounds does not prove every exact bounded argmax scheme impossible.\n')
emit('## P4 compact metadata')
emit('Disposition: '+p4status+'. Prepared implementation sprint/scripts/implement_metadata.py is only applied if all baseline backend comparisons pass, protecting an unvalidated production format from depending on a failing simulator. If smoke/proof checks fail, exact pre-change source bytes are restored. See engineering_status.json for whether it actually ran.\n')
emit('If applied: packer stores q=RNE(original_multiplier,16) as u16 plus original shift u8 for GEMV tensors; embedding scale metadata stays u32+u8. RTL G_SCALES consumes three bytes, expands {q,16b0}; embedding E_SCALES still consumes five. Classifier tile offsets and board/default image addresses are regenerated from the packer. Math proof: RNE((q<<16)/2^16)=q, exactly the old GEMV effective multiplier; subsequent products, shifts and div127 are unchanged. Packer conservatively requires q<32768 although stored in u16, because the unchanged GEMV interprets the expanded 32-bit multiplier as signed. The measured model range fits this constraint; accepting arbitrary u16 would be unsound. Generic flash models consume the new packed image automatically, rather than supplying hand-authored expected data. New top fault omits <<16 and must be detected.\n')
emit('Validation commands: python -m pytest sprint/scripts/test_compact_metadata.py -q compares the integer golden on original versus independently decoded compact metadata over generated tie and random cases; original clean/mutation runners and touched-unit fault suites are then rerun. No change to runq_runtime.py or fixed.py. Full acceptance requires every listed producer/fault to pass, not just the smoke.\n')
rows=[]
for name,before,after in [('260K QSPI','e2e_qspi_verilator.json','260k_metadata.json'),('15M QSPI',None,'15m_metadata.json')]:
 a=load(before,{}) if before else {};b=load(after,{})
 rows.append([name,a.get('stream_bytes_per_token','NOT MEASURED'),b.get('stream_bytes_per_token','NOT MEASURED'),a.get('cycles_per_token_mean','NOT MEASURED'),b.get('cycles_per_token_mean','NOT MEASURED'),f"{50e6/b['cycles_per_token_mean']:.5f}" if b.get('cycles_per_token_mean') else 'NOT MEASURED'])
table(['Config','Before B/token MEASURED','After B/token MEASURED','Before cycles/token MEASURED','After cycles/token MEASURED','50MHz tok/s DERIVED'],rows)
emit('The 15M baseline in 15m_small.json is IDEAL 1 B/clk and cannot serve as a measured pre-change QSPI baseline. Historical 11.63MB/token and estimated 1.07 tok/s are not substituted for a fresh QSPI measurement. Research predicted 949,248B saving/pass; only actual producer output above supports a new measured claim.\n')
emit('## P5 serial overlap')
emit('TIMEBOXED design-only. Independent pf_active/pf_left prefetch and st_ready already run during RMS/attention/SwiGLU/quant until FIFO fills. Thus the research inference that all those phase clocks imply idle flash was too strong. No unmeasured enable parameter was added. See sprint/P5_PREFETCH_REVIEW.md. FIFO sizing and occupancy instrumentation remain next experiments, not a delivered speedup.\n')
emit('## P6 regeneration and strict gate')
emit('Engineering commands and return codes: sprint/data/engineering_status.json, with timestamped command lines in SPRINT_LOG.md. Every artifact is produced by its runner; there are no copied/hash-patched artifacts. Check the final full strict gate log sprint/data/strict_gate_final.log; any unmet item remains unmet. Physical results are NOT RUN until OWNER_RUN2.ps1 executes. No alternate relaxed gate was introduced.\n')
rows=[]
for k,v in es.items():
 if isinstance(v,dict) and 'rc' in v:rows.append([k,v['rc'],f"{v.get('seconds',0):.2f}",' '.join(v.get('command',[]))])
table(['Job','Exit MEASURED','Wall seconds MEASURED','Exact command'],rows)
emit('## Diff and numbers versus audit-fixes')
r=subprocess.run(['git','diff','e8d9282','--stat'],capture_output=True,text=True);emit('```text\n'+r.stdout+'```')
emit('Before: implicit unit TB time precision and compiler-version-only provenance. After: shared backend, explicit precision, mandatory runtime/tool identities. Representative GEMV cycle difference 101.2654 ->100.2654 is a TB sampling correction and both new backends must agree; not a hardware performance gain. Production arithmetic source diff must stay empty for firmware/host/numerics_v2/runq_runtime.py and fixed.py. New metadata measurements, if complete, are separated from research estimates above.\n')
emit('## OWNER_RUN2.ps1')
emit('NOT RUN: `powershell -ExecutionPolicy Bypass -File OWNER_RUN2.ps1`. Requires Vivado2025.2, the models junction, generated LUTs, matching final packed image, and successful final fast artifact regeneration. All jobs sequential; build/owner_logs2/<job>.log records START/EXIT, first failure throws, strict gate is last. No bitstream generation.\n')
emit('```powershell\n'+Path('OWNER_RUN2.ps1').read_text()+'```')
emit('## Next three items')
table(['Priority','Work','Concrete acceptance criterion'],[['1','Resolve every backend difference and run final physical regeneration','All unit clean/mutation SUMMARY/mismatch/cycle fields identical; real 8-prompt regression passes; OWNER_RUN2 finishes with strict unmet[]'],['2','Validate compact metadata end-to-end and route it','Independent metadata decode matches every integer output, wrong-expansion fault detected; fresh QSPI bytes/cycles; routed board timing/resource budget fits'],['3','Screen tighter progressive classifier bounds before more RTL','A bound must actually skip blocks on the same 50x120 dataset, preserve exact argmax/max-logit and improve byte/cycle estimates after metadata and seek costs']])
emit('## NOT RUN / incomplete list')
emit('- OWNER_RUN2 Vivado and final physical gate: exact command above.\n- If P1/P4 acceptance is incomplete, metadata hardware optimization is not declared complete merely because a patch script exists.\n- Full baud/reset/context boundary expansion, hardware bounded argmax, FIFO-depth A/B prefetch experiment: NOT RUN; no feature or result claimed.\n- `python tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi --gen 8 --sim icarus --output sprint/data/15m_qspi_icarus.json`: NOT RUN (long Icarus experiment).\n')
Path('SPRINT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
Path('sprint/data/final_summary.txt').write_text('\n'.join(summary),encoding='utf-8')
print('SUMMARY report written; P1 complete=',complete_p1,'P4=',p4status,'P3=',go)
