"""Fail-closed Phase-4a completion gate for VMATMUL/VLAYOUT."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'bench/results/phase4a_vector_evidence.json'

def run(cmd,bad):
 print('RUN:', ' '.join(cmd),flush=True)
 if subprocess.run(cmd,cwd=ROOT).returncode:bad.append(' '.join(cmd))

def main():
 bad=[];py=sys.executable
 for gate in ('tools/phase1_gate.py','tools/phase2_gate.py','tools/phase3_gate.py'):run([py,gate],bad)
 run([py,'tools/run_phase4a_evidence.py'],bad)
 if not RESULT.is_file():bad.append('Phase-4a evidence missing')
 else:
  try:r=json.loads(RESULT.read_text(encoding='utf-8'))
  except Exception as e:bad.append('invalid Phase-4a evidence: '+str(e));r={}
  mat=r.get('matmul',{});rtl=mat.get('rtl',{})
  if r.get('status')!='passed':bad.append('Phase-4a evidence status')
  if mat.get('sim_cases',0)<2000 or rtl.get('cases',0)<2000:bad.append('VMATMUL random-case minimum')
  if mat.get('families',0)<14 or set(mat.get('edge_cases',()))!={'max_magnitude','K=1','M=N=1','causal_row_0'}:bad.append('VMATMUL family/edge coverage')
  for name,item in (('VMATMUL RTL',rtl),('VLAYOUT',r.get('layout',{})),('reference score/AV',r.get('reference_score_av',{})),('full top',r.get('full_top',{}))):
   if item.get('mismatches',1)!=0 or item.get('passed') is not True:bad.append(name+' mismatch')
  lay=r.get('layout',{})
  if lay.get('cases',0)<20 or lay.get('round_trips',0)<5 or not lay.get('tails_and_padding'):bad.append('VLAYOUT layout/tail coverage')
  ref=r.get('reference_score_av',{})
  if ref.get('sequences',0)<8 or ref.get('score_cases',0)<24 or ref.get('av_cases',0)<24:bad.append('reference score/AV coverage')
  if r.get('full_top',{}).get('cases',0)<1:bad.append('full-top VMATMUL coverage')
 synth=ROOT/'bench/results/boolean_vec_unit_synth.json'
 if not synth.is_file():bad.append('synthesis artifact missing')
 else:
  s=json.loads(synth.read_text(encoding='utf-8'));one=s.get('modes',{}).get('1',{})
  if s.get('part')!='xc7s50csga324-1' or s.get('core_clock_hz')!=50000000:bad.append('wrong synthesis target')
  if any(one.get(k,999999)>v for k,v in (('lut',32600),('bram',75),('dsp',120))):bad.append('synthesis utilization limit')
  if one.get('wns_ns',-999)<0 or s.get('vector_unit_ooc',{}).get('wns_ns',-999)<1:bad.append('synthesis timing target')
 dirty=subprocess.check_output(['git','status','--porcelain','--','firmware/host','rtl','tools','bench/results/phase4a_vector_evidence.json'],cwd=ROOT,text=True).strip()
 if dirty:bad.append('uncommitted scoped changes')
 for x in bad:print('UNMET:',x)
 return int(bool(bad))
if __name__=='__main__':raise SystemExit(main())
