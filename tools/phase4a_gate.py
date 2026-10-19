"""Fail-closed Phase-4a completion gate for VMATMUL/VLAYOUT."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'bench/results/phase4a_vector_evidence.json'

def head_vector_blobs():
 paths=['rtl/top/top.sv','rtl/top/top_board.sv']+[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'rtl/vector').glob('*.sv'))]
 return {p:subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=ROOT,text=True).strip() for p in paths}

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
  if lay.get('cases',0)<36 or lay.get('round_trips',0)<8 or not lay.get('tails_and_padding') or lay.get('physical_legacy_load_round_trips',0)<3 or set(lay.get('mode_cases',{}))!=set('01234567') or any(not lay['mode_cases'].get(str(i),0) for i in range(8)):bad.append('VLAYOUT layout/tail/mode coverage')
  ref=r.get('reference_score_av',{})
  if ref.get('sequences',0)<512 or ref.get('score_cases',0)<1536 or ref.get('av_cases',0)<1536:bad.append('reference score/AV coverage')
  if r.get('full_top',{}).get('cases',0)<1:bad.append('full-top VMATMUL coverage')
  f4=r.get('f4_legacy_gemm',{})
  if (f4.get('cases',0)<1 or f4.get('shape')!='4x8x8' or not f4.get('passed') or
      any(f4.get(k,1)!=0 for k in ('numpy_mismatches','isa_mismatches','rtl_mismatches'))):
   bad.append('F4 VLAYOUT/legacy-GEMM boundary proof')
  dec=r.get('decode_errors',{});classes=dec.get('classes',{})
  required={'matmul_bad_record_kind','matmul_reserved_record_bits','matmul_invalid_stride','matmul_illegal_causal_shape','matmul_prohibited_overlap','layout_bad_record_kind','layout_reserved_record_bits','layout_invalid_stride','layout_physical_shape','layout_prohibited_overlap'}
  if not required.issubset(classes) or dec.get('mismatches',1)!=0 or not dec.get('passed'):bad.append('VMATMUL/VLAYOUT decode-error coverage')
 synth=ROOT/'bench/results/boolean_vec_unit_synth.json'
 if not synth.is_file():bad.append('synthesis artifact missing')
 else:
  s=json.loads(synth.read_text(encoding='utf-8'));one=s.get('modes',{}).get('1',{})
  if s.get('part')!='xc7s50csga324-1' or s.get('core_clock_hz')!=50000000:bad.append('wrong synthesis target')
  if s.get('rtl_head_blobs')!=head_vector_blobs():bad.append('synthesis stale')
  if any(one.get(k,999999)>v for k,v in (('lut',32600),('bram',75),('dsp',120))):bad.append('synthesis utilization limit')
  if one.get('wns_ns',-999)<1 or s.get('vector_unit_ooc',{}).get('wns_ns',-999)<1:bad.append('synthesis timing target')
 dirty=subprocess.check_output(['git','status','--porcelain','--','firmware/host','rtl','tools','bench/results/phase4a_vector_evidence.json'],cwd=ROOT,text=True).strip()
 if dirty:bad.append('uncommitted scoped changes')
 for x in bad:print('UNMET:',x)
 return int(bool(bad))
if __name__=='__main__':raise SystemExit(main())
