"""Generate build/decode/top_decode_15m_wrap.sv: top_decode with the stories15M image parameters (build/decode/decode_image_15m/params.json,
written by pack_decode_image.py --model 15m), MAX_SEQ 128, KV4, the 128-position RoPE table. Synthesized by synth_ooc.py as 'top_decode_15m'.
Run gen_qspi_wrap.py afterwards for the variant with qspi_reader."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
KEYS = ['D', 'H', 'NL', 'NH', 'NKV', 'HS', 'VOCAB', 'CLS4', 'WQ4', 'WK4', 'WV4', 'WO4', 'W14', 'W34', 'W24', 'CLS_BASE', 'EMB_SCALE_BASE',
        'LAYERS_BASE', 'IMAGE_BYTES', 'CLS_TILE_BYTES', 'CLS_KB', 'CLS_G', 'RECIP_D', 'NORM_MULT', 'EPS', 'GFB', 'NORM_SHIFT', 'SILU_MULT',
        'SWIGLU_MULT', 'SCORE_MULT', 'SILU_SHIFT', 'SWIGLU_SHIFT', 'SCORE_SHIFT']
pj = json.loads((ROOT / 'build/decode/decode_image_15m/params.json').read_text())
params = ', '.join(f'.{k}({pj[k]})' for k in KEYS)
params += ', .MAX_SEQ(128), .KV4(1), .ROPE_CS("rtl/decode/luts/rope_cs_q15_h48_p128.mem"), .ROPE_DEPTH(3072)'
out = f"""module top_decode_15m (
    input logic clk, rst, start, input logic [15:0] token, input logic [6:0] pos,
    output logic seek_req, output logic [31:0] seek_addr, input logic st_valid, input logic [7:0] st_data,
    output logic st_ready, busy, done, output logic [15:0] token_out, output logic logit_valid,
    output logic [15:0] logit_idx, output logic signed [31:0] logit_val);
  top_decode #({params}) u (.*);
endmodule
"""
(ROOT / 'build/decode/top_decode_15m_wrap.sv').write_text(out)
