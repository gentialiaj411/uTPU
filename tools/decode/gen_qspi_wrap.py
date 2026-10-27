"""Generate build/decode/top_decode_15m_qspi_wrap.sv: the 15M top_decode (parameters copied from
build/decode/top_decode_15m_wrap.sv) with qspi_reader on its flash port and QSPI pins as the only flash-side ports."""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
src = (ROOT / 'build/decode/top_decode_15m_wrap.sv').read_text()
params = re.search(r'top_decode (#\(.*\)) u \(', src, re.S).group(1)
out = f"""module top_decode_15m_qspi (
    input logic clk, rst, start, input logic [15:0] token, input logic [6:0] pos,
    output logic busy, done, output logic [15:0] token_out, output logic logit_valid,
    output logic [15:0] logit_idx, output logic signed [31:0] logit_val,
    output logic sclk, cs_n, output logic [3:0] io_o, io_oe, input logic [3:0] io_i);
  logic seek_req, st_valid, st_ready; logic [31:0] seek_addr; logic [7:0] st_data;
  top_decode {params} u (.*);
  qspi_reader qrd (.*);
endmodule
"""
(ROOT / 'build/decode/top_decode_15m_qspi_wrap.sv').write_text(out)
