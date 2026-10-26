module act_quant #(parameter MAX_GROUP=64) (
 input logic clk,input logic rst,input logic start,
 input logic in_valid,input logic in_last,input logic signed [31:0] raw_in,
 output logic in_ready, output logic code_valid, output logic signed [7:0] code_out,
 output logic scale_valid, output logic [31:0] scale_out, output logic done);
 typedef enum logic [3:0] {IDLE,CAP,PEAK,DIV,EMIT,FIN} st_t;
 st_t st; logic signed [31:0] mem [0:MAX_GROUP-1]; logic [6:0] count, peak_i, emit_i;
 logic [31:0] peak; logic [31:0] dividend, divisor, quot; logic [32:0] rem; logic [5:0] div_i;
 logic signed [63:0] prod; logic signed [63:0] rounded; logic [31:0] recip;
 assign in_ready=(st==CAP || st==IDLE);
 function automatic signed [63:0] rne24(input signed [63:0] x);
   reg neg; reg [63:0] m,q,r,half; begin
    neg=x<0; m=neg ? -x : x; q=m>>24; r=m & ((64'd1<<24)-1); half=64'd1<<23;
`ifdef ACTQ_BAD_TIE
    if (r>=half) q=q+1;
`else
    if ((r>half)||((r==half)&&(q[0]))) q=q+1;
`endif
    rne24=neg ? -$signed(q) : $signed(q);
   end
 endfunction
 function automatic [31:0] recip_calc(input [31:0] d);
   reg [32:0] rr; reg [31:0] qq; reg [31:0] num; integer j;
   begin rr=0; qq=0; num=(32'd127<<24); for(j=31;j>=0;j=j-1) begin rr={rr[31:0],num[j]}; if(rr>=d) begin rr=rr-d; qq[j]=1'b1; end end if((rr<<1)>=d) qq=qq+1'b1;
`ifdef ACTQ_BAD_PEAK
   if(qq>0) qq=qq-1'b1;
`endif
   recip_calc=qq; end
 endfunction
 always_ff @(posedge clk) begin
  if(rst) begin st<=IDLE; count<=0; peak_i<=0; emit_i<=0; code_valid<=0; scale_valid<=0; done<=0; end else begin
   code_valid<=0; scale_valid<=0; done<=0;
   case(st)
    IDLE: if(start) begin count<=0; st<=CAP; end
    CAP: if(in_valid) begin mem[count]<=raw_in; count<=count+1; if(in_last) begin peak<=1; peak_i<=0; st<=PEAK; end end
    PEAK: begin if (($signed(mem[peak_i])<0 ? -$signed(mem[peak_i]) : $signed(mem[peak_i])) > peak) peak <= ($signed(mem[peak_i])<0 ? -$signed(mem[peak_i]) : $signed(mem[peak_i])); if(peak_i+1>=count) begin divisor<= (($signed(mem[peak_i])<0 ? -$signed(mem[peak_i]) : $signed(mem[peak_i])) > peak) ? ($signed(mem[peak_i])<0 ? -$signed(mem[peak_i]) : $signed(mem[peak_i])) : peak; recip<=recip_calc((($signed(mem[peak_i])<0 ? -$signed(mem[peak_i]) : $signed(mem[peak_i])) > peak) ? ($signed(mem[peak_i])<0 ? -$signed(mem[peak_i]) : $signed(mem[peak_i])) : peak); emit_i<=0; st<=EMIT; end else peak_i<=peak_i+1; end
    DIV: begin
      rem <= {rem[31:0],dividend[31]}; dividend <= {dividend[30:0],1'b0};
      if ({rem[31:0],dividend[31]} >= divisor) begin rem <= {rem[31:0],dividend[31]}-divisor; quot <= {quot[30:0],1'b1}; end else quot <= {quot[30:0],1'b0};
      if(div_i==31) begin recip <= ({rem[31:0],dividend[31]} >= divisor) ? {quot[30:0],1'b1} : {quot[30:0],1'b0}; emit_i<=0; st<=EMIT; end else div_i<=div_i+1;
    end
    EMIT: begin
      prod <= $signed(mem[emit_i]) * $signed({1'b0,recip}); rounded <= rne24($signed(mem[emit_i]) * $signed({1'b0,recip}));
      code_out <= (rne24($signed(mem[emit_i]) * $signed({1'b0,recip})) > 127) ? 127 : (rne24($signed(mem[emit_i]) * $signed({1'b0,recip})) < -127 ? -127 : rne24($signed(mem[emit_i]) * $signed({1'b0,recip})));
`ifdef ACTQ_MUTATE
      if(emit_i==5) code_out <= (((rne24($signed(mem[emit_i]) * $signed({1'b0,recip})) > 127) ? 127 : (rne24($signed(mem[emit_i]) * $signed({1'b0,recip})) < -127 ? -127 : rne24($signed(mem[emit_i]) * $signed({1'b0,recip})))) ^ 8'sd1);
`endif
`ifdef ACTQ_BAD_TIE
      if(emit_i==0) code_out <= code_out + 1'sd1;
`endif
      code_valid<=1; if(emit_i+1>=count) begin scale_out<=peak; scale_valid<=1; st<=FIN; end else emit_i<=emit_i+1;
    end
    FIN: begin done<=1; st<=IDLE; end
   endcase
  end
 end
endmodule
