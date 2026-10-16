// One-buffer-port controller for the v1 VEC operations.  It intentionally
// serializes words: the surrounding top FSM owns the port while busy.
module vector_sequencer #(
    parameter ADDRESS_SIZE = 16,
    parameter BUFFER_WORDS = 32768
) (
    input logic clk, input logic rst, input logic start,
    input logic [3:0] subop, input logic [ADDRESS_SIZE-1:0] dst, src0, param,
    input logic [15:0] length, input logic [15:0] buf_rdata,
    output logic busy, done, error,
    output logic buf_we, buf_re, buf_store_en,
    output logic [ADDRESS_SIZE-1:0] buf_addr,
    output logic [15:0] buf_wdata
);
    localparam [3:0] ADD=0, NORM=1, POS=2, SOFT=3;
    localparam [5:0] IDLE=0, PR=1, PW0=2, PW1=3, CHECK=4,
                     GR=5, GW0=6, GW1=7, AR=8, AW0=9, AW1=10,
                     BR=11, BW0=12, BW1=13, WRITE=14, WW=15,
                     NAR=16, NAW0=17, NAW1=18, NWRITE=19, NWW=20,
                     SR=21, SW0=22, SW1=23, SWRITE=24, SWW=25,
                     FINISH=26, ALIGN_PIPE=27, ADD_PIPE=28, RNE_PIPE=29,
                     SAT_PIPE=30, NORM_SUM_LO=31, NORM_SUM_HI=32,
                     NORM_PREP=33, NORM_LUT_ADDR=34, NORM_LUT_READ=35,
                     NORM_NEWTON_MUL0=36, NORM_NEWTON_MUL1=37,
                     NORM_NEWTON_T=38, NORM_NEWTON_R=39,
                     NORM_NEWTON_RNE=40, NORM_SCALE_MUL0=41,
                     NORM_SCALE_MUL1=42, NORM_SCALE_RNE=43,
                     NORM_SCALE_SAT=44, SOFT_MAX_INIT=45, SOFT_MAX=46,
                     SOFT_EXP_ADDR=47, SOFT_EXP_READ=48, SOFT_EXP_SUM=49,
                     SOFT_DIV_START=50, SOFT_DIV_WAIT=51, SOFT_DIV_PACK=52,
                     NORM_LOG_INIT=53, NORM_LOG_SCAN=54;
    logic [5:0] state;
    logic [15:0] record [0:7];
    logic signed [7:0] a [0:47], b [0:47], gamma [0:47], score [0:31];
    logic [15:0] norm_word, soft_word;
    logic signed [7:0] ea, eb, eo;
    logic [5:0] gamma_shift, valid_count;
    // Add/position operations may span 768 packed words.  Keep the row-local
    // datapath index separate by taking index[5:0] at the serial ALU.
    logic [9:0] index;
    logic [5:0] row;
    logic [15:0] word_a, word_b;
    logic signed [8:0] aligned_a_lo, aligned_a_hi, aligned_b_lo, aligned_b_hi;
    logic signed [9:0] add_lo, add_hi, rounded_lo, rounded_hi;
    logic [15:0] add_word;
    logic [31:0] square_sum;
    logic [15:0] words_total;
    logic [15:0] square_word;
    logic [5:0] norm_element;
    logic [15:0] rsqrt_q14 [0:191];
    logic [15:0] exp_q15 [0:128];
    logic [31:0] norm_mean;
    logic [4:0] norm_exponent;
    logic [4:0] norm_log_probe;
    logic [16:0] norm_mantissa;
    logic [7:0] norm_bin;
    logic signed [31:0] norm_r0, norm_t, norm_r1;
    logic signed [47:0] newton_mul0, newton_mul1;
    logic signed [63:0] newton_product, scale_product;
    logic signed [15:0] scale_ag;
    logic signed [63:0] scale_rounded;
    logic [7:0] norm_low_byte;
    logic [5:0] soft_element;
    logic signed [8:0] soft_max, soft_rel;
    logic [7:0] soft_lut_index;
    logic [21:0] soft_sum;
    logic [15:0] soft_exp, soft_values [0:31];
    logic [7:0] soft_low_byte, soft_byte;
    logic divider_start, divider_busy, divider_done;
    logic [23:0] divider_numerator, divider_quotient;
    logic [21:0] divider_denominator;

    initial begin
`ifdef SYNTHESIS
        $readmemh("rsqrt_q14.mem", rsqrt_q14);
        $readmemh("exp_q15.mem", exp_q15);
`else
        $readmemh("rtl/vector/rsqrt_q14.mem", rsqrt_q14);
        $readmemh("rtl/vector/exp_q15.mem", exp_q15);
`endif
    end
    restoring_divider u_softmax_divider(
        .clk(clk), .rst(rst), .start(divider_start), .numerator(divider_numerator),
        .denominator(divider_denominator), .busy(divider_busy), .done(divider_done),
        .quotient(divider_quotient));

    function automatic signed [63:0] rne_shift(input signed [63:0] v, input integer sh);
        reg signed [63:0] mag,q,rem,half;
        begin
            if (sh <= 0) rne_shift=v;
            else begin mag=v<0?-v:v; q=mag>>>sh; rem=mag-(q<<<sh); half=64'sd1<<<(sh-1);
                if ((rem>half)||((rem==half)&&q[0])) q=q+1;
                rne_shift=v<0?-q:q; end
        end
    endfunction
    function automatic signed [7:0] sat8(input signed [63:0] v);
        if(v>127) sat8=127; else if(v < -128) sat8=-128; else sat8=v[7:0];
    endfunction
    function automatic [31:0] rne_div48(input [31:0] v);
        reg [31:0] q, rem;
        begin
            q=v/48; rem=v%48;
            if((2*rem>48)||((2*rem==48)&&q[0])) q=q+1;
            rne_div48=q;
        end
    endfunction
    function automatic [31:0] even_power_of_two(input [4:0] exponent);
        begin
            case (exponent)
                30: even_power_of_two=32'h40000000;
                28: even_power_of_two=32'h10000000;
                26: even_power_of_two=32'h04000000;
                24: even_power_of_two=32'h01000000;
                22: even_power_of_two=32'h00400000;
                20: even_power_of_two=32'h00100000;
                18: even_power_of_two=32'h00040000;
                16: even_power_of_two=32'h00010000;
                14: even_power_of_two=32'h00004000;
                12: even_power_of_two=32'h00001000;
                10: even_power_of_two=32'h00000400;
                 8: even_power_of_two=32'h00000100;
                 6: even_power_of_two=32'h00000040;
                 4: even_power_of_two=32'h00000010;
                 2: even_power_of_two=32'h00000004;
                default: even_power_of_two=32'h00000001;
            endcase
        end
    endfunction
    function automatic range_ok(input [ADDRESS_SIZE-1:0] base, input [15:0] words);
        reg [ADDRESS_SIZE:0] finish;
        begin finish={1'b0,base}+words; range_ok=(words!=0)&&(finish<=BUFFER_WORDS); end
    endfunction
    function automatic overlap(input [ADDRESS_SIZE-1:0] abase, input [15:0] awords,
                               input [ADDRESS_SIZE-1:0] bbase, input [15:0] bwords);
        reg [ADDRESS_SIZE:0] aend,bend;
        begin aend={1'b0,abase}+awords; bend={1'b0,bbase}+bwords;
              overlap=({1'b0,abase}<bend)&&({1'b0,bbase}<aend); end
    endfunction
    function automatic exp_ok(input signed [7:0] v);
        begin exp_ok=(v>=-31)&&(v<=31); end
    endfunction

    always_comb begin
        buf_we=0; buf_re=0; buf_store_en=0; buf_addr=0; buf_wdata=0;
        divider_start=0; divider_numerator=0; divider_denominator=soft_sum;
        case(state)
            PR: begin buf_re=1; buf_store_en=1; buf_addr=param+index; end
            GR: begin buf_re=1; buf_store_en=1; buf_addr=record[1]+index; end
            AR, NAR: begin buf_re=1; buf_store_en=1; buf_addr=src0 + ((subop==NORM) ? row*24+index : index); end
            BR: begin buf_re=1; buf_store_en=1; buf_addr=(subop==POS ? record[1] : record[1])+index; end
            SR: begin buf_re=1; buf_store_en=1; buf_addr=src0+row*16+index; end
            WRITE: begin buf_we=1; buf_store_en=1; buf_addr=dst+index; buf_wdata=add_word; end
            NWRITE: begin buf_we=1; buf_store_en=1; buf_addr=dst+row*24+(norm_element>>1); buf_wdata=norm_word; end
            SWRITE: begin buf_we=1; buf_store_en=1; buf_addr=dst+row*16+(soft_element>>1); buf_wdata=soft_word; end
            SOFT_DIV_START: if(soft_element<valid_count) begin
                divider_start=1;
                divider_numerator=({soft_values[soft_element],8'b0}-soft_values[soft_element])+(soft_sum>>1);
            end
            default: ;
        endcase
    end

    always_ff @(posedge clk) begin
        if(rst) begin state<=IDLE; busy<=0; done<=0; error<=0; index<=0; row<=0; end
        else begin
            done<=0;
            case(state)
                IDLE: if(start) begin busy<=1; error<=0; index<=0; row<=0; square_sum<=0; words_total<=length>>1; state<=PR; end
                PR: state<=PW0; PW0: state<=PW1;
                PW1: begin record[index]<=buf_rdata; if(index==7) begin index<=0; state<=CHECK; end else begin index<=index+1; state<=PR; end end
                CHECK: begin
                    ea<=record[2][7:0]; eb<=record[2][15:8]; eo<=record[3][7:0]; gamma_shift<=record[2][5:0];
                    // This is deliberately the runtime validation equivalent
                    // of isa_simulator._execute_vec: no malformed record may
                    // begin a buffer write.
                    if (!range_ok(dst,words_total) || !range_ok(src0,words_total) ||
                        !range_ok(param,16'd8) ||
                        (overlap(dst,words_total,src0,words_total) && dst!=src0) ||
                        overlap(dst,words_total,param,16'd8) || overlap(src0,words_total,param,16'd8) ||
                        (subop==NORM && (record[0]!=16'd2 || (record[2]&16'hff00)!=0 ||
                          record[3]!=0 || record[4]!=0 || record[5]!=0 || record[6]!=0 || record[7]!=0 ||
                          record[2][7:0]>31 || !range_ok(record[1],16'd24) ||
                          overlap(dst,words_total,record[1],16'd24) || overlap(src0,words_total,record[1],16'd24))) ||
                        (subop==ADD && (record[0]!=16'd1 || (record[3]&16'hff00)!=0 ||
                          record[4]!=0 || record[5]!=0 || record[6]!=0 || record[7]!=0 ||
                          !range_ok(record[1],words_total) ||
                          (overlap(dst,words_total,record[1],words_total) && dst!=record[1]) ||
                          !exp_ok(record[2][7:0]) || !exp_ok(record[2][15:8]) || !exp_ok(record[3][7:0]) ||
                          $signed(record[3][7:0]) < $signed(record[2][7:0]) || $signed(record[3][7:0]) < $signed(record[2][15:8]))) ||
                        (subop==POS && (record[0]!=16'd3 || (record[3]&16'hff00)!=0 ||
                          record[4]!=0 || record[5]!=0 || record[6]!=0 || record[7]!=0 ||
                          !range_ok(record[1],16'd768) || overlap(dst,words_total,record[1],16'd768) ||
                          overlap(src0,words_total,record[1],16'd768) || !exp_ok(record[2][7:0]) ||
                          !exp_ok(record[2][15:8]) || !exp_ok(record[3][7:0]) ||
                          $signed(record[3][7:0]) < $signed(record[2][7:0]) || $signed(record[3][7:0]) < $signed(record[2][15:8]))) ||
                        (subop==SOFT && (record[0]!=16'd4 || record[2]!=0 || record[3]!=0 || record[4]!=0 ||
                          record[5]!=0 || record[6]!=0 || record[7]!=0 || !range_ok(record[1],16'd512) ||
                          overlap(dst,words_total,record[1],16'd512) || overlap(src0,words_total,record[1],16'd512)))) begin
                        error<=1; state<=FINISH;
                    end else if(subop==NORM) state<=GR; else if(subop==SOFT) state<=SR; else state<=AR;
                end
                GR: state<=GW0; GW0: state<=GW1;
                GW1: begin gamma[index*2]<=buf_rdata[7:0]; gamma[index*2+1]<=buf_rdata[15:8]; if(index==23) begin index<=0; state<=NAR; end else begin index<=index+1; state<=GR; end end
                AR: state<=AW0; AW0: state<=AW1;
                AW1: begin word_a<=buf_rdata; if(subop==NORM) state<=NORM_SUM_LO; else state<=BR; end
                NAR: state<=NAW0; NAW0: state<=NAW1;
                NAW1: begin
                    a[index*2]<=buf_rdata[7:0]; a[index*2+1]<=buf_rdata[15:8];
                    square_word<=buf_rdata;
                    state<=NORM_SUM_LO;
                end
                // One registered multiply-accumulate per element.  The
                // packed buffer word is split across two successive states.
                NORM_SUM_LO: begin
                    square_sum<=square_sum+
                        $signed({{24{square_word[7]}},square_word[7:0]})*
                        $signed({{24{square_word[7]}},square_word[7:0]});
                    state<=NORM_SUM_HI;
                end
                NORM_SUM_HI: begin
                    square_sum<=square_sum+
                        $signed({{24{square_word[15]}},square_word[15:8]})*
                        $signed({{24{square_word[15]}},square_word[15:8]});
                    if(index==23) begin index<=0; state<=NORM_PREP; end
                    else begin index<=index+1; state<=NAR; end
                end
                NORM_PREP: begin norm_mean<=rne_div48(square_sum)+1; state<=NORM_LOG_INIT; end
                NORM_LOG_INIT: begin norm_log_probe<=30; state<=NORM_LOG_SCAN; end
                // One descending threshold compare per cycle selects
                // floor(log2(v)) rounded down to an even exponent.
                NORM_LOG_SCAN: begin
                    if(norm_mean>=even_power_of_two(norm_log_probe)) begin
                        norm_exponent<=norm_log_probe;
                        state<=NORM_LUT_ADDR;
                    end else if(norm_log_probe==0) begin
                        norm_exponent<=0;
                        state<=NORM_LUT_ADDR;
                    end else norm_log_probe<=norm_log_probe-2;
                end
                NORM_LUT_ADDR: begin
                    if(rne_shift($signed({1'b0,norm_mean})<<<14,norm_exponent)<16384)
                        norm_mantissa<=16384;
                    else if(rne_shift($signed({1'b0,norm_mean})<<<14,norm_exponent)>65535)
                        norm_mantissa<=65535;
                    else norm_mantissa<=rne_shift($signed({1'b0,norm_mean})<<<14,norm_exponent);
                    state<=NORM_LUT_READ;
                end
                NORM_LUT_READ: begin
                    norm_bin<=(norm_mantissa-16384)>>8;
                    state<=NORM_NEWTON_MUL0;
                end
                NORM_NEWTON_MUL0: begin
                    norm_r0<=rsqrt_q14[norm_bin];
                    newton_mul0<=$signed({1'b0,norm_mantissa})*$signed(rsqrt_q14[norm_bin]);
                    state<=NORM_NEWTON_MUL1;
                end
                NORM_NEWTON_MUL1: begin
                    newton_mul1<=newton_mul0*norm_r0;
                    state<=NORM_NEWTON_T;
                end
                NORM_NEWTON_T: begin norm_t<=rne_shift(newton_mul1,28); state<=NORM_NEWTON_R; end
                NORM_NEWTON_R: begin
                    newton_product<=norm_r0*(49152-norm_t);
                    state<=NORM_NEWTON_RNE;
                end
                NORM_NEWTON_RNE: begin
                    norm_r1<=rne_shift(newton_product,15);
                    norm_element<=0;
                    state<=NORM_SCALE_MUL0;
                end
                NORM_SCALE_MUL0: begin
                    scale_ag<=$signed(a[norm_element])*$signed(gamma[norm_element]);
                    state<=NORM_SCALE_MUL1;
                end
                NORM_SCALE_MUL1: begin scale_product<=scale_ag*norm_r1; state<=NORM_SCALE_RNE; end
                NORM_SCALE_RNE: begin
                    scale_rounded<=rne_shift(scale_product,14+gamma_shift+(norm_exponent>>1));
                    state<=NORM_SCALE_SAT;
                end
                NORM_SCALE_SAT: begin
                    if(!norm_element[0]) begin
                        norm_low_byte<=sat8(scale_rounded);
                        norm_element<=norm_element+1;
                        state<=NORM_SCALE_MUL0;
                    end else begin
                        norm_word<={sat8(scale_rounded),norm_low_byte};
                        state<=NWRITE;
                    end
                end
                BR: state<=BW0; BW0: state<=BW1;
                BW1: begin word_b<=buf_rdata; state<=ALIGN_PIPE; end
                // Align, add, and saturation are separate registers so the
                // VADD/VPOS write path has no element-wide combinational cone.
                ALIGN_PIPE: begin
                    aligned_a_lo<=rne_shift($signed(word_a[7:0]),eo-ea);
                    aligned_a_hi<=rne_shift($signed(word_a[15:8]),eo-ea);
                    aligned_b_lo<=rne_shift($signed(word_b[7:0]),eo-eb);
                    aligned_b_hi<=rne_shift($signed(word_b[15:8]),eo-eb);
                    state<=ADD_PIPE;
                end
                ADD_PIPE: begin
                    add_lo<=aligned_a_lo+aligned_b_lo;
                    add_hi<=aligned_a_hi+aligned_b_hi;
                    state<=RNE_PIPE;
                end
                // Alignment already implements the contract's RNE shifts;
                // retain a registered rounding boundary before saturation.
                RNE_PIPE: begin rounded_lo<=add_lo; rounded_hi<=add_hi; state<=SAT_PIPE; end
                SAT_PIPE: begin add_word<={sat8(rounded_hi),sat8(rounded_lo)}; state<=WRITE; end
                WRITE: state<=WW;
                WW: if(index+1>=words_total) begin state<=FINISH; end else begin index<=index+1; state<=AR; end
                NWRITE: state<=NWW;
                NWW: if(norm_element==47) begin
                    if(row+1>=words_total/24) state<=FINISH;
                    else begin row<=row+1; index<=0; square_sum<=0; state<=NAR; end
                end else begin norm_element<=norm_element+1; state<=NORM_SCALE_MUL0; end
                SR: state<=SW0; SW0: state<=SW1;
                SW1: begin
                    score[index*2]<=buf_rdata[7:0];
                    score[index*2+1]<=buf_rdata[15:8];
                    if(index==15) begin
                        valid_count<=row+1;
                        index<=0;
                        state<=SOFT_MAX_INIT;
                    end else begin
                        index<=index+1;
                        state<=SR;
                    end
                end
                // Causal rows are reduced serially.  ``valid_count`` is the
                // loop bound, so masked elements never reach either max or
                // the EXP/SUM pass.
                SOFT_MAX_INIT: begin
                    soft_element<=0;
                    soft_max<=-128;
                    state<=SOFT_MAX;
                end
                SOFT_MAX: begin
                    if ($signed(score[soft_element]) > soft_max)
                        soft_max<=$signed(score[soft_element]);
                    if(soft_element+1>=valid_count) begin
                        soft_element<=0;
                        soft_sum<=0;
                        state<=SOFT_EXP_ADDR;
                    end else soft_element<=soft_element+1;
                end
                // The subtract/clamp and lookup-address register keep the
                // LUT path out of the accumulator cycle.
                SOFT_EXP_ADDR: begin
                    soft_rel<=$signed(score[soft_element])-soft_max;
                    if (($signed(score[soft_element])-soft_max) < -128)
                        soft_lut_index<=8'd128;
                    else
                        soft_lut_index<=soft_max-$signed(score[soft_element]);
                    state<=SOFT_EXP_READ;
                end
                SOFT_EXP_READ: begin
                    soft_exp<=exp_q15[soft_lut_index];
                    state<=SOFT_EXP_SUM;
                end
                SOFT_EXP_SUM: begin
                    soft_values[soft_element]<=soft_exp;
                    soft_sum<=soft_sum+soft_exp;
                    if(soft_element+1>=valid_count) begin
                        soft_element<=0;
                        state<=SOFT_DIV_START;
                    end else begin
                        soft_element<=soft_element+1;
                        state<=SOFT_EXP_ADDR;
                    end
                end
                // A quotient is retired only after the iterative divider
                // reports done.  Masked positions bypass it with a zero.
                SOFT_DIV_START: begin
                    if(soft_element<valid_count) state<=SOFT_DIV_WAIT;
                    else begin soft_byte<=0; state<=SOFT_DIV_PACK; end
                end
                SOFT_DIV_WAIT: if(divider_done) begin
                    soft_byte<=divider_quotient[7:0];
                    state<=SOFT_DIV_PACK;
                end
                SOFT_DIV_PACK: begin
                    if(!soft_element[0]) begin
                        soft_low_byte<=soft_byte;
                        soft_element<=soft_element+1;
                        state<=SOFT_DIV_START;
                    end else begin
                        soft_word<={soft_byte,soft_low_byte};
                        state<=SWRITE;
                    end
                end
                SWRITE: state<=SWW;
                SWW: begin
                    if(soft_element==31) begin
                        if(row==31) state<=FINISH;
                        else begin row<=row+1; index<=0; state<=SR; end
                    end else begin
                        soft_element<=soft_element+1;
                        state<=SOFT_DIV_START;
                    end
                end
                FINISH: begin busy<=0; done<=1; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
