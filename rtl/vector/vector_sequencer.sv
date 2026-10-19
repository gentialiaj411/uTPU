// One-buffer-port controller for the v1 VEC operations.  It intentionally
// serializes words: the surrounding top FSM owns the port while busy.
module vector_sequencer #(
    parameter ADDRESS_SIZE = 16,
    parameter BUFFER_WORDS = 32768
) (
    input logic clk, input logic rst, input logic start,
    input logic [3:0] subop, input logic causal, input logic [ADDRESS_SIZE-1:0] dst, src0, param,
    input logic [15:0] length, input logic [15:0] buf_rdata,
    output logic busy, done, error,
    output logic buf_we, buf_re, buf_store_en,
    output logic [ADDRESS_SIZE-1:0] buf_addr,
    output logic [15:0] buf_wdata
);
    localparam [3:0] ADD=0, NORM=1, POS=2, SOFT=3, MAT=4, LAYOUT=5;
    localparam [6:0] IDLE=0, PR=1, PW0=2, PW1=3, CHECK=4,
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
                     NORM_SCALE_MUL1=42, NORM_SCALE_RNE_INPUT=43,
                     NORM_SCALE_RNE=44, NORM_SCALE_SAT=45, SOFT_MAX_INIT=46,
                     SOFT_MAX=47, SOFT_EXP_ADDR=48, SOFT_EXP_READ=49,
                     SOFT_EXP_SUM=50, SOFT_DIV_START=51, SOFT_DIV_WAIT=52,
                     SOFT_DIV_PACK=53, NORM_LOG_INIT=54, NORM_LOG_SCAN=55,
                     MAT_A=56, MAT_AW0=57, MAT_AW1=58, MAT_B=59, MAT_BW0=60,
                     MAT_BW1=61, MAT_MUL=62, MAT_ACC=63, MAT_CLIP=64,
                     MAT_SCALE_MUL=65, MAT_SHIFT=66, MAT_SAT=67, MAT_OUT_READ=68,
                     MAT_OUT_WAIT0=69, MAT_OUT_WAIT1=70, MAT_WRITE=71,
                     LAY_ZERO=72, LAY_READ=73, LAY_READ_WAIT0=74,
                     LAY_READ_WAIT1=75, LAY_READ_WAIT2=76, LAY_OUT_READ=77,
                     LAY_OUT_WAIT0=78, LAY_OUT_WAIT1=79, LAY_OUT_WAIT2=80,
                     LAY_WRITE=81, MAT_WRITE_COMMIT=82,
                     PR_WAIT2=83, GR_WAIT2=84, AR_WAIT2=85, BR_WAIT2=86,
                     NAR_WAIT2=87, SR_WAIT2=88, MAT_A_WAIT2=89,
                     MAT_B_WAIT2=90, MAT_OUT_WAIT2=91,
                     NORM_MEAN_ADD=92, NORM_LOG_DECIDE=93,
                     NORM_MANT_RNE=94, NORM_MANT_SAT=95,
                     NORM_NEWTON_T_RNE=96, NORM_NEWTON_RNE_INPUT=97;
    logic [6:0] state;
    logic buf_we_c, buf_re_c, buf_store_en_c;
    logic [ADDRESS_SIZE-1:0] buf_addr_c;
    logic [15:0] buf_wdata_c;
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
    logic [31:0] norm_mean_div;
    logic [15:0] words_total;
    logic [15:0] square_word;
    logic [5:0] norm_element;
    logic [15:0] rsqrt_q14 [0:191];
    logic [15:0] exp_q15 [0:128];
    logic [31:0] norm_mean;
    logic [4:0] norm_exponent;
    logic [4:0] norm_log_probe;
    logic [31:0] norm_log_threshold;
    logic norm_log_ge;
    logic [16:0] norm_mantissa;
    logic [63:0] norm_mantissa_shift_c, norm_mantissa_input_c;
    logic [63:0] norm_mantissa_shift_q, norm_mantissa_rounded;
    logic norm_mantissa_guard_q, norm_mantissa_sticky_q, norm_mantissa_lsb_q;
    logic [63:0] norm_mantissa_guard_input_c;
    logic [7:0] norm_bin;
    logic signed [31:0] norm_r0, norm_t, norm_r1;
    logic signed [47:0] newton_mul0, newton_mul1;
    logic signed [47:0] newton_t_rounded;
    logic signed [47:0] newton_t_shifted_q;
    logic signed [63:0] newton_r_shifted_q;
    logic newton_t_neg_q, newton_t_guard_q, newton_t_sticky_q, newton_t_lsb_q;
    logic newton_r_neg_q, newton_r_guard_q, newton_r_sticky_q, newton_r_lsb_q;
    logic [47:0] newton_t_abs_c, newton_t_shift_c;
    logic [63:0] newton_r_abs_c, newton_r_shift_c;
    logic signed [63:0] newton_product;
    // RMSNorm scale bound: |int8*int8|*r1_q14 <= 16129*16384 < 2^29.
    // Signed 32 bits therefore cover every RMSNorm suite extreme; the
    // maximum RNE shift is 14+31+(30/2)=60.
    logic signed [31:0] scale_product, scale_round_input;
    logic signed [15:0] scale_ag;
    logic signed [31:0] scale_rounded;
    logic [5:0] scale_shift_amt;
    logic [31:0] scale_abs_c, scale_shifted_c;
    logic scale_neg_c, scale_guard_c, scale_sticky_c, scale_lsb_c;
    logic scale_neg_q, scale_guard_q, scale_sticky_q, scale_lsb_q;
    logic [31:0] scale_shifted_q;
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
    logic [5:0] mat_m, mat_kdim, mat_n, mat_i, mat_j, mat_k;
    logic [7:0] mat_a_stride, mat_b_stride;
    logic mat_transpose, mat_a_unsigned, mat_b_unsigned, mat_causal;
    logic signed [8:0] mat_a_value, mat_b_value;
    logic signed [31:0] mat_acc;
    logic signed [31:0] mat_product;
    logic signed [31:0] mat_clipped;
    logic signed [63:0] mat_scaled, mat_shifted;
    logic signed [7:0] mat_result;
    logic [15:0] mat_out_word;
    logic [ADDRESS_SIZE-1:0] mat_write_addr_q;
    logic [15:0] mat_write_data_q;
    logic [5:0] lay_rows;
    logic [8:0] lay_cols, lay_r, lay_c;
    logic [7:0] lay_src_stride, lay_dst_stride;
    logic [2:0] lay_mode;
    logic [31:0] lay_zero_word, lay_dest_words;
    logic signed [7:0] lay_value;
    logic [15:0] lay_out_word;

    function automatic [31:0] lay_lin_index(input [5:0] r, input [8:0] c, input [8:0] cols);
        begin lay_lin_index=((r/4)*cols+c)*4+(r%4); end
    endfunction
    function automatic [31:0] lay_lfin_index(input [5:0] r, input [8:0] c, input [5:0] rows);
        begin lay_lfin_index=((c/8)*rows+r)*8+(c%8); end
    endfunction
    function automatic [31:0] lay_wt_index(input [5:0] r, input [8:0] c, input [8:0] cols);
        begin lay_wt_index=((r/8)*cols+c)*8+(r%8); end
    endfunction
    // Physical stream consumed by legacy LOAD for B=4: eight K lanes for
    // batch lane 0, then lane 1, etc., per eight-column K tile.
    function automatic [31:0] lay_load_index(input [5:0] r, input [8:0] c, input [8:0] cols);
        begin lay_load_index=(r/4)*cols*4+(c/8)*32+(r%4)*8+(c%8); end
    endfunction
    // Fixed v1 attention merge: VMATMUL writes [head][row][Dh=16], while
    // legacy LOAD consumes the merged [row][H*Dh=48] physical stream.
    function automatic [31:0] lay_head_index(input [5:0] r, input [8:0] c, input [5:0] rows);
        begin lay_head_index=(c/16)*rows*16+r*16+(c%16); end
    endfunction
    function automatic [31:0] lay_src_index;
        begin
            case(lay_mode)
                0,2,6: lay_src_index=lay_r*lay_src_stride+lay_c;
                1: lay_src_index=lay_lfin_index(lay_r,lay_c,lay_rows);
                3: lay_src_index=lay_lin_index(lay_r,lay_c,lay_cols);
                4: lay_src_index=lay_head_index(lay_r,lay_c,lay_rows);
                5: lay_src_index=lay_lfin_index(lay_r,lay_c,lay_rows);
                default: lay_src_index=lay_load_index(lay_r,lay_c,lay_cols);
            endcase
        end
    endfunction
    function automatic [31:0] lay_dst_index;
        begin
            case(lay_mode)
                0: lay_dst_index=lay_lin_index(lay_r,lay_c,lay_cols);
                1,3: lay_dst_index=lay_r*lay_dst_stride+lay_c;
                2: lay_dst_index=lay_wt_index(lay_r,lay_c,lay_cols);
                4,5: lay_dst_index=lay_load_index(lay_r,lay_c,lay_cols);
                6: lay_dst_index=lay_load_index(lay_r,lay_c,lay_cols);
                default: lay_dst_index=lay_r*lay_dst_stride+lay_c;
            endcase
        end
    endfunction
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
        buf_we_c=0; buf_re_c=0; buf_store_en_c=0; buf_addr_c=0; buf_wdata_c=0;
        divider_start=0; divider_numerator=0; divider_denominator=soft_sum;
        case(state)
            PR: begin buf_re_c=1; buf_store_en_c=1; buf_addr_c=param+index; end
            GR: begin buf_re_c=1; buf_store_en_c=1; buf_addr_c=record[1]+index; end
            AR, NAR: begin buf_re_c=1; buf_store_en_c=1; buf_addr_c=src0 + ((subop==NORM) ? row*24+index : index); end
            BR: begin buf_re_c=1; buf_store_en_c=1; buf_addr_c=(subop==POS ? record[1] : record[1])+index; end
            SR: begin buf_re_c=1; buf_store_en_c=1; buf_addr_c=src0+row*16+index; end
            WRITE: begin buf_we_c=1; buf_store_en_c=1; buf_addr_c=dst+index; buf_wdata_c=add_word; end
            NWRITE: begin buf_we_c=1; buf_store_en_c=1; buf_addr_c=dst+row*24+(norm_element>>1); buf_wdata_c=norm_word; end
            SWRITE: begin buf_we_c=1; buf_store_en_c=1; buf_addr_c=dst+row*16+(soft_element>>1); buf_wdata_c=soft_word; end
            MAT_A: begin buf_re_c=1; buf_store_en_c=1;
                buf_addr_c=src0+((mat_i*mat_a_stride+mat_k)>>1); end
            MAT_B: begin buf_re_c=1; buf_store_en_c=1;
                if(mat_transpose) buf_addr_c=record[1]+((mat_j*mat_b_stride+mat_k)>>1);
                else buf_addr_c=record[1]+((mat_k*mat_b_stride+mat_j)>>1);
            end
            MAT_OUT_READ: begin buf_re_c=1; buf_store_en_c=1;
                buf_addr_c=dst+((mat_i*mat_n+mat_j)>>1); end
            MAT_WRITE: begin end
            MAT_WRITE_COMMIT: begin buf_we_c=1; buf_store_en_c=1;
                buf_addr_c=mat_write_addr_q; buf_wdata_c=mat_write_data_q; end
            LAY_ZERO: begin buf_we_c=1; buf_store_en_c=1; buf_addr_c=dst+lay_zero_word; buf_wdata_c=0; end
            LAY_READ: begin buf_re_c=1; buf_store_en_c=1; buf_addr_c=src0+(lay_src_index()>>1); end
            LAY_OUT_READ: begin buf_re_c=1; buf_store_en_c=1; buf_addr_c=dst+(lay_dst_index()>>1); end
            LAY_WRITE: begin buf_we_c=1; buf_store_en_c=1; buf_addr_c=dst+(lay_dst_index()>>1); buf_wdata_c=lay_out_word; end
            SOFT_DIV_START: if(soft_element<valid_count) begin
                divider_start=1;
                divider_numerator=({soft_values[soft_element],8'b0}-soft_values[soft_element])+(soft_sum>>1);
            end
            default: ;
        endcase
    end

    // First half of the staged RMSNorm RNE: barrel shift and capture the
    // exact guard/sticky/LSB bits needed for round-to-nearest-even.
    always_comb begin
        scale_shift_amt = 6'(14 + gamma_shift + (norm_exponent >> 1));
        scale_neg_c = scale_product[31];
        scale_abs_c = scale_neg_c ? -scale_product : scale_product;
        scale_shifted_c = scale_shift_amt >= 32 ? 32'd0 : (scale_abs_c >> scale_shift_amt);
        scale_guard_c = 1'b0;
        scale_sticky_c = 1'b0;
        scale_lsb_c = (scale_shift_amt < 32) ? scale_shifted_c[0] : 1'b0;
        if (scale_shift_amt != 0 && scale_shift_amt <= 32) begin
            scale_guard_c = scale_abs_c[scale_shift_amt-1];
            if (scale_shift_amt > 1)
                scale_sticky_c = |(scale_abs_c & ((32'h1 << (scale_shift_amt-1))-1));
        end else if (scale_shift_amt > 32) begin
            scale_sticky_c = |scale_abs_c;
        end
    end

    // RMSNorm statistics are deliberately staged: one barrel shift captures
    // quotient/guard/sticky/LSB, the next state performs the RNE add, and the
    // third state clamps the mantissa.  The shift amount is formed once here.
    always_comb begin
        norm_mantissa_input_c = ({32'd0,norm_mean} << 14);
        norm_mantissa_shift_c = norm_mantissa_input_c >> norm_exponent;
        norm_mantissa_guard_input_c = 64'd0;
        if (norm_exponent != 0)
            norm_mantissa_guard_input_c = norm_mantissa_input_c >> (norm_exponent-1);

        if (newton_mul1[47]) newton_t_abs_c = -newton_mul1;
        else newton_t_abs_c = newton_mul1;
        newton_t_shift_c = newton_t_abs_c >> 28;
        if (newton_product[63]) newton_r_abs_c = -newton_product;
        else newton_r_abs_c = newton_product;
        newton_r_shift_c = newton_r_abs_c >> 15;
    end

    // Single register stage on the complete vector buffer interface.
    always_ff @(posedge clk) begin
        if (rst) begin
            buf_we <= 1'b0; buf_re <= 1'b0; buf_store_en <= 1'b0;
            buf_addr <= '0; buf_wdata <= '0;
        end else begin
            buf_we <= buf_we_c; buf_re <= buf_re_c; buf_store_en <= buf_store_en_c;
            buf_addr <= buf_addr_c; buf_wdata <= buf_wdata_c;
        end
    end

    always_ff @(posedge clk) begin
        if(rst) begin state<=IDLE; busy<=0; done<=0; error<=0; index<=0; row<=0; end
        else begin
            done<=0;
            case(state)
                IDLE: if(start) begin busy<=1; error<=0; index<=0; row<=0; square_sum<=0; words_total<=length>>1; state<=PR; end
                PR: state<=PW0; PW0: state<=PW1; PW1: state<=PR_WAIT2;
                PR_WAIT2: begin record[index]<=buf_rdata; if(index==7) begin index<=0; state<=CHECK; end else begin index<=index+1; state<=PR; end end
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
                          overlap(dst,words_total,record[1],16'd512) || overlap(src0,words_total,record[1],16'd512))) ||
                        (subop==MAT && (record[0]!=16'd5 || record[7]!=0 || (record[4]&16'hfff8)!=0 ||
                          (record[6]&16'hffe0)!=0 || record[3][7:0] < (((record[2]>>5)&5'h1f)+1) ||
                          record[3][15:8] < ((record[2][15] ? ((record[2]>>5)&5'h1f) : ((record[2]>>10)&5'h1f))+1) ||
                          record[3][0] || record[3][8] || (causal != record[4][2]) ||
                          (causal && !((record[2][15] && (record[2][4:0]==record[2][14:10])) ||
                                       (record[4][0] && !record[2][15] && (record[2][4:0]==record[2][9:5])))))) ||
                        (subop==LAYOUT && (record[0]!=16'd6 || (record[1]&16'he000)!=0 ||
                          (record[2]&16'hfff8)!=0 ||
                          ((record[2][2:0]>=3'd4) && ((record[1][4:0]+1)%4!=0 || (((record[1]>>5)&8'hff)+1)%8!=0)) ||
                          (record[2][2:0]==3'd4 && ((record[1][4:0]+1)!=32 || (((record[1]>>5)&8'hff)+1)!=48)) ||
                          record[3][7:0] < (((record[1]>>5)&8'hff)+1) || record[3][15:8] < (((record[1]>>5)&8'hff)+1) ||
                          record[3][0] || record[3][8] || record[4]!=0 || record[5]!=0 || record[6]!=0 || record[7]!=0))) begin
                        error<=1; state<=FINISH;
                    end else if(subop==NORM) state<=GR;
                    else if(subop==SOFT) state<=SR;
                    else if(subop==MAT) begin
                        mat_m<=record[2][4:0]+1; mat_kdim<=record[2][9:5]+1; mat_n<=record[2][14:10]+1;
                        mat_a_stride<=record[3][7:0]; mat_b_stride<=record[3][15:8]; mat_transpose<=record[2][15];
                        mat_a_unsigned<=record[4][0]; mat_b_unsigned<=record[4][1]; mat_causal<=record[4][2];
                        mat_i<=0; mat_j<=0; mat_k<=0; mat_acc<=0; state<=MAT_A;
                    end else if(subop==LAYOUT) begin
                        lay_rows<=record[1][4:0]+1; lay_cols<=record[1][12:5]+1; lay_mode<=record[2][2:0];
                        lay_src_stride<=record[3][7:0]; lay_dst_stride<=record[3][15:8]; lay_r<=0; lay_c<=0; lay_zero_word<=0;
                        if(record[2][2:0]==0) lay_dest_words<=((((record[1][4:0]+1+3)/4)*((record[1][12:5]+1))*4+1)>>1);
                        else if(record[2][2:0]==1 || record[2][2:0]==3 || record[2][2:0]==7) lay_dest_words<=(((record[1][4:0]+1)*record[3][15:8]+1)>>1);
                        else lay_dest_words<=((((record[1][4:0]+1+7)/8)*(record[1][12:5]+1)*8+1)>>1);
                        state<=LAY_ZERO;
                    end else state<=AR;
                end
                GR: state<=GW0; GW0: state<=GW1; GW1: state<=GR_WAIT2;
                GR_WAIT2: begin gamma[index*2]<=buf_rdata[7:0]; gamma[index*2+1]<=buf_rdata[15:8]; if(index==23) begin index<=0; state<=NAR; end else begin index<=index+1; state<=GR; end end
                AR: state<=AW0; AW0: state<=AW1; AW1: state<=AR_WAIT2;
                AR_WAIT2: begin word_a<=buf_rdata; if(subop==NORM) state<=NORM_SUM_LO; else state<=BR; end
                NAR: state<=NAW0; NAW0: state<=NAW1; NAW1: state<=NAR_WAIT2;
                NAR_WAIT2: begin
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
                NORM_PREP: begin norm_mean_div<=rne_div48(square_sum); state<=NORM_MEAN_ADD; end
                NORM_MEAN_ADD: begin norm_mean<=norm_mean_div+1; state<=NORM_LOG_INIT; end
                NORM_LOG_INIT: begin norm_log_probe<=30; state<=NORM_LOG_SCAN; end
                // One descending threshold compare per cycle selects
                // floor(log2(v)) rounded down to an even exponent.
                NORM_LOG_SCAN: begin
                    norm_log_threshold<=even_power_of_two(norm_log_probe);
                    state<=NORM_LOG_DECIDE;
                end
                NORM_LOG_DECIDE: begin
                    if(norm_mean>=norm_log_threshold) begin
                        norm_exponent<=norm_log_probe;
                        state<=NORM_LUT_ADDR;
                    end else if(norm_log_probe==0) begin
                        norm_exponent<=0;
                        state<=NORM_LUT_ADDR;
                    end else begin norm_log_probe<=norm_log_probe-2; state<=NORM_LOG_SCAN; end
                end
                NORM_LUT_ADDR: begin
                    norm_mantissa_shift_q<=norm_mantissa_shift_c;
                    norm_mantissa_guard_q<=norm_exponent!=0 && norm_mantissa_guard_input_c[0];
                    norm_mantissa_sticky_q<=norm_exponent>1 &&
                        |(norm_mantissa_input_c & ((64'h1 << (norm_exponent-1))-1));
                    norm_mantissa_lsb_q<=norm_mantissa_shift_c[0];
                    state<=NORM_MANT_RNE;
                end
                NORM_MANT_RNE: begin
                    norm_mantissa_rounded<=norm_mantissa_shift_q+
                        ((norm_mantissa_guard_q && (norm_mantissa_sticky_q || norm_mantissa_lsb_q)) ? 64'd1 : 64'd0);
                    state<=NORM_MANT_SAT;
                end
                NORM_MANT_SAT: begin
                    if(norm_mantissa_rounded<64'd16384) norm_mantissa<=16384;
                    else if(norm_mantissa_rounded>64'd65535) norm_mantissa<=65535;
                    else norm_mantissa<=norm_mantissa_rounded[16:0];
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
                NORM_NEWTON_T: begin
                    newton_t_shifted_q<=newton_t_shift_c;
                    newton_t_neg_q<=newton_mul1[47];
                    newton_t_guard_q<=newton_mul1[47] ? newton_t_abs_c[27] : newton_mul1[27];
                    newton_t_sticky_q<=newton_mul1[47] ? |newton_t_abs_c[26:0] : |newton_mul1[26:0];
                    newton_t_lsb_q<=newton_t_shift_c[0];
                    state<=NORM_NEWTON_T_RNE;
                end
                NORM_NEWTON_T_RNE: begin
                    norm_t<=newton_t_neg_q
                        ? -$signed(newton_t_shifted_q + ((newton_t_guard_q && (newton_t_sticky_q || newton_t_lsb_q)) ? 48'd1 : 48'd0))
                        : $signed(newton_t_shifted_q + ((newton_t_guard_q && (newton_t_sticky_q || newton_t_lsb_q)) ? 48'd1 : 48'd0));
                    state<=NORM_NEWTON_R;
                end
                NORM_NEWTON_R: begin
                    newton_product<=norm_r0*(49152-norm_t);
                    state<=NORM_NEWTON_RNE_INPUT;
                end
                NORM_NEWTON_RNE_INPUT: begin
                    newton_r_shifted_q<=newton_r_shift_c;
                    newton_r_neg_q<=newton_product[63];
                    newton_r_guard_q<=newton_product[63] ? newton_r_abs_c[14] : newton_product[14];
                    newton_r_sticky_q<=newton_product[63] ? |newton_r_abs_c[13:0] : |newton_product[13:0];
                    newton_r_lsb_q<=newton_r_shift_c[0];
                    state<=NORM_NEWTON_RNE;
                end
                NORM_NEWTON_RNE: begin
                    norm_r1<=newton_r_neg_q
                        ? -$signed(newton_r_shifted_q + ((newton_r_guard_q && (newton_r_sticky_q || newton_r_lsb_q)) ? 64'd1 : 64'd0))
                        : $signed(newton_r_shifted_q + ((newton_r_guard_q && (newton_r_sticky_q || newton_r_lsb_q)) ? 64'd1 : 64'd0));
                    norm_element<=0;
                    state<=NORM_SCALE_MUL0;
                end
                NORM_SCALE_MUL0: begin
                    scale_ag<=$signed(a[norm_element])*$signed(gamma[norm_element]);
                    state<=NORM_SCALE_MUL1;
                end
                NORM_SCALE_MUL1: begin
                    scale_product<=scale_ag*norm_r1;
                    state<=NORM_SCALE_RNE_INPUT;
                end
                // Keep the variable-shift RNE cone off the product register's
                // timing path.  This register is intentionally separate from
                // scale_product: Vivado previously reported this endpoint as
                // the vector unit's critical 52-level path.
                NORM_SCALE_RNE_INPUT: begin
                    scale_round_input<=scale_product;
                    scale_neg_q<=scale_neg_c;
                    scale_shifted_q<=scale_shifted_c;
                    scale_guard_q<=scale_guard_c;
                    scale_sticky_q<=scale_sticky_c;
                    scale_lsb_q<=scale_lsb_c;
                    state<=NORM_SCALE_RNE;
                end
                NORM_SCALE_RNE: begin
                    scale_rounded<=scale_neg_q
                        ? -$signed(scale_shifted_q + ((scale_guard_q && (scale_sticky_q || scale_lsb_q)) ? 32'd1 : 32'd0))
                        : $signed(scale_shifted_q + ((scale_guard_q && (scale_sticky_q || scale_lsb_q)) ? 32'd1 : 32'd0));
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
                BR: state<=BW0; BW0: state<=BW1; BW1: state<=BR_WAIT2;
                BR_WAIT2: begin word_b<=buf_rdata; state<=ALIGN_PIPE; end
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
                SR: state<=SW0; SW0: state<=SW1; SW1: state<=SR_WAIT2;
                SR_WAIT2: begin
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
                // VMATMUL is deliberately scalar: buffer operands, multiply,
                // accumulation, and requantization each retire in a distinct
                // register stage.  This keeps every stage to one wide operator.
                MAT_A: state<=MAT_AW0;
                MAT_AW0: state<=MAT_AW1; MAT_AW1: state<=MAT_A_WAIT2;
                MAT_A_WAIT2: begin
                    if(mat_a_unsigned) mat_a_value<={1'b0,((mat_i*mat_a_stride+mat_k)&1) ? buf_rdata[15:8] : buf_rdata[7:0]};
                    else mat_a_value<=$signed(((mat_i*mat_a_stride+mat_k)&1) ? buf_rdata[15:8] : buf_rdata[7:0]);
                    state<=MAT_B;
                end
                MAT_B: state<=MAT_BW0;
                MAT_BW0: state<=MAT_BW1; MAT_BW1: state<=MAT_B_WAIT2;
                MAT_B_WAIT2: begin
                    if(mat_b_unsigned) mat_b_value<={1'b0,(((mat_transpose ? mat_j*mat_b_stride+mat_k : mat_k*mat_b_stride+mat_j)&1)) ? buf_rdata[15:8] : buf_rdata[7:0]};
                    else mat_b_value<=$signed((((mat_transpose ? mat_j*mat_b_stride+mat_k : mat_k*mat_b_stride+mat_j)&1)) ? buf_rdata[15:8] : buf_rdata[7:0]);
                    state<=MAT_MUL;
                end
                MAT_MUL: begin mat_product<=mat_a_value*mat_b_value; state<=MAT_ACC; end
                MAT_ACC: begin
                    if(mat_causal && mat_transpose && mat_j>mat_i) begin
                        mat_result<=0; state<=MAT_OUT_READ;
                    end else if(mat_k+1>=((mat_causal && mat_a_unsigned && !mat_transpose) ? mat_i+1 : mat_kdim)) begin
                        mat_acc<=mat_acc+mat_product; state<=MAT_CLIP;
                    end else begin
                        mat_acc<=mat_acc+mat_product; mat_k<=mat_k+1; state<=MAT_A;
                    end
                end
                MAT_CLIP: begin
                    if(mat_acc>32'sd8388607) mat_clipped<=32'sd8388607;
                    else if(mat_acc<-32'sd8388608) mat_clipped<=-32'sd8388608;
                    else mat_clipped<=mat_acc;
                    state<=MAT_SCALE_MUL;
                end
                MAT_SCALE_MUL: begin mat_scaled<=mat_clipped*$signed({1'b0,record[5]}); state<=MAT_SHIFT; end
                MAT_SHIFT: begin mat_shifted<=mat_scaled >>> record[6][4:0]; state<=MAT_SAT; end
                MAT_SAT: begin mat_result<=sat8(mat_shifted); mat_acc<=0; state<=MAT_OUT_READ; end
                MAT_OUT_READ: state<=MAT_OUT_WAIT0;
                MAT_OUT_WAIT0: state<=MAT_OUT_WAIT1; MAT_OUT_WAIT1: state<=MAT_OUT_WAIT2;
                MAT_OUT_WAIT2: begin
                    if((mat_i*mat_n+mat_j)&1) mat_out_word<={mat_result,buf_rdata[7:0]};
                    else mat_out_word<={buf_rdata[15:8],mat_result};
                    state<=MAT_WRITE;
                end
                MAT_WRITE: begin
                    mat_write_addr_q<=dst+((mat_i*mat_n+mat_j)>>1);
                    mat_write_data_q<=mat_out_word;
                    state<=MAT_WRITE_COMMIT;
                end
                MAT_WRITE_COMMIT: begin
                    mat_k<=0; mat_acc<=0;
                    if(mat_j+1>=mat_n) begin
                        mat_j<=0;
                        if(mat_i+1>=mat_m) state<=FINISH;
                        else begin mat_i<=mat_i+1; state<=MAT_A; end
                    end else begin mat_j<=mat_j+1; state<=MAT_A; end
                end
                // VLAYOUT clears the complete destination (including tile
                // padding) then copies one code per pass through the port.
                LAY_ZERO: begin
                    if(lay_zero_word+1>=lay_dest_words) begin lay_r<=0; lay_c<=0; state<=LAY_READ; end
                    else lay_zero_word<=lay_zero_word+1;
                end
                LAY_READ: state<=LAY_READ_WAIT0;
                LAY_READ_WAIT0: state<=LAY_READ_WAIT1; LAY_READ_WAIT1: state<=LAY_READ_WAIT2;
                LAY_READ_WAIT2: begin
                    lay_value<=$signed((lay_src_index()&1) ? buf_rdata[15:8] : buf_rdata[7:0]);
                    state<=LAY_OUT_READ;
                end
                LAY_OUT_READ: state<=LAY_OUT_WAIT0;
                LAY_OUT_WAIT0: state<=LAY_OUT_WAIT1; LAY_OUT_WAIT1: state<=LAY_OUT_WAIT2;
                LAY_OUT_WAIT2: begin
                    if(lay_dst_index()&1) lay_out_word<={lay_value,buf_rdata[7:0]};
                    else lay_out_word<={buf_rdata[15:8],lay_value};
                    state<=LAY_WRITE;
                end
                LAY_WRITE: begin
                    if(lay_c+1>=lay_cols) begin
                        lay_c<=0;
                        if(lay_r+1>=lay_rows) state<=FINISH;
                        else begin lay_r<=lay_r+1; state<=LAY_READ; end
                    end else begin lay_c<=lay_c+1; state<=LAY_READ; end
                end
                FINISH: begin busy<=0; done<=1; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
